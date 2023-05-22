# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from math import ceil, floor
import random

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig, 
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyOTMixCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    use_kl: bool = field(
        default=False,
        metadata={"help:": "use kl loss"},
    )
    use_ot: bool = field(
        default=False,
        metadata={"help:": "use ot for mixup"},
    )
    ot_loss: bool = field(
        default=False,
        metadata={"help:": "use ot loss"},
    )
    kl_weight: float = field(
        default=1,
        metadata={"help:": "kl loss weight"},
    )
    kl_st: bool = field(
        default=False,
        metadata={"help:": "kl loss for st and mixed sequence"},
    )
    kl_mt: bool = field(
        default=False,
        metadata={"help:": "kl loss for mt and mixed sequence"},
    )
    ot_weight: float = field(
        default=0.1,
        metadata={"help:": "ot loss weight"},
    )
    ot_position: str = field(
        default="encoder_out",
        metadata={"help:": "position of ot, choose from encoder_in, encoder_out"},
    )
    ot_type: str = field(
        default="L2",
        metadata={"help:": "type of ot, choose from L2, cosine"},
    )
    ot_window: bool = field(
        default=False,
        metadata={"help:": "use window strategy for ot"},
    )
    ot_window_size: int = field(
        default=1,
        metadata={"help:": "window size for OT loss"},
    )
    mix_prob: float = field(
        default=0.2,
        metadata={"help:": "probability of mixing the two sentences"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_otmix", dataclass=LabelSmoothedCrossEntropyOTMixCriterionConfig
)
class LabelSmoothedCrossEntropyOTMixCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        use_kl=False,
        use_ot=False,
        ot_loss=False,
        kl_weight=1,
        kl_st=False,
        kl_mt=False,
        ot_weight=0.1,
        ot_position="encoder_out",
        ot_type="L2",
        ot_window=False,
        ot_window_size=1,
        mix_prob=0.2,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.use_kl = use_kl
        self.use_ot = use_ot
        self.ot_loss = ot_loss
        self.kl_weight = kl_weight
        self.kl_st = kl_st
        self.kl_mt = kl_mt
        self.ot_weight = ot_weight
        self.ot_position = ot_position
        self.ot_type = ot_type
        self.ot_window = ot_window
        self.ot_window_size = ot_window_size
        self.mix_prob = mix_prob

    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        audio_output = model(**audio_input)
        out = audio_output[1]["encoder_out"]
        audio_embedding, audio_encoder_out, audio_padding_mask = out["encoder_embedding"][0], out["encoder_out"][0], out["encoder_padding_mask"][0]
        loss, _ = self.compute_loss(model, audio_output, sample, reduce=reduce)
        return loss, out, audio_output, audio_embedding, audio_encoder_out.transpose(0, 1), audio_padding_mask
    
    def forward_mt(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        text_output = model(**text_input)
        out = text_output[1]["encoder_out"]
        text_embedding, text_encoder_out, text_padding_mask = out["encoder_embedding"][0], out["encoder_out"][0], out["encoder_padding_mask"][0]
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss, out, text_output, text_embedding, text_encoder_out.transpose(0, 1), text_padding_mask
    
    def forward_decoder(self, model, sample, encoder_out, reduce):
        decoder_input = {
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "encoder_out": encoder_out,
        }
        decoder_output = model.decoder(**decoder_input)
        loss, _ = self.compute_loss(model, decoder_output, sample, reduce=reduce)
        return loss, decoder_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        mix_loss = torch.Tensor([0]).cuda()
        kl_loss, kl_s, kl_t = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        ot_loss = torch.Tensor([0]).cuda()
        st_size, mt_size, mix_size, kl_size, ot_size = 0, 0, 0, 0, 0
        st_output, mt_output = None, None
        if self.training:
            st_loss, st_enc_out, st_output, audio_embedding, audio_encoder_out, audio_padding_mask = self.forward_st(model, sample, reduce) # encoder_out & embedding: [B, T, C], padding_mask: [B, T]
            mt_loss, mt_enc_out, mt_output, text_embedding, text_encoder_out, text_padding_mask = self.forward_mt(model, sample, reduce)
            st_lprobs, _, st_probs = self.get_lprobs_and_target(model, st_output, sample)
            mt_lprobs, _, mt_probs = self.get_lprobs_and_target(model, mt_output, sample)
            loss = st_loss + mt_loss
            if self.use_kl and self.use_ot:
                seq_len = self.get_seq_len(sample, st_output, "tgt_len").to(loss.device)
                ot_st, ot_st_loss = self.get_ot_matrix(audio_embedding, audio_padding_mask, text_embedding, text_padding_mask, seq_len) # [B, T]
                # ot_st, ot_st_loss = self.get_ot_matrix(audio_encoder_out, audio_padding_mask, text_encoder_out, text_padding_mask, seq_len) # [B, T]
                mixup = self.get_mixed_sequence(x=audio_encoder_out, y=text_encoder_out, ot=ot_st, p=self.mix_prob) # [B, T, C]
                mix_encoder_out = st_enc_out
                mix_encoder_out["encoder_out"][0] = mixup.transpose(0, 1)
                mix_loss, mix_output = self.forward_decoder(model, sample, mix_encoder_out, reduce)
                mix_lprobs, _, mix_probs = self.get_lprobs_and_target(model, mix_output, sample)
                kl_s = 0.5 * (self.compute_kl_loss(mix_lprobs, st_probs) + self.compute_kl_loss(st_lprobs, mix_probs))
                kl_t = 0.5 * (self.compute_kl_loss(mix_lprobs, mt_probs) + self.compute_kl_loss(mt_lprobs, mix_probs))
                if self.kl_st:
                    kl_loss = kl_loss + kl_s
                if self.kl_mt:
                    kl_loss = kl_loss + kl_t
                loss = loss + self.kl_weight * kl_loss
                if self.ot_loss:
                    ot_loss = ot_st_loss
                    loss = loss + self.ot_weight * ot_loss
            elif self.use_kl:
                kl_s = self.compute_kl_loss(st_lprobs, mt_probs)
                kl_t = self.compute_kl_loss(mt_lprobs, st_probs)
                kl_loss = 0.5 * (kl_s + kl_t)
                loss = loss + self.kl_weight * kl_loss
            else:
                pass
            st_size = mt_size = kl_size = mix_size = sample_size = sample["ntokens"]
        else:
            st_loss, st_enc_out, st_output, audio_embedding, audio_encoder_out, audio_padding_mask = self.forward_st(model, sample, reduce)
            loss = st_loss
            st_size = sample_size = sample["ntokens"]

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "st_sample_size": st_size,
            "mt_loss": mt_loss.data,
            "mt_sample_size": mt_size,
            "mix_loss": mix_loss.data,
            "mix_sample_size": mix_size,
            "kl_loss": kl_loss.data,
            "kl_s": kl_s.data,
            "kl_t": kl_t.data,
            "kl_sample_size": kl_size,
            "ot_loss": ot_loss.data,
            "ot_sample_size": ot_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, st_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = probs.log()
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), probs.view(-1, probs.size(-1))

    def get_seq_len(self, sample, audio_output, seq_len_type="none"):
        if seq_len_type == "none":
            bsz = sample["target"].size(0)
            return torch.ones(bsz)
        elif seq_len_type == "audio_len":
            padding_mask = audio_output["encoder_out"]["encoder_padding_mask"][0]
            padding_mask = (~padding_mask).float()
            seq_len = padding_mask.sum(dim=1)
            return seq_len
        elif seq_len_type == "src_len":
            seq_len = sample["net_input"]["source_lengths"].float()
            return seq_len
        elif seq_len_type == "tgt_len":
            seq_len = sample["target_lengths"].float()
            return seq_len

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target, _ = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def distance(self, x, y, type):
        len1, len2 = x.size(-2), y.size(-2)
        bsz, dim = x.size(0), x.size(-1)
        tx = x.unsqueeze(dim=-2).expand(bsz, len1, len2, dim)
        ty = y.unsqueeze(dim=-3).expand(bsz, len1, len2, dim)
        if type == "L2":
            dist = torch.linalg.norm(tx - ty, dim=-1)
            return dist
        else:
            sim = F.cosine_similarity(tx, ty, dim=-1)
            return 1. - sim

    def get_ot_matrix(self, x, x_padding_mask, y, y_padding_mask, seq_len):
        # x: [B, T1, C], x_padding_mask: [B, T1]
        # y: [B, T2, C], y_padding_mask: [B, T2]
        x_len = (~x_padding_mask).float().sum(dim=1)
        y_len = (~y_padding_mask).float().sum(dim=1)
        dist = self.distance(x, y, type=self.ot_type)
        dist = dist.masked_fill(x_padding_mask.unsqueeze(-1), 6e4).masked_fill(y_padding_mask.unsqueeze(-2), 6e4)
        weight = torch.norm(x, dim=-1) / torch.norm(x, dim=-1).sum(dim=-1, keepdim=True)
        weight = weight.masked_fill(x_padding_mask, 0.)
        if self.ot_window:
            window_mask = torch.zeros_like(dist) # [B, T1, T2]
            for i in range(dist.size(0)):
                ratio = x_len[i] / y_len[i]
                window = self.ot_window_size
                for j in range(dist.size(1)):
                    mid = j / ratio
                    window_mask[i, j, max(0, floor(mid - window)): min(dist.size(2), ceil(mid + window))] = 1.
            window_mask = window_mask.masked_fill(x_padding_mask.unsqueeze(-1), 0.).masked_fill(y_padding_mask.unsqueeze(-2), 0.)
            window_mask = ~window_mask.to(bool)
            dist = dist.masked_fill(window_mask, 6e4)
        ot = dist.min(dim=-1)[1]
        ot_loss = dist.min(dim=-1)[0] * weight.detach().clone() # dist: [B, T1, T2], weight: [B, T1]
        ot_loss = (ot_loss.sum(dim=-1) * seq_len).sum() # [B x T1] -> [B] -> []
        return ot, ot_loss

    def get_mixed_sequence(self, x, y, ot, p=0.2):
        # x: [B, T1, C], y: [B, T2, C], ot: [B, T1]
        mixed = torch.zeros_like(x)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                if random.random() < 1 - p:
                    mixed[i, j, :] = x[i, j, :]
                else:
                    mixed[i, j, :] = y[i, ot[i, j], :]
        return mixed

    def compute_kl_loss(self, xl, y):
        kl = F.kl_div(xl, y, reduction="none")
        return kl.sum()

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target, _ = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        mix_loss_sum = sum(log.get("mix_loss", 0) for log in logging_outputs)
        mix_sample_size = sum(log.get("mix_sample_size", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        kl_s_sum = sum(log.get("kl_s", 0) for log in logging_outputs)
        kl_t_sum = sum(log.get("kl_t", 0) for log in logging_outputs)
        ot_loss_sum = sum(log.get("ot_loss", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        kl_sample_size = sum(log.get("kl_sample_size", 0) for log in logging_outputs)
        ot_sample_size = sum(log.get("ot_sample_size", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "mix_loss", mix_loss_sum / mix_sample_size / math.log(2) if mix_sample_size != 0 else 0, mix_sample_size, round=3
        )
        metrics.log_scalar(
            "kl_loss", kl_loss_sum / kl_sample_size / math.log(2) if kl_sample_size != 0 else 0, kl_sample_size, round=3
        )
        metrics.log_scalar(
            "kl_s", kl_s_sum / kl_sample_size / math.log(2) if kl_sample_size != 0 else 0, kl_sample_size, round=3
        )
        metrics.log_scalar(
            "kl_t", kl_t_sum / kl_sample_size / math.log(2) if kl_sample_size != 0 else 0, kl_sample_size, round=3
        )
        metrics.log_scalar(
            "ot_loss", ot_loss_sum / ot_sample_size / math.log(2) if ot_sample_size != 0 else 0, ot_sample_size, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
