#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from fairseq import checkpoint_utils, utils, tasks
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)

logger = logging.getLogger(__name__)


@register_model("hubert_ot")
class HuBERTOTModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config_yaml="config.yaml",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config_yaml,
            **kwargs,
        )
        return S2THubInterface(x["args"], x["task"], x["models"][0])

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--encoder-freezing-updates",
            type=int,
            metavar="N",
            help="freeze encoder for first N updates",
        )
        # HuBERT
        parser.add_argument(
            "--hubert-model-path",
            type=str,
            metavar="STR",
            help="path/to/hubert/model"
        )
        parser.add_argument(
            "--freeze-hubert",
            action="store_true",
            help="if we want to freeze the hubert features"
        )
        # MT
        parser.add_argument(
            "--mt-model-path",
            type=str,
            metavar="STR",
            help="path/to/mt/model"
        )

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        return HuBERTOTEncoder(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder_embed_tokens = decoder_embed_tokens
        args.tgt_dict_size = len(task.target_dictionary)
        encoder = cls.build_encoder(args, task, encoder_embed_tokens)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)

        mt_model_path = getattr(args, "mt_model_path", "")
        if mt_model_path != "":
            mt_model = cls.load_mt(cls, mt_model_path)
            encoder_dict = OrderedDict()
            decoder_dict = OrderedDict()
            for key in mt_model:
                if key.startswith("encoder"):
                    if key[len("encoder") + 1 :].startswith("layers"):
                        _key = "transformer_" + key[len("encoder") + 1 :]
                    else:
                        _key = key[len("encoder") + 1 :]
                    encoder_dict[_key] = mt_model[key]
                if key.startswith("decoder"):
                    _key = key[len("decoder") + 1 :]
                    decoder_dict[_key] = mt_model[key]
            encoder.load_state_dict(encoder_dict, strict=False)
            decoder.load_state_dict(decoder_dict, strict=False)

        return cls(encoder, decoder)

    def load_mt(self, mt_path):
        ckpt = checkpoint_utils.load_checkpoint_to_cpu(mt_path)
        mt_model = ckpt["model"]
        return mt_model

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def get_ctc_target(self, sample: Optional[Dict[str, Tensor]]):
        return sample["target"], sample["target_lengths"]

    def get_ctc_output(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        sample: Optional[Dict[str, Tensor]],
    ):
        encoder_out = net_output[1]["encoder_out"]["encoder_out"][0]
        logits = self.encoder.ctc_proj(encoder_out)  # T x B x C
        out = utils.log_softmax(logits.float(), dim=-1)
        padding_mask = net_output[1]["encoder_out"]["encoder_padding_mask"]
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens

    def forward(self, src_tokens, src_lengths, prev_output_tokens, mode="st"):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, mode=mode)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class HuBERTOTEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, dictionary=None, embed_tokens=None):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = dictionary.pad()
        
        # HuBERT model
        self.hubert_model_path = getattr(args, "hubert_model_path", None)
        assert self.hubert_model_path is not None
        self.hubert_model, self.hubert_args = self.load_hubert(self.hubert_model_path)
        self.freeze_hubert = getattr(args, "freeze_hubert", False)
        if self.freeze_hubert:
            for param in self.hubert_model.parameters():
                param.requires_grad = False

        if args.conv_kernel_sizes:
            self.subsampler = Conv1dSubsampler(
                self.hubert_args.model.encoder_embed_dim,
                args.conv_channels,
                args.decoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
        else:
            self.subsampler = None

        self.embed_tokens = embed_tokens
        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_tokens.embedding_dim, export=export)
        else:
            self.layernorm_embedding = None
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.ctc_proj = None
        if getattr(args, "ctc_weight", 0.0) > 0.0:
            self.ctc_proj = nn.Linear(args.encoder_embed_dim, args.tgt_dict_size)

    def load_hubert(self, ckpt):
        ckpt = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
        hubert_args = ckpt["cfg"]
        hubert_task = tasks.setup_task(hubert_args.task)
        if "task_state" in ckpt:
            hubert_task.load_state_dict(ckpt["task_state"])
        model = hubert_task.build_model(hubert_args.model)
        model.load_state_dict(ckpt["model"], strict=True)
        model.remove_pretraining_modules()
        return model, hubert_args

    def get_hubert_features(self, audio, audio_lengths):
        padding_mask = lengths_to_padding_mask(audio_lengths)
        hubert_args = {
            "source": audio,
            "padding_mask": padding_mask,
            "mask": False,
        } 
        x, padding_mask = self.hubert_model.extract_features(**hubert_args)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        return x, padding_mask, output_length

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def _forward(self, src_tokens, src_lengths, mode="st", return_all_hiddens=False):
        if mode == "st":
            x, encoder_padding_mask, input_lengths = self.get_hubert_features(src_tokens, src_lengths)
            if self.subsampler is not None:
                x, input_lengths = self.subsampler(x, input_lengths)
                encoder_padding_mask = lengths_to_padding_mask(input_lengths)
                x = x.transpose(0, 1) # T x B x C -> B x T x C
            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)
            x = self.dropout_module(x)
        elif mode == "mt": # mt
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
            has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
            x, _ = self.forward_embedding(src_tokens)
            if has_pads:
                x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        else: # mixup
            x = src_tokens
            encoder_padding_mask = lengths_to_padding_mask(src_lengths)

        encoder_embedding = x
        x = x.transpose(0, 1)  # B x T x C -> T x B x C

        encoder_states = []

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward(self, src_tokens, src_lengths, mode="st", return_all_hiddens=False):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens, src_lengths, mode, return_all_hiddens=return_all_hiddens
                )
        else:
            x = self._forward(
                src_tokens, src_lengths, mode, return_all_hiddens=return_all_hiddens
            )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        extra = {"encoder_out": encoder_out} if incremental_state is None else None
        return x, extra


@register_model_architecture(model_name="hubert_ot", arch_name="hubert_ot_pre")
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # HuBERT
    args.hubert_model_path = getattr(args, "hubert_model_path", "")
    args.freeze_hubert = getattr(args, "freeze_hubert", False)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # MT model
    args.mt_model_path = getattr(args, "mt_model_path", "")
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)


@register_model_architecture(model_name="hubert_ot", arch_name="hubert_ot_post")
def hubert_ot_post(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    base_architecture(args)
