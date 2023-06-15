# CMOT

This is a PyTorch implementation for ACL 2023 main conference paper "CMOT: Cross-modal Mixup via Optimal Transport for Speech Translation".

## Dependencies

- Python version >= 3.8

- [Pytorch](http://pytorch.org/)

- torchaudio version >= 0.8.0

- **To install fairseq version 0.12.2** and develop locally:

  ```bash
  cd fairseq
  pip install --editable ./
  ```

## Train your own model

### 1. Data Preparation

#### MuST-C Dataset

Download [MuST-C](https://ict.fbk.eu/must-c/) v1.0 dataset. Place the dataset in `MUSRC_ROOT`.

#### HuBERT Model

Download [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) model.

#### WMT Dataset (optional)

Download WMT [13](https://www.statmt.org/wmt13/translation-task.html) / [14](https://www.statmt.org/wmt14/translation-task.html) / [16](https://www.statmt.org/wmt16/translation-task.html) dataset.

#### OPUS Dataset (optional)

Download [OPUS](http://opus.nlpl.eu/opus-100.php) dataset.

### 2. Preprocess

```bash
python cmot/preprocess/prep_mustc_data_joint.py \
  --tgt-lang ${LANG} --data-root ${MUSTC_ROOT} \
  --task st --yaml-filename config_st_raw_joint.yaml \
  --vocab-type unigram --vocab-size 10000 \
  --use-audio-input
```

### 3. MT Pretraining

We pretrain the model with 4 GPUs.

```bash
python fairseq/fairseq_cli/train.py ${DATA} \
    --no-progress-bar --fp16 --memory-efficient-fp16 \
    --arch transformer --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 --max-update 250000 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0 \
    --seed 1 --update-freq 1 \
    --log-interval 10 \
    --validate-interval 1 --save-interval 1 \
    --save-interval-updates 1000 --keep-interval-updates 10 \
    --save-dir ${MT_SAVE_DIR} --tensorboard-logdir ${LOG_DIR} \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend=legacy_ddp \
    |& tee -a ${LOG_DIR}/train.log
```

Here,

- `DATA` is the directory of preprocessed MT data,
- `MT_SAVE_DIR` denotes the directory to save the MT checkpoints,
- `LOG_DIR` denotes the directory to save logs.

### 4. Training

```bash
prob=0.2
kl_weight=2
python fairseq/fairseq_cli/train.py ${MUSTC_ROOT}/en-${LANG} \
    --no-progress-bar --fp16 --memory-efficient-fp16 \
    --config-yaml config_st_raw_joint.yaml --train-subset train_st_raw_joint --valid-subset dev_st_raw \
    --save-dir ${ST_SAVE_DIR} \
    --max-tokens 2000000 --max-source-positions 900000 --batch-size 32 --max-target-positions 1024  --max-tokens-text 4096 \
    --max-update 60000 --log-interval 10 --num-workers 4 \
    --task speech_and_text --criterion label_smoothed_cross_entropy_otmix \
    --use-kl --kl-st --kl-mt --kl-weight ${kl_weight} \
    --use-ot --ot-type L2 --ot-position encoder_out --ot-window --ot-window-size 10 --mix-prob ${prob} \
    --label-smoothing 0.1 --report-accuracy \
    --arch hubert_ot_post --layernorm-embedding --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --hubert-model-path ${HUBERT_MODEL} --mt-model-path ${MT_MODEL} \
    --clip-norm 0.0 --seed 1 --update-freq 2 \
    --tensorboard-logdir ${LOG_DIR} \
    --ddp-backend=legacy_ddp \
    --skip-invalid-size-inputs-valid-test \
    |& tee -a $LOG_DIR/train.log
```

Here,

- `MUSTC_ROOT` is the root directory of MuST-C dataset,
- `LANG` denotes language id (select from de / fr / ru / es / it / ro / pt / nl),
- `MT_MODEL` is the path of pretrained MT model,
- `ST_SAVE_DIR` denotes the directory to save the ST checkpoints,
- `LOG_DIR` denotes the directory to save logs.

We set `update-freq=2` to simulate 8 GPUs with 4 GPUs. 

### 5. Inference

First, average the checkpoints:

```bash
number=10
CHECKPOINT_FILENAME=checkpoint_avg${number}.pt
python fairseq/scripts/average_checkpoints.py \
    --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints ${number} \
    --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
```

Then run inference (taking en-de as an example):

```bash
LANG=de
ckpt=avg10
CHECKPOINT_FILENAME=checkpoint_${ckpt}.pt
lenpen=1.2
BEAM=8
python fairseq/fairseq_cli/generate.py ${MUSTC_ROOT}/en-${LANG} \
  --config-yaml config_st_raw_joint.yaml --gen-subset tst-COMMON_st_raw --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} --lenpen ${lenpen} \
  --max-tokens 1000000 --max-source-positions 1000000 --beam $BEAM --scoring sacrebleu \
  > $RES_DIR/$res.${ckpt}.lp${lenpen}.beam-${BEAM}
```

