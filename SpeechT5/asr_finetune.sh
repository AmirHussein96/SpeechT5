#!/usr/bin/env bash

set -eou pipefail

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


root_dir=/export/fs06/ahussei6/multimodal
data_dir=${root_dir}/data
ckpt_dir=${root_dir}/models
lm_data_dir=${data_dir}/raw/librispeech-lm-corpus
spm_model=${ckpt_dir}/spm_char.model
expdir=${root_dir}/exp

stage=1
stop_stage=1

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Run the pre-training script..."
    JOBID=$(date +%Y%m%d%H%M%S)
    # JOBID=speecht5_base_down
    DATA_ROOT=${data_dir}/pretrain100
    SAVE_DIR=${expdir}/finetune100_base/${JOBID}
    TRAIN_SET="speech_train100"
    VALID_SET="dev_clean"
    LABEL_DIR=$DATA_ROOT
    BPE_TOKENIZER=$spm_model #double check this
    USER_DIR=speecht5
    # PT_CHECKPOINT_PATH=$root_dir/exp/pretrain2/20250102141244/checkpoint_best.pt
    # PT_CHECKPOINT_PATH=$root_dir/my_models/speecht5_base.pt
    PT_CHECKPOINT_PATH=$root_dir/exp/pretrain/base/checkpoint_best.pt    # the base model is our pretained version for 200k steps.


    mkdir -p ${SAVE_DIR}
    fairseq-train ${DATA_ROOT} \
        --save-dir ${SAVE_DIR} \
        --tensorboard-logdir ${SAVE_DIR} \
        --train-subset ${TRAIN_SET} \
        --valid-subset ${VALID_SET} \
        --hubert-label-dir ${LABEL_DIR} \
        --distributed-world-size 4 \
        --distributed-port 0 \
        --ddp-backend legacy_ddp \
        --user-dir speecht5 \
        --log-format json \
        --seed 1 \
        --fp16 \
        \
        --task speecht5 \
        --t5-task s2t \
        --sample-rate 16000 \
        --num-workers 20 \
        --max-tokens 3200000 \
        --update-freq 2 \
        --bpe-tokenizer ${BPE_TOKENIZER} \
        \
        --criterion speecht5 \
        --report-accuracy \
        --zero-infinity \
        --ce-weight 0.5 \
        --ctc-weight 0.5 \
        --sentence-avg \
        \
        --optimizer adam \
        --adam-betas "(0.9, 0.98)" \
        --adam-eps 1e-08 \
        --weight-decay 0.1 \
        --clip-norm 25.0 \
        --lr 0.00006 \
        --lr-scheduler tri_stage \
        --phase-ratio "[0.1, 0.4, 0.5]" \
        --final-lr-scale 0.05 \
        \
        --max-update 80000 \
        --max-text-positions 600 \
        --required-batch-size-multiple 1 \
        --save-interval-updates 3000 \
        --skip-invalid-size-inputs-valid-test \
        \
        --arch t5_transformer_base_asr \
        --share-input-output-embed \
        --find-unused-parameters \
        --bert-init \
        --relative-position-embedding \
        --freeze-encoder-updates 13000 \
        \
        --keep-last-epochs 10 \
        --feature-grad-mult 1.0 \
        --best-checkpoint-metric s2t_accuracy \
        --maximize-best-checkpoint-metric \
        --finetune-from-model ${PT_CHECKPOINT_PATH}
fi
