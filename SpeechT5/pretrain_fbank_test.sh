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
n_cluster=500 # Default is 500 from the SpeechT5 paper
km_path=${data_dir}/kmean100h #kmeans_model.pt
lm_data_dir=${data_dir}/raw/librispeech-lm-corpus
spm_model=${ckpt_dir}/spm_char.model
expdir=${root_dir}/exp

stage=1
stop_stage=1

tsv_dir=${data_dir}/tsv
feat_dir=${data_dir}/hubert_features #hubert_features
text_dir=${data_dir}/text
nshard=1
lab_dir=${data_dir}/hubert_km_labels #hubert_km_labels

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Run the pre-training script..."
    # JOBID=$(date +%Y%m%d%H%M%S)
    DATA_ROOT=${data_dir}/pretrain
    SAVE_DIR=${expdir}/pretrain/fbank_test
    LABEL_DIR=${lab_dir}
    TRAIN_SET="speech_train|train"
    VALID_SET="speech_valid|valid"

    mkdir -p ${SAVE_DIR}

    fairseq-train ${DATA_ROOT} \
        --save-dir ${SAVE_DIR} \
        --tensorboard-logdir ${SAVE_DIR} \
        --train-subset ${TRAIN_SET} \
        --valid-subset ${VALID_SET} \
        --hubert-label-dir ${LABEL_DIR} \
        --distributed-world-size 1 \
        --distributed-port 0 \
        --ddp-backend legacy_ddp \
        --user-dir speecht5 \
        --encoder-speech-prenet mel \
        --log-format json \
        --seed 1337 \
        --fp16 \
        \
        --task speecht5 \
        --t5-task pretrain \
        --label-rates 50 \
        --sample-rate 16000 \
        --random-crop \
        \
        --num-workers 10 \
        --max-tokens 4000000 \
        --max-speech-sample-size 250000 \
        --update-freq 4 \
        --batch-ratio "[1,0.0086]" \
        \
        --criterion speecht5 \
        --optimizer adam \
        --reset-optimizer \
        --adam-betas "(0.9, 0.98)" \
        --adam-eps 1e-06 \
        --weight-decay 0.01 \
        --power 1 \
        --clip-norm 5.0 \
        --lr 0.0002 \
        --lr-scheduler polynomial_decay \
        \
        --max-update 400000 \
        --warmup-updates 64000 \
        --total-num-update 400000 \
        --save-interval-updates 3000 \
        --skip-invalid-size-inputs-valid-test \
        --required-batch-size-multiple 1 \
        \
        --arch t5_transformer_base \
        --share-input-output-embed \
        --find-unused-parameters \
        --bert-init \
        --relative-position-embedding \
        --use-codebook \
        --codebook-prob 0.1 \
        --loss-weights="[10,0.1]" \
        --max-text-positions 600
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Continue pre-training script..."
    # JOBID=$(date +%Y%m%d%H%M%S)
    DATA_ROOT=${data_dir}/pretrain
    SAVE_DIR=${expdir}/pretrain/base
    LABEL_DIR=${lab_dir}
    TRAIN_SET="speech_train|train"
    VALID_SET="speech_valid|valid"

    mkdir -p ${SAVE_DIR}

    fairseq-train ${DATA_ROOT} \
        --restore-file checkpoint_last.pt \
        --save-dir ${SAVE_DIR} \
        --tensorboard-logdir ${SAVE_DIR} \
        --train-subset ${TRAIN_SET} \
        --valid-subset ${VALID_SET} \
        --hubert-label-dir ${LABEL_DIR} \
        --distributed-world-size 4\
        --encoder-speech-prenet mel \
        --distributed-port 0 \
        --ddp-backend legacy_ddp \
        --user-dir speecht5 \
        --log-format json \
        --seed 1337 \
        --fp16 \
        \
        --task speecht5 \
        --t5-task pretrain \
        --label-rates 50 \
        --sample-rate 16000 \
        --random-crop \
        \
        --num-workers 20 \
        --max-tokens 5000000 \
        --max-speech-sample-size 300000 \
        --update-freq 2 \
        --batch-ratio "[1,0.0086]" \
        \
        --criterion speecht5 \
        --optimizer adam \
        --adam-betas "(0.9, 0.98)" \
        --adam-eps 1e-06 \
        --weight-decay 0.01 \
        --power 1 \
        --clip-norm 5.0 \
        --lr 0.0002 \
        --lr-scheduler polynomial_decay \
        \
        --max-update 300000 \
        --warmup-updates 64000 \
        --total-num-update 300000 \
        --save-interval-updates 3000 \
        --skip-invalid-size-inputs-valid-test \
        --required-batch-size-multiple 1 \
        \
        --arch t5_transformer_base \
        --share-input-output-embed \
        --find-unused-parameters \
        --bert-init \
        --relative-position-embedding \
        --use-codebook \
        --codebook-prob 0.1 \
        --loss-weights="[10,0.1]" \
        --max-text-positions 600
fi

# python -m pdb $(which fairseq-train) ${DATA_ROOT} \
#     --save-dir ${SAVE_DIR} \
#     --tensorboard-logdir ${SAVE_DIR} \
#     --train-subset ${TRAIN_SET} \
#     --valid-subset ${VALID_SET} \
#     --hubert-label-dir ${LABEL_DIR} \
#     --distributed-world-size 1 \
#     --distributed-port 0 \
#     --ddp-backend legacy_ddp \
#     --user-dir speecht5 \
#     --log-format json \
#     --seed 1337 \
#     --fp16 \
#     --task speecht5 \
#     --t5-task pretrain \
#     --label-rates 50 \
#     --sample-rate 16000 \
#     --random-crop \
#     --num-workers 30 \
#     --max-tokens 1400000 \
#     --max-speech-sample-size 250000 \
#     --update-freq 2 \
#     --batch-ratio "[1,0.0086]" \
#     --criterion speecht5 \
#     --optimizer adam \
#     --reset-optimizer \
#     --adam-betas "(0.9, 0.98)" \
#     --adam-eps 1e-06 \
#     --weight-decay 0.01 \
#     --power 1 \
#     --clip-norm 5.0 \
#     --lr 0.0002 \
#     --lr-scheduler polynomial_decay \
#     --max-update 800000 \
#     --warmup-updates 64000 \
#     --total-num-update 800000 \
#     --save-interval-updates 3000 \
#     --skip-invalid-size-inputs-valid-test \
#     --required-batch-size-multiple 1 \
#     --arch t5_transformer_base \
#     --share-input-output-embed \
#     --find-unused-parameters \
#     --bert-init \
#     --relative-position-embedding \
#     --use-codebook \
#     --codebook-prob 0.1 \
#     --loss-weights="[10,0.1]" \
#     --max-text-positions 600




# python $(which fairseq-train) ${DATA_ROOT} \
#     --save-dir ${SAVE_DIR} \
#     --tensorboard-logdir ${SAVE_DIR} \
#     --train-subset ${TRAIN_SET} \
#     --valid-subset ${VALID_SET} \
#     --hubert-label-dir ${LABEL_DIR} \
#     --encoder-speech-prenet mel \
#     --distributed-world-size 1 \
#     --distributed-port 0 \
#     --ddp-backend legacy_ddp \
#     --user-dir speecht5 \
#     --log-format json \
#     --seed 1337 \
#     --fp16 \
#     --task speecht5 \
#     --t5-task pretrain \
#     --label-rates 50 \
#     --sample-rate 16000 \
#     --random-crop \
#     --num-workers 20 \
#     --max-tokens 5000000 \
#     --max-speech-sample-size 300000 \
#     --update-freq 2 \
#     --batch-ratio "[1,0.0086]" \
#     --criterion speecht5 \
#     --optimizer adam \
#     --reset-optimizer \
#     --adam-betas "(0.9, 0.98)" \
#     --adam-eps 1e-06 \
#     --weight-decay 0.01 \
#     --power 1 \
#     --clip-norm 5.0 \
#     --lr 0.0002 \
#     --lr-scheduler polynomial_decay \
#     --max-update 800000 \
#     --warmup-updates 64000 \
#     --total-num-update 800000 \
#     --save-interval-updates 3000 \
#     --skip-invalid-size-inputs-valid-test \
#     --required-batch-size-multiple 1 \
#     --arch t5_transformer_base \
#     --share-input-output-embed \
#     --find-unused-parameters \
#     --bert-init \
#     --relative-position-embedding \
#     --use-codebook \
#     --codebook-prob 0.1 \
#     --loss-weights="[10,0.1]" \
#     --max-text-positions 600