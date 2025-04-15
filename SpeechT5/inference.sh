#!/bin/bash

root_dir=/export/fs06/ahussei6/multimodal
data_dir=${root_dir}/data
ckpt_dir=${root_dir}/models
spm_model=${ckpt_dir}/spm_char.model
expdir=${root_dir}/exp


# CHECKPOINT_PATH=/export/fs06/ahussei6/multimodal/exp/finetune100/20250109184810/checkpoint_best.pt # finetuned from the downloaded model
# CHECKPOINT_PATH=/export/fs06/ahussei6/multimodal/exp/finetune100_base/20250122100247/checkpoint_best.pt  # finetuned from the base pretrained model 200k iterations
CHECKPOINT_PATH=/export/fs06/ahussei6/multimodal/exp/finetune100_fbank/20250330120404/checkpoint_best.pt 
DATA_ROOT=${data_dir}/pretrain100
SUBSETS="dev_clean dev_other test-clean test-other"  # List of subsets
# SUBSETS="test-other"  # List of subsets
BPE_TOKENIZER=$spm_model
LABEL_DIR=$DATA_ROOT
USER_DIR=speecht5
BEAM=10 #10
MAX_TOKENS=4000000
BATCH_SIZE=
CTC_WEIGHT=0
LM_WEIGHT=0
JOBID=$(date +%Y%m%d%H%M%S)

# Loop over all subsets
for SUBSET in $SUBSETS; do
  echo "Processing subset: ${SUBSET}"
  

 if [ "$CTC_WEIGHT" != "0" ]; then
    SAVE_DIR=${expdir}/inference100/ctc_${CTC_WEIGHT}
    mkdir -p ${SAVE_DIR}
    MAX_TOKENS=
    BATCH_SIZE=1
    echo "max tokens: ${MAX_TOKENS}"
    echo "batch size: ${BATCH_SIZE}"
    fairseq-generate ${DATA_ROOT} \
      --gen-subset ${SUBSET} \
      --bpe-tokenizer ${BPE_TOKENIZER} \
      --user-dir ${USER_DIR} \
      --task speecht5 \
      --t5-task s2t \
      --model-parallel-size 1 \
      --path ${CHECKPOINT_PATH} \
      --hubert-label-dir ${LABEL_DIR} \
      --ctc-weight ${CTC_WEIGHT} \
      --lm-weight ${LM_WEIGHT} \
      --batch-size ${BATCH_SIZE} \
      --beam ${BEAM} \
      --scoring wer \
      --max-len-a 0 \
      --max-len-b 620 \
      --sample-rate 16000 \
      --num-workers 2 \
      | tail -n 1 > ${SAVE_DIR}/${SUBSET}_last_line.log
else
  # Run fairseq-generate and save only the last line of output
    SAVE_DIR=${expdir}/inference100/${CTC_WEIGHT}

    mkdir -p ${SAVE_DIR}
    fairseq-generate ${DATA_ROOT} \
    --gen-subset ${SUBSET} \
    --bpe-tokenizer ${BPE_TOKENIZER} \
    --user-dir ${USER_DIR} \
    --task speecht5 \
    --t5-task s2t \
    --model-parallel-size 1 \
    --path ${CHECKPOINT_PATH} \
    --hubert-label-dir ${LABEL_DIR} \
    --lm-weight ${LM_WEIGHT} \
    --beam ${BEAM} \
    --max-tokens ${MAX_TOKENS} \
    --scoring wer \
    --max-len-a 0 \
    --max-len-b 620 \
    --sample-rate 16000 \
    --num-workers 10 \
    | tail -n 1 > ${SAVE_DIR}/${SUBSET}_last_line.log
    #  \
  
  echo "Finished processing subset: ${SUBSET}. Last line saved to ${SAVE_DIR}/${SUBSET}_last_line.log"
  fi
done
