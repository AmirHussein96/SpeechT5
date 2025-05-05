#!/usr/bin/env bash

set -eou pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
root_dir=/export/fs06/ahussei6/multimodal
data_dir=${root_dir}/data   # directory to store the kmeans model, hubert kmeans labels
                            # hubert extracted features, xvectors.zip, additional text data
                            # and all the prepared data for training and finetuning
ckpt_dir=${root_dir}/models
n_cluster=500 # Default is 500 from the SpeechT5 paper
km_path=${data_dir}/kmean100h #${data_dir}/kmeans_model.pt
lm_data_dir=${data_dir}/raw/librispeech-lm-corpus   # additional text data
spm_model=${ckpt_dir}/spm_char.model
train_split=0.95
org_data_dir=/export/corpora5/LibriSpeech

stage=5
stop_stage=5

train_sets="train-clean-100 train-clean-360 train-other-500"    # specifying the training set
dev_sets="dev-clean dev-other"
test_sets="test-clean test-other"

tsv_dir=${data_dir}/tsv
feat_dir=${data_dir}/hubert_features
text_dir=${data_dir}/text

nshard=1
lab_dir=${data_dir}/hubert_km_labels
ckpt_path=${ckpt_dir}/hubert_base_ls960.pt
layer=6
rank=0
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then

    log "Stage 0: Prepare HuBERT features using the base model and layer ${layer}..."
    mkdir -p ${feat_dir}
    mkdir -p ${tsv_dir}/raw/valid
    mkdir -p ${tsv_dir}/raw/train

    # Process the valid set
    for split in ${dev_sets}; do
        
        # Check if the symlink already exists
        target=${org_data_dir}/${split} 
        link=${tsv_dir}/raw/valid/${split}
        if [ ! -L "$link" ]; then
            # Create a proxy directory for the split
            ln -sfv "$target" "$link"
        else
            echo "Symlink already exists: $link"
        fi
    done
    for split in ${test_sets}; do
        # Create a proxy directory for the split
        target=${org_data_dir}/${split} 
        link=${tsv_dir}/raw/test/${split}
        if [ ! -L "$link" ]; then
            # Create a proxy directory for the split
            ln -sfv "$target" "$link"
        else
            echo "Symlink already exists: $link"
        fi
         # prepare fairseq tsv 
        python SpeechT5/SpeechT5/fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/test/${split} --dest ${tsv_dir}/test --ext flac --valid-percent 0
        # # Rename the file for training
        cp ${tsv_dir}/test/train.tsv ${tsv_dir}/${split}.tsv
    done

    # prepare all valid sets (dev-clean and dev-other) and produce one tsv file
    python fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/valid --dest ${tsv_dir}/valid --ext flac --valid-percent 0
    # # Rename the file for training
    cp ${tsv_dir}/valid/train.tsv ${tsv_dir}/speech_valid.tsv

    # Process the training set
    for split in ${train_sets}; do
        # Create a proxy directory for the split
        target=${org_data_dir}/${split} 
        link=${tsv_dir}/raw/train/${split}
        if [ ! -L "$link" ]; then
            # Create a proxy directory for the split
            ln -sfv "$target" "$link"
        else
            echo "Symlink already exists: $link"
        fi
    done
    python SpeechT5/SpeechT5/fairseq/examples/wav2vec/wav2vec_manifest.py ${tsv_dir}/raw/train --dest ${tsv_dir}/train --ext flac --valid-percent 0
    # Rename the file for training
    cp ${tsv_dir}/train/train.tsv ${tsv_dir}/speech_train.tsv

    # Add speaker embedding to the last column
    # for split in "speech_train" "speech_valid"; do
    for split in "speech_train" "speech_valid"; do
        python SpeechT5/SpeechT5/scripts/integrate_spkembs.py \
            -i ${tsv_dir}/${split}.tsv \
            --dset librispeech \
            --xvectors ${data_dir}/xvectors.zip \
            -o ${tsv_dir}/${split}_spk.tsv

        # Generate the hubert features
        python SpeechT5/SpeechT5/fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Fit a k-means model to the HuBERT features..."
    python SpeechT5/SpeechT5/fairseq/examples/hubert/simple_kmeans/learn_kmeans.py \
        ${feat_dir} \
        speech_train \
        ${nshard} \
        ${km_path} \
        ${n_cluster} \
        --percent 0.08
    # python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py data/hubert_features speech_train 1 data/kmeans_model.pt 500 --percent 0.05 --max_iter 1
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Assign cluster IDs to the HuBERT features..."
    rank=0
    set -x

    for split in "speech_train" "speech_valid"; do
        python SpeechT5/SpeechT5/fairseq/examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
        for rank in $(seq 0 $((nshard - 1))); do
            cat $lab_dir/${split}_${rank}_${nshard}.km
        done >$lab_dir/${split}.km
    done

    # Generate the dict.km.txt file for the k-means labels on the training split
    python SpeechT5/SpeechT5/scripts/generate_dict.py \
        -i ${lab_dir}/speech_train.km \
        -o ${lab_dir}/dict.km.txt
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Prepare the text data..."
    mkdir -p ${text_dir}
    # Step 1: Dump the lowercased text data to a single txt file
    # Combine all .txt files recursively into a single file
    find ${lm_data_dir}/corpus -type f -name "*.txt" -exec cat {} + >${text_dir}/tmp.txt

    # Convert the combined file to lowercase
    tr '[:upper:]' '[:lower:]' <${text_dir}/tmp.txt >${text_dir}/tmp2.txt

    # Remove blank lines from the file
    grep -v '^$' ${text_dir}/tmp2.txt >${text_dir}/text

    # Optional: Remove the intermediate file if not needed
    rm ${text_dir}/tmp.txt
    rm ${text_dir}/tmp2.txt
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Prepare the text data..."
    # Perform train/dev split of the text data
    python SpeechT5/SpeechT5/scripts/train_dev_split.py \
        -i "${text_dir}/text" \
        -o "${text_dir}"

    # Tokenize the text data
    for split in "text_train" "text_valid"; do
        python SpeechT5/SpeechT5/fairseq/scripts/spm_encode.py \
            --model ${spm_model} \
            --output_format=piece \
            --inputs ${text_dir}/${split}.txt \
            --outputs ${text_dir}/${split}.token
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Prepare the text data using Fairseq..."
    DEST_DIR="${text_dir}/bins"
    DICT="/home/cxiao7/research/mult5/SpeechT5/SpeechT5/models/dict.txt"

    # Create destination directory if it doesn't exist
    mkdir -p "${DEST_DIR}"

    fairseq-preprocess \
        --only-source \
        --trainpref "${text_dir}/text_train.token" \
        --validpref "${text_dir}/text_valid.token" \
        --destdir "${DEST_DIR}" \
        --srcdict "${DICT}" \
        --workers 20
fi



# prepare the labels for finetuning:
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    python SpeechT5/SpeechT5/fairseq/examples/wav2vec/libri_labels.py $tsv_dir/speech_train.tsv --output-dir $tsv_dir --output-name speech_train_libri

    # python SpeechT5/SpeechT5/fairseq/examples/wav2vec/libri_labels.py $tsv_dir/speech_valid.tsv --output-dir $tsv_dir --output-name valid_trans
    # for split in "speech_valid" "dev_clean" "dev_other" "test_clean" "test_other"; do
     for split in "test-clean" "test-other"; do
        python SpeechT5/SpeechT5/fairseq/examples/wav2vec/libri_labels.py $tsv_dir/${split}.tsv --output-dir $tsv_dir --output-name ${split}
    done

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Stage 6: Finalize the pretrain data folder..."

    pretrain_data_dir=${data_dir}/pretrain
    mkdir -p ${pretrain_data_dir}

    # Link the text and speech pretrain data
    ln -sfv ${text_dir}/bins/* ${pretrain_data_dir}
    ln -sfv ${tsv_dir}/speech_valid_spk.tsv ${pretrain_data_dir}/speech_valid.tsv
    ln -sfv ${tsv_dir}/speech_train_spk.tsv ${pretrain_data_dir}/speech_train.tsv
    ln -sfv $tsv_dir/speech_valid.ltr $pretrain_data_dir
    ln -sfv $tsv_dir/speech_valid.wrd $pretrain_data_dir
    ln -sfv $tsv_dir/speech_train.ltr $pretrain_data_dir
    ln -sfv $tsv_dir/speech_train.wrd $pretrain_data_dir
    ln -sfv $tsv_dir/test* $pretrain_data_dir
    ln -sfv $tsv_dir/test-other.* $pretrain_data_dir
fi