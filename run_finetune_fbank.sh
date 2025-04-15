#!/usr/bin/env bash

#SBATCH --job-name=speecht5ft
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task 20
#SBATCH --mem=60GB
#SBATCH -o output-%j.txt #standard output file
#SBATCH -e errors-%j.txt #standard error file

. ~/.bashrc
echo "Running a process on $(hostname)"
conda activate t5 
echo "conda env: $CONDA_DEFAULT_ENV"
echo "python: $(which python)"

cd /export/fs06/ahussei6/multimodal/SpeechT5/SpeechT5
# bash pretrain.sh
# bash asr_finetune_fbank.sh
bash inference.sh 