#!/usr/bin/env bash

#SBATCH --job-name=speecht5
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task 80
#SBATCH --mem=200GB
#SBATCH -o output-%j.txt #standard output file
#SBATCH -e errors-%j.txt #standard error file

. ~/.bashrc
echo "Running a process on $(hostname)"
conda activate t5 
echo "conda env: $CONDA_DEFAULT_ENV"
echo "python: $(which python)"

cd /export/fs06/ahussei6/multimodal/SpeechT5/SpeechT5
# bash pretrain.sh
bash pretrain_fbank.sh