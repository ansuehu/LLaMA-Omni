#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=Extract_code # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=4 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=3-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/extract-code.out.log
#SBATCH --error=.slurm/extract-code.err.log
#SBATCH --chdir=/home/andoni.sudupe/LLaMA-Omni/


python dataset_generation/extract_units.py \
    --dataset_repo Ansu/Instruct_S2S_eu \
    --audio_path /scratch/andoni.sudupe/Instruct_S2S_eu/answers/ \
    --kmeans_path /scratch/andoni.sudupe/Instruct_S2S_eu/kmeans/basque_hubert_k1000_L9.pt \
    --model_repo Ansu/mHubert-basque-k1000-L9 \
    --layer 9 \
    --split train \
    --push_repo Ansu/Instruct_S2U_eu
