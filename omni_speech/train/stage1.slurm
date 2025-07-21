#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=Omni-Stage1 # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/Omni-Stage1.out.log
#SBATCH --error=.slurm/Omni-Stage1.err.log
#SBATCH --chdir=/home/andoni.sudupe/LLaMA-Omni/



python omni_speech/train/stage1.py \
    --model-path Llama-3.2-1B-Instruct \
    --question-file data.json \
    --answer-file answer.json \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 128 \

