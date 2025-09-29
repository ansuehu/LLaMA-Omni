#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=Omni-Stage1-ddp # Name of the process
#SBATCH --gres=gpu:2 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=32GB # RAM memory needed (8-16GB)
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/Omni-Stage1-ddp-%j.out.log
#SBATCH --error=.slurm/Omni-Stage1-ddp-%j.err.log
#SBATCH --chdir=/home/andoni.sudupe/LLaMA-Omni/

export SSL_CERT_FILE=~/cacert.pem
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
MODEL_PATH=HiTZ/Latxa-Llama-3.1-8B-Instruct
QUESTION_FILE=/scratch/andoni.sudupe/Instruct_S2S_eu/data.json
OUTPUT_FOLDER=$SLURM_JOB_ID

torchrun \
     --nproc_per_node=$GPUS_PER_NODE \
     --master_port=29500 \
     omni_speech/train/stage1-ddp.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
     --output-folder $OUTPUT_FOLDER \

