#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=make_dataset # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=4 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=3-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/extract-code.out.log
#SBATCH --error=.slurm/extract-code.err.log
#SBATCH --chdir=/home/andoni.sudupe/LLaMA-Omni/


python dataset_generation/change_format.py