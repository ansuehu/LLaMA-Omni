#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=Instruction_rewriting # Name of the process
#SBATCH --array=0-9        # for 10 shards
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=4 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/Instruction_rewriting_%A_%a.out.log
#SBATCH --error=.slurm/Instruction_rewriting_%A_%a.err.log
#SBATCH --chdir=/home/andoni.sudupe/LLaMA-Omni/

SHARD_ID=${SLURM_ARRAY_TASK_ID}
INPUT_PATH="/home/andoni.sudupe/LLaMA-Omni/data/shards/shard_${SHARD_ID}"
INSTRUCTION_PATH="/home/andoni.sudupe/LLaMA-Omni/data/output/translated_instruction_${SHARD_ID}.json"
ANSWER_PATH="/home/andoni.sudupe/LLaMA-Omni/data/output/translated_answer_${SHARD_ID}.json"

start_time=$(date +%s)

# Run the Python script
python3 ./dataset_generation/instruction_rewriting.py \
  --input_path "$INPUT_PATH" \
  --instruction_path "$INSTRUCTION_PATH" \
  --answer_path "$ANSWER_PATH" 
  # --num_examples "$NUM_EXAMPLES"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: $(date -ud "@$elapsed" +%T)"
