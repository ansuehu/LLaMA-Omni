#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=Instruction_rewriting # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/Instruction_rewriting.out.log
#SBATCH --error=.slurm/Instruction_rewriting.err.log
#SBATCH --chdir=/home/andoni.sudupe/LLaMA-Omni/

INPUT_PATH=${1:-HiTZ/Magpie-Llama-3.1-8B-Instruct-Filtered-translated-1M}
OUTPUT_PATH=${2:-dataset_generation/rewritten_instructions.json}
NUM_EXAMPLES=${3:-10}

echo "Input path: $INPUT_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Number of examples: $NUM_EXAMPLES"

# Run the Python script
python3 ./dataset_generation/instruction_rewriting.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_examples "$NUM_EXAMPLES"
