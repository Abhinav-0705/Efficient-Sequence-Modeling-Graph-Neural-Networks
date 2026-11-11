#!/bin/bash
#SBATCH --job-name=anlp3_eval
#SBATCH --output=logs/slurm_eval_%j.out
#SBATCH --error=logs/slurm_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

export HF_TOKEN="SET_YOUR_HF_TOKEN"
export WANDB_API_KEY="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"
export HF_REPO="Abhi07rolex/anlp_assign3"

PYTHON="/home2/abhinav.chatrathi/miniconda3/envs/anlp_clean/bin/python"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT/src"

DATA_DIR="$HOME/data_new"

ROLL_NUMBER="${1:-2023XXXX}"

echo "======================================================"
echo "  ANLP Assignment 3 -- FINAL EVALUATION (ALL 4 MODELS)"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : ${SLURMD_NODENAME}"
echo "  Roll Number : ${ROLL_NUMBER}"
echo "  Started     : $(date)"
echo "======================================================"

mkdir -p "$PROJECT_ROOT/output" "$PROJECT_ROOT/output/part1"

$PYTHON -u evaluate.py \
    --model all \
    --data_dir "$DATA_DIR" \
    --split test \
    --device cuda \
    --batch_size 32 \
    --output_dir "$PROJECT_ROOT/output" \
    --ckpt_dir "$PROJECT_ROOT/checkpoints" \
    --hf_repo_id "$HF_REPO" \
    --save_preds \
    --heldout \
    --roll_number "$ROLL_NUMBER"

echo "Evaluation complete at $(date)!"

