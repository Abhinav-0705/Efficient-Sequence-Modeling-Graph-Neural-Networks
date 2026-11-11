#!/bin/bash
#SBATCH --job-name=anlp3_partB
#SBATCH --output=logs/slurm_gnn_%j.out
#SBATCH --error=logs/slurm_gnn_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

PYTHON="/home2/abhinav.chatrathi/miniconda3/envs/anlp_clean/bin/python"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT/src"

DATA_DIR="$PROJECT_ROOT/data/UPFD"

echo "======================================================"
echo "  ANLP Assignment 3 -- PART B (GNN & FEATURE ABLATION)"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : ${SLURMD_NODENAME}"
echo "  Started     : $(date)"
echo "======================================================"

mkdir -p "$PROJECT_ROOT/output/part2" "$PROJECT_ROOT/checkpoints/part2"

$PYTHON -u train_gnn.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$PROJECT_ROOT/output/part2" \
    --ckpt_dir "$PROJECT_ROOT/checkpoints/part2" \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.005 \
    --hidden_dim 128 \
    --device cuda

echo "Part B training and evaluation complete at $(date)!"

