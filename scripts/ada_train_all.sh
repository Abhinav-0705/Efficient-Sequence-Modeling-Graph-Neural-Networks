#!/bin/bash
#SBATCH --job-name=anlp3_all
#SBATCH --output=logs/slurm_all_%j.out
#SBATCH --error=logs/slurm_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

export HF_TOKEN="SET_YOUR_HF_TOKEN"
export WANDB_API_KEY="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"
export HF_REPO="Abhi07rolex/anlp_assign3"
export WANDB_MODE="offline"

PYTHON="/home2/abhinav.chatrathi/miniconda3/envs/anlp_clean/bin/python"

echo "======================================================"
echo "  ANLP Assignment 3 -- ALL MODELS RUN"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Node     : ${SLURMD_NODENAME}"
echo "  Started  : $(date)"
echo "  Python   : $($PYTHON --version 2>&1)"
echo "======================================================"

cd ${SLURM_SUBMIT_DIR}
echo "Working dir: $(pwd)"

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
SRC="$PROJECT_ROOT/src"

DATA_DIR=""
if [ -d "$HOME/data_new" ]; then
    DATA_DIR="$HOME/data_new"
elif [ -d "$PROJECT_ROOT/data_new" ]; then
    DATA_DIR="$PROJECT_ROOT/data_new"
elif [ -d "$PROJECT_ROOT/../data_new" ]; then
    DATA_DIR="$PROJECT_ROOT/../data_new"
elif [ -d "$PROJECT_ROOT/dataset" ]; then
    DATA_DIR="$PROJECT_ROOT/dataset"
fi

echo "Data dir: $DATA_DIR"

ROLL_NUMBER="${1:-2023XXXX}"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/output" "$PROJECT_ROOT/output/part1"

cd "$SRC"

for MODEL in hybrid; do
    echo ""
    echo ">>> Starting ${MODEL} at $(date)"
    $PYTHON -u train.py \
        --model "$MODEL" \
        --data_dir "$DATA_DIR" \
        --tokenizer_dir "$PROJECT_ROOT/tokenizers" \
        --save_dir "$PROJECT_ROOT/checkpoints" \
        --output_dir "$PROJECT_ROOT/output" \
        --hf_token "$HF_TOKEN" \
        --hf_repo_id "$HF_REPO" \
        --device cuda \
        --epochs 10 \
        --batch_size 32 \
        --max_train_samples 50000 \
        --d_model 256 \
        --num_heads 8 \
        --num_layers 4 \
        --ffn_dim 1024 \
        --max_src_len 256 \
        --max_tgt_len 256 \
        --lr 3e-4 \
        --warmup_ratio 0.05 \
        --weight_decay 1e-2 \
        --dropout 0.1 \
        --label_smooth 0.1 \
        --num_workers 4 \
        --d_state 16 \
        --d_conv 4 \
        --expand 2 \
        --wandb_project anlp-assign3
    echo ">>> Finished ${MODEL} at $(date)"
done

echo ""
echo "=== All 4 models finished training. Running final evaluation ==="
$PYTHON -u evaluate.py \
    --model all \
    --data_dir "$DATA_DIR" \
    --split test \
    --device cuda \
    --batch_size 32 \
    --output_dir "$PROJECT_ROOT/output" \
    --ckpt_dir "$PROJECT_ROOT/checkpoints" \
    --save_preds \
    --heldout \
    --roll_number "$ROLL_NUMBER"

echo "All complete at $(date)!"
