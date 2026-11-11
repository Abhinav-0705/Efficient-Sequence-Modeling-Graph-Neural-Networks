#!/bin/bash
#SBATCH --job-name=anlp3_train
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

export HF_TOKEN="SET_YOUR_HF_TOKEN"
export WANDB_API_KEY="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"
export HF_REPO="Abhi07rolex/anlp_assign3"

MODEL=${1:-"transformer"}

echo "======================================================"
echo "  ANLP Assignment 3 -- Model: ${MODEL}"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Node     : ${SLURMD_NODENAME}"
echo "  Started  : $(date)"
echo "======================================================"

cd ${SLURM_SUBMIT_DIR}
echo "Working dir: $(pwd)"

module purge 2>/dev/null || true
module load cuda/11.8 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate anlp_clean 2>/dev/null || conda activate anlp_env 2>/dev/null || conda activate base 2>/dev/null || true
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    conda activate anlp_clean 2>/dev/null || conda activate anlp_env 2>/dev/null || conda activate base 2>/dev/null || true
fi

if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch on compute node..."
    pip3 install --user --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
fi

pip3 install --user --no-cache-dir wandb huggingface_hub matplotlib

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
SRC="$PROJECT_ROOT/src"

DATA_DIR=""
if [ -d "$PROJECT_ROOT/dataset" ]; then
    DATA_DIR="$PROJECT_ROOT/dataset"
elif [ -d "$PROJECT_ROOT/data" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
elif [ -d "$PROJECT_ROOT/data_new" ]; then
    DATA_DIR="$PROJECT_ROOT/data_new"
elif [ -d "$PROJECT_ROOT/../data_new" ]; then
    DATA_DIR="$PROJECT_ROOT/../data_new"
elif [ -d "$HOME/data_new" ]; then
    DATA_DIR="$HOME/data_new"
else
    DATA_DIR="$PROJECT_ROOT/data_new"
fi

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/output"

cd "$SRC"
python3 -u train.py \
    --model "$MODEL" \
    --data_dir "$DATA_DIR" \
    --tokenizer_dir "$PROJECT_ROOT/tokenizers" \
    --save_dir "$PROJECT_ROOT/checkpoints" \
    --output_dir "$PROJECT_ROOT/output" \
    --hf_token "$HF_TOKEN" \
    --hf_repo_id "$HF_REPO" \
    --device cuda \
    --epochs 30 \
    --batch_size 64 \
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

EXIT_CODE=$?
echo ""
echo "Finished ${MODEL} at $(date) exit_code=${EXIT_CODE}"
exit ${EXIT_CODE}
