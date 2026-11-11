#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
else
    DATA_DIR="$PROJECT_ROOT/data_new"
fi

export WANDB_API_KEY="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"
export HF_REPO="Abhi07rolex/anlp_assign3"

cd "$SRC"

COMMON_ARGS="
    --data_dir $DATA_DIR
    --tokenizer_dir $PROJECT_ROOT/tokenizers
    --save_dir $PROJECT_ROOT/checkpoints
    --output_dir $PROJECT_ROOT/outputs
    --hf_repo_id $HF_REPO
    --device cuda
    --epochs 10
    --batch_size 32
    --max_train_samples 50000
    --d_model 256
    --num_heads 8
    --num_layers 4
    --ffn_dim 1024
    --max_src_len 256
    --max_tgt_len 256
    --lr 3e-4
    --warmup_ratio 0.05
    --weight_decay 1e-2
    --dropout 0.1
    --label_smooth 0.1
    --num_workers 4
    --wandb_project anlp-assign3
"

echo "=== Training Transformer ==="
python train.py --model transformer $COMMON_ARGS "$@"

echo "=== Training Mamba ==="
python train.py --model mamba $COMMON_ARGS --d_state 16 --d_conv 4 --expand 2 "$@"

echo "=== Training DeltaNet ==="
python train.py --model deltanet $COMMON_ARGS "$@"

echo "=== Training Hybrid ==="
python train.py --model hybrid $COMMON_ARGS --d_state 16 --d_conv 4 --expand 2 "$@"

echo "=== All models trained. Running final evaluation ==="
python evaluate.py \
    --model all \
    --data_dir "$DATA_DIR" \
    --split test \
    --device cuda \
    --batch_size 32 \
    --output_dir "$PROJECT_ROOT/outputs" \
    --ckpt_dir "$PROJECT_ROOT/checkpoints" \
    --save_preds \
    "$@"
