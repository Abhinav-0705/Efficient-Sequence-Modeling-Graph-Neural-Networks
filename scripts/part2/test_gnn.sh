#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$PROJECT_ROOT/src"

DATA_DIR=""
if [ -d "$PROJECT_ROOT/data/UPFD" ]; then
    DATA_DIR="$PROJECT_ROOT/data/UPFD"
elif [ -d "$HOME/anlp_assign3/data/UPFD" ]; then
    DATA_DIR="$HOME/anlp_assign3/data/UPFD"
elif [ -d "$PROJECT_ROOT/../data/UPFD" ]; then
    DATA_DIR="$PROJECT_ROOT/../data/UPFD"
else
    DATA_DIR="$PROJECT_ROOT/data/UPFD"
fi

cd "$SRC"

python train_gnn.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$PROJECT_ROOT/output/part2" \
    --ckpt_dir "$PROJECT_ROOT/checkpoints/part2" \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.005 \
    --hidden_dim 128 \
    "$@"

