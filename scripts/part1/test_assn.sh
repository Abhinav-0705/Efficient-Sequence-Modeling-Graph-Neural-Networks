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

cd "$SRC"

python evaluate.py \
    --model all \
    --data_dir "$DATA_DIR" \
    --split test \
    --device cuda \
    --batch_size 32 \
    --output_dir "$PROJECT_ROOT/output" \
    --hf_repo_id "Abhi07rolex/anlp_assign3" \
    --from_hf \
    --save_preds \
    "$@"
