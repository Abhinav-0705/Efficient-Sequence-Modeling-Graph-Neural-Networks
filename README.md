# ANLP Assignment 3 — Cipher Decryption with SSM Architectures

## Models Implemented
- **Transformer** (Encoder-Decoder with MHA + RMSNorm)
- **Mamba** (Selective State Space Model — from scratch, no `mamba_ssm`)
- **DeltaNet** (Linear Transformer with Delta Rule — from scratch, no `flash-linear-attention`)
- **Mamba-DeltaNet Hybrid** (Interleaved; see `docs/DESIGN.md`)

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables:
```bash
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_key"
```

## Data Layout

```
data_new/
├── train/
│   ├── cipher.jsonl
│   └── cleaned.jsonl
├── validation/
│   ├── cipher.jsonl
│   └── cleaned.jsonl
└── test/
    ├── cipher.jsonl
    └── cleaned.jsonl
```

Place `data_new/` as a sibling of this `anlp_assign3/` folder.

## Training

```bash
bash scripts/part1/test_training.sh
```

Or train individual models:
```bash
cd src
python train.py --model transformer --epochs 30 --data_dir ../data_new
python train.py --model mamba      --epochs 30 --data_dir ../data_new
python train.py --model deltanet   --epochs 30 --data_dir ../data_new
python train.py --model hybrid     --epochs 30 --data_dir ../data_new
```

## Inference (from HuggingFace)

```bash
bash scripts/part1/test_assn.sh
```

Or manually:
```bash
cd src
python evaluate.py --model all --from_hf --hf_repo_id Abhi07rolex/anlp_assign3 --data_dir ../data_new
```

## Metrics Reported
- Character-level Accuracy
- Word-level Accuracy
- Precision, Recall, F1 (character-level)
- Levenshtein Distance
- Parameter Count
- FLOPs

## HuggingFace
Models pushed to: `Abhi07rolex/anlp_assign3`

## WandB
Project: `anlp-assign3`

## Notes for Ada GPU
- Use `--device cuda --use_amp` (default) for mixed precision training
- Recommended batch size: 64 for d_model=256
- All models fit well within 10GB VRAM
- Mamba's sequential scan is the bottleneck; consider reducing `max_src_len` if OOM

