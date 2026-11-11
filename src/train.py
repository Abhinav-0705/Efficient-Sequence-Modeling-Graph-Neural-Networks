import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloader, build_tokenizers, CharTokenizer
from models import TransformerSeq2Seq, MambaSeq2Seq, DeltaNetSeq2Seq, HybridSeq2Seq
from utils import compute_all_metrics, count_parameters, compute_flops, plot_training_curves, plot_metric_bars

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from huggingface_hub import HfApi, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class TrainConfig:
    model_name: str = "transformer"
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    max_src_len: int = 256
    max_tgt_len: int = 256
    batch_size: int = 64
    lr: float = 3e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    label_smooth: float = 0.1
    epochs: int = 30
    data_dir: str = "../data_new"
    tokenizer_dir: str = "tokenizers"
    save_dir: str = "checkpoints"
    output_dir: str = "outputs"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    wandb_project: str = "anlp-assign3"
    wandb_entity: Optional[str] = None
    hf_token: Optional[str] = None
    hf_repo_id: str = "Abhi07rolex/anlp_assign3"
    smoke_test: bool = False
    use_amp: bool = True
    max_train_samples: Optional[int] = 50000


def build_model(cfg: TrainConfig, src_vocab_size: int, tgt_vocab_size: int) -> nn.Module:
    kwargs = dict(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        max_seq_len=max(cfg.max_src_len, cfg.max_tgt_len),
        norm_type="rmsnorm",
        pad_idx=0,
    )
    if cfg.model_name == "transformer":
        return TransformerSeq2Seq(**kwargs, ffn_dim=cfg.ffn_dim)
    elif cfg.model_name == "mamba":
        return MambaSeq2Seq(**kwargs, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)
    elif cfg.model_name == "deltanet":
        return DeltaNetSeq2Seq(**kwargs, ffn_dim=cfg.ffn_dim)
    elif cfg.model_name == "hybrid":
        return HybridSeq2Seq(**kwargs, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand, ffn_dim=cfg.ffn_dim)
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")


def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
    return mask


def make_pad_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2).float()


def label_smooth_loss(logits: torch.Tensor, targets: torch.Tensor, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0) -> torch.Tensor:
    B, T, V = logits.shape
    log_probs = F.log_softmax(logits.reshape(-1, V), dim=-1)
    targets_flat = targets.reshape(-1)
    with torch.no_grad():
        smooth_target = torch.zeros_like(log_probs).fill_(smoothing / (V - 1))
        smooth_target.scatter_(1, targets_flat.unsqueeze(1), 1.0 - smoothing)
        smooth_target[targets_flat == pad_idx] = 0.0
    loss = -(smooth_target * log_probs).sum(dim=-1)
    num_tokens = (targets_flat != pad_idx).sum().float()
    return loss.sum() / num_tokens.clamp(min=1.0)


def get_lr_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def greedy_decode(model, src, src_mask, tgt_bos, tgt_eos, max_len, device):
    model.eval()
    with torch.no_grad():
        is_transformer = hasattr(model, "tgt_pe")
        memory = model.encode(src, src_mask) if is_transformer else model.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), tgt_bos, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            T_tgt = ys.size(1)
            if is_transformer:
                tgt_mask = make_causal_mask(T_tgt, device)
                hidden = model.decode(ys, memory, tgt_mask=tgt_mask, mem_mask=src_mask)
            else:
                hidden = model.decode(ys, memory, mem_mask=src_mask)
            logits = model.output_proj(hidden)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = next_token.masked_fill(done, tgt_eos)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            done = done | (next_token == tgt_eos)
            if done.all():
                break
    return ys


def evaluate(model, loader, plain_tokenizer, device, cfg, max_batches=None, max_decode_batches=4):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_pad_mask(src).to(device)
            T_tgt = tgt_inp.size(1)
            tgt_mask = make_causal_mask(T_tgt, device) * make_pad_mask(tgt_inp).to(device)
            logits = model(src, tgt_inp, src_mask=src_mask, tgt_mask=tgt_mask, mem_mask=src_mask)
            loss = label_smooth_loss(logits, tgt_out, plain_tokenizer.vocab_size, cfg.label_smooth)
            total_loss += loss.item()
            n_batches += 1
            if max_decode_batches is None or i < max_decode_batches:
                pred_ids = greedy_decode(model, src, src_mask, plain_tokenizer.bos_id, plain_tokenizer.eos_id, cfg.max_tgt_len, device)
                for pred_seq, tgt_seq in zip(pred_ids.cpu().tolist(), tgt.cpu().tolist()):
                    pred_str = plain_tokenizer.decode([t for t in pred_seq if t != plain_tokenizer.bos_id])
                    tgt_str = plain_tokenizer.decode([t for t in tgt_seq if t != plain_tokenizer.bos_id])
                    all_preds.append(pred_str)
                    all_targets.append(tgt_str)
    metrics = compute_all_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics, all_preds, all_targets


def upload_to_hf(cfg: TrainConfig, ckpt_path: str, metrics: dict):
    if not HF_AVAILABLE:
        return
    token = cfg.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        api = HfApi(token=token)
        create_repo(cfg.hf_repo_id, token=token, exist_ok=True, private=False)
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=os.path.basename(ckpt_path),
            repo_id=cfg.hf_repo_id,
        )
        meta_path = ckpt_path.replace(".pt", "_metrics.json")
        with open(meta_path, "w") as f:
            json.dump(metrics, f, indent=2)
        api.upload_file(
            path_or_fileobj=meta_path,
            path_in_repo=os.path.basename(meta_path),
            repo_id=cfg.hf_repo_id,
        )
        print(f"[HF] Uploaded {cfg.model_name} to {cfg.hf_repo_id}")
    except Exception as e:
        print(f"[HF] Upload failed: {e}")


def main(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    tok_cipher_path = os.path.join(cfg.tokenizer_dir, "cipher_vocab.json")
    tok_plain_path = os.path.join(cfg.tokenizer_dir, "plain_vocab.json")

    if os.path.exists(tok_cipher_path) and os.path.exists(tok_plain_path):
        cipher_tok = CharTokenizer.load(tok_cipher_path)
        plain_tok = CharTokenizer.load(tok_plain_path)
        print(f"[data] Loaded tokenizers: cipher_vocab={cipher_tok.vocab_size} plain_vocab={plain_tok.vocab_size}")
    else:
        print("[data] Building tokenizers from training data...")
        cipher_tok, plain_tok = build_tokenizers(cfg.data_dir, cfg.tokenizer_dir)

    max_batches_smoke = 3 if cfg.smoke_test else None

    train_loader = get_dataloader("train", cfg.data_dir, cipher_tok, plain_tok, batch_size=cfg.batch_size, max_src_len=cfg.max_src_len, max_tgt_len=cfg.max_tgt_len, num_workers=cfg.num_workers, max_samples=cfg.max_train_samples)
    val_loader = get_dataloader("validation", cfg.data_dir, cipher_tok, plain_tok, batch_size=cfg.batch_size, max_src_len=cfg.max_src_len, max_tgt_len=cfg.max_tgt_len, num_workers=cfg.num_workers)

    model = build_model(cfg, cipher_tok.vocab_size, plain_tok.vocab_size)
    model = model.to(device)

    param_info = count_parameters(model)
    flops = compute_flops(model, cfg.max_src_len, cfg.max_tgt_len, cfg.d_model, 1)
    print(f"[model] {cfg.model_name}  params={param_info['total']:,}  trainable={param_info['trainable']:,}  flops={flops:,}")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_lr_scheduler(optimizer, total_steps, warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    run = None
    if WANDB_AVAILABLE:
        try:
            run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.model_name,
                config=asdict(cfg),
                reinit="finish_previous",
                settings=wandb.Settings(init_timeout=60),
            )
            wandb.log({"param_count": param_info["total"], "flops": flops})
        except Exception as e:
            print(f"[wandb] init failed ({e}), continuing without wandb")

    history = {"train_loss": [], "val_loss": [], "val_char_acc": [], "val_f1": []}
    best_val_loss = float("inf")
    best_ckpt = os.path.join(cfg.save_dir, f"{cfg.model_name}_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        t0 = time.time()
        for batch_idx, (src, tgt) in enumerate(train_loader):
            if cfg.smoke_test and batch_idx >= max_batches_smoke:
                break
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_pad_mask(src).to(device)
            T_tgt = tgt_inp.size(1)
            tgt_mask = make_causal_mask(T_tgt, device) * make_pad_mask(tgt_inp).to(device)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp and device.type == "cuda"):
                logits = model(src, tgt_inp, src_mask=src_mask, tgt_mask=tgt_mask, mem_mask=src_mask)
                loss = label_smooth_loss(logits, tgt_out, plain_tok.vocab_size, cfg.label_smooth)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            n_train += 1
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"[{cfg.model_name}] Epoch {epoch:02d}/{cfg.epochs} [{batch_idx + 1:5d}/{len(train_loader)}]  loss={loss.item():.4f}", flush=True)

        train_loss /= max(n_train, 1)
        val_metrics, _, _ = evaluate(model, val_loader, plain_tok, device, cfg, max_batches=max_batches_smoke)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_char_acc"].append(val_metrics["char_accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        elapsed = time.time() - t0
        print(f"[{cfg.model_name}] Epoch {epoch:03d}/{cfg.epochs}  train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  char_acc={val_metrics['char_accuracy']:.4f}  f1={val_metrics['f1']:.4f}  ({elapsed:.1f}s)")

        if WANDB_AVAILABLE and run:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_char_acc": val_metrics["char_accuracy"],
                "val_word_acc": val_metrics["word_accuracy"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_levenshtein": val_metrics["levenshtein"],
                "lr": scheduler.get_last_lr()[0],
            })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": asdict(cfg),
                "cipher_vocab": cipher_tok.vocab,
                "plain_vocab": plain_tok.vocab,
                "val_metrics": val_metrics,
            }, best_ckpt)
            print(f"  → Saved best checkpoint (val_loss={best_val_loss:.4f})")

    print(f"\n[{cfg.model_name}] Loading best checkpoint for final test evaluation...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loader = get_dataloader("test", cfg.data_dir, cipher_tok, plain_tok, batch_size=cfg.batch_size, max_src_len=cfg.max_src_len, max_tgt_len=cfg.max_tgt_len, num_workers=cfg.num_workers)
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, plain_tok, device, cfg)

    test_metrics["param_count"] = param_info["total"]
    test_metrics["flops"] = flops

    print(f"\n[{cfg.model_name}] TEST METRICS:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    result_path = os.path.join(cfg.output_dir, f"{cfg.model_name}_results.json")
    with open(result_path, "w") as f:
        json.dump({
            "model": cfg.model_name,
            "config": asdict(cfg),
            "params": param_info,
            "flops": flops,
            "test": test_metrics,
            "history": history,
        }, f, indent=2)

    if WANDB_AVAILABLE and run:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        run.finish()

    upload_to_hf(cfg, best_ckpt, test_metrics)
    return test_metrics


def parse_args():
    p = argparse.ArgumentParser(description="ANLP Assignment 3 Training")
    p.add_argument("--model", type=str, default="transformer", choices=["transformer", "mamba", "deltanet", "hybrid"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--max_src_len", type=int, default=256)
    p.add_argument("--max_tgt_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--data_dir", type=str, default="../data_new")
    p.add_argument("--tokenizer_dir", type=str, default="tokenizers")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb_project", type=str, default="anlp-assign3")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--hf_repo_id", type=str, default="Abhi07rolex/anlp_assign3")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=50000)
    args = p.parse_args()
    cfg = TrainConfig(
        model_name=args.model,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        label_smooth=args.label_smooth,
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        save_dir=args.save_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        hf_repo_id=args.hf_repo_id,
        smoke_test=args.smoke_test,
        use_amp=not args.no_amp,
        max_train_samples=args.max_train_samples,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)

