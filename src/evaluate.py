import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CharTokenizer, get_dataloader, resolve_split_paths, load_jsonl_with_ids
from models import TransformerSeq2Seq, MambaSeq2Seq, DeltaNetSeq2Seq, HybridSeq2Seq
from utils import compute_all_metrics, count_parameters, compute_flops, plot_metric_bars
from train import make_pad_mask, make_causal_mask, greedy_decode, TrainConfig


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["cfg"]
    cfg = TrainConfig(**cfg_dict)
    cipher_tok = CharTokenizer(ckpt["cipher_vocab"])
    plain_tok = CharTokenizer(ckpt["plain_vocab"])
    model_name = cfg.model_name
    kwargs = dict(
        src_vocab_size=cipher_tok.vocab_size,
        tgt_vocab_size=plain_tok.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=0.0,
        max_seq_len=max(cfg.max_src_len, cfg.max_tgt_len),
        norm_type="rmsnorm",
        pad_idx=0,
    )
    if model_name == "transformer":
        model = TransformerSeq2Seq(**kwargs, ffn_dim=cfg.ffn_dim)
    elif model_name == "mamba":
        model = MambaSeq2Seq(**kwargs, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)
    elif model_name == "deltanet":
        model = DeltaNetSeq2Seq(**kwargs, ffn_dim=cfg.ffn_dim)
    elif model_name == "hybrid":
        model = HybridSeq2Seq(**kwargs, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand, ffn_dim=cfg.ffn_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model, cipher_tok, plain_tok, cfg


def load_model_from_hf(repo_id: str, filename: str, device: torch.device):
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_model_from_checkpoint(ckpt_path, device)


def run_inference(model, cipher_tok, plain_tok, cfg, data_dir, split, device, batch_size=32):
    loader = get_dataloader(
        split, data_dir, cipher_tok, plain_tok,
        batch_size=batch_size,
        max_src_len=cfg.max_src_len,
        max_tgt_len=cfg.max_tgt_len,
        shuffle=False,
        num_workers=0,
    )
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = make_pad_mask(src).to(device)
            pred_ids = greedy_decode(
                model, src, src_mask,
                plain_tok.bos_id, plain_tok.eos_id,
                cfg.max_tgt_len, device,
            )
            for pred_seq, tgt_seq in zip(pred_ids.cpu().tolist(), tgt.cpu().tolist()):
                pred_str = plain_tok.decode([t for t in pred_seq if t != plain_tok.bos_id])
                tgt_str = plain_tok.decode([t for t in tgt_seq if t != plain_tok.bos_id])
                all_preds.append(pred_str)
                all_targets.append(tgt_str)
    metrics = compute_all_metrics(all_preds, all_targets)
    param_info = count_parameters(model)
    flops = compute_flops(model, cfg.max_src_len, cfg.max_tgt_len, cfg.d_model, 1)
    metrics["param_count"] = param_info["total"]
    metrics["flops"] = flops
    return metrics, all_preds, all_targets


def main():
    p = argparse.ArgumentParser(description="ANLP Assignment 3 Inference")
    p.add_argument("--model", type=str, default="all", choices=["transformer", "mamba", "deltanet", "hybrid", "all"])
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--data_dir", type=str, default="../data_new")
    p.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_dir", type=str, default="output")
    p.add_argument("--hf_repo_id", type=str, default="Abhi07rolex/anlp_assign3")
    p.add_argument("--from_hf", action="store_true")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--heldout", action="store_true")
    p.add_argument("--roll_number", type=str, default=os.environ.get("ROLL_NUMBER", "2023XXXX"))
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "part1"), exist_ok=True)

    models_to_eval = ["hybrid", "mamba", "deltanet", "transformer"] if args.model == "all" else [args.model]
    all_results = {}
    last_preds = None

    for model_name in models_to_eval:
        print(f"\n[eval] Evaluating {model_name}...")
        filename = f"{model_name}_best.pt"
        try:
            if args.from_hf:
                model, cipher_tok, plain_tok, cfg = load_model_from_hf(args.hf_repo_id, filename, device)
            else:
                ckpt_path = os.path.join(args.ckpt_dir, filename)
                model, cipher_tok, plain_tok, cfg = load_model_from_checkpoint(ckpt_path, device)
        except Exception as e:
            print(f"  [skip] Could not load {model_name}: {e}")
            continue

        metrics, preds, targets = run_inference(model, cipher_tok, plain_tok, cfg, args.data_dir, args.split, device, args.batch_size)
        all_results[model_name] = metrics
        last_preds = preds

        print(f"  Results for {model_name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        if args.save_preds or args.heldout:
            pred_path = os.path.join(args.output_dir, f"{model_name}_{args.split}_preds.json")
            with open(pred_path, "w") as f:
                json.dump({"preds": preds, "targets": targets}, f, indent=2)

    if args.heldout and last_preds is not None:
        c_path, _ = resolve_split_paths(args.data_dir, args.split)
        ids, _ = load_jsonl_with_ids(c_path)
        heldout_out = os.path.join(args.output_dir, "part1", f"heldout_result_{args.roll_number}.jsonl")
        with open(heldout_out, "w", encoding="utf-8") as f:
            for sid, ptxt in zip(ids, last_preds):
                f.write(json.dumps({"id": sid, "text": ptxt}, ensure_ascii=False) + "\n")
        print(f"\n[eval] Saved heldout predictions to {heldout_out}")

    out_path = os.path.join(args.output_dir, f"all_{args.split}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] Saved all metrics to {out_path}")

    plot_metric_bars(all_results, os.path.join(args.output_dir, "test_metrics_comparison.png"))

    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import HfApi, create_repo
            api = HfApi(token=token)
            create_repo(args.hf_repo_id, token=token, exist_ok=True, private=False)
            for model_name in models_to_eval:
                ckpt_file = os.path.join(args.ckpt_dir, f"{model_name}_best.pt")
                if os.path.exists(ckpt_file):
                    api.upload_file(path_or_fileobj=ckpt_file, path_in_repo=f"{model_name}_best.pt", repo_id=args.hf_repo_id)
                    print(f"[HF] Uploaded {model_name}_best.pt to {args.hf_repo_id}")
            if os.path.exists(out_path):
                api.upload_file(path_or_fileobj=out_path, path_in_repo="all_test_metrics.json", repo_id=args.hf_repo_id)
        except Exception as e:
            print(f"[HF] Upload failed: {e}")

    print("\n=== FINAL SUMMARY ===")
    header = f"{'Model':<12} {'CharAcc':>8} {'WordAcc':>8} {'F1':>8} {'Levenshtein':>13} {'Params':>12} {'FLOPs':>14}"
    print(header)
    print("-" * len(header))
    for name, m in all_results.items():
        print(f"{name:<12} {m.get('char_accuracy',0):>8.4f} {m.get('word_accuracy',0):>8.4f} {m.get('f1',0):>8.4f} {m.get('levenshtein',0):>13.2f} {m.get('param_count',0):>12,} {m.get('flops',0):>14,}")


if __name__ == "__main__":
    main()

