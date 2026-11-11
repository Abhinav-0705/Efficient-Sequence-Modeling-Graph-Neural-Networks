import argparse
import copy
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

from gnn_models import build_gnn_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, loader, device) -> Tuple[float, float, float, float]:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ptr = getattr(batch, "ptr", None)
            out = model(batch.x, batch.edge_index, batch.batch, ptr=ptr)
            preds = out.argmax(dim=-1).cpu().numpy()
            targets = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return acc, prec, rec, f1


def train_single_seed(
    model_name: str,
    feature: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    in_dim: int,
    seed: int,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    save_dir: str = None
) -> Tuple[Dict[str, float], dict]:
    set_seed(seed)
    model = build_gnn_model(model_name, in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            ptr = getattr(batch, "ptr", None)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, ptr=ptr)
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader, device)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"{model_name}_{feature}_seed{seed}.pt")
        torch.save({
            "model_state": best_model_state,
            "model_name": model_name,
            "feature": feature,
            "seed": seed,
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "test_metrics": {
                "accuracy": test_acc,
                "precision": test_prec,
                "recall": test_rec,
                "f1": test_f1
            }
        }, ckpt_path)

    metrics = {
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "f1": test_f1
    }
    return metrics, best_model_state


def _normalize_dataset(train_data, val_data, test_data):
    raw = train_data._data.x if hasattr(train_data, "_data") else train_data.data.x
    feat_mean = raw.mean(dim=0, keepdim=True)
    feat_std = raw.std(dim=0, keepdim=True).clamp(min=1e-5)
    for ds in [train_data, val_data, test_data]:
        d = ds._data if hasattr(ds, "_data") else ds.data
        d.x = (d.x - feat_mean) / feat_std


def run_experiment(
    model_name: str,
    feature: str,
    data_dir: str,
    seeds: List[int],
    device: torch.device,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 0.001,
    hidden_dim: int = 128,
    save_dir: str = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    print(f"\n>>> Running {model_name.upper()} with feature='{feature}' across {len(seeds)} seeds...")

    is_bigcn = model_name.lower().replace("_", "").replace("-", "") == "bigcn"
    transform = None if is_bigcn else ToUndirected()

    train_dataset = UPFD(root=data_dir, name="gossipcop", feature=feature, split="train", transform=transform)
    val_dataset = UPFD(root=data_dir, name="gossipcop", feature=feature, split="val", transform=transform)
    test_dataset = UPFD(root=data_dir, name="gossipcop", feature=feature, split="test", transform=transform)

    _normalize_dataset(train_dataset, val_dataset, test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    in_dim = train_dataset.num_features

    seed_results = []
    for s in seeds:
        metrics, _ = train_single_seed(
            model_name=model_name,
            feature=feature,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            in_dim=in_dim,
            seed=s,
            device=device,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            save_dir=save_dir
        )
        print(f"  [Seed {s}] Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
        seed_results.append(metrics)

    mean_metrics = {}
    std_metrics = {}
    for k in ["accuracy", "precision", "recall", "f1"]:
        vals = [r[k] for r in seed_results]
        mean_metrics[k] = float(np.mean(vals))
        std_metrics[k] = float(np.std(vals))

    print(f"  [MEAN ± STD] Acc: {mean_metrics['accuracy']:.4f}±{std_metrics['accuracy']:.4f} | F1: {mean_metrics['f1']:.4f}±{std_metrics['f1']:.4f}")
    return mean_metrics, std_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/UPFD")
    p.add_argument("--output_dir", type=str, default="output/part2")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/part2")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ta", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    arch_models = ["root_mlp", "gcn", "graphsage", "gat", "bigcn"]
    ablation_features = ["profile", "spacy", "bert", "content"]

    arch_results_mean = {}
    arch_results_full = {}

    ablation_results_mean = {}
    ablation_results_full = {}

    for m in arch_models:
        mean_m, std_m = run_experiment(
            model_name=m,
            feature="bert",
            data_dir=args.data_dir,
            seeds=args.seeds,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            save_dir=args.ckpt_dir
        )
        arch_results_mean[m] = mean_m
        arch_results_full[m] = {
            "mean": mean_m,
            "std": std_m
        }
        if m == "graphsage":
            ablation_results_mean["bert"] = mean_m
            ablation_results_full["bert"] = {
                "mean": mean_m,
                "std": std_m
            }

    for f in ablation_features:
        if f == "bert":
            continue
        mean_f, std_f = run_experiment(
            model_name="graphsage",
            feature=f,
            data_dir=args.data_dir,
            seeds=args.seeds,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            save_dir=args.ckpt_dir
        )
        ablation_results_mean[f] = mean_f
        ablation_results_full[f] = {
            "mean": mean_f,
            "std": std_f
        }

    final_results = {
        "architecture_comparison": arch_results_mean,
        "feature_ablation": ablation_results_mean
    }

    final_full = {
        "architecture_comparison": arch_results_full,
        "feature_ablation": ablation_results_full
    }

    out_filename = "results_ta.json" if args.ta else "results.json"
    out_path = os.path.join(args.output_dir, out_filename)
    with open(out_path, "w") as fp:
        json.dump(final_results, fp, indent=2)
    print(f"\n[Done] Saved mean results to {out_path}")

    full_path = os.path.join(args.output_dir, "results_mean_std.json")
    with open(full_path, "w") as fp:
        json.dump(final_full, fp, indent=2)

    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: ARCHITECTURE COMPARISON (BERT FEATURES)")
    print("=" * 65)
    print(f"{'Model':<15} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'F1-Score':<14}")
    print("-" * 65)
    for m in arch_models:
        mn = arch_results_full[m]["mean"]
        st = arch_results_full[m]["std"]
        print(f"{m:<15} {mn['accuracy']:.4f}±{st['accuracy']:.4f}  {mn['precision']:.4f}±{st['precision']:.4f}  {mn['recall']:.4f}±{st['recall']:.4f}  {mn['f1']:.4f}±{st['f1']:.4f}")

    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: FEATURE ABLATION (GRAPHSAGE BACKBONE)")
    print("=" * 65)
    print(f"{'Feature':<15} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'F1-Score':<14}")
    print("-" * 65)
    for f in ablation_features:
        mn = ablation_results_full[f]["mean"]
        st = ablation_results_full[f]["std"]
        print(f"{f:<15} {mn['accuracy']:.4f}±{st['accuracy']:.4f}  {mn['precision']:.4f}±{st['precision']:.4f}  {mn['recall']:.4f}±{st['recall']:.4f}  {mn['f1']:.4f}±{st['f1']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()

