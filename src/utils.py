import os
import math
from typing import List, Dict, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def char_accuracy(preds: List[str], targets: List[str]) -> float:
    total = 0
    correct = 0
    for p, t in zip(preds, targets):
        length = max(len(p), len(t))
        p_pad = p.ljust(length, "\x00")
        t_pad = t.ljust(length, "\x01")
        for a, b in zip(p_pad, t_pad):
            total += 1
            correct += int(a == b)
    return correct / max(total, 1)


def word_accuracy(preds: List[str], targets: List[str]) -> float:
    total = 0
    correct = 0
    for p, t in zip(preds, targets):
        p_words = p.split()
        t_words = t.split()
        length = max(len(p_words), len(t_words))
        p_pad = p_words + [""] * (length - len(p_words))
        t_pad = t_words + [""] * (length - len(t_words))
        for pw, tw in zip(p_pad, t_pad):
            total += 1
            correct += int(pw == tw)
    return correct / max(total, 1)


def precision_recall_f1(preds: List[str], targets: List[str]) -> Dict[str, float]:
    all_chars = set()
    for t in targets:
        all_chars.update(t)
    for p in preds:
        all_chars.update(p)

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for p, t in zip(preds, targets):
        for c in all_chars:
            tp = sum(1 for pc, tc in zip(p, t) if pc == c and tc == c)
            fp = sum(1 for pc in p if pc == c) - tp
            fn = sum(1 for tc in t if tc == c) - tp
            tp_total += tp
            fp_total += fp
            fn_total += fn

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_levenshtein(preds: List[str], targets: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(levenshtein_distance(p, t) for p, t in zip(preds, targets)) / len(preds)


def sequence_accuracy(preds: List[str], targets: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(p == t for p, t in zip(preds, targets)) / len(preds)


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def compute_flops(model, src_len: int, tgt_len: int, d_model: int, batch_size: int = 1) -> int:
    total = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module.weight, "shape"):
            w = module.weight
            if len(w.shape) == 2:
                total += 2 * batch_size * src_len * w.shape[0] * w.shape[1]
    return total


def compute_all_metrics(preds: List[str], targets: List[str]) -> Dict[str, float]:
    has_targets = any(len(t) > 0 for t in targets)
    if not has_targets:
        return {
            "char_accuracy": 0.0,
            "word_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "levenshtein": 0.0,
            "sequence_accuracy": 0.0,
        }
    prf = precision_recall_f1(preds, targets)
    return {
        "char_accuracy": char_accuracy(preds, targets),
        "word_accuracy": word_accuracy(preds, targets),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "levenshtein": mean_levenshtein(preds, targets),
        "sequence_accuracy": sequence_accuracy(preds, targets),
    }


def plot_training_curves(histories: Dict, save_path: str = "outputs/training_curves.png"):
    if plt is None:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Dynamics", fontsize=14, fontweight="bold")
    metric_keys = ["train_loss", "val_loss", "val_char_acc"]
    metric_titles = ["Train Loss", "Val Loss", "Val Char Accuracy"]
    colors = {"transformer": "#4C72B0", "mamba": "#DD8452", "deltanet": "#55A868", "hybrid": "#C44E52"}
    for ax, key, title in zip(axes, metric_keys, metric_titles):
        for model_name, history in histories.items():
            if key in history:
                ax.plot(history[key], label=model_name, color=colors.get(model_name), linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(results: Dict, save_path: str = "outputs/test_metrics.png"):
    if plt is None:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics = ["char_accuracy", "word_accuracy", "f1", "levenshtein"]
    models = list(results.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    fig.suptitle("Test Metrics Comparison", fontsize=14, fontweight="bold")
    for ax, metric in zip(axes, metrics):
        vals = [results.get(m, {}).get(metric, 0.0) for m in models]
        bars = ax.bar(models, vals, color=colors[:len(models)], edgecolor="white")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")
        ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
