"""Shared helpers for all model scripts: data loading, label encoding, metrics, curves, saving."""
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, CLEAN_DIR, MODELS_DIR, RESULTS_DIR, SEED

LABEL2ID = {c: i for i, c in enumerate(CLASSES)}  # fixed order from config.CLASSES
ID2LABEL = {i: c for c, i in LABEL2ID.items()}
SPLIT_FILES = {"temporal": "complaints_temporal.csv", "random": "complaints_random.csv"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    """MPS if it works, else CPU."""
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps") + 1
            return torch.device("mps")
        except Exception as e:  # pragma: no cover
            print(f"MPS failed ({e}); falling back to CPU")
    return torch.device("cpu")


def load_splits(scheme):
    """Return dict split -> DataFrame (with integer column 'y') for 'temporal' or 'random'."""
    df = pd.read_csv(CLEAN_DIR / SPLIT_FILES[scheme], dtype=str, keep_default_na=False)
    df["is_transcript"] = df.is_transcript.eq("True")
    df["y"] = df.label.map(LABEL2ID)
    assert df.y.notna().all(), "unknown label"
    return {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(CLASSES)), zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=range(len(CLASSES)), zero_division=0)),
        "n": int(len(y_true)),
    }


def per_class_report(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    rows = [{"class": c, **{k: rep[c][k] for k in ["precision", "recall", "f1-score", "support"]}} for c in CLASSES]
    return pd.DataFrame(rows).rename(columns={"f1-score": "f1"})


def run_dir(model, scheme):
    d = RESULTS_DIR / model / scheme
    d.mkdir(parents=True, exist_ok=True)
    return d


def ckpt_path(model, scheme, suffix=".pt"):
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR / f"{model}_{scheme}{suffix}"


def plot_curves(history, path, title):
    """history: list of dicts with epoch, train_loss, val_loss (optional), val_macro_f1."""
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h.epoch, h.train_loss, marker="o", label="train loss")
    if "val_loss" in h:
        ax[0].plot(h.epoch, h.val_loss, marker="o", label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(h.epoch, h.val_macro_f1, marker="o", color="#2a7f62")
    best = h.val_macro_f1.idxmax()
    ax[1].scatter([h.epoch[best]], [h.val_macro_f1[best]], color="red", zorder=3, label=f"best (epoch {h.epoch[best]})")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation macro-F1"); ax[1].legend(); ax[1].set_title("Validation macro-F1")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_run(out_dir, config, history, val_metrics, test_metrics, test_report, test_pred_df, extra):
    """Write config.json, history.csv, metrics.json, per_class_test.csv, test_predictions.csv, curves.png."""
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    metrics = {"val": val_metrics, "test": test_metrics, **extra}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    test_report.to_csv(out_dir / "per_class_test.csv", index=False)
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    plot_curves(history, out_dir / "curves.png", f"{config['model']} ({config['split_scheme']} split)")
