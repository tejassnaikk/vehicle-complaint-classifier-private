"""Phase 5.1: TF-IDF (1-2 grams, 30k features, fit on train only) -> MLP (512, 128).

Usage: python src/05_train_mlp.py --split temporal|random
Model selection / early stopping use validation macro-F1 only. Test is evaluated once at the end
with the best checkpoint.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, SEED, ID2LABEL, ckpt_path, compute_metrics, get_device, load_splits,
                    per_class_report, run_dir, save_run, set_seed)
import pandas as pd

CFG = dict(model="mlp", max_features=30000, ngram_range=[1, 2], hidden=[512, 128], dropout=0.3,
           lr=1e-3, batch_size=128, max_epochs=30, patience=3, seed=SEED)


class MLP(nn.Module):
    def __init__(self, d_in, hidden, n_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], n_out))

    def forward(self, x):
        return self.net(x)


def to_dense(X, idx, device):
    return torch.from_numpy(X[idx].toarray().astype(np.float32)).to(device)


@torch.no_grad()
def predict(model, X, y, device, loss_fn, chunk=1024):
    model.eval()
    logits, loss = [], 0.0
    for i in range(0, X.shape[0], chunk):
        idx = np.arange(i, min(i + chunk, X.shape[0]))
        out = model(to_dense(X, idx, device))
        loss += loss_fn(out, torch.from_numpy(y[idx]).to(device)).item() * len(idx)
        logits.append(out.cpu())
    return torch.cat(logits), loss / X.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["temporal", "random"], required=True)
    args = ap.parse_args()
    scheme = args.split
    set_seed()
    device = get_device()
    t0 = time.time()
    out_dir = run_dir("mlp", scheme)
    data = load_splits(scheme)
    print(f"[{scheme}] device={device}  train/val/test = "
          f"{len(data['train'])}/{len(data['val'])}/{len(data['test'])}", flush=True)

    vec = TfidfVectorizer(ngram_range=tuple(CFG["ngram_range"]), max_features=CFG["max_features"], dtype=np.float32)
    X = {"train": vec.fit_transform(data["train"].text)}  # fit on train only
    X["val"], X["test"] = vec.transform(data["val"].text), vec.transform(data["test"].text)
    y = {s: data[s].y.values.astype(np.int64) for s in data}
    print(f"TF-IDF features: {X['train'].shape[1]:,}", flush=True)
    with open(ckpt_path("mlp", scheme, "_tfidf.pkl"), "wb") as f:
        pickle.dump(vec, f)

    model = MLP(X["train"].shape[1], CFG["hidden"], len(CLASSES), CFG["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    loss_fn = nn.CrossEntropyLoss()
    gen = np.random.default_rng(SEED)
    best_f1, best_epoch, bad, history = -1.0, 0, 0, []
    ck = ckpt_path("mlp", scheme)

    for epoch in range(1, CFG["max_epochs"] + 1):
        model.train()
        perm = gen.permutation(X["train"].shape[0])
        tot = 0.0
        for i in range(0, len(perm), CFG["batch_size"]):
            idx = perm[i:i + CFG["batch_size"]]
            xb, yb = to_dense(X["train"], idx, device), torch.from_numpy(y["train"][idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        logits, vloss = predict(model, X["val"], y["val"], device, loss_fn)
        vm = compute_metrics(y["val"], logits.argmax(1).numpy())
        history.append(dict(epoch=epoch, train_loss=tot / len(perm), val_loss=vloss,
                            val_accuracy=vm["accuracy"], val_macro_f1=vm["macro_f1"]))
        print(f"epoch {epoch:2d}  train_loss {tot/len(perm):.4f}  val_loss {vloss:.4f}  "
              f"val_acc {vm['accuracy']:.4f}  val_macroF1 {vm['macro_f1']:.4f}", flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = vm["macro_f1"], epoch, 0
            torch.save(model.state_dict(), ck)
        else:
            bad += 1
            if bad >= CFG["patience"]:
                print(f"early stop: no val macro-F1 gain for {CFG['patience']} epochs", flush=True)
                break

    # Final evaluation with the best checkpoint. Test is touched exactly once, here.
    model.load_state_dict(torch.load(ck, map_location=device))
    vlog, _ = predict(model, X["val"], y["val"], device, loss_fn)
    val_m = compute_metrics(y["val"], vlog.argmax(1).numpy())
    tlog, _ = predict(model, X["test"], y["test"], device, loss_fn)
    prob = torch.softmax(tlog, 1)
    pred = prob.argmax(1).numpy()
    test_m = compute_metrics(y["test"], pred)
    rep = per_class_report(y["test"], pred)
    pred_df = pd.DataFrame({"odino": data["test"].odino, "y_true": y["test"], "y_pred": pred,
                            "true_label": [ID2LABEL[i] for i in y["test"]],
                            "pred_label": [ID2LABEL[i] for i in pred],
                            "confidence": prob.max(1).values.numpy(),
                            "is_transcript": data["test"].is_transcript})
    wall = time.time() - t0
    extra = dict(best_epoch=best_epoch, epochs_run=len(history), wall_time_s=round(wall, 1), device=str(device),
                 checkpoint=str(ck.name), n_features=int(X["train"].shape[1]))
    cfg = {**CFG, "split_scheme": scheme}
    save_run(out_dir, cfg, history, val_m, test_m, rep, pred_df, extra)
    print(f"\nBEST epoch {best_epoch} of {len(history)} run | wall {wall:.0f}s")
    print(f"VAL  acc {val_m['accuracy']:.4f}  macroF1 {val_m['macro_f1']:.4f}")
    print(f"TEST acc {test_m['accuracy']:.4f}  macroF1 {test_m['macro_f1']:.4f}")
    print("Weakest 3 classes on test (F1):\n" + rep.sort_values("f1").head(3)[["class", "precision", "recall", "f1", "support"]].to_string(index=False))


if __name__ == "__main__":
    main()
