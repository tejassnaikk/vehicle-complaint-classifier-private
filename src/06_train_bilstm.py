"""Phase 5.2: BiLSTM text classifier with its own tokenizer.

Usage:
  python src/06_train_bilstm.py --bench                      # time 1 epoch on MPS and on CPU, no saving
  python src/06_train_bilstm.py --split temporal --device cpu|mps
Model: embedding 128 (trainable) -> 2-layer bidirectional LSTM (hidden 128, packed sequences) ->
masked max-pool + mean-pool over time -> dropout 0.3 -> linear. Early stopping on validation macro-F1.
Test is evaluated once, with the best checkpoint.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, ID2LABEL, SEED, ckpt_path, compute_metrics, get_device, load_splits,
                    per_class_report, run_dir, save_run, set_seed)

CFG = dict(model="bilstm", vocab_max=30000, vocab_min_freq=2, max_len=256, embed_dim=128, hidden=128,
           num_layers=2, dropout=0.3, lr=1e-3, batch_size=64, grad_clip=1.0, max_epochs=15, patience=3,
           tokenizer=r"[a-z0-9]+(?:'[a-z]+)?", seed=SEED)
PAD, UNK = 0, 1
TOKEN_RE = re.compile(CFG["tokenizer"])


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts):
    cnt = Counter(t for x in texts for t in tokenize(x))
    words = [w for w, c in cnt.most_common() if c >= CFG["vocab_min_freq"]][:CFG["vocab_max"] - 2]
    return {w: i + 2 for i, w in enumerate(words)}  # 0 = PAD, 1 = UNK


def encode(texts, vocab):
    """Return padded id matrix (N, max_len), lengths, and token counts before truncation."""
    L = CFG["max_len"]
    ids = np.zeros((len(texts), L), dtype=np.int64)
    lens = np.zeros(len(texts), dtype=np.int64)
    full = np.zeros(len(texts), dtype=np.int64)
    unk = tot = 0
    for i, t in enumerate(texts):
        toks = tokenize(t)
        full[i] = len(toks)
        seq = [vocab.get(w, UNK) for w in toks[:L]]
        ids[i, :len(seq)] = seq
        lens[i] = max(len(seq), 1)  # at least 1 so packing never sees an empty sequence
        unk += sum(1 for s in seq if s == UNK)
        tot += len(seq)
    return ids, lens, full, unk / max(tot, 1)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_out):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, CFG["embed_dim"], padding_idx=PAD)
        self.lstm = nn.LSTM(CFG["embed_dim"], CFG["hidden"], num_layers=CFG["num_layers"], batch_first=True,
                            bidirectional=True, dropout=CFG["dropout"])
        self.drop = nn.Dropout(CFG["dropout"])
        self.out = nn.Linear(4 * CFG["hidden"], n_out)  # max-pool + mean-pool, each 2*hidden

    def forward(self, x, lens):
        T = x.shape[1]
        packed = pack_padded_sequence(self.emb(x), lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=T)
        mask = (torch.arange(T, device=x.device)[None, :] < lens[:, None]).unsqueeze(-1)  # padding excluded
        mx = h.masked_fill(~mask, float("-inf")).max(1).values
        mean = (h * mask).sum(1) / lens[:, None].to(h.dtype)
        return self.out(self.drop(torch.cat([mx, mean], dim=1)))


def batch(ids, lens, idx, device):
    m = int(lens[idx].max())
    return (torch.from_numpy(ids[idx, :m]).to(device), torch.from_numpy(lens[idx]).to(device))


def train_epoch(model, opt, loss_fn, ids, lens, y, device, gen):
    model.train()
    perm = gen.permutation(len(ids))
    tot = 0.0
    for i in range(0, len(perm), CFG["batch_size"]):
        idx = perm[i:i + CFG["batch_size"]]
        xb, lb = batch(ids, lens, idx, device)
        opt.zero_grad()
        loss = loss_fn(model(xb, lb), torch.from_numpy(y[idx]).to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        opt.step()
        tot += loss.item() * len(idx)
    return tot / len(perm)


@torch.no_grad()
def predict(model, ids, lens, y, device, loss_fn, chunk=256):
    model.eval()
    logits, loss = [], 0.0
    for i in range(0, len(ids), chunk):
        idx = np.arange(i, min(i + chunk, len(ids)))
        xb, lb = batch(ids, lens, idx, device)
        out = model(xb, lb)
        loss += loss_fn(out, torch.from_numpy(y[idx]).to(device)).item() * len(idx)
        logits.append(out.cpu())
    return torch.cat(logits), loss / len(ids)


def prepare(scheme):
    data = load_splits(scheme)
    vocab = build_vocab(data["train"].text)  # train only
    enc = {s: encode(data[s].text.tolist(), vocab) for s in data}
    y = {s: data[s].y.values.astype(np.int64) for s in data}
    return data, vocab, enc, y


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def bench():
    data, vocab, enc, y = prepare("temporal")
    ids, lens, _, _ = enc["train"]
    print(f"vocab {len(vocab)+2}; benchmarking 1 full training epoch ({len(ids)} texts, batch {CFG['batch_size']})", flush=True)
    for name in ["mps", "cpu"]:
        device = torch.device(name)
        set_seed()
        model = BiLSTM(len(vocab) + 2, len(CLASSES)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
        t0 = time.time()
        try:
            loss = train_epoch(model, opt, nn.CrossEntropyLoss(), ids, lens, y["train"], device, np.random.default_rng(SEED))
            sync(device)
            print(f"{name}: 1 epoch = {time.time()-t0:.1f}s  (train loss {loss:.4f})", flush=True)
        except Exception as e:
            print(f"{name}: FAILED after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--split", choices=["temporal", "random"])
    ap.add_argument("--device", choices=["mps", "cpu"], default=None)
    args = ap.parse_args()
    if args.bench:
        return bench()
    scheme = args.split
    device = torch.device(args.device) if args.device else get_device()
    set_seed()
    t0 = time.time()
    out_dir = run_dir("bilstm", scheme)
    data, vocab, enc, y = prepare(scheme)
    V = len(vocab) + 2
    trunc = {s: float((enc[s][2] > CFG["max_len"]).mean()) for s in enc}
    print(f"[{scheme}] device={device} vocab={V} | texts truncated at {CFG['max_len']} tokens: "
          + ", ".join(f"{s} {v:.1%}" for s, v in trunc.items())
          + f" | UNK rate val {enc['val'][3]:.1%}", flush=True)
    ckpt_vocab = ckpt_path("bilstm", scheme, "_vocab.json")
    ckpt_vocab.write_text(json.dumps(vocab))

    model = BiLSTM(V, len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    loss_fn = nn.CrossEntropyLoss()
    gen = np.random.default_rng(SEED)
    ck = ckpt_path("bilstm", scheme)
    best_f1, best_epoch, bad, history = -1.0, 0, 0, []
    for epoch in range(1, CFG["max_epochs"] + 1):
        te = time.time()
        tl = train_epoch(model, opt, loss_fn, enc["train"][0], enc["train"][1], y["train"], device, gen)
        logits, vl = predict(model, enc["val"][0], enc["val"][1], y["val"], device, loss_fn)
        vm = compute_metrics(y["val"], logits.argmax(1).numpy())
        history.append(dict(epoch=epoch, train_loss=tl, val_loss=vl, val_accuracy=vm["accuracy"],
                            val_macro_f1=vm["macro_f1"], epoch_time_s=round(time.time() - te, 1)))
        print(f"epoch {epoch:2d}  train_loss {tl:.4f}  val_loss {vl:.4f}  val_acc {vm['accuracy']:.4f}  "
              f"val_macroF1 {vm['macro_f1']:.4f}  ({time.time()-te:.0f}s)", flush=True)
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
    vlog, _ = predict(model, enc["val"][0], enc["val"][1], y["val"], device, loss_fn)
    val_m = compute_metrics(y["val"], vlog.argmax(1).numpy())
    tlog, _ = predict(model, enc["test"][0], enc["test"][1], y["test"], device, loss_fn)
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
                 checkpoint=ck.name, vocab_size=V, pct_truncated=trunc,
                 n_params=int(sum(p.numel() for p in model.parameters())))
    save_run(out_dir, {**CFG, "split_scheme": scheme, "device": str(device)}, history, val_m, test_m, rep, pred_df, extra)
    print(f"\nBEST epoch {best_epoch} of {len(history)} run | wall {wall:.0f}s")
    print(f"VAL  acc {val_m['accuracy']:.4f}  macroF1 {val_m['macro_f1']:.4f}")
    print(f"TEST acc {test_m['accuracy']:.4f}  macroF1 {test_m['macro_f1']:.4f}")
    print("Weakest 3 classes on test (F1):\n" + rep.sort_values("f1").head(3)[["class", "precision", "recall", "f1", "support"]].to_string(index=False))


if __name__ == "__main__":
    main()
