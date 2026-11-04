"""Phase 5.3: fine-tune distilbert-base-uncased (temporal split; random split added at Tejas's request).

Usage:
  python src/07_train_distilbert.py --bench                    # time 50 training steps; no saving
  python src/07_train_distilbert.py --full --split temporal|random   # 3 epochs max, val macro-F1 each epoch, test once
Max length 256, batch 16, lr 2e-5, weight decay 0.01, linear warmup then linear decay.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, ID2LABEL, LABEL2ID, SEED, ckpt_path, compute_metrics, get_device, load_splits,
                    per_class_report, run_dir, save_run, set_seed)

SCHEME = "temporal"
CFG = dict(model="distilbert", pretrained="distilbert-base-uncased", max_len=256, batch_size=16, lr=2e-5,
           weight_decay=0.01, warmup_ratio=0.1, max_epochs=3, eval_batch_size=64, seed=SEED)


def tokenize(tok, texts):
    enc = tok(texts, truncation=True, max_length=CFG["max_len"], padding=False)
    return enc["input_ids"]


def make_batch(ids_list, y, idx, pad_id, device):
    seqs = [ids_list[i] for i in idx]
    m = max(len(s) for s in seqs)  # dynamic padding to the longest in the batch
    x = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), m), dtype=torch.long)
    for j, s in enumerate(seqs):
        x[j, :len(s)] = torch.tensor(s)
        att[j, :len(s)] = 1
    return x.to(device), att.to(device), torch.from_numpy(y[idx]).to(device)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def predict(model, ids_list, y, pad_id, device, n=None):
    model.eval()
    N = len(ids_list) if n is None else n
    # sort by length so eval batches are tight; restore original order afterwards
    order = np.argsort([len(s) for s in ids_list[:N]])
    logits = np.zeros((N, len(CLASSES)), dtype=np.float32)
    loss_sum = 0.0
    for i in range(0, N, CFG["eval_batch_size"]):
        idx = order[i:i + CFG["eval_batch_size"]]
        x, att, yb = make_batch(ids_list, y, idx, pad_id, device)
        out = model(input_ids=x, attention_mask=att).logits
        loss_sum += torch.nn.functional.cross_entropy(out, yb, reduction="sum").item()
        logits[idx] = out.float().cpu().numpy()
    return torch.from_numpy(logits), loss_sum / N


def setup():
    set_seed()
    device = get_device()
    data = load_splits(SCHEME)
    tok = AutoTokenizer.from_pretrained(CFG["pretrained"])
    ids = {s: tokenize(tok, data[s].text.tolist()) for s in data}
    y = {s: data[s].y.values.astype(np.int64) for s in data}
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG["pretrained"], num_labels=len(CLASSES), id2label=ID2LABEL, label2id=LABEL2ID).to(device)
    return device, data, tok, ids, y, model


def make_opt(model, total_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": CFG["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = AdamW(groups, lr=CFG["lr"])
    sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup_ratio"] * total_steps), total_steps)
    return opt, sched


def bench():
    device, data, tok, ids, y, model = setup()
    n_train = len(ids["train"])
    steps_per_epoch = int(np.ceil(n_train / CFG["batch_size"]))
    total = steps_per_epoch * CFG["max_epochs"]
    lens = np.array([len(s) for s in ids["train"]])
    print(f"device={device}  train {n_train}  steps/epoch {steps_per_epoch}  total steps (3 epochs) {total}")
    print(f"token lengths (train, after truncation to {CFG['max_len']}): mean {lens.mean():.0f}, "
          f"median {np.median(lens):.0f}; truncated {np.mean([len(s) >= CFG['max_len'] for s in ids['train']]):.1%}", flush=True)
    opt, sched = make_opt(model, total)
    gen = np.random.default_rng(SEED)
    perm = gen.permutation(n_train)
    model.train()
    times = []
    for step in range(50):
        idx = perm[step * CFG["batch_size"]:(step + 1) * CFG["batch_size"]]
        t0 = time.time()
        x, att, yb = make_batch(ids["train"], y["train"], idx, tok.pad_token_id, device)
        loss = model(input_ids=x, attention_mask=att, labels=yb).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        sync(device)
        times.append(time.time() - t0)
        if step % 10 == 9:
            print(f"  step {step+1:2d}  loss {loss.item():.4f}  last-10 avg {np.mean(times[-10:]):.3f}s/step", flush=True)
    steady = np.mean(times[10:])
    print(f"\n50 steps: total {sum(times):.1f}s, mean {np.mean(times):.3f}s/step; steady (steps 11-50) {steady:.3f}s/step")
    # time a slice of eval: first 640 val texts
    n_eval = 640
    t0 = time.time()
    predict(model, ids["val"], y["val"], tok.pad_token_id, device, n=n_eval)
    sync(device)
    per_text = (time.time() - t0) / n_eval
    val_pass = per_text * len(ids["val"])
    print(f"eval: {n_eval} val texts in {per_text*n_eval:.1f}s -> full val pass (7125) ~ {val_pass:.0f}s; test pass (7320) ~ {per_text*len(ids['test']):.0f}s")
    train_s = steady * total
    print(f"\nPROJECTION for 3 epochs: training {train_s/60:.0f} min + 3 val passes {3*val_pass/60:.1f} min + 1 test pass "
          f"{per_text*len(ids['test'])/60:.1f} min = {(train_s + 3*val_pass + per_text*len(ids['test']))/60:.0f} min "
          f"({(train_s + 3*val_pass + per_text*len(ids['test']))/3600:.1f} h)")
    print(f"MPS memory allocated: {torch.mps.current_allocated_memory()/1e9:.2f} GB" if device.type == "mps" else "")


def full():
    t0 = time.time()
    device, data, tok, ids, y, model = setup()
    out_dir = run_dir("distilbert", SCHEME)
    pad = tok.pad_token_id
    n_train = len(ids["train"])
    steps_per_epoch = int(np.ceil(n_train / CFG["batch_size"]))
    total = steps_per_epoch * CFG["max_epochs"]
    opt, sched = make_opt(model, total)
    gen = np.random.default_rng(SEED)
    ck = ckpt_path("distilbert", SCHEME)
    best_f1, best_epoch, history = -1.0, 0, []
    print(f"device={device} steps/epoch {steps_per_epoch} total {total}", flush=True)
    for epoch in range(1, CFG["max_epochs"] + 1):
        te = time.time()
        model.train()
        perm = gen.permutation(n_train)
        tot = 0.0
        for k, i in enumerate(range(0, n_train, CFG["batch_size"])):
            idx = perm[i:i + CFG["batch_size"]]
            x, att, yb = make_batch(ids["train"], y["train"], idx, pad, device)
            loss = model(input_ids=x, attention_mask=att, labels=yb).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item() * len(idx)
            if (k + 1) % 250 == 0:
                print(f"  epoch {epoch} step {k+1}/{steps_per_epoch}  running train loss {tot/(i+len(idx)):.4f}  "
                      f"elapsed {(time.time()-te)/60:.1f} min", flush=True)
        logits, vl = predict(model, ids["val"], y["val"], pad, device)
        vm = compute_metrics(y["val"], logits.argmax(1).numpy())
        history.append(dict(epoch=epoch, train_loss=tot / n_train, val_loss=vl, val_accuracy=vm["accuracy"],
                            val_macro_f1=vm["macro_f1"], epoch_time_s=round(time.time() - te, 1)))
        print(f"epoch {epoch}  train_loss {tot/n_train:.4f}  val_loss {vl:.4f}  val_acc {vm['accuracy']:.4f}  "
              f"val_macroF1 {vm['macro_f1']:.4f}  ({(time.time()-te)/60:.1f} min)", flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch = vm["macro_f1"], epoch
            torch.save(model.state_dict(), ck)

    # Final evaluation with the best checkpoint. Test is touched exactly once, here.
    model.load_state_dict(torch.load(ck, map_location=device))
    vlog, _ = predict(model, ids["val"], y["val"], pad, device)
    val_m = compute_metrics(y["val"], vlog.argmax(1).numpy())
    tlog, _ = predict(model, ids["test"], y["test"], pad, device)
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
                 checkpoint=ck.name, n_params=int(sum(p.numel() for p in model.parameters())))
    save_run(out_dir, {**CFG, "split_scheme": SCHEME, "device": str(device)}, history, val_m, test_m, rep, pred_df, extra)
    print(f"\nBEST epoch {best_epoch} of {len(history)} run | wall {wall/60:.0f} min")
    print(f"VAL  acc {val_m['accuracy']:.4f}  macroF1 {val_m['macro_f1']:.4f}")
    print(f"TEST acc {test_m['accuracy']:.4f}  macroF1 {test_m['macro_f1']:.4f}")
    print("Weakest 3 classes on test (F1):\n" + rep.sort_values("f1").head(3)[["class", "precision", "recall", "f1", "support"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bench", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--split", choices=["temporal", "random"], default="temporal")
    a = ap.parse_args()
    SCHEME = a.split
    bench() if a.bench else full()
