"""Training module: train/validate loops, LR scheduling, early stopping, checkpointing."""
import time
from pathlib import Path

import torch
import torch.nn as nn

from config import Config
from utils import save_json


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, cfg: Config, device) -> dict:
    """Train with early stopping on validation loss. Returns history + timing."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val, bad_epochs, best_epoch = float("inf"), 0, 0
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, lr)):
            history[k].append(v)
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr:.2e}")

        if va_loss < best_val:
            best_val, best_epoch, bad_epochs = va_loss, epoch, 0
            torch.save(model.state_dict(), cfg.checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
                break

    history["best_epoch"] = best_epoch
    history["epochs_run"] = len(history["train_loss"])
    history["training_time_sec"] = round(time.time() - start, 2)
    save_json(history, cfg.history_path)
    return history
