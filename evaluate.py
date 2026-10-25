"""Evaluation module: computes metrics on the test set and stores them as JSON."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from config import Config, CLASS_NAMES
from model import count_parameters
from utils import save_json


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_preds, all_labels, total_loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item()
        all_preds.append(out.argmax(1).cpu())
        all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return preds, labels, total_loss / len(labels)


def evaluate_and_save(model, test_loader, cfg: Config, device, history: dict | None = None) -> dict:
    preds, labels, test_loss = predict(model, test_loader, device)

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    per_class = {name: {"precision": round(report[name]["precision"], 4),
                        "recall": round(report[name]["recall"], 4),
                        "f1_score": round(report[name]["f1-score"], 4),
                        "support": int(report[name]["support"])}
                 for name in CLASS_NAMES}

    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))

    metrics = {
        "dataset": "FashionMNIST",
        "model": "FashionCNN",
        "num_parameters": count_parameters(model),
        "num_test_samples": int(len(labels)),
        "test_loss": round(test_loss, 4),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "macro_avg": {"precision": round(p_macro, 4), "recall": round(r_macro, 4), "f1_score": round(f_macro, 4)},
        "weighted_avg": {"precision": round(p_w, 4), "recall": round(r_w, 4), "f1_score": round(f_w, 4)},
        "per_class": per_class,
        "confusion_matrix": {"labels": CLASS_NAMES, "matrix": cm.tolist()},
        "training": ({"best_epoch": history["best_epoch"],
                      "epochs_run": history["epochs_run"],
                      "training_time_sec": history["training_time_sec"]} if history else None),
        "config": cfg.to_dict(),
    }
    save_json(metrics, cfg.metrics_path)
    return metrics
