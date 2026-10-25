"""Entry point: wires the modules together.

Usage:
    python main.py                       # defaults
    python main.py --epochs 30 --lr 5e-4 --batch-size 64
"""
import argparse
from pathlib import Path

import torch

from config import Config
from data_loader import get_dataloaders
from evaluate import evaluate_and_save
from model import FashionCNN, count_parameters
from preprocessing import save_sample_grid
from train import fit
from utils import get_device, set_seed


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="FashionMNIST CNN")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--data-root", type=str, default=cfg.data_root)
    p.add_argument("--output-dir", type=str, default=cfg.output_dir)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    a = p.parse_args()
    cfg.epochs, cfg.batch_size, cfg.lr = a.epochs, a.batch_size, a.lr
    cfg.data_root, cfg.output_dir, cfg.num_workers = a.data_root, a.output_dir, a.num_workers
    return cfg


def main(cfg: Config | None = None):
    cfg = cfg or parse_args()
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    # 1) Data + image preprocessing
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    print(f"Train/Val/Test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")
    save_sample_grid(train_loader.dataset, cfg, Path(cfg.output_dir) / "preprocessed_samples.png")

    # 2) Model
    model = FashionCNN(cfg).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # 3) Train
    history = fit(model, train_loader, val_loader, cfg, device)

    # 4) Evaluate best checkpoint on the test set -> JSON
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
    metrics = evaluate_and_save(model, test_loader, cfg, device, history)
    print(f"\nTest accuracy: {metrics['accuracy']:.4f} | macro F1: {metrics['macro_avg']['f1_score']:.4f}")
    print(f"Metrics saved to {cfg.metrics_path}")


if __name__ == "__main__":
    main()
