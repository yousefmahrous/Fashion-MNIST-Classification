"""Image-processing module: builds the transform pipelines applied to every image.

Training pipeline (with augmentation):
    RandomCrop(28, pad=2) -> RandomHorizontalFlip -> RandomRotation
    -> ToTensor (uint8 [0,255] -> float [0,1], shape 1x28x28)
    -> Normalize(mean, std) -> RandomErasing

Evaluation pipeline (deterministic, no augmentation):
    ToTensor -> Normalize(mean, std)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

from config import Config, CLASS_NAMES


def get_train_transform(cfg: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(28, padding=cfg.crop_padding),
        transforms.RandomHorizontalFlip(p=cfg.hflip_prob),
        transforms.RandomRotation(cfg.rotation_degrees),
        transforms.ToTensor(),
        transforms.Normalize((cfg.mean,), (cfg.std,)),
        transforms.RandomErasing(p=cfg.erasing_prob, scale=(0.02, 0.1)),
    ])


def get_eval_transform(cfg: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((cfg.mean,), (cfg.std,)),
    ])


def denormalize(img, cfg: Config):
    """Undo normalization so a tensor can be displayed as an image."""
    return img * cfg.std + cfg.mean


def save_sample_grid(dataset, cfg: Config, path, n: int = 16) -> None:
    """Save a grid of preprocessed samples so the pipeline can be inspected visually."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    for ax, i in zip(axes.flat, range(n)):
        img, label = dataset[i]
        ax.imshow(denormalize(img, cfg).squeeze(0).clamp(0, 1), cmap="gray")
        ax.set_title(CLASS_NAMES[label], fontsize=7)
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
