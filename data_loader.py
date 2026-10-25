"""Data module: downloads FashionMNIST, splits train/val, builds DataLoaders."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from config import Config
from preprocessing import get_train_transform, get_eval_transform


def get_dataloaders(cfg: Config):
    """Return (train_loader, val_loader, test_loader).

    Two copies of the training set are created so that the training subset gets
    augmentation while the validation subset uses the clean eval transform.
    Both use the SAME shuffled indices, so there is no leakage between them.
    """
    train_aug = datasets.FashionMNIST(cfg.data_root, train=True, download=True,
                                      transform=get_train_transform(cfg))
    train_clean = datasets.FashionMNIST(cfg.data_root, train=True, download=True,
                                        transform=get_eval_transform(cfg))
    test_set = datasets.FashionMNIST(cfg.data_root, train=False, download=True,
                                     transform=get_eval_transform(cfg))

    n_total = len(train_aug)
    n_val = int(n_total * cfg.val_split)
    gen = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n_total, generator=gen).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = Subset(train_aug, train_idx)
    val_set = Subset(train_clean, val_idx)

    pin = torch.cuda.is_available()
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=pin)
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader
