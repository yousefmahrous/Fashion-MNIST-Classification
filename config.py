"""Central configuration for the FashionMNIST CNN project."""
from dataclasses import dataclass, asdict
from pathlib import Path

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


@dataclass
class Config:
    # Paths
    data_root: str = "data"
    output_dir: str = "outputs"

    # Data
    val_split: float = 0.1          # fraction of the 60k train set used for validation
    batch_size: int = 128
    num_workers: int = 2
    mean: float = 0.2860            # FashionMNIST train-set statistics
    std: float = 0.3530

    # Augmentation (training only)
    rotation_degrees: int = 10
    crop_padding: int = 2
    hflip_prob: float = 0.5
    erasing_prob: float = 0.1

    # Model
    num_classes: int = 10
    dropout_conv: float = 0.25
    dropout_fc: float = 0.5

    # Training
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5               # early-stopping patience (epochs)
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def metrics_path(self) -> Path:
        return Path(self.output_dir) / "metrics.json"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "best_model.pt"

    @property
    def history_path(self) -> Path:
        return Path(self.output_dir) / "history.json"
