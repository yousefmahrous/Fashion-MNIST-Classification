# Fashion-MNIST Classification

A PyTorch-based image classification project that trains a convolutional neural network (CNN) to classify the 10 classes in the Fashion-MNIST dataset. The project includes data loading, preprocessing and augmentation, model definition, training loop with early stopping, evaluation, and metrics export.

The repository is designed to be easy to run and inspect, with all important hyperparameters centralized in a single configuration class and with outputs saved automatically to an `outputs/` directory.

## Table of Contents

- [Overview](#overview)
- [Project Features](#project-features)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training Workflow](#training-workflow)
- [Evaluation and Outputs](#evaluation-and-outputs)
- [Generated Files](#generated-files)
- [Requirements](#requirements)
- [Notes](#notes)

## Overview

This project uses the Fashion-MNIST dataset, which contains 70,000 grayscale images of clothing items split into:

- 60,000 training images
- 10,000 test images

The model predicts one of the following categories:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The implementation follows a modular structure with separate files for:

- configuration
- data loading
- preprocessing/augmentation
- model definition
- training
- evaluation
- utilities
- main execution entry point

## Project Features

- Fashion-MNIST dataset loading through `torchvision.datasets`
- Train/validation split with deterministic randomization using a fixed seed
- Preprocessing pipeline with normalization and augmentation
- Custom CNN model built with PyTorch
- Batch normalization, dropout, and ReLU activations
- Learning rate scheduling with `ReduceLROnPlateau`
- Early stopping based on validation loss
- Best checkpoint saving
- Test-set evaluation with accuracy, precision, recall, F1, confusion matrix, and per-class metrics
- JSON metrics export and saved training history
- Sample image grid export for inspecting preprocessing output

## Repository Structure

```text
Fashion-MNIST-Classification/
├── config.py
├── data_loader.py
├── evaluate.py
├── main.py
├── model.py
├── preprocessing.py
├── requirements.txt
├── train.py
├── utils.py
├── outputs/
│   ├── best_model.pt
│   ├── history.json
│   ├── metrics.json
│   └── preprocessed_samples.png
└── README.md
```

### File Descriptions

| File | Purpose |
| --- | --- |
| `config.py` | Stores the project configuration, dataset paths, model settings, training hyperparameters, and output paths using a `Config` dataclass |
| `data_loader.py` | Downloads the dataset, splits it into train/validation sets, and builds `DataLoader` objects |
| `preprocessing.py` | Defines training and evaluation transforms, performs normalization, augmentation, and saves a sample preview grid |
| `model.py` | Defines the custom CNN architecture (`FashionCNN`) and parameter counting helper |
| `train.py` | Implements the training loop, validation loop, early stopping, checkpointing, LR scheduling, and history logging |
| `evaluate.py` | Computes evaluation metrics and saves results to JSON |
| `utils.py` | Provides helpers for device selection, seeding, JSON saving, and general utility functions |
| `main.py` | Runs the end-to-end workflow: dataset setup, training, evaluation, and output saving |
| `requirements.txt` | Lists the project dependencies |
| `outputs/` | Stores model checkpoints, evaluation metrics, history, and preprocessing examples |

## Dataset

The project uses the Fashion-MNIST dataset, which is one of the standard benchmark datasets for image classification. It contains grayscale images of clothing items with the following class labels:

- `T-shirt/top`
- `Trouser`
- `Pullover`
- `Dress`
- `Coat`
- `Sandal`
- `Shirt`
- `Sneaker`
- `Bag`
- `Ankle boot`

The `data_loader.py` script downloads the dataset automatically if it is not already present in the configured data directory. The training set is split into a validation subset using a fixed random permutation generated from the configured seed.

## Model Architecture

The architecture is defined in `model.py` and uses a CNN with three convolutional blocks followed by a classifier head.

### Convolutional Blocks

The model uses repeated convolutional layers with:

- 3x3 kernels
- Batch normalization
- ReLU activation
- Max pooling
- Dropout

The structure is roughly:

- Block 1: 32 channels
- Block 2: 64 channels
- Block 3: 128 channels

### Classifier Head

After feature extraction, the model flattens the output and passes it through:

- Linear layer (`128 * 3 * 3` -> `256`)
- Batch normalization
- ReLU
- Dropout
- Final linear output for 10 classes

This results in a compact yet effective model for Fashion-MNIST classification.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yousefmahrous/Fashion-MNIST-Classification.git
cd Fashion-MNIST-Classification
pip install -r requirements.txt
```

## Quick Start

Run the complete training and evaluation pipeline:

```bash
python main.py
```

You can also override common settings:

```bash
python main.py --epochs 30 --lr 5e-4 --batch-size 64
```

Available CLI args include:

- `--epochs`
- `--batch-size`
- `--lr`
- `--data-root`
- `--output-dir`
- `--num-workers`

## Configuration

The project uses the `Config` dataclass in `config.py` to centralize all training and model settings.

Important configuration values include:

```python
@dataclass
class Config:
    data_root: str = "data"
    output_dir: str = "outputs"
    val_split: float = 0.1
    batch_size: int = 128
    num_workers: int = 2
    mean: float = 0.2860
    std: float = 0.3530
    rotation_degrees: int = 10
    crop_padding: int = 2
    hflip_prob: float = 0.5
    erasing_prob: float = 0.1
    num_classes: int = 10
    dropout_conv: float = 0.25
    dropout_fc: float = 0.5
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
```

### Why this matters

- `val_split` controls the validation set size
- `batch_size` affects GPU/CPU memory usage and training speed
- `mean` and `std` are used for image normalization
- `rotation_degrees`, `crop_padding`, and `hflip_prob` control augmentation strength
- `dropout_conv` and `dropout_fc` reduce overfitting
- `patience` controls early stopping behavior
- `seed` ensures reproducibility

## Training Workflow

When you run `python main.py`, the following sequence happens:

1. A `Config` instance is created and updated from CLI arguments
2. A deterministic random seed is set
3. The available device is selected (`cuda` if available, otherwise `cpu`)
4. Data loaders are created for train, validation, and test sets
5. A sample image grid is generated and saved
6. The CNN model is initialized and moved to the target device
7. Training begins with `fit()` in `train.py`
8. Validation metrics are computed each epoch
9. The learning rate is adjusted using `ReduceLROnPlateau`
10. Early stopping is applied based on validation loss
11. The best model checkpoint is saved
12. The fully trained model is evaluated on the test set
13. Metrics and training history are saved to JSON files

### Training Details

The `fit()` function uses:

- `CrossEntropyLoss`
- `AdamW` optimizer
- learning rate scheduling
- early stopping

During training, both train and validation loss/accuracy are logged and written to `outputs/history.json`.

## Evaluation and Outputs

After training, the model is evaluated on the test set and the following metrics are exported:

- accuracy
- macro precision
- macro recall
- macro F1-score
- weighted precision
- weighted recall
- weighted F1-score
- per-class metrics
- confusion matrix
- dataset/model metadata
- training summary information

The evaluation logic lives in `evaluate.py` and writes results to `outputs/metrics.json`.

### Example Outputs

The project saves these files under `outputs/`:

- `best_model.pt` — checkpoint for the best model found during training
- `history.json` — training and validation history over epochs
- `metrics.json` — final evaluation metrics
- `preprocessed_samples.png` — visual example of the transformed dataset

## Generated Files

When running the script for the first time, the project will generate an `outputs/` directory containing:

```text
outputs/
├── best_model.pt
├── history.json
├── metrics.json
└── preprocessed_samples.png
```

If the directory already exists, the files are overwritten as needed.

## Requirements

The repository requires the following Python packages:

```text
torch>=2.1
torchvision>=0.16
numpy
scikit-learn
matplotlib
```

These are listed in `requirements.txt`.

## Notes

- This project is intentionally modular, making it easy to extend or replace parts of the pipeline.
- All configuration lives in one place for easier experimentation.
- The use of a validation split and early stopping makes the training process more stable and reproducible.
- The repo is suitable as a learning project for PyTorch-based computer vision and image classification workflows.

## License

This project does not appear to include an explicit license file in the repository. If you plan to reuse or distribute it, please check the repository permissions or add an appropriate license before publishing or sharing it publicly.

## Contributing

You can extend the project by:

- experimenting with different CNN structures
- adding more advanced data augmentation
- implementing custom callbacks
- trying alternative optimizers or schedulers
- evaluating more performance metrics
- saving training plots and confusion matrix visualizations

## Summary

This repository implements an end-to-end workflow for Fashion-MNIST classification using a deep CNN in PyTorch. It is a strong example of a compact, organized machine learning project with data preprocessing, training, validation, early stopping, checkpointing, and metrics analysis all integrated into a modular codebase.
