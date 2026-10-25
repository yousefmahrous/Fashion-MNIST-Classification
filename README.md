# FashionMNIST CNN

## Structure
| Module | Responsibility |
|---|---|
| `config.py` | All hyperparameters and paths in one `Config` dataclass |
| `preprocessing.py` | Image-processing pipelines (normalization + augmentation) and sample preview |
| `data_loader.py` | Download, train/val split, DataLoaders |
| `model.py` | CNN architecture (`FashionCNN`) |
| `train.py` | Training loop, LR scheduler, early stopping, checkpointing |
| `evaluate.py` | Test-set metrics -> `outputs/metrics.json` |
| `utils.py` | Seed, device, JSON helpers |
| `main.py` | Runs the full pipeline |

## Run
```bash
pip install -r requirements.txt
python main.py                     # or: python main.py --epochs 30 --lr 5e-4
```

## Outputs (`outputs/`)
- `metrics.json` – accuracy, loss, macro/weighted precision/recall/F1, per-class metrics, confusion matrix, training summary, config
- `history.json` – per-epoch train/val loss and accuracy
- `best_model.pt` – best checkpoint (by validation loss)
- `preprocessed_samples.png` – visual check of the preprocessing pipeline
