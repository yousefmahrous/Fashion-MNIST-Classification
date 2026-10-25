"""Model module: the CNN architecture.

Input  : 1 x 28 x 28 grayscale image
Block 1: [Conv3x3(32)-BN-ReLU] x2 -> MaxPool -> Dropout   (28 -> 14)
Block 2: [Conv3x3(64)-BN-ReLU] x2 -> MaxPool -> Dropout   (14 -> 7)
Block 3: [Conv3x3(128)-BN-ReLU]   -> MaxPool -> Dropout   (7  -> 3)
Head   : Flatten -> Linear(1152, 256) -> BN -> ReLU -> Dropout -> Linear(256, 10)
"""
import torch
import torch.nn as nn

from config import Config


def conv_block(in_ch: int, out_ch: int, n_convs: int, dropout: float) -> nn.Sequential:
    layers = []
    for i in range(n_convs):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    layers += [nn.MaxPool2d(2), nn.Dropout2d(dropout)]
    return nn.Sequential(*layers)


class FashionCNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, 2, cfg.dropout_conv),
            conv_block(32, 64, 2, cfg.dropout_conv),
            conv_block(64, 128, 1, cfg.dropout_conv),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_fc),
            nn.Linear(256, cfg.num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
