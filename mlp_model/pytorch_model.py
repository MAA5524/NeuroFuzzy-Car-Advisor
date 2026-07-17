import torch
import torch.nn as nn


class CarPriceMLP(nn.Module):
    """
    Deep MLP for car price regression.

    Design principles for small tabular datasets (~1700 samples):
    - NO BatchNorm: causes train/eval discrepancy via running stats
    - Small Dropout (0.1): just enough to prevent overfitting
    - ReLU: simple and effective for tabular regression
    - Kaiming init: correct weight initialization for ReLU networks
    - Modest size: avoids over-parameterization relative to dataset size

    Architecture: input_dim -> 64 -> 32 -> 1
    """

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1: 64 hidden units
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            # Layer 2: 32 hidden units
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output layer
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """He (Kaiming) initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)
