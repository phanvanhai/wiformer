import torch
import torch.nn as nn
from .base import BaseModel

class MLP(BaseModel):
    """
    MLP baseline.

    Input
    -----
    (batch, C, T)

    Output
    ------
    (batch, num_classes)
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

        input_dim = input_shape[0] * input_shape[1]
        self.fc = nn.Sequential(
            nn.Linear(
                input_dim,
                1024,
            ),
            nn.ReLU(),
            nn.Linear(
                1024,
                128,
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):
        x = x.view(
            x.size(0),
            -1,
        )

        x = self.fc(x)
        return x
