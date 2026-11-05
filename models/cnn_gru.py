import torch
import torch.nn as nn

from .base import BaseModel


class CNN_GRU(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
        )

        channels, sub, time = input_shape

        self.cnn = nn.Sequential(
            # Conv block 1
            nn.Conv2d(
                channels,
                128,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Dropout(0.6),

            # Conv block 2
            nn.Conv2d(
                128,
                128,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        # Pooling không padding:
        # 56 -> 28 -> 14
        # 30 -> 15 -> 7
        sub_out = (sub // 2) // 2

        self.gru = nn.GRU(
            input_size=128 * sub_out,
            hidden_size=256,
            batch_first=True,
        )

        self.fc = nn.Linear(
            256,
            num_classes,
        )

    def forward(self, x):
        # Input:
        # (B, 12, 56, 1000)

        x = self.cnn(x)

        # sub=56:
        # (B, 128, 14, 250)
        #
        # sub=30:
        # (B, 128, 7, 250)

        # TimeDistributed Flatten
        x = x.permute(0, 3, 1, 2)

        # sub=56 -> (B, 250, 1792)
        # sub=30 -> (B, 250, 896)
        x = x.flatten(2)

        x, _ = self.gru(x)

        # Last timestep
        x = x[:, -1]

        return self.fc(x)