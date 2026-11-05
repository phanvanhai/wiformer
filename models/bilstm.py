import torch.nn as nn

from .base import BaseModel


class BiLSTM(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        hidden_dim=64,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
        )

        # UT_HAR input is typically (batch, 90, 250)
        # where 90 = feature dim, 250 = time steps.
        input_dim = input_shape[0]

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=False,
        )

        # Match the architecture you provided, but keep it generic
        # instead of hard-coding 7 output classes.
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Dataset format: (B, C, T) = (B, 90, 250)
        # LSTM expects: (T, B, C) = (250, B, 90)
        x = x.permute(2, 0, 1)

        _, (ht, _) = self.lstm(x)

        # ht[-1] -> last direction hidden state, shape: (B, hidden_dim)
        output = self.fc(ht[-1])

        return output
