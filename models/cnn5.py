import torch
import torch.nn as nn


class CNN5(nn.Module):
    def __init__(
        self,
        dataset,
        num_classes,
    ):
        super().__init__()
        dataset = dataset.lower()

        if dataset == "ut_har":
            input_size = (250, 90)
        elif dataset == "sshar_esp":
            input_size = (250, 168)
        elif dataset == "sshar_asus":
            input_size = (250, 672)
        else:
            raise ValueError(
                f"Unknown dataset: {dataset}"
            )

        self.encoder = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=7,
                stride=(3, 1),
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                32,
                64,
                kernel_size=(5, 4),
                stride=(2, 2),
                padding=(1, 0),
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                64,
                96,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                1,
                input_size[0],
                input_size[1],
            )
            dummy = self.encoder(dummy)
            feature_dim = dummy.numel()

        print(
            "LeNet feature size:",
            feature_dim,
        )

        self.fc = nn.Sequential(
            nn.Linear(
                feature_dim,
                128,
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):
        x = x.permute(
            0,
            2,
            1,
        ).unsqueeze(1)

        x = self.encoder(x)
        x = torch.flatten(
            x,
            1,
        )
        x = self.fc(x)

        return x
