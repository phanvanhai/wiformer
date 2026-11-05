import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .bidirectional_mamba import FusionModel


def get_bimamba_config(dataset, input_shape, num_classes):
    dataset = dataset.lower()
    channels = input_shape[1]
    if dataset == "ut_har":
        return {
            "depth": 8,
            "embed_dim": input_shape[0],
            "channels": channels,
            "num_classes": num_classes,
            "in_channels": input_shape[0],
            "out_channels": input_shape[0],
            "kernel_size": 5,
            "groups": 3,
        }
    elif dataset == "sshar_esp":
        return {
            "depth": 2,
            "embed_dim": input_shape[0],
            "channels": channels,
            "num_classes": num_classes,
            "in_channels": input_shape[0],
            "out_channels": input_shape[0],
            "kernel_size": 5,
            "groups": input_shape[0] // 28,
        }
    elif dataset == "sshar_asus":
        return {
            "depth": 2,
            "embed_dim": input_shape[0],
            "channels": channels,
            "num_classes": num_classes,
            "in_channels": input_shape[0],
            "out_channels": input_shape[0],
            "kernel_size": 5,
            "groups": input_shape[0] // 28,
        }
    elif dataset == "xrf55":
        return {
            "depth": 2,
            "embed_dim": input_shape[0],
            "channels": channels,
            "num_classes": num_classes,
            "in_channels": input_shape[0],
            "out_channels": input_shape[0],
            "kernel_size": 5,
            "groups": 9,
        }
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}"
        )


class BiMamba(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        dataset=None,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
        )

        config = get_bimamba_config(
            dataset=dataset or "unknown",
            input_shape=input_shape,
            num_classes=num_classes,
        )

        self.model = FusionModel(**config)

    def forward(self, x):
        if (
            x.ndim == 3
            and x.shape[1] == self.input_shape[1]
            and x.shape[2] == self.input_shape[0]
        ):
            x = x.transpose(1, 2)
        return self.model(x)
