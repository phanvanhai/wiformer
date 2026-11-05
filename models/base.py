import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class of all benchmark models.
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError
