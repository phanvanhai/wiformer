"""Benchmark model package."""

from .mlp import MLP
from .cnn5 import CNN5
from .resnet18 import ResNet18
from .resnet1d import resnet18_1d
from .bilstm import BiLSTM
from .vit import ViT
from .cnn_gru import CNN_GRU
from .bimamba import BiMamba
from .wiformer import WiFormer

SUPPORTED_MODELS = [
    "mlp",
    "cnn5",
    "bilstm",
    "vit",
    "resnet18",
    "resnet1d",
    "cnn_gru",
    "bimamba",
    "wiformer",
]


def list_models():
    return SUPPORTED_MODELS.copy()


def get_model(model_name, input_shape, num_classes, **kwargs):
    model_name = model_name.lower()
    if model_name == "mlp":
        return MLP(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    if model_name == "cnn5":
        return CNN5(
            dataset=kwargs.get("dataset"),
            num_classes=num_classes,
        )

    if model_name == "bilstm":
        return BiLSTM(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    if model_name == "vit":
        return ViT(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    if model_name == "resnet18":
        return ResNet18(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    if model_name == "resnet1d":
        return resnet18_1d(
            inchannel=input_shape[0],
            num_classes=num_classes,
        )

    if model_name == "cnn_gru":
        return CNN_GRU(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    if model_name == "bimamba":
        return BiMamba(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    if model_name == "wiformer":
        return WiFormer(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported model: {model_name}\n"
        f"Supported models: {SUPPORTED_MODELS}"
    )
