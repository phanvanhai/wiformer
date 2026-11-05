import torch
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(
        self,
        input_shape=None,
        num_classes=22,
        **kwargs,
    ):
        super(ResNet18, self).__init__()

        self.reshape = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(15, 23), stride=(3, 9), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 23), stride=1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        try:
            import torchvision.models as torchvision_models
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for ResNet18. Install torchvision or remove resnet18 from the benchmark models."
            ) from exc

        resnet = torchvision_models.resnet18(weights=None)

        self.max_pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        if x.dim() == 5:
            b, c1, c2, h, w = x.shape
            x = x.view(b, c1 * c2, h, w)

        x = self.reshape(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
