import torch
import torch.nn as nn
import torchvision.models as models

try:
    from src.unet import UNet
except ModuleNotFoundError:
    from unet import UNet


class _NamedBackbone(nn.Module):
    """
    Wraps ResNet layers as named attributes so named_parameters()
    returns 'layer3.x.x' style keys, while forward() skips fc.
    """
    def __init__(self, resnet):
        super().__init__()
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool
        # fc intentionally excluded

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x  # [B, 2048, 1, 1]


class TomatoFusionModel(nn.Module):
    def __init__(self, num_classes=4, backbone="resnet50"):
        super().__init__()

        self.unet = UNet(in_channels=3, out_channels=1)

        if backbone == "resnet50":
            base      = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.cnn  = _NamedBackbone(base)   # named layers, no fc
            cnn_dim   = 2048

        elif backbone == "densenet121":
            base      = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.cnn  = base.features
            cnn_dim   = 1024

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.fusion_head = nn.Sequential(
            nn.Linear(cnn_dim + 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        mask     = self.unet(x)
        lpf      = mask.mean(dim=[1, 2, 3]).unsqueeze(1)       # [B, 1]
        features = self.cnn(x)
        features = features.view(features.size(0), -1)         # [B, 2048]
        fused    = torch.cat([lpf, features], dim=1)           # [B, 2049]
        return self.fusion_head(fused)

    def unfreeze_backbone(self, layers=("layer3", "layer4")):
        for name, param in self.cnn.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = True
        print(f"[INFO] Unfroze CNN layers: {list(layers)}")

    def get_leaf_density(self, x):
        self.eval()
        with torch.no_grad():
            mask = self.unet(x)
        return round(mask.mean(dim=[1, 2, 3]).item(), 4)