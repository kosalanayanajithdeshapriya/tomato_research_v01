import torch
import torch.nn as nn
import torchvision.models as models
from unet import UNet


class TomatoFusionModel(nn.Module):
    def __init__(self, num_classes=4, backbone="resnet50"):
        super().__init__()

        self.unet = UNet(in_channels=3, out_channels=1)

        if backbone == "resnet50":
            base              = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.cnn_features = nn.Sequential(*list(base.children())[:-1])
            cnn_dim           = 2048
        elif backbone == "densenet121":
            base              = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.cnn_features = base.features
            cnn_dim           = 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.cnn = self.cnn_features  # alias for train.py unfreeze_backbone

        for param in self.cnn_features.parameters():
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
        features = self.cnn_features(x)
        features = features.view(features.size(0), -1)         # [B, 2048]
        fused    = torch.cat([lpf, features], dim=1)           # [B, 2049]
        return self.fusion_head(fused)

    def unfreeze_backbone(self, layers=("layer3", "layer4")):
        layer_map = {"layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7}
        for layer in layers:
            idx = layer_map.get(layer, -1)
            if idx >= 0:
                for param in self.cnn_features[idx].parameters():
                    param.requires_grad = True
        print(f"[INFO] Unfroze CNN layers: {list(layers)}")

    def get_leaf_density(self, x):
        self.eval()
        with torch.no_grad():
            mask = self.unet(x)
        return round(mask.mean(dim=[1, 2, 3]).item(), 4)