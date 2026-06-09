import os
import sys

# ── Fix src module resolution (works locally and in CI) ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.models as models
from src.unet import UNet   # ← correct, NOT from src.model


class TomatoFusionModel(nn.Module):
    def __init__(self, num_classes=4, backbone="resnet50"):
        super().__init__()

        # ── Branch 1: U-Net ───────────────────────────────────────────────
        self.unet = UNet(in_channels=3, out_channels=1)

        # ── Branch 2: CNN Backbone ────────────────────────────────────────
        if backbone == "resnet50":
            base         = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
            self.cnn     = base
            self.cnn.fc  = nn.Identity()
            cnn_dim      = 2048

        elif backbone == "densenet121":
            base                = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1
            )
            self.cnn            = base
            self.cnn.classifier = nn.Identity()
            cnn_dim             = 1024

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze ALL CNN params — Phase 1 trains only fusion_head
        for param in self.cnn.parameters():
            param.requires_grad = False

        # ── Feature Fusion Head ───────────────────────────────────────────
        # cnn outputs [B, 2048], unet lpf outputs [B, 1] → concat = [B, 2049]
        self.fusion_head = nn.Sequential(
            nn.Linear(cnn_dim + 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # ── Branch 1: Leaf Area Density ───────────────────────────────────
        mask = self.unet(x)                            # [B, 1, 224, 224]
        lpf  = mask.mean(dim=[1, 2, 3]).unsqueeze(1)   # [B, 1]

        # ── Branch 2: Visual Features ─────────────────────────────────────
        features = self.cnn(x)                         # [B, 2048]
        if features.dim() == 4:
            features = features.view(features.size(0), -1)

        # ── Fusion ────────────────────────────────────────────────────────
        fused = torch.cat([lpf, features], dim=1)      # [B, 2049]
        return self.fusion_head(fused)                 # [B, num_classes]

    def unfreeze_backbone(self, layers=("layer3", "layer4")):
        for name, param in self.cnn.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = True
        print(f"[INFO] Unfroze CNN layers: {list(layers)}")

    def get_leaf_density(self, x):
        self.eval()
        with torch.no_grad():
            mask = self.unet(x)
            lpf  = mask.mean(dim=[1, 2, 3])
        return round(lpf.item(), 4)