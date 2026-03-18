"""
model.py
--------
Brahmi character classifier: ResNet-50 (feature extractor) + classifier head.

Architecture:
  Input Image (B, 3, 224, 224)
        │
  ResNet-50 backbone (pretrained ImageNet, partial freezing)
        │
  Global Average Pooling  →  (B, 2048)
        │
  Dropout → FC(512) → BN → ReLU → Dropout → FC(num_classes)
        │
  Class logits  (B, num_classes)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# ResNet50 Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ResNet50Features(nn.Module):
    """
    Wraps pretrained ResNet-50 and exposes its feature map output.

    ResNet-50 outputs (B, 2048, 7, 7) for 224x224 input.
    """

    def __init__(self, freeze_blocks: int = 3):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # ResNet50 components:
        # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,  # block 1
            backbone.layer2,  # block 2
            backbone.layer3,  # block 3
            backbone.layer4,  # block 4
        )
        self.out_channels = 2048

        # Freeze early layers
        # layers in self.features: 0..7
        # we can freeze layer1, layer2 etc.
        # i=0..3 are the stem (conv1..maxpool)
        # i=4 is layer1, i=5 is layer2, i=6 is layer3, i=7 is layer4
        
        freeze_until = 4 + freeze_blocks # if freeze_blocks=2, freeze until layer2
        
        for i, block in enumerate(self.features):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)   # (B, 2048, 7, 7)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet50 Classifier
# ─────────────────────────────────────────────────────────────────────────────

class ResNet50Classifier(nn.Module):
    """
    Full ResNet-50 image classifier for Brahmi character recognition.

    Architecture:
        ResNet50Features  →  AdaptiveAvgPool2d(1)
        → Flatten  → Dropout(p)  → Linear(2048, hidden)
        → BatchNorm1d  → GELU  → Dropout(p)  → Linear(hidden, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_blocks: int = 2,
    ):
        super().__init__()
        self.backbone = ResNet50Features(freeze_blocks=freeze_blocks)
        cnn_out = self.backbone.out_channels      # 2048

        self.pool = nn.AdaptiveAvgPool2d(1)       # (B, 2048, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (B, 2048)
            nn.Dropout(p=dropout),
            nn.Linear(cnn_out, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Kaiming init for FC layers, constant init for BN."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone(images)    # (B, 2048, 7, 7)
        x = self.pool(x)             # (B, 2048, 1, 1)
        return self.classifier(x)    # (B, num_classes)


if __name__ == "__main__":
    model = ResNet50Classifier(num_classes=214)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)   # (4, 214)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
