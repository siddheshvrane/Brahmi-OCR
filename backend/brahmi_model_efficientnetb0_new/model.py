"""
model.py
--------
Brahmi character classifier: EfficientNet-B0 (feature extractor) + classifier head.

Architecture:
  Input Image (B, 3, 224, 224)
        │
  EfficientNet-B0 backbone (pretrained ImageNet, partial freezing)
        │
  Global Average Pooling  →  (B, 1280)
        │
  Dropout → FC(512) → BN → ReLU → Dropout → FC(num_classes)
        │
  Class logits  (B, num_classes)

Key differences from MobileNetV2 version:
  - Uses EfficientNet-B0 (compound scaling, MBConv blocks, SE attention)
  - EfficientNet-B0 last-layer channels: 1280 (same as MobileNetV2, easier comparison)
  - Deeper classifier head with BatchNorm for stability
  - No Transformer decoder — pure image classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNetB0 Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetB0Features(nn.Module):
    """
    Wraps pretrained EfficientNet-B0 and exposes its feature map output.

    EfficientNet-B0 'features' block outputs (B, 1280, 7, 7) for 224x224 input.
    Early blocks are optionally frozen to speed training and prevent overfitting.

    Args:
        freeze_blocks: number of MBConv blocks to freeze (0=all trainable, 7=all frozen).
                       EfficientNet-B0 has 9 feature sub-modules (0..8).
    """

    def __init__(self, freeze_blocks: int = 4):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # features: Sequential of 9 blocks (0=stem, 1–8=MBConv groups)
        self.features = backbone.features  # Output: (B, 1280, 7, 7)
        self.out_channels = backbone.features[-1][0].out_channels  # 1280

        # Freeze the first `freeze_blocks` feature blocks
        for i, block in enumerate(self.features):
            if i < freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)   # (B, 1280, 7, 7)


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNetB0 Classifier
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetB0Classifier(nn.Module):
    """
    Full EfficientNet-B0 image classifier for Brahmi character recognition.

    Architecture:
        EfficientNetB0Features  →  AdaptiveAvgPool2d(1)
        → Flatten  → Dropout(p)  → Linear(1280, hidden)
        → BatchNorm1d  → GELU  → Dropout(p)  → Linear(hidden, num_classes)

    Args:
        num_classes:   number of Brahmi character classes (e.g., 214)
        hidden_dim:    intermediate FC layer width (default 512)
        dropout:       dropout probability (default 0.3)
        freeze_blocks: EfficientNet-B0 blocks to freeze (default 4 out of 9)
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_blocks: int = 4,
    ):
        super().__init__()
        self.backbone = EfficientNetB0Features(freeze_blocks=freeze_blocks)
        cnn_out = self.backbone.out_channels      # 1280

        self.pool = nn.AdaptiveAvgPool2d(1)       # (B, 1280, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (B, 1280)
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
        """
        Args:
            images: (B, 3, H, W) RGB image tensor, ImageNet-normalised
        Returns:
            logits: (B, num_classes)
        """
        x = self.backbone(images)    # (B, 1280, 7, 7)
        x = self.pool(x)             # (B, 1280, 1, 1)
        return self.classifier(x)    # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = EfficientNetB0Classifier(num_classes=214)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)   # (4, 214)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
