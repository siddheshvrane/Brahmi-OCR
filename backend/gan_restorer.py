"""
Brahmi Character Restoration Module
Uses exact code from brahmi_inference.py (Brahmi_Model_Export).
damage_type is fixed to "binary" which is the best-performing mode.

Pipeline per crop:
  1. Grayscale + resize to 256x256
  2. Auto damage detection (shape-based, score=0.85)
  3. Prepare model input EXACTLY as training (binary mode)
  4. GAN inference (UNetGenerator: img + mask + sobel edges)
  5. Return as RGB PIL Image for downstream compatibility
"""

import os
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ============================================================================
# MODEL ARCHITECTURE — exact copy from brahmi_inference.py
# ============================================================================

class _ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_c, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.e1 = _ConvBlock(in_channels, 64,  normalize=False)
        self.e2 = _ConvBlock(64,  128)
        self.e3 = _ConvBlock(128, 256)
        self.e4 = _ConvBlock(256, 512)
        self.e5 = _ConvBlock(512, 512)
        self.e6 = _ConvBlock(512, 512)
        self.d6 = _UpBlock(512,  512, dropout=True)
        self.d5 = _UpBlock(1024, 512, dropout=True)
        self.d4 = _UpBlock(1024, 256)
        self.d3 = _UpBlock(512,  128)
        self.d2 = _UpBlock(256,  64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x);  e2 = self.e2(e1); e3 = self.e3(e2)
        e4 = self.e4(e3); e5 = self.e5(e4); e6 = self.e6(e5)
        d6 = self.d6(e6)
        d5 = self.d5(torch.cat([d6, e5], 1))
        d4 = self.d4(torch.cat([d5, e4], 1))
        d3 = self.d3(torch.cat([d4, e3], 1))
        d2 = self.d2(torch.cat([d3, e2], 1))
        return self.final(torch.cat([d2, e1], 1))


# ============================================================================
# PREPROCESSING — exact copy from brahmi_inference.py
# ============================================================================

def _compute_sobel(img):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                       device=img.device).view(1, 1, 3, 3)
    ky  = kx.transpose(2, 3)
    i01 = (img + 1.) * .5
    gx  = F.conv2d(i01, kx, padding=1)
    gy  = F.conv2d(i01, ky, padding=1)
    return ((torch.sqrt(gx**2 + gy**2 + 1e-8) * .25).clamp(0, 1) * 2.) - 1.


def _prepare_model_input(img_np: np.ndarray, mask_np: np.ndarray):
    """
    Binary damage mode (hardcoded — best performing):
      Masked pixels → set to 128 (= 0.0 in [-1,1] space).
      soft_mask = mask (1.0 where damaged).
    Exactly mirrors BrahmiDataset.__getitem__ during training.
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    mask_f    = mask_np.astype(np.float32)           # [0,1] float
    erased_np = img_np.copy().astype(np.float32)
    erased_np[mask_f > 0.5] = 128.0                  # 128 → 0.0 in [-1,1]
    img_t     = tf(Image.fromarray(erased_np.astype(np.uint8))).unsqueeze(0)
    soft_mask = mask_f

    mask_t = torch.from_numpy(soft_mask).unsqueeze(0).unsqueeze(0).float()
    return img_t, mask_t


@torch.no_grad()
def _run_model(G, img_t, mask_t, device):
    img   = img_t.to(device)
    mask  = mask_t.to(device)
    edges = _compute_sobel(img)
    amp   = device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=amp):
        out = G(torch.cat([img, mask, edges], 1))
    return (out[0, 0].cpu().float().numpy() * .5 + .5).clip(0, 1)


# ============================================================================
# DAMAGE DETECTION — exact copy from brahmi_inference.py (score=0.85)
# ============================================================================

def _detect_damage(img_np, low=100, high=220, min_area=400, dilation=12):
    H, W = img_np.shape
    f    = img_np.astype(np.float32)
    raw  = ((f >= low) & (f <= high)).astype(np.uint8)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cln  = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cln)

    dmg = np.zeros((H, W), dtype=np.uint8)
    kept, rejected = [], []

    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        bw   = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh   = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            rejected.append((area, 0.0, f"too_small({area})"))
            continue
        aspect    = max(bw, bh) / (min(bw, bh) + 1e-5)
        img_frac  = area / (H * W)
        diag_span = np.sqrt(bw**2 + bh**2) / np.sqrt(H**2 + W**2)
        blob      = (labels == lbl).astype(np.uint8)
        cnts, _   = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity  = 0.5
        if cnts:
            cnt      = max(cnts, key=cv2.contourArea)
            ha       = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / (ha + 1e-5)
        score = 0.; notes = []
        if img_frac > 0.025:  score += 0.25; notes.append(f"large({img_frac:.2f})")
        if img_frac > 0.06:   score += 0.15; notes.append("very_large")
        if aspect   > 2.5:    score += 0.20; notes.append(f"elongated({aspect:.1f}x)")
        if aspect   > 5.0:    score += 0.15; notes.append("very_elongated")
        if diag_span > 0.30:  score += 0.20; notes.append(f"spans({diag_span:.2f})")
        if diag_span > 0.55:  score += 0.15; notes.append("crosses_image")
        if 0.25 < solidity < 0.92: score += 0.10; notes.append(f"sol({solidity:.2f})")
        if score >= 0.45:
            dmg[labels == lbl] = 1
            kept.append((area, score, " + ".join(notes) if notes else "no_match"))
        else:
            rejected.append((area, score, " + ".join(notes) if notes else "no_match"))

    print(f"   Components: {n-1} | kept: {len(kept)} | rejected: {len(rejected)}")

    if dilation > 0 and dmg.any():
        kd  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation*2+1, dilation*2+1))
        dmg = cv2.dilate(dmg, kd)

    return dmg.astype(np.float32)


# ============================================================================
# RESTORER CLASS — public interface for app.py
# ============================================================================

class GANRestorer:
    """
    Brahmi character restoration pipeline backed by the exported epoch_0250 model.
    Compatible with app.py: restore(pil_img) -> RGB PIL Image (256x256).
    Uses binary damage_type exclusively (best-performing mode).
    """

    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)

        print(f"Loading GAN Restorer from {model_path}...")
        G = UNetGenerator(3, 1).to(self.device)
        ck = torch.load(model_path, map_location=self.device, weights_only=False)
        G.load_state_dict(ck["G_state"] if "G_state" in ck else ck)
        print(f"✓ Model loaded (epoch {ck.get('epoch', '?')}) on {self.device}")
        G.eval()
        self.G = G

    def needs_restoration(self, pil_img):
        """
        Returns True if the crop contains any significant damage blob.

        Uses the full _detect_damage() scoring heuristic (same as restore())
        with dilation=0 so we detect both:
          - Rectangular gray boxes  (large area → high img_frac score)
          - Crack/line damage        (thin but long → high aspect + diag_span score)

        The 0.45 score threshold inside _detect_damage() prevents false positives
        from normal ink anti-aliasing (small, compact, low-span blobs).
        """
        img_np = np.array(
            pil_img.convert('L').resize((256, 256), Image.LANCZOS)
        )
        mask_f = _detect_damage(img_np, dilation=0)   # score-based detection, no dilation

        if mask_f.max() == 0:
            print("   Damage check: no damage → SKIP")
            return False

        coverage = float(mask_f.mean())
        print(f"   Damage check: {coverage*100:.1f}% detected → RESTORE")
        return True


    def _get_damage_mask(self, img_np, dilation=3):
        """Run damage detection on a 256x256 grayscale uint8 array.

        dilation=3 (tight) is used for inpainting — prevents mask from expanding
        into loop interiors and surrounding clean strokes. The larger default=12
        from brahmi_inference.py was tuned for full inscription images, not crops.

        Returns (mask_f, coverage) where coverage is fraction in [0,1]."""
        mask_f   = _detect_damage(img_np, dilation=dilation)
        coverage = float(mask_f.mean())
        return mask_f, coverage

    def restore(self, pil_img):
        """
        Restore a damaged character crop.
        Args:
            pil_img: PIL Image (any mode, any size)
        Returns:
            Restored PIL Image in RGB mode (256x256)
        """
        # Convert to grayscale 256x256
        img_np = np.array(
            pil_img.convert('L').resize((256, 256), Image.LANCZOS)
        )  # uint8 [0..255]

        # Auto-detect damage
        mask_f, coverage = self._get_damage_mask(img_np)

        if mask_f.max() == 0:
            # No damage mask at all — return original
            return pil_img.convert('L').resize((256, 256), Image.LANCZOS).convert('RGB')

        # Prepare input exactly as training (binary mode)
        img_t, mask_t = _prepare_model_input(img_np, mask_f)

        # Run model — GAN output covers the whole image
        restored = _run_model(self.G, img_t, mask_t, self.device)   # float [0,1]

        # ── INPAINTING COMPOSITE ──────────────────────────────────────────────
        # Rule 1: Only replace pixels INSIDE the damage mask.
        #         Feather mask edges with Gaussian blur to avoid hard seams.
        # Rule 2: Inside the mask, only accept GAN output if it is DARKER than
        #         the original (i.e., GAN adds ink). If GAN predicts lighter,
        #         keep the original — this prevents GAN hallucinating strokes
        #         inside white loop interiors even if the dilated mask covers them.
        original_norm = img_np.astype(np.float32) / 255.0            # [0,1]

        # Tight mask feathering (sigma 1px — mask is already tight at dilation=3)
        mask_smooth = cv2.GaussianBlur(
            mask_f.astype(np.float32), ksize=(5, 5), sigmaX=1.0
        ).clip(0.0, 1.0)

        # Ink-only constraint: within the mask, take whichever is darker
        # original if original is already darker (existing stroke stays)
        # GAN output if GAN is darker (GAN adds missing ink)
        restored_constrained = np.minimum(original_norm, restored)

        composite = original_norm * (1.0 - mask_smooth) + restored_constrained * mask_smooth
        out_img   = (composite * 255).clip(0, 255).astype(np.uint8)
        # ─────────────────────────────────────────────────────────────────────

        return Image.fromarray(out_img, mode='L').convert('RGB')

