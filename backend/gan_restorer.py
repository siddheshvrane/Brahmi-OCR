"""
Brahmi Character Restoration Module
Uses the correct Pix2Pix UNetGenerator architecture matching the checkpoint.
Pipeline per crop:
  1. Grayscale conversion
  2. Damage detection (white holes near dark strokes)
  3. Canny edge extraction
  4. GAN inference (input: grayscale + damage mask + edges)
  5. Post-processing: invert + Otsu threshold on damaged regions + morphological closing
  6. Return as RGB PIL Image for downstream compatibility
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ============================================================================
# MODEL ARCHITECTURE — must match the checkpoint exactly
# ============================================================================

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 64, normalize=False)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.enc5 = self._conv_block(512, 512)
        self.enc6 = self._conv_block(512, 512)
        self.dec6 = self._upconv_block(512, 512)
        self.dec5 = self._upconv_block(1024, 512)
        self.dec4 = self._upconv_block(1024, 256)
        self.dec3 = self._upconv_block(512, 128)
        self.dec2 = self._upconv_block(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def _conv_block(self, in_c, out_c, normalize=True):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def _upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        d6 = self.dec6(e6)
        d5 = self.dec5(torch.cat([d6, e5], 1))
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        return self.final(torch.cat([d2, e1], 1))


# ============================================================================
# RESTORER CLASS
# ============================================================================

class GANRestorer:
    """
    Brahmi character restoration pipeline.
    Drop-in replacement for the previous GANRestorer.
    Compatible with app.py: restore(pil_img) -> RGB PIL Image (256x256).
    """

    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.model = UNetGenerator(in_channels=3, out_channels=1).to(self.device)

        print(f"Loading GAN Restorer from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['G_state'])
        self.model.eval()

        # Single-channel normalization (model trained on grayscale)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        print(f"GAN Restorer loaded successfully on {self.device}.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def restore(self, pil_img):
        """
        Restore a damaged character crop.
        Args:
            pil_img: PIL Image (any mode, any size)
        Returns:
            Restored PIL Image in RGB mode (256x256)
        """
        # Work in grayscale
        gray = pil_img.convert('L')
        original_size = gray.size
        gray_256 = gray.resize((256, 256), Image.Resampling.LANCZOS)
        img_np = np.array(gray_256)  # uint8 [0..255]

        # Step 1: Detect damage (white holes near black strokes)
        damage_mask = self._detect_damage(img_np)

        # Step 2: Run the GAN
        restored_np = self._run_inference(gray_256, damage_mask)

        # Step 3: Post-process → fill holes, threshold
        final_np = self._post_process(img_np, restored_np, damage_mask)

        # Return as 256x256 RGB PIL (for downstream compatibility with app.py)
        rgb = Image.fromarray(final_np, mode='L').convert('RGB')
        return rgb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_damage(self, img_np):
        """
        Return binary float32 mask where 1 = damaged (white holes near dark strokes).
        """
        # White pixels are potential damage
        damage_mask = (img_np > 200).astype(np.float32)

        # Only flag white pixels that are near actual black strokes
        strokes = (img_np < 50).astype(np.uint8)
        kernel = np.ones((7, 7), np.uint8)
        stroke_region = cv2.dilate(strokes, kernel, iterations=1)
        damage_mask = damage_mask * stroke_region

        # Remove tiny noise
        kernel_small = np.ones((3, 3), np.uint8)
        damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, kernel_small)
        return damage_mask

    def _run_inference(self, gray_pil, damage_mask):
        """
        Build the 3-channel input and run the generator.
        Channels: [grayscale image | damage mask | canny edges]
        Returns: uint8 numpy array (256x256).
        """
        # Channel 1 — grayscale image (normalised -1..1)
        I_damaged = self.transform(gray_pil).unsqueeze(0).to(self.device)  # [1,1,256,256]

        # Channel 2 — damage mask
        mask_t = torch.from_numpy(damage_mask).unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,256,256]

        # Channel 3 — canny edges (normalised -1..1)
        img_np = np.array(gray_pil)
        edges = cv2.Canny(img_np, 50, 150).astype(np.float32) / 255.0
        edges_t = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0).to(self.device) * 2.0 - 1.0

        # Concatenate along channel dim → [1, 3, 256, 256]
        input_t = torch.cat([I_damaged, mask_t, edges_t], dim=1)

        with torch.no_grad():
            output_t = self.model(input_t)  # [1, 1, 256, 256]  in [-1, 1]

        # De-normalise
        out_np = output_t[0].cpu().squeeze().numpy()     # [-1, 1]
        out_np = np.clip(out_np * 0.5 + 0.5, 0, 1)      # [0, 1]
        out_np = (out_np * 255).astype(np.uint8)
        return out_np

    def _post_process(self, original_np, restored_np, damage_mask):
        """
        Merge restored pixels back into original using damage mask.
        Steps:
          1. Invert model output (model predicts repair regions)
          2. Otsu threshold on damaged pixels only
          3. Morphological closing to fill remaining small holes
        """
        restored_inverted = 255 - restored_np
        mask_regions = damage_mask > 0.5

        if mask_regions.sum() > 0:
            damaged_pixels = restored_inverted[mask_regions].reshape(-1, 1).astype(np.uint8)
            threshold, _ = cv2.threshold(
                damaged_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            final = original_np.copy()
            final[mask_regions] = np.where(
                restored_inverted[mask_regions] < threshold,
                0,      # black stroke
                255     # white background
            )

            # Morphological closing: fills small hollow areas inside strokes
            strokes = (final < 50).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, kernel)
            final = ((1 - closed) * 255).astype(np.uint8)
        else:
            # No damage detected — return original untouched
            final = original_np.copy()

        return final.astype(np.uint8)
