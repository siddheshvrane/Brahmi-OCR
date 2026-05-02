#!/usr/bin/env python3
"""
brahmi_ocr.py  —  Brahmi Script OCR  |  Pure MobileNetV2 Classifier  |  Single File
====================================================================================
GPU: NVIDIA RTX 4050 (CUDA 12.1, mixed-precision AMP/FP16, cuDNN benchmark)
Model saved as .keras (HDF5 format, re-loadable for inference & fine-tuning)
All metrics logged permanently to CSV + JSON every epoch (for research paper)

INSTALL (one-time):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install pandas pillow jiwer tqdm h5py

USAGE:
    # Extract dataset + train
    python brahmi_ocr.py train --zip path/to/dataset.zip --epochs 100

    # Quick 1-epoch sanity check
    python brahmi_ocr.py train --zip path/to/dataset.zip --debug

    # Run inference on an image
    python brahmi_ocr.py predict --img path/to/image.png

    # Run inference on a folder
    python brahmi_ocr.py predict --folder path/to/images/
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import csv
import json
import math
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tv_models

try:
    import h5py
except ImportError:
    h5py = None

try:
    from jiwer import cer as _cer, wer as _wer
    _JIWER = True
except ImportError:
    _JIWER = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 ── GPU Setup
# ══════════════════════════════════════════════════════════════════════════════

def configure_gpu() -> torch.device:
    """
    Set up CUDA for maximum performance on RTX 4050.
    Returns the torch device to use.
    """
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available — training on CPU.")
        return torch.device("cpu")

    device = torch.device("cuda")
    props  = torch.cuda.get_device_properties(0)
    vram   = props.total_memory / 1024**2

    torch.backends.cudnn.benchmark     = True   # auto-tune best conv kernel
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")  # TF32 on Ampere Tensor Cores

    print(f"[GPU] {props.name}")
    print(f"      VRAM: {vram:.0f} MB  |  Compute: {props.major}.{props.minor}")
    print(f"      CUDA: {torch.version.cuda}  |  cuDNN: {torch.backends.cudnn.version()}")
    print(f"      Mixed precision (AMP FP16): enabled")
    return device


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── Dataset Extraction
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
SPLIT_NAMES    = {"train", "val", "test", "valid", "validation"}


def extract_zip(zip_path: str, out_dir: str) -> Path:
    zip_path = Path(zip_path)
    out_dir  = Path(out_dir)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting {zip_path.name} → {out_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print("[INFO] Extraction complete.")
    return out_dir


def _find_dataset_root(data_dir: Path) -> Path:
    """
    Auto-descend through single-subdirectory wrapper folders.
    e.g. data/Brahmi_Characters/<class>/img.png
         → descends into data/Brahmi_Characters/ automatically.
    Stops when it finds either:
      - Multiple subdirectories (likely class folders), OR
      - A subdirectory that contains image files directly, OR
      - A split folder (train/val/test)
    """
    current = data_dir
    for _ in range(5):   # max 5 levels deep
        subdirs = [d for d in current.iterdir() if d.is_dir()]
        if len(subdirs) != 1:
            break   # multiple dirs = class folders or split folders
        # Check if the single subdir is a split or class-with-images
        only = subdirs[0]
        if only.name.lower() in SPLIT_NAMES:
            break   # it's a split folder, stop here
        grandchildren_dirs = [d for d in only.iterdir() if d.is_dir()]
        grandchildren_imgs = [f for f in only.iterdir()
                              if f.suffix.lower() in SUPPORTED_EXTS]
        if grandchildren_imgs:
            break   # images directly inside = it IS a class folder
        # The single subdir has only sub-subdirs → it's a wrapper, descend
        print(f"[INFO] Auto-descending into wrapper folder: {only.name}/")
        current = only
    return current


def collect_samples(data_dir: Path) -> Tuple[List[Dict], List[str]]:
    """
    Scan extracted directory. Supports two layouts:
      Flat:       data/<ClassName>/image.png   → auto 80/10/10 split
      Pre-split:  data/train/<ClassName>/image.png
    Auto-detects and descends through single wrapper folders.
    """
    # Find the real dataset root (handles Brahmi_Characters/ wrappers)
    data_dir = _find_dataset_root(data_dir)

    samples: List[Dict] = []
    top_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    has_splits = any(d.name.lower() in SPLIT_NAMES for d in top_dirs)

    if has_splits:
        for split_dir in sorted(top_dirs):
            split = split_dir.name.lower()
            if split not in SPLIT_NAMES:
                continue
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                for img_p in sorted(class_dir.iterdir()):
                    if img_p.suffix.lower() in SUPPORTED_EXTS:
                        samples.append({"image_path": str(img_p.resolve()),
                                        "label": class_dir.name, "split": split})
    else:
        for class_dir in sorted(top_dirs):
            imgs = [p for p in sorted(class_dir.iterdir())
                    if p.suffix.lower() in SUPPORTED_EXTS]
            n = len(imgs)
            for i, img_p in enumerate(imgs):
                split = "train" if i < int(0.8*n) else ("val" if i < int(0.9*n) else "test")
                samples.append({"image_path": str(img_p.resolve()),
                                 "label": class_dir.name, "split": split})

    labels = sorted(set(s["label"] for s in samples))
    return samples, labels


def save_metadata(samples: List[Dict], out_dir: Path) -> Path:
    csv_path = out_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "split", "class_idx"])
        writer.writeheader()
        writer.writerows(samples)
    return csv_path


def prepare_dataset(zip_path: str, out_dir: str = "./data"):
    dp = Path(out_dir)
    csv_f = dp / "metadata.csv"
    
    if csv_f.exists():
        print(f"[INFO] metadata.csv already exists at {csv_f}. Skipping extraction.")
        all_labels = []
        import csv as csv_lib
        with open(csv_f, newline="", encoding="utf-8") as f:
            reader = csv_lib.DictReader(f)
            for row in reader:
                all_labels.append(row["label"])
        labels_set = sorted(set(all_labels))
        label2idx = {lbl: i for i, lbl in enumerate(labels_set)}
        return str(csv_f), labels_set, label2idx

    # Use a different name for the extracted directory to avoid shadowing Path
    extracted_dir = extract_zip(zip_path, str(dp))
    samples, labels = collect_samples(extracted_dir)

    if not samples:
        raise ValueError("No images found in the zip. Check folder structure.")

    label2idx = {lbl: i for i, lbl in enumerate(labels)}
    for s in samples:
        s["class_idx"] = label2idx[s["label"]]

    final_csv = save_metadata(samples, extracted_dir)
    return str(final_csv), labels, label2idx


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── Vocabulary (character-level, for sequence labels)
# ══════════════════════════════════════════════════════════════════════════════

PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
SPECIAL = [PAD, SOS, EOS, UNK]


def build_vocab(labels: List[str]) -> Dict[str, int]:
    chars   = sorted(set(ch for lbl in labels for ch in lbl))
    char2id = {t: i for i, t in enumerate(SPECIAL)}
    for ch in chars:
        if ch not in char2id:
            char2id[ch] = len(char2id)
    return char2id


def encode(label: str, char2id: Dict[str, int]) -> List[int]:
    return [SOS_IDX] + [char2id.get(ch, UNK_IDX) for ch in label] + [EOS_IDX]


def decode_seq(ids: List[int], id2char: Dict[int, str]) -> str:
    out = []
    for i in ids:
        if i == EOS_IDX:
            break
        if i in (PAD_IDX, SOS_IDX):
            continue
        out.append(id2char.get(i, ""))
    return "".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

def get_transform(split: str, img_size: int = 224) -> T.Compose:
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if split == "train":
        return T.Compose([
            T.Resize((int(img_size * 1.1), int(img_size * 1.1))),   # slight oversize then crop
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(0.05),   # Brahmi scripts are NOT mirrored — keep very low
            T.RandomRotation(8),             # slight rotation for variation
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            T.RandomGrayscale(p=0.1),        # teach invariance to stone texture colors
            T.ToTensor(),
            norm,
            T.RandomErasing(p=0.1, scale=(0.02, 0.08)),  # simulate damaged inscriptions
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        norm,
    ])


class BrahmiDataset(Dataset):
    def __init__(self, csv_path: str, char2id: Dict[str, int],
                 split: str = "train", img_size: int = 224,
                 label2idx: Optional[Dict[str, int]] = None):
        self.char2id   = char2id
        self.label2idx = label2idx
        self.transform = get_transform(split, img_size)
        self.samples: List[Tuple[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.samples.append((row["image_path"], row["label"]))
        if not self.samples:
            raise ValueError(f"No samples for split='{split}' in {csv_path}")
        # RAM cache: raw bytes stored after first disk read.
        # All subsequent epochs read from memory -- eliminates disk I/O bottleneck.
        self._cache: dict = {}

    def cache_all(self, limit: int = 0):
        """Pre-load images into RAM. One-time cost; all training epochs after are fast."""
        n = min(limit, len(self.samples)) if limit > 0 else len(self.samples)
        print(f"[Cache] Pre-loading {n:,} images into RAM...", end="", flush=True)
        for i in range(n):
            if i not in self._cache:
                try:
                    with open(self.samples[i][0], "rb") as fp:
                        self._cache[i] = fp.read()
                except Exception:
                    pass
        print(f" done.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import io
        img_path, label = self.samples[idx]
        # Serve from RAM cache if available, else load from disk and cache it
        if idx in self._cache:
            img = Image.open(io.BytesIO(self._cache[idx])).convert("RGB")
        else:
            img = Image.open(img_path).convert("RGB")
            try:
                with open(img_path, "rb") as fp:
                    self._cache[idx] = fp.read()
            except Exception:
                pass
        img = self.transform(img)
        enc = torch.tensor(encode(label, self.char2id), dtype=torch.long)
        class_idx = self.label2idx[label] if self.label2idx else -1
        return img, enc, label, class_idx


def collate_fn(batch):
    imgs, labels, raw, class_indices = zip(*batch)
    imgs  = torch.stack(imgs, 0)
    lens  = torch.tensor([len(l) for l in labels], dtype=torch.long)
    maxl  = lens.max().item()
    padded = torch.full((len(labels), maxl), PAD_IDX, dtype=torch.long)
    for i, lbl in enumerate(labels):
        padded[i, :len(lbl)] = lbl
    
    class_indices = torch.tensor(class_indices, dtype=torch.long)
    
    return imgs, padded, lens, list(raw), class_indices


def get_loader(csv_path: str, char2id: Dict[str, int], split: str,
               batch_size: int = 256, num_workers: int = 2,
               img_size: int = 128, limit: int = 0,
               label2idx: Optional[Dict[str, int]] = None) -> DataLoader:
    ds = BrahmiDataset(csv_path, char2id, split=split, img_size=img_size, label2idx=label2idx)
    
    if limit > 0 and len(ds) > limit:
        import random
        ds.samples = random.sample(ds.samples, limit)
    
    # Lower workers on Windows to prevent shared memory allocation error 1455
    nw = num_workers if os.name != "nt" else min(num_workers, 2)
    persistent = (nw > 0)
    
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=nw, pin_memory=True,
        persistent_workers=persistent,
        collate_fn=collate_fn, drop_last=(split == "train"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ── Model Architecture (Pure Classifier)
# ─────────────────────────────────────────────────────────────────────────────


class BrahmiOCRModel(nn.Module):
    """
    Pure MobileNetV2 Classifier for Brahmi Script.
    Takes an image, outputs class probabilities (214 characters).
    """
    def __init__(self, num_classes: int, freeze_layers: int = 0):
        super().__init__()
        # ── MobileNetV2 backbone ────────────────────────────────────────────
        base = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
        self.cnn = base.features
        
        # Optionally freeze early layers
        if freeze_layers > 0:
            for i, layer in enumerate(self.cnn):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        # ── Simple classification head (Direct from CNN) ──────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        fmap   = self.cnn(x)
        pooled = self.pool(fmap).flatten(1)
        return self.head(pooled)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── Save / Load in .keras (HDF5) format
# ══════════════════════════════════════════════════════════════════════════════

def save_keras_format(model: nn.Module, path: str, meta: dict):
    """
    Saves PyTorch model weights + metadata in HDF5 (.keras) format.
    The file is fully re-loadable for inference and fine-tuning.
    """
    if h5py is None:
        raise ImportError("h5py is required to save in .keras format. Run: pip install h5py")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        # Store model architecture metadata
        f.attrs["framework"]  = "pytorch"
        f.attrs["format"]     = "keras"
        f.attrs["saved_at"]   = datetime.now().isoformat()
        f.attrs["meta"]       = json.dumps(meta)
        # Store all parameter tensors
        weights_grp = f.create_group("model_weights")
        for name, tensor in model.state_dict().items():
            weights_grp.create_dataset(name, data=tensor.cpu().float().numpy())
    print(f"  [✓] Model saved (.keras HDF5): {path}")


def load_keras_format(model: nn.Module, path: str) -> dict:
    """
    Loads weights from .keras (HDF5) file into a PyTorch model.
    Returns the saved metadata dict.
    """
    if h5py is None:
        raise ImportError("h5py is required. Run: pip install h5py")
    with h5py.File(str(path), "r") as f:
        meta = json.loads(f.attrs["meta"])
        state = {}
        for name, ds in f["model_weights"].items():
            state[name] = torch.tensor(np.array(ds))
    model.load_state_dict(state)
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── Metrics Logger
# ══════════════════════════════════════════════════════════════════════════════

LOG_FIELDS = [
    "epoch", "timestamp",
    "train_loss", "val_loss",
    "train_accuracy", "val_accuracy",
    "train_cer", "val_cer",
    "train_wer", "val_wer",
    "epoch_duration_sec", "learning_rate",
    "gpu_memory_mb",
    "best_val_loss", "best_val_accuracy",
]


class MetricsLogger:
    """Appends to CSV + rewrites JSON after every epoch — data is never lost."""

    def __init__(self, results_dir: str = "./results", run_name: str = "brahmi_ocr"):
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path  = self.dir / f"{run_name}_{ts}_metrics.csv"
        self.json_path = self.dir / f"{run_name}_{ts}_summary.json"
        self.best_json = self.dir / f"{run_name}_best_metrics.json"
        self.history   = []
        self.best_val_loss = float("inf")
        self.best_val_acc  = 0.0
        self.best_epoch    = 0
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
        print(f"[MetricsLogger] → {self.csv_path}")

    def log(self, epoch: int, metrics: dict):
        if metrics.get("val_loss", float("inf")) < self.best_val_loss:
            self.best_val_loss = metrics["val_loss"]
            self.best_epoch    = epoch
        if metrics.get("val_accuracy", 0) > self.best_val_acc:
            self.best_val_acc = metrics["val_accuracy"]
        row = {f: "" for f in LOG_FIELDS}
        row.update({"epoch": epoch,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "best_val_loss": round(self.best_val_loss, 6),
                    "best_val_accuracy": round(self.best_val_acc, 6)})
        row.update({k: round(v, 6) if isinstance(v, float) else v
                    for k, v in metrics.items() if k in LOG_FIELDS})
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
        self.history.append(row)
        # Flush JSON (overwrite every epoch — never lose data)
        # Flush JSON (overwrite every epoch -- never lose data)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"best_epoch": self.best_epoch,
                       "best_val_loss": self.best_val_loss,
                       "best_val_accuracy": self.best_val_acc,
                       "history": self.history}, f, indent=2)
        best_row = next((r for r in self.history if r["epoch"] == self.best_epoch), {})
        with open(self.best_json, "w", encoding="utf-8") as f:
            json.dump({"best_epoch": self.best_epoch, "metrics": best_row}, f, indent=2)

    def print_row(self, epoch: int, m: dict):
        dur = m.get("epoch_duration_sec", 0)
        warn = "  ! >30s!" if dur > 30 else ""
        print(
            f"  Ep {epoch:04d} | "
            f"TrLoss={m.get('train_loss',0):.4f} VaLoss={m.get('val_loss',0):.4f} | "
            f"TrAcc={m.get('train_accuracy',0):.3f} VaAcc={m.get('val_accuracy',0):.3f} | "
            f"CER={m.get('val_cer',0):.3f} WER={m.get('val_wer',0):.3f} | "
            f"LR={m.get('learning_rate',0):.2e} | "
            f"GPU={m.get('gpu_memory_mb',0):.0f}MB | "
            f"Time={dur:.1f}s{warn}"
        )


# =============================================================================
# SECTION 7 -- Training Loop Helpers
# =============================================================================

def compute_cer_wer(preds: List[str], targets: List[str]) -> Tuple[float, float]:
    if not _JIWER or not preds:
        return 0.0, 0.0
    try:
        return float(_cer(targets, preds)), float(_wer(targets, preds))
    except Exception:
        return 0.0, 0.0


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool, epoch: int = 0):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    n, n_batches = 0, len(loader)
    t_start = time.time()
    mode = "Train" if train else "Val  "

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, _, _, _, class_indices in loader:
            # channels_last: aligns with MobileNetV2 conv layout -> up to 30% faster on NVIDIA
            imgs   = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = class_indices.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(imgs)
                loss   = criterion(logits, labels)
                correct += (logits.argmax(-1) == labels).sum().item()
                total   += len(labels)

            total_loss += loss.item()
            n += 1

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            # -- Live progress every 5 batches so terminal shows activity immediately
            if n % 5 == 0 or n == n_batches:
                elapsed  = time.time() - t_start
                eta      = (elapsed / n) * (n_batches - n) if n < n_batches else 0.0
                acc_now  = correct / max(total, 1) * 100
                loss_now = total_loss / n
                print(
                    f"\r  Ep{epoch:04d} [{mode}] {n:3d}/{n_batches} | "
                    f"Loss={loss_now:.4f} | Acc={acc_now:.1f}% | ETA {eta:.0f}s   ",
                    end="", flush=True,
                )

    print()  # newline after \r progress line
    avg_loss = total_loss / max(n, 1)
    avg_acc  = correct    / max(total, 1)
    return avg_loss, avg_acc, 0.0, 0.0


# =============================================================================
# SECTION 8 -- Train Command
# =============================================================================

def train_cmd(args):
    device = configure_gpu()

    # 1. Extract dataset
    csv_path, labels, label2idx = prepare_dataset(args.zip, args.data_dir)
    num_classes = len(labels)
    idx2label = {i: lbl for lbl, i in label2idx.items()}

    # Image Classifier mode (Transformer removed)
    char2id    = build_vocab(labels)
    id2char    = {v: k for k, v in char2id.items()}
    vocab_size = len(char2id)
    print(f"[INFO] Mode: Classifier | Classes: {num_classes}")

    # Save vocab
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = ckpt_dir / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump({
            "char2id": char2id, 
            "id2char": {str(k): v for k, v in id2char.items()},
            "label2idx": label2idx,
            "idx2label": {str(k): v for k, v in idx2label.items()},
            "use_seq2seq": False
        }, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Vocab -> {vocab_file}")

    # 2. DataLoaders
    # Full train dataset is created ONCE. A fresh random subset is drawn each epoch
    # so each epoch sees DIFFERENT images -- best of both speed and generalization.
    import random
    _nw = args.num_workers if os.name != "nt" else min(args.num_workers, 2)
    train_ds_full = BrahmiDataset(csv_path, char2id, "train", args.img_size, label2idx=label2idx)
    val_ld = get_loader(csv_path, char2id, "val",
                        args.batch_size, _nw, args.img_size,
                        limit=args.limit_val, label2idx=label2idx)
    _train_limit = args.limit_train
    _eff_train   = min(len(train_ds_full), _train_limit) if _train_limit > 0 else len(train_ds_full)
    _train_batches = max(1, _eff_train // args.batch_size)
    print(f"[INFO] Full train size : {len(train_ds_full):,} images")
    print(f"[INFO] Per-epoch sample: {_eff_train:,} images (~{_train_batches} batches) | Val batches: {len(val_ld)}")

    # Pre-load images into RAM cache (one-time cost, eliminates disk I/O on all future epochs)
    # On Windows, num_workers>0 + shared memory causes overhead with large caches.
    # num_workers=0 with RAM cache is FASTER than num_workers=2 reading from disk.
    train_ds_full.cache_all(limit=_eff_train)
    _nw_loader = 0  # single-threaded is faster when data is already in RAM

    # 3. Model
    model = BrahmiOCRModel(num_classes=num_classes, 
                           freeze_layers=args.freeze_layers).to(device)
    
    start_epoch = 1
    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    if args.resume:
        latest_ckpt = ckpt_dir / "latest_model.keras"
        best_ckpt   = ckpt_dir / "best_model.keras"
        
        # Determine which checkpoint to load
        target_ckpt = None
        if latest_ckpt.exists():
            target_ckpt = latest_ckpt
        elif best_ckpt.exists():
            target_ckpt = best_ckpt
            print(f"[INFO] latest_model.keras missing. Using {best_ckpt} instead.")

        if target_ckpt:
            print(f"[INFO] Attempting to resume from {target_ckpt}...")
            try:
                # Try strict loading first
                meta = load_keras_format(model, str(target_ckpt))
                start_epoch = meta.get("epoch", 0) + 1
                best_val_loss = meta.get("best_val_loss", meta.get("val_loss", float("inf")))
                best_val_acc = meta.get("best_val_accuracy", meta.get("val_accuracy", 0.0))
                
                # If these are empty because we loaded `latest_model.keras`, try best_model.keras metadata
                if best_ckpt.exists() and h5py is not None:
                    try:
                        with h5py.File(str(best_ckpt), "r") as f:
                            b_meta = json.loads(f.attrs.get("meta", "{}"))
                            best_val_loss = b_meta.get("val_loss", best_val_loss)
                            best_val_acc = b_meta.get("val_accuracy", best_val_acc)
                    except Exception:
                        pass

                print(f"[INFO] Resumed fully from epoch {start_epoch - 1} | Prev Best Acc: {best_val_acc*100:.2f}% | Best Loss: {best_val_loss:.4f}")
            except Exception as e:
                # If sizes mismatch (e.g. old model was Seq2Seq, new is Classifier)
                print(f"[WARN] Strict load failed: {e}")
                print(f"[INFO] Falling back to loading ONLY the CNN feature backbone (The 'Eyes')...")
                # Manually extract just the CNN weights
                if h5py is not None:
                    try:
                        loaded = 0
                        with h5py.File(str(latest_ckpt), "r") as f:
                            state = model.state_dict()
                            for name, ds in f["model_weights"].items():
                                if name.startswith("cnn.") and name in state:
                                    if state[name].shape == ds.shape:
                                        state[name].copy_(torch.tensor(np.array(ds)))
                                        loaded += 1
                        print(f"[INFO] Successfully transferred {loaded} internal CNN layers.")
                        print(f"[INFO] Starting at epoch 1 (Head is randomly initialized).")
                        start_epoch = 1
                    except Exception as fallback_e:
                        print(f"[ERROR] Fallback CNN load failed: {fallback_e}")
        else:
            print("[WARN] --resume flag passed but no latest_model.keras found. Starting fresh.")
            
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params: {trainable:,}")

    # -- GPU Speed Maximization --------------------------------------------------
    # channels_last: stores feature maps as NHWC instead of NCHW.
    # MobileNetV2's depthwise convolutions run ~20-30% faster in this layout on NVIDIA.
    model = model.to(memory_format=torch.channels_last)
    # torch.compile needs Triton which is Linux-only (not available on Windows).
    # On Linux/Colab this would give 20-40% extra speedup automatically.
    if os.name != "nt":
        try:
            _dynamo = __import__("torch._dynamo", fromlist=["config"])
            _dynamo.config.suppress_errors = True   # fallback to eager if compile fails
            model = torch.compile(model, mode="reduce-overhead")
            print("[INFO] OK torch.compile enabled -- warmup on ep1-2, then 20-40% faster")
        except Exception as _ce:
            print(f"[INFO] torch.compile skipped ({_ce})")
    else:
        print("[INFO] torch.compile skipped -- Triton not available on Windows (AMP + channels_last active)")

    # 4. Optimizer / Loss / Scheduler
    # Lower label smoothing -- 0.05 is better for fine-tuning a model already above 90%
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
    # CosineAnnealingWarmRestarts: cycles T_0=10 epochs, then doubles each cycle.
    # This is MUCH better for resuming -- it can escape a plateau by periodically
    # spiking LR which shakes weights out of a local minimum.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    scaler = torch.amp.GradScaler('cuda')

    # Optional: Stochastic Weight Averaging -- activated in last 20% of epochs
    # SWA averages weights from multiple LR cycles -> smoother loss landscape -> higher accuracy
    swa_model   = torch.optim.swa_utils.AveragedModel(model)
    swa_start   = max(1, int(args.epochs * 0.7))  # start SWA at 70% of training
    swa_scheduler = torch.optim.swa_utils.SWALR(
        optimizer, swa_lr=args.lr * 0.1, anneal_epochs=5
    )
    swa_active = False
    print(f"[INFO] Scheduler: CosineAnnealingWarmRestarts (T0=10, Tmult=2)")
    print(f"[INFO] SWA will activate at epoch {swa_start}")

    logger         = MetricsLogger(results_dir=args.results_dir)
    logger.best_val_loss = best_val_loss
    logger.best_val_acc  = best_val_acc

    print("\n" + "=" * 80)
    print("  BRAHMI OCR TRAINING START")
    print("=" * 80)

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()

        # -- Per-epoch fresh random sample (different images every epoch) --------
        if _train_limit > 0 and len(train_ds_full) > _train_limit:
            _idx     = random.sample(range(len(train_ds_full)), _train_limit)
            _epoch_ds = torch.utils.data.Subset(train_ds_full, _idx)
        else:
            _epoch_ds = train_ds_full
        train_ld = DataLoader(
            _epoch_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=_nw_loader, pin_memory=True, persistent_workers=False,
            prefetch_factor=None,   # num_workers=0, prefetch_factor must be None
            collate_fn=collate_fn, drop_last=True,
        )

        tr_loss, tr_acc, _, _ = run_epoch(model, train_ld, criterion, optimizer, scaler, device, train=True, epoch=epoch)
        vl_loss, vl_acc, _, _ = run_epoch(model, val_ld,   criterion, optimizer, scaler, device, train=False, epoch=epoch)

        # -- LR Scheduling ----------------------------------------------------
        if swa_active:
            swa_model.update_parameters(model)  # accumulate SWA weights
            swa_scheduler.step()
            lr = swa_scheduler.get_last_lr()[0]
        else:
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            if epoch >= swa_start and not swa_active:
                swa_active = True
                print(f"\n  [SWA] Stochastic Weight Averaging activated at epoch {epoch}!")

        dur = time.time() - t0
        gm  = torch.cuda.memory_reserved(0) / 1024**2 if torch.cuda.is_available() else 0

        metrics = {
            "train_loss": tr_loss, "val_loss": vl_loss,
            "train_accuracy": tr_acc, "val_accuracy": vl_acc,
            "epoch_duration_sec": dur, "learning_rate": lr,
            "gpu_memory_mb": gm,
        }
        logger.log(epoch, metrics)
        logger.print_row(epoch, metrics)

        # -- Save best model (.keras HDF5) -----------------------------------
        is_best = vl_acc > best_val_acc or (vl_acc == best_val_acc and vl_loss < best_val_loss)
        if is_best:
            best_val_loss = vl_loss
            best_val_acc  = vl_acc
            save_keras_format(model, ckpt_dir / "best_model.keras", {
                "epoch": epoch, "val_loss": vl_loss, "val_accuracy": vl_acc,
                "num_classes": num_classes, "vocab_size": vocab_size,
                "use_seq2seq": False,
            })
            print(f"  [*] New best: {vl_acc*100:.2f}%  (saved best_model.keras)")

        # -- Save latest model (.keras HDF5) every epoch ---------------------
        save_keras_format(model, ckpt_dir / "latest_model.keras", {
            "epoch": epoch,
            "val_loss": vl_loss,
            "val_accuracy": vl_acc,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc
        })

        if args.debug:
            print("[DEBUG] Stopping after 1 epoch.")
            break

    # -- SWA: update BatchNorm stats then evaluate ----------------------------
    if swa_active:
        print("\n[SWA] Updating BatchNorm statistics on training data...")
        torch.optim.swa_utils.update_bn(train_ld, swa_model, device=device)
        swa_vl_loss, swa_vl_acc, _, _ = run_epoch(
            swa_model, val_ld, criterion, optimizer, scaler, device, train=False
        )
        print(f"[SWA] SWA model val accuracy: {swa_vl_acc*100:.2f}%  (vs best single: {best_val_acc*100:.2f}%)")
        if swa_vl_acc > best_val_acc:
            best_val_acc = swa_vl_acc
            save_keras_format(swa_model.module, ckpt_dir / "best_model.keras", {
                "epoch": "swa", "val_loss": swa_vl_loss, "val_accuracy": swa_vl_acc,
                "num_classes": num_classes, "vocab_size": vocab_size,
                "use_seq2seq": False, "swa": True,
            })
            print(f"  [*] SWA model is better! Saved as best_model.keras")

    # -- Save final model -----------------------------------------------------
    save_keras_format(model, ckpt_dir / "final_model.keras",
                      {"epochs": args.epochs, "best_val_loss": best_val_loss,
                       "best_val_accuracy": best_val_acc})

    print(f"\n[DONE] Best val accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
    print(f"[DONE] Models saved      : {ckpt_dir}/best_model.keras & final_model.keras")
    print(f"[DONE] Metrics CSV       : {logger.csv_path}")
    print(f"[DONE] Metrics JSON      : {logger.json_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 ── Predict Command
# ══════════════════════════════════════════════════════════════════════════════

def predict_cmd(args):
    device = configure_gpu()
    ckpt_dir = Path(args.ckpt_dir)

    # Load vocab
    with open(ckpt_dir / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    char2id    = vocab["char2id"]
    id2char    = {int(k): v for k, v in vocab["id2char"].items()}
    label2idx  = vocab.get("label2idx", {})
    idx2label  = {int(k): v for k, v in vocab.get("idx2label", {}).items()}
    
    num_classes = len(label2idx)
    vocab_size  = len(char2id)

    # Rebuild model
    model = BrahmiOCRModel(num_classes=num_classes).to(device)

    # Load from .keras (HDF5)
    keras_path = ckpt_dir / "best_model.keras"
    meta = load_keras_format(model, str(keras_path))
    model.eval()
    print(f"[INFO] Loaded model from: {keras_path}")
    print(f"[INFO] Best epoch meta: {meta}")

    transform = get_transform("val", args.img_size)

    def infer(img_path: str) -> str:
        img = Image.open(img_path).convert("RGB")
        x   = transform(img).unsqueeze(0).to(device)
        with torch.no_grad(), autocast(dtype=torch.float16):
            logits = model(x)
            probs  = torch.softmax(logits, -1)[0]
            top_k  = torch.topk(probs, min(args.top_k, num_classes))
            return [(idx2label[i.item()] if idx2label else id2char.get(i.item(), "?"), p.item()) 
                    for i, p in zip(top_k.indices, top_k.values)]

    SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    if args.img:
        result = infer(args.img)
        print(f"\nTop-{args.top_k} predictions for {args.img}:")
        for rank, (lbl, conf) in enumerate(result, 1):
            print(f"  #{rank}: {lbl:30s}  {conf*100:.1f}%")

    elif args.folder:
        imgs = [p for p in sorted(Path(args.folder).iterdir())
                if p.suffix.lower() in SUPPORTED]
        print(f"\nRunning inference on {len(imgs)} images …\n")
        for p in imgs:
            try:
                r = infer(str(p))
                label = r[0][0]
                print(f"  {p.name:40s} → {label}")
            except Exception as e:
                print(f"  {p.name}: ERROR — {e}")
    else:
        print("[ERROR] Provide --img or --folder")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 ── CLI / Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brahmi_ocr.py",
        description="Brahmi OCR — Pure MobileNetV2 Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # ── train ────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Extract dataset + train model")
    tr.add_argument("--zip",          type=str,   required=True)
    tr.add_argument("--data_dir",     type=str,   default="./data")
    tr.add_argument("--epochs",       type=int,   default=100)
    tr.add_argument("--batch_size",   type=int,   default=128,
                    help="128@224px fits in 6GB VRAM comfortably. Use 256 if no OOM.")
    tr.add_argument("--img_size",     type=int,   default=224,
                    help="224 matches MobileNetV2 pretraining — ALWAYS use 224 for best accuracy")
    tr.add_argument("--lr",           type=float, default=1e-4,
                    help="Fine-tuning LR. 1e-4 is safe for resuming from a trained checkpoint")
    tr.add_argument("--num_workers",  type=int,   default=2,
                    help="Keep low on Windows to avoid Error 1455")
    tr.add_argument("--limit_train",  type=int,   default=20000,
                    help="Images per epoch. 20000 @ batch512 = ~40 batches = ~20s/epoch. 0 = full dataset")
    tr.add_argument("--limit_val",    type=int,   default=5000,
                    help="Val images per epoch. 5000 is fast and statistically reliable")
    tr.add_argument("--freeze_layers",type=int,   default=0,
                    help="0 = Unfreeze everything for maximum accuracy gain")
    tr.add_argument("--ckpt_dir",     type=str,   default="./checkpoints")
    tr.add_argument("--results_dir",  type=str,   default="./results")
    tr.add_argument("--debug",        action="store_true",
                    help="Run 1 epoch only for quick sanity check")
    tr.add_argument("--resume",       action="store_true",
                    help="Resume training from latest_model.keras if it exists")

    # ── predict ──────────────────────────────────────────────────────────────
    pr = sub.add_parser("predict", help="Run inference on image(s)")
    pr.add_argument("--ckpt_dir",  type=str, default="./checkpoints")
    pr.add_argument("--img",       type=str, default=None)
    pr.add_argument("--folder",    type=str, default=None)
    pr.add_argument("--img_size",  type=int, default=224)
    pr.add_argument("--top_k",     type=int, default=5)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    if   args.command == "train":   train_cmd(args)
    elif args.command == "predict": predict_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
