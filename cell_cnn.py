import argparse
from pathlib import Path
import random
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MinimalCellCNN(nn.Module):
    def __init__(self, num_classes=2, patch_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, patch_size, patch_size)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def preprocess_patch_for_cnn(patch_bgr, patch_size=64):
    if patch_bgr.shape[0] != patch_size or patch_bgr.shape[1] != patch_size:
        patch_bgr = cv2.resize(patch_bgr, (patch_size, patch_size))
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    patch_rgb = patch_rgb.astype(np.float32) / 255.0
    patch_chw = np.transpose(patch_rgb, (2, 0, 1))
    return torch.from_numpy(patch_chw)

def extract_patch(image_bgr, cx, cy, patch_size=64):
    h, w, _ = image_bgr.shape
    half = patch_size // 2
    x1, y1 = max(0, cx-half), max(0, cy-half)
    x2, y2 = min(w, cx+half), min(h, cy+half)
    patch = image_bgr[y1:y2, x1:x2]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size))
    return patch

class CellPatchDataset(Dataset):
    def __init__(self,
                 data_root="ocelot_testing_data",
                 split="train",
                 max_images=None,
                 pos_per_image=50,
                 neg_per_image=50,
                 patch_size=64,
                 min_neg_dist=20.0,
                 seed=42):

        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.min_neg_dist = min_neg_dist

        random.seed(seed)
        np.random.seed(seed)

        images_dir = self.data_root / "images" / split / "cell"
        ann_dir = self.data_root / "annotations" / split / "cell"

        image_paths = sorted(images_dir.glob("*.jpg"))
        if max_images:
            image_paths = image_paths[:max_images]

        self.samples = []

        for img_path in tqdm(image_paths):
            ann_path = ann_dir / (img_path.stem + ".csv")
            if not ann_path.exists():
                continue

            ann = np.genfromtxt(ann_path, delimiter=",")
            if ann.size == 0:
                continue
            if ann.ndim == 1:
                ann = ann.reshape(1, -1)
            positives = [(int(x), int(y)) for x, y, _ in ann]

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w, _ = img.shape

            for (px, py) in random.sample(positives,
                                          min(pos_per_image, len(positives))):
                self.samples.append((img_path, px, py, 1))

            neg_count = 0
            attempts = 0
            while neg_count < neg_per_image and attempts < neg_per_image * 10:
                attempts += 1
                rx, ry = random.randint(0, w-1), random.randint(0, h-1)
                if min(np.hypot(rx-px, ry-py) for px, py in positives) >= min_neg_dist:
                    self.samples.append((img_path, rx, ry, 0))
                    neg_count += 1

        print(f"[{split}] Samples: {len(self.samples)}")
        self.cache_img_path = None
        self.cache_img = None

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        if self.cache_img_path != path:
            self.cache_img = cv2.imread(str(path))
            self.cache_img_path = path
        return self.cache_img

    def __getitem__(self, idx):
        img_path, x, y, label = self.samples[idx]
        img = self._load_img(img_path)
        patch = extract_patch(img, x, y, self.patch_size)
        tensor = preprocess_patch_for_cnn(patch, self.patch_size)
        return tensor, torch.tensor(label)

def train_epoch(model, loader, opt, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total

def eval_epoch(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="ocelot_testing_data")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos", type=int, default=50)
    p.add_argument("--neg", type=int, default=50)
    p.add_argument("--out", default="minimal_cell_cnn.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = CellPatchDataset(args.data_root, "train",
                                pos_per_image=args.pos,
                                neg_per_image=args.neg,
                                patch_size=args.patch_size)
    val_ds   = CellPatchDataset(args.data_root, "val",
                                pos_per_image=args.pos,
                                neg_per_image=args.neg,
                                patch_size=args.patch_size,
                                max_images=50)

    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds, batch_size=args.batch,
                          shuffle=False, num_workers=4)

    model = MinimalCellCNN(2, args.patch_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tr_loss, tr_acc = train_epoch(model, train_dl, opt, device)
        va_loss, va_acc = eval_epoch(model, val_dl, device)

        print(f"Train: Loss={tr_loss:.3f} Acc={tr_acc:.3f}")
        print(f"Val:   Loss={va_loss:.3f} Acc={va_acc:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), args.out)
            print(f" ✓ Saved best model to {args.out}")

    print(f"Done. Best val acc = {best_acc:.3f}")

if __name__ == "__main__":
    main()
