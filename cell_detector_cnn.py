import argparse
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CellDetectorCNN(nn.Module):
    def __init__(self, num_classes=2, patch_size=64):
        super().__init__()
        self.patch_size = patch_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, patch_size, patch_size)
            feat = self.features(dummy)
            self.flat_size = feat.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def augment_patch(patch_bgr, p_flip=0.5, p_rotate=0.5, p_color=0.8):
    patch = patch_bgr.copy()

    if random.random() < p_flip:
        patch = cv2.flip(patch, 1)

    if random.random() < p_flip:
        patch = cv2.flip(patch, 0)

    if random.random() < p_rotate:
        k = random.randint(1, 3)
        patch = np.rot90(patch, k).copy()

    if random.random() < p_color:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
        patch = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return patch


class CellPatchDataset(Dataset):
    def __init__(
        self,
        data_root="ocelot_testing_data",
        split="train",
        patch_size=64,
        pos_per_image=100,
        hard_neg_per_image=150,
        easy_neg_per_image=10,
        augment=True,
        seed=42,
        max_images=None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == "train")

        random.seed(seed)
        np.random.seed(seed)

        images_dir = self.data_root / "images" / split / "cell"
        ann_dir = self.data_root / "annotations" / split / "cell"

        image_paths = sorted(images_dir.glob("*.jpg"))
        if max_images:
            image_paths = image_paths[:max_images]

        self.samples = []

        print(f"Loading {split} dataset...")
        for img_path in tqdm(image_paths, desc=f"Preparing {split}"):
            ann_path = ann_dir / (img_path.stem + ".csv")
            if not ann_path.exists():
                continue

            ann = np.genfromtxt(ann_path, delimiter=",")
            if ann.size == 0:
                continue
            if ann.ndim == 1:
                ann = ann.reshape(1, -1)

            cell_centers = [(int(x), int(y)) for x, y, _ in ann]
            if len(cell_centers) == 0:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            half = patch_size // 2

            pos_samples = random.sample(
                cell_centers,
                min(pos_per_image, len(cell_centers))
            )
            for (cx, cy) in pos_samples:
                if half <= cx < w - half and half <= cy < h - half:
                    self.samples.append((img_path, cx, cy, 1))

            hard_neg_count = 0
            attempts = 0
            target_type1 = hard_neg_per_image // 3
            while hard_neg_count < target_type1 and attempts < hard_neg_per_image * 30:
                attempts += 1
                cx, cy = random.choice(cell_centers)
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(8, 15)
                nx = int(cx + dist * np.cos(angle))
                ny = int(cy + dist * np.sin(angle))

                if not (half <= nx < w - half and half <= ny < h - half):
                    continue

                min_dist = min(np.hypot(nx - px, ny - py) for px, py in cell_centers)
                if min_dist >= 8:
                    self.samples.append((img_path, nx, ny, 0))
                    hard_neg_count += 1

            hard_neg_count2 = 0
            attempts = 0
            target_type2 = hard_neg_per_image // 3
            while hard_neg_count2 < target_type2 and attempts < hard_neg_per_image * 30:
                attempts += 1
                cx, cy = random.choice(cell_centers)
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(15, 30)
                nx = int(cx + dist * np.cos(angle))
                ny = int(cy + dist * np.sin(angle))

                if not (half <= nx < w - half and half <= ny < h - half):
                    continue

                min_dist = min(np.hypot(nx - px, ny - py) for px, py in cell_centers)
                if min_dist >= 12:
                    self.samples.append((img_path, nx, ny, 0))
                    hard_neg_count2 += 1

            hard_neg_count3 = 0
            attempts = 0
            target_type3 = hard_neg_per_image // 3
            while hard_neg_count3 < target_type3 and attempts < hard_neg_per_image * 30:
                attempts += 1
                cx, cy = random.choice(cell_centers)
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(30, 50)
                nx = int(cx + dist * np.cos(angle))
                ny = int(cy + dist * np.sin(angle))

                if not (half <= nx < w - half and half <= ny < h - half):
                    continue

                min_dist = min(np.hypot(nx - px, ny - py) for px, py in cell_centers)
                if min_dist >= 15:
                    self.samples.append((img_path, nx, ny, 0))
                    hard_neg_count3 += 1

            easy_neg_count = 0
            attempts = 0
            while easy_neg_count < easy_neg_per_image and attempts < easy_neg_per_image * 20:
                attempts += 1
                rx = random.randint(half, w - half - 1)
                ry = random.randint(half, h - half - 1)

                min_dist = min(np.hypot(rx - px, ry - py) for px, py in cell_centers)
                if min_dist >= 60:
                    self.samples.append((img_path, rx, ry, 0))
                    easy_neg_count += 1

        pos_count = sum(1 for s in self.samples if s[3] == 1)
        neg_count = len(self.samples) - pos_count
        print(f"[{split}] Total samples: {len(self.samples)} "
              f"(pos: {pos_count}, neg: {neg_count})")

        self._cache_path = None
        self._cache_img = None

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        if self._cache_path != path:
            self._cache_img = cv2.imread(str(path))
            self._cache_path = path
        return self._cache_img

    def __getitem__(self, idx):
        img_path, cx, cy, label = self.samples[idx]

        img = self._load_image(img_path)
        half = self.patch_size // 2

        patch = img[cy - half:cy + half, cx - half:cx + half].copy()

        if self.augment:
            patch = augment_patch(patch)

        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_float = patch_rgb.astype(np.float32) / 255.0
        patch_tensor = torch.from_numpy(patch_float.transpose(2, 0, 1))

        return patch_tensor, torch.tensor(label, dtype=torch.long)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for patches, labels in tqdm(loader, desc="Training", leave=False):
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(patches)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * patches.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += patches.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for patches, labels in tqdm(loader, desc="Evaluating", leave=False):
            patches = patches.to(device)
            labels = labels.to(device)

            logits = model(patches)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * patches.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += patches.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return total_loss / total, correct / total, precision, recall, f1


def sliding_window_detect(
    model,
    image_bgr,
    device,
    patch_size=64,
    stride=8,
    confidence_threshold=0.6,
    nms_radius=20,
):
    model.eval()
    h, w = image_bgr.shape[:2]
    half = patch_size // 2

    positions = []
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            positions.append((x, y))

    batch_size = 256
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i:i + batch_size]
            patches = []

            for (x, y) in batch_positions:
                patch = image_bgr[y - half:y + half, x - half:x + half]
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_float = patch_rgb.astype(np.float32) / 255.0
                patch_tensor = patch_float.transpose(2, 0, 1)
                patches.append(patch_tensor)

            batch_tensor = torch.from_numpy(np.stack(patches)).to(device)
            logits = model(batch_tensor)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    detections = []
    for (x, y), prob in zip(positions, all_probs):
        if prob >= confidence_threshold:
            detections.append((x, y, prob))

    detections.sort(key=lambda d: d[2], reverse=True)

    kept = []
    suppressed = set()

    for i, (x, y, prob) in enumerate(detections):
        if i in suppressed:
            continue

        kept.append((x, y, prob))

        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue
            x2, y2, _ = detections[j]
            dist = np.hypot(x - x2, y - y2)
            if dist < nms_radius:
                suppressed.add(j)

    return kept


def detect_cells_in_image(
    model,
    image_path,
    device,
    patch_size=64,
    stride=8,
    confidence_threshold=0.6,
    nms_radius=20,
):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    detections = sliding_window_detect(
        model, image_bgr, device,
        patch_size=patch_size,
        stride=stride,
        confidence_threshold=confidence_threshold,
        nms_radius=nms_radius,
    )

    return detections, image_bgr


def evaluate_detection(
    model,
    data_root,
    split,
    device,
    patch_size=64,
    stride=8,
    confidence_threshold=0.6,
    nms_radius=20,
    distance_threshold=15,
    max_images=None,
):
    data_root = Path(data_root)
    images_dir = data_root / "images" / split / "cell"
    ann_dir = data_root / "annotations" / split / "cell"

    image_paths = sorted(images_dir.glob("*.jpg"))
    if max_images:
        image_paths = image_paths[:max_images]

    all_results = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for img_path in tqdm(image_paths, desc=f"Evaluating {split}"):
        ann_path = ann_dir / (img_path.stem + ".csv")

        gt_cells = []
        if ann_path.exists():
            ann = np.genfromtxt(ann_path, delimiter=",")
            if ann.size > 0:
                if ann.ndim == 1:
                    ann = ann.reshape(1, -1)
                gt_cells = [(int(x), int(y)) for x, y, _ in ann]

        detections, _ = detect_cells_in_image(
            model, img_path, device,
            patch_size=patch_size,
            stride=stride,
            confidence_threshold=confidence_threshold,
            nms_radius=nms_radius,
        )

        detected_cells = [(x, y) for x, y, _ in detections]

        gt_matched = [False] * len(gt_cells)
        det_matched = [False] * len(detected_cells)

        for i, (dx, dy) in enumerate(detected_cells):
            best_dist = float('inf')
            best_j = -1
            for j, (gx, gy) in enumerate(gt_cells):
                if gt_matched[j]:
                    continue
                dist = np.hypot(dx - gx, dy - gy)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0 and best_dist <= distance_threshold:
                gt_matched[best_j] = True
                det_matched[i] = True

        tp = sum(det_matched)
        fp = len(detected_cells) - tp
        fn = len(gt_cells) - sum(gt_matched)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        all_results.append({
            'image': img_path.name,
            'gt_count': len(gt_cells),
            'det_count': len(detected_cells),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0

    return {
        'per_image': all_results,
        'aggregate': {
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'precision': agg_precision,
            'recall': agg_recall,
            'f1': agg_f1,
        }
    }


def visualize_detections(image_bgr, detections, gt_cells=None, output_path=None):
    vis = image_bgr.copy()

    if gt_cells:
        for (gx, gy) in gt_cells:
            cv2.circle(vis, (gx, gy), 8, (0, 0, 255), 2)

    for (x, y, conf) in detections:
        cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), 2)
        cv2.putText(vis, f"{conf:.2f}", (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    if output_path:
        cv2.imwrite(str(output_path), vis)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Train cell detector CNN")
    parser.add_argument("--data-root", default="ocelot_testing_data")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pos-per-image", type=int, default=100)
    parser.add_argument("--hard-neg-per-image", type=int, default=150)
    parser.add_argument("--easy-neg-per-image", type=int, default=10)
    parser.add_argument("--output", default="cell_detector.pt")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.eval_only:
        model_path = args.model_path or args.output
        print(f"Loading model from {model_path}")
        model = CellDetectorCNN(num_classes=2, patch_size=args.patch_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        print("\nEvaluating on validation set...")
        results = evaluate_detection(
            model, args.data_root, "val", device,
            patch_size=args.patch_size,
            stride=8,
            confidence_threshold=0.5,
            nms_radius=12,
            distance_threshold=15,
        )

        print(f"\nAggregate Results:")
        print(f"  Precision: {results['aggregate']['precision']:.3f}")
        print(f"  Recall:    {results['aggregate']['recall']:.3f}")
        print(f"  F1 Score:  {results['aggregate']['f1']:.3f}")
        print(f"  Total TP: {results['aggregate']['total_tp']}, "
              f"FP: {results['aggregate']['total_fp']}, "
              f"FN: {results['aggregate']['total_fn']}")
        return

    print("\nPreparing training dataset...")
    train_ds = CellPatchDataset(
        data_root=args.data_root,
        split="train",
        patch_size=args.patch_size,
        pos_per_image=args.pos_per_image,
        hard_neg_per_image=args.hard_neg_per_image,
        easy_neg_per_image=args.easy_neg_per_image,
        augment=True,
        max_images=args.max_train_images,
    )

    print("\nPreparing validation dataset...")
    val_ds = CellPatchDataset(
        data_root=args.data_root,
        split="val",
        patch_size=args.patch_size,
        pos_per_image=args.pos_per_image,
        hard_neg_per_image=args.hard_neg_per_image,
        easy_neg_per_image=args.easy_neg_per_image,
        augment=False,
        max_images=args.max_val_images,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = CellDetectorCNN(num_classes=2, patch_size=args.patch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*60)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.3f}")

        val_loss, val_acc, precision, recall, f1 = eval_epoch(model, val_loader, device)
        print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.3f}")
        print(f"        Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best model (F1: {f1:.3f})")

    print(f"\n{'='*60}")
    print(f"Training complete! Best validation F1: {best_f1:.3f}")
    print(f"Model saved to: {args.output}")

    print("\nRunning sliding window evaluation on validation set...")
    model.load_state_dict(torch.load(args.output, map_location=device))

    results = evaluate_detection(
        model, args.data_root, "val", device,
        patch_size=args.patch_size,
        stride=8,
        confidence_threshold=0.6,
        nms_radius=20,
        distance_threshold=15,
        max_images=30,
    )

    print(f"\nSliding Window Detection Results (30 val images):")
    print(f"  Precision: {results['aggregate']['precision']:.3f}")
    print(f"  Recall:    {results['aggregate']['recall']:.3f}")
    print(f"  F1 Score:  {results['aggregate']['f1']:.3f}")


if __name__ == "__main__":
    main()
