import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from project_utils import calculate_detection_metrics


DEFAULT_TUNING_PARAMS = {
    'dist_transform_threshold': 0.05,
    'use_color_mask_only': False,
    'morph_kernel_size': 3,
    'morph_open_iterations': 1,
    'morph_close_iterations': 2,
    'adaptive_block_size': 31,
    'adaptive_C': 8,
    'aspect_ratio_max': 6.0,
    'compactness_min': 0.3,
    'blob_detector_enabled': True,
    'blob_min_circularity': 0.1,
    'blob_min_convexity': 0.1,
    'blob_min_inertia': 0.1,
    'blob_merge_distance': 10,
}


ORGAN_PARAMS = {
    'bladder': {
        'hsv_lower': np.array([130, 90, 50]),
        'hsv_upper': np.array([157, 180, 170]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'endometrium': {
        'hsv_lower': np.array([133, 70, 70]),
        'hsv_upper': np.array([169, 175, 200]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'head-and-neck': {
        'hsv_lower': np.array([120, 70, 65]),
        'hsv_upper': np.array([165, 200, 200]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'kidney': {
        'hsv_lower': np.array([130, 75, 50]),
        'hsv_upper': np.array([160, 170, 175]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'prostate': {
        'hsv_lower': np.array([122, 40, 90]),
        'hsv_upper': np.array([158, 140, 215]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'stomach': {
        'hsv_lower': np.array([128, 45, 85]),
        'hsv_upper': np.array([158, 150, 210]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    },
    'default': {
        'hsv_lower': np.array([125, 60, 60]),
        'hsv_upper': np.array([160, 180, 200]),
        'min_area': 200,
        'max_area': 2000,
        'cell_size_range': (10, 40),
    }
}


def get_organ_from_metadata(image_path, data_root='ocelot_testing_data'):
    img_id = Path(image_path).stem
    metadata_path = Path(data_root) / 'metadata.json'
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if img_id in metadata['sample_pairs']:
            return metadata['sample_pairs'][img_id].get('organ', 'default')
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
    return 'default'


def detect_cells_multi_stage(image_bgr, params, tuning_params=None):
    if tuning_params is None:
        tuning_params = DEFAULT_TUNING_PARAMS
    else:
        merged_params = DEFAULT_TUNING_PARAMS.copy()
        merged_params.update(tuning_params)
        tuning_params = merged_params

    morph_kernel_size = tuning_params['morph_kernel_size']
    morph_open_iterations = tuning_params['morph_open_iterations']
    morph_close_iterations = tuning_params['morph_close_iterations']
    adaptive_block_size = tuning_params['adaptive_block_size']
    adaptive_C = tuning_params['adaptive_C']
    use_color_mask_only = tuning_params['use_color_mask_only']
    dist_transform_threshold = tuning_params['dist_transform_threshold']
    aspect_ratio_max = tuning_params['aspect_ratio_max']
    compactness_min = tuning_params['compactness_min']
    blob_detector_enabled = tuning_params['blob_detector_enabled']
    blob_min_circularity = tuning_params['blob_min_circularity']
    blob_min_convexity = tuning_params['blob_min_convexity']
    blob_min_inertia = tuning_params['blob_min_inertia']
    blob_merge_distance = tuning_params['blob_merge_distance']

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(image_hsv, params['hsv_lower'], params['hsv_upper'])

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_small, iterations=morph_open_iterations)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_small, iterations=morph_close_iterations)

    if use_color_mask_only:
        combined_mask = color_mask
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=adaptive_block_size,
            C=adaptive_C
        )
        combined_mask = cv2.bitwise_and(color_mask, adaptive_thresh)

    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    dist_transform = cv2.distanceTransform(combined_mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_transform, dist_transform_threshold * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        sure_fg, connectivity=8
    )

    detected_centroids = []
    min_area = params['min_area']
    max_area = params['max_area']

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if min_area <= area <= max_area:
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            bbox_area = w * h
            compactness = area / bbox_area if bbox_area > 0 else 0

            if aspect_ratio < aspect_ratio_max and compactness > compactness_min:
                detected_centroids.append(centroids[label])

    if blob_detector_enabled:
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.filterByArea = True
        blob_params.minArea = params['min_area']
        blob_params.maxArea = params['max_area']
        blob_params.filterByCircularity = True
        blob_params.minCircularity = blob_min_circularity
        blob_params.filterByConvexity = True
        blob_params.minConvexity = blob_min_convexity
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = blob_min_inertia

        detector = cv2.SimpleBlobDetector_create(blob_params)
        keypoints = detector.detect(color_mask)

        for kp in keypoints:
            kp_point = np.array([kp.pt[0], kp.pt[1]])
            too_close = False
            for centroid in detected_centroids:
                dist = np.linalg.norm(kp_point - centroid)
                if dist < blob_merge_distance:
                    too_close = True
                    break
            if not too_close:
                detected_centroids.append(kp_point)

    return detected_centroids, {
        'color_mask': color_mask,
        'combined_mask': combined_mask,
        'dist_transform': (dist_transform * 255).astype(np.uint8),
        'sure_fg': sure_fg
    }


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


def load_cnn_model(model_path, patch_size=64, device=None, num_classes=2):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MinimalCellCNN(num_classes=num_classes, patch_size=patch_size)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def extract_patch(image_bgr, cx, cy, patch_size=64):
    h, w, _ = image_bgr.shape
    half = patch_size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    patch = image_bgr[y1:y2, x1:x2]

    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

    return patch


def preprocess_patch_for_cnn(patch_bgr, patch_size=64):
    if patch_bgr.shape[0] != patch_size or patch_bgr.shape[1] != patch_size:
        patch_bgr = cv2.resize(patch_bgr, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

    patch_rgb = patch_rgb.astype(np.float32) / 255.0

    patch_chw = np.transpose(patch_rgb, (2, 0, 1))

    patch_tensor = torch.from_numpy(patch_chw).unsqueeze(0)
    return patch_tensor


def filter_centroids_with_cnn(
    image_bgr,
    centroids,
    cnn_model,
    device='cpu',
    patch_size=64,
    positive_class_index=1,
    prob_threshold=0.3
):
    if cnn_model is None or len(centroids) == 0:
        return centroids, [1.0] * len(centroids)

    cnn_model.eval()
    filtered_centroids = []
    scores = []

    with torch.no_grad():
        for c in centroids:
            cx, cy = int(c[0]), int(c[1])
            patch = extract_patch(image_bgr, cx, cy, patch_size=patch_size)
            patch_tensor = preprocess_patch_for_cnn(patch, patch_size=patch_size).to(device)

            logits = cnn_model(patch_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            p_pos = float(probs[positive_class_index])

            if p_pos >= prob_threshold:
                filtered_centroids.append(np.array([cx, cy], dtype=np.float32))
                scores.append(p_pos)

    return filtered_centroids, scores


def process_cell_advanced_cnn(
    image_path,
    annotations: pd.DataFrame,
    image=None,
    data_root='ocelot_testing_data',
    distance_threshold=100,
    tuning_params=None,
    cnn_model=None,
    cnn_model_path=None,
    cnn_device=None,
    cnn_patch_size=64,
    cnn_positive_class_index=1,
    cnn_prob_threshold=0.5
):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    organ = get_organ_from_metadata(image_path, data_root)
    params = ORGAN_PARAMS.get(organ, ORGAN_PARAMS['default'])

    if cnn_model is None and cnn_model_path is not None:
        print(f"Loading CNN model from {cnn_model_path}")
        cnn_model, cnn_device = load_cnn_model(
            cnn_model_path,
            patch_size=cnn_patch_size,
            device=cnn_device,
            num_classes=2
        )

    print(f"Processing {Path(image_path).name} as {organ} organ (advanced + CNN)")

    if tuning_params is not None:
        print(f"  Using custom tuning parameters:")
        for key, value in tuning_params.items():
            print(f"    {key}: {value}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    detected_centroids, debug_images = detect_cells_multi_stage(image_bgr, params, tuning_params)

    print(f"  Initial detections from multi-stage: {len(detected_centroids)}")

    cnn_used = cnn_model is not None
    cnn_scores = None

    if cnn_used:
        if cnn_device is None:
            cnn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cnn_model.to(cnn_device)

        detected_centroids, cnn_scores = filter_centroids_with_cnn(
            image_bgr,
            detected_centroids,
            cnn_model,
            device=cnn_device,
            patch_size=cnn_patch_size,
            positive_class_index=cnn_positive_class_index,
            prob_threshold=cnn_prob_threshold
        )
        print(f"  After CNN filtering (threshold={cnn_prob_threshold:.2f}): {len(detected_centroids)} detections")

    detected_count = len(detected_centroids)
    actual_count = len(annotations)

    metrics = calculate_detection_metrics(detected_centroids, annotations, distance_threshold)

    print(f"  Organ: {organ}, Detected: {detected_count}, Actual: {actual_count}")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, "
          f"F1: {metrics['f1_score']:.3f}, Accuracy: {metrics['accuracy']:.3f}")

    cv2.imwrite(
        str(results_dir / f"{Path(image_path).stem}_cell_advcnn_1_color_mask.png"),
        debug_images['color_mask']
    )
    cv2.imwrite(
        str(results_dir / f"{Path(image_path).stem}_cell_advcnn_2_combined.png"),
        debug_images['combined_mask']
    )
    cv2.imwrite(
        str(results_dir / f"{Path(image_path).stem}_cell_advcnn_3_distance.png"),
        debug_images['dist_transform']
    )
    cv2.imwrite(
        str(results_dir / f"{Path(image_path).stem}_cell_advcnn_4_centers.png"),
        debug_images['sure_fg']
    )

    overlay = image_bgr.copy()

    for i, centroid in enumerate(detected_centroids):
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), 2)

        if cnn_scores is not None:
            score = cnn_scores[i]
            cv2.putText(
                overlay, f"{score:.2f}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

    if annotations is not None and len(annotations) > 0:
        for _, row in annotations.iterrows():
            cx, cy = int(row['x']), int(row['y'])
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), 2)

    cv2.putText(overlay, f"Organ: {organ} (Adv+CNN)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"Detected: {detected_count} | Actual: {actual_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"P: {metrics['precision']:.2f} R: {metrics['recall']:.2f} "
                         f"F1: {metrics['f1_score']:.2f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    overlay_path = results_dir / f"{Path(image_path).stem}_cell_advcnn_5_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    original_path = results_dir / f"{Path(image_path).stem}_cell_advcnn_6_original.png"
    cv2.imwrite(str(original_path), image_bgr)

    return {
        'image_name': Path(image_path).name,
        'organ': organ,
        'detected_components': detected_count,
        'actual_components': actual_count,
        'difference': abs(detected_count - actual_count),
        'cnn_used': cnn_used,
        'cnn_prob_threshold': cnn_prob_threshold if cnn_used else None,
        **metrics
    }
