import pandas as pd
import numpy as np
import cv2
from pathlib import Path


def calculate_detection_metrics(detected_centroids, ground_truth_annotations, distance_threshold=100):
    if len(ground_truth_annotations) == 0:
        return {
            'true_positives': 0,
            'false_positives': len(detected_centroids),
            'false_negatives': 0,
            'precision': 0.0 if len(detected_centroids) > 0 else 1.0,
            'recall': 1.0,
            'f1_score': 0.0,
            'accuracy': 0.0 if len(detected_centroids) > 0 else 1.0,
            'matched_detections': 0,
            'unmatched_detections': len(detected_centroids),
            'unmatched_ground_truth': 0
        }

    if len(detected_centroids) == 0:
        return {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(ground_truth_annotations),
            'precision': 1.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'matched_detections': 0,
            'unmatched_detections': 0,
            'unmatched_ground_truth': len(ground_truth_annotations)
        }

    gt_points = ground_truth_annotations[['x', 'y']].values
    detected_points = np.array(detected_centroids)

    matched_gt = set()
    matched_det = set()

    for det_idx, det_point in enumerate(detected_points):
        min_dist = float('inf')
        closest_gt_idx = -1

        for gt_idx, gt_point in enumerate(gt_points):
            if gt_idx in matched_gt:
                continue

            dist = np.linalg.norm(det_point - gt_point)
            if dist < min_dist and dist <= distance_threshold:
                min_dist = dist
                closest_gt_idx = gt_idx

        if closest_gt_idx != -1:
            matched_gt.add(closest_gt_idx)
            matched_det.add(det_idx)

    true_positives = len(matched_gt)
    false_positives = len(detected_centroids) - true_positives
    false_negatives = len(ground_truth_annotations) - true_positives

    precision = true_positives / len(detected_centroids) if len(detected_centroids) > 0 else 0.0
    recall = true_positives / len(ground_truth_annotations) if len(ground_truth_annotations) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    total = true_positives + false_positives + false_negatives
    accuracy = true_positives / total if total > 0 else 0.0

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'matched_detections': true_positives,
        'unmatched_detections': false_positives,
        'unmatched_ground_truth': false_negatives
    }


def generate_accuracy_mosaics(df, method_name, results_dir='results'):
    results_path = Path(results_dir)

    if 'accuracy' not in df.columns or len(df) == 0:
        print("Cannot generate mosaics: 'accuracy' column missing or no data")
        return

    df_filtered = df[(df['accuracy'] >= 0.10) & (df['accuracy'] <= 0.95)].copy()

    if len(df_filtered) == 0:
        print("No images in the accuracy range [10%, 95%] for mosaics")
        return

    print(f"Generating mosaics from {len(df_filtered)} images (filtered from {len(df)} total, excluding <10% and >95% accuracy)")

    df_sorted = df_filtered.sort_values('accuracy', ascending=False).reset_index(drop=True)

    quartiles = {
        'q4_best': (0.0, 0.25, 'Best (Top 25%)'),
        'q3_good': (0.25, 0.50, 'Good (50-75%)'),
        'q2_fair': (0.50, 0.75, 'Fair (25-50%)'),
        'q1_worst': (0.75, 1.0, 'Worst (Bottom 25%)')
    }

    for quartile_name, (lower, upper, label) in quartiles.items():
        lower_idx = int(len(df_sorted) * lower)
        upper_idx = int(len(df_sorted) * upper)

        if upper_idx <= lower_idx:
            continue

        quartile_df = df_sorted.iloc[lower_idx:upper_idx]

        sample_size = min(4, len(quartile_df))
        if sample_size == 0:
            continue

        indices = np.linspace(0, len(quartile_df) - 1, sample_size, dtype=int)
        sampled_images = quartile_df.iloc[indices]

        images = []
        for _, row in sampled_images.iterrows():
            image_stem = Path(row['image_name']).stem

            overlay_patterns = [
                f"{image_stem}_{method_name}_*_overlay.png",
                f"{image_stem}_cell_*_overlay.png",
                f"{image_stem}_*_overlay.png"
            ]

            overlay_path = None
            for pattern in overlay_patterns:
                matches = list(results_path.glob(pattern))
                if matches:
                    overlay_path = matches[0]
                    break

            if overlay_path and overlay_path.exists():
                img = cv2.imread(str(overlay_path))
                if img is not None:
                    images.append((img, row['accuracy'], row['image_name']))

        if len(images) == 0:
            print(f"No overlay images found for {quartile_name}")
            continue

        mosaic = create_mosaic_grid(images, grid_size=2, tile_size=512)

        title_height = 60
        mosaic_with_title = np.ones((mosaic.shape[0] + title_height, mosaic.shape[1], 3), dtype=np.uint8) * 255
        mosaic_with_title[title_height:, :] = mosaic

        title_text = f"{method_name.upper()} - {label}"
        cv2.putText(mosaic_with_title, title_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        output_path = results_path / f"{method_name}_mosaic_{quartile_name}.png"
        cv2.imwrite(str(output_path), mosaic_with_title)
        print(f"  Saved mosaic: {output_path.name}")


def create_mosaic_grid(images, grid_size=3, tile_size=512):
    mosaic = np.ones((grid_size * tile_size, grid_size * tile_size, 3), dtype=np.uint8) * 240

    for idx, (img, accuracy, name) in enumerate(images):
        if idx >= grid_size * grid_size:
            break

        row = idx // grid_size
        col = idx % grid_size

        resized = cv2.resize(img, (tile_size, tile_size))

        label_bg_height = 35
        cv2.rectangle(resized, (0, 0), (tile_size, label_bg_height), (0, 0, 0), -1)

        accuracy_color = (0, 255, 0) if accuracy > 0.8 else (0, 165, 255) if accuracy > 0.6 else (0, 0, 255)
        text = f"{Path(name).stem[:20]} | Acc: {accuracy:.3f}"
        cv2.putText(resized, text, (5, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, accuracy_color, 2)

        y_start = row * tile_size
        y_end = (row + 1) * tile_size
        x_start = col * tile_size
        x_end = (col + 1) * tile_size

        mosaic[y_start:y_end, x_start:x_end] = resized

    return mosaic


def save_cell_results(results_list, output_path='results/component_comparison.csv'):
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(output_path, index=False)
        return output_path


def analyze_cell_results(csv_path: str):
    df = pd.read_csv(csv_path)

    csv_filename = csv_path.split("/")[-1]
    method_name = csv_filename.replace("_comparison.csv", "")

    stats_dir = Path('results') / method_name / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)

    stats_path = stats_dir / f'{csv_filename.split(".")[0]}_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DETECTION RESULTS ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        if 'difference' in df.columns:
            differences = df['difference']
            f.write("COMPONENT COUNT DIFFERENCES:\n")
            f.write(f"  Mean difference: {differences.mean():.2f}\n")
            f.write(f"  Std deviation: {differences.std():.2f}\n")
            f.write(f"  Min difference: {differences.min()}\n")
            f.write(f"  Max difference: {differences.max()}\n")
            f.write(f"  Median difference: {differences.median():.2f}\n\n")

        if 'precision' in df.columns and 'recall' in df.columns and 'f1_score' in df.columns:
            f.write("DETECTION ACCURACY METRICS:\n")
            f.write(f"  Mean Precision: {df['precision'].mean():.4f}\n")
            f.write(f"  Mean Recall: {df['recall'].mean():.4f}\n")
            f.write(f"  Mean F1 Score: {df['f1_score'].mean():.4f}\n")
            f.write(f"  Mean Accuracy: {df['accuracy'].mean():.4f}\n\n")

            f.write("DETECTION COUNTS:\n")
            f.write(f"  Total True Positives: {df['true_positives'].sum()}\n")
            f.write(f"  Total False Positives: {df['false_positives'].sum()}\n")
            f.write(f"  Total False Negatives: {df['false_negatives'].sum()}\n\n")

            f.write("PER-IMAGE STATISTICS:\n")
            f.write(f"  Precision - Min: {df['precision'].min():.4f}, Max: {df['precision'].max():.4f}, Std: {df['precision'].std():.4f}\n")
            f.write(f"  Recall    - Min: {df['recall'].min():.4f}, Max: {df['recall'].max():.4f}, Std: {df['recall'].std():.4f}\n")
            f.write(f"  F1 Score  - Min: {df['f1_score'].min():.4f}, Max: {df['f1_score'].max():.4f}, Std: {df['f1_score'].std():.4f}\n")
            f.write(f"  Accuracy  - Min: {df['accuracy'].min():.4f}, Max: {df['accuracy'].max():.4f}, Std: {df['accuracy'].std():.4f}\n\n")

        if 'organ' in df.columns:
            f.write("ORGAN-SPECIFIC METRICS:\n")
            organ_groups = df.groupby('organ')
            for organ, group in organ_groups:
                f.write(f"\n  {organ.upper()}:\n")
                f.write(f"    Images: {len(group)}\n")
                if 'precision' in group.columns:
                    f.write(f"    Precision: {group['precision'].mean():.4f}\n")
                    f.write(f"    Recall: {group['recall'].mean():.4f}\n")
                    f.write(f"    F1 Score: {group['f1_score'].mean():.4f}\n")
                    f.write(f"    Accuracy: {group['accuracy'].mean():.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Analysis saved to: {stats_path}")

    if 'accuracy' in df.columns and len(df) >= 4:
        print(f"\nGenerating accuracy quartile mosaics for {method_name}...")
        method_results_dir = Path('results') / method_name
        generate_accuracy_mosaics(df, method_name, results_dir=str(method_results_dir))
    elif 'accuracy' in df.columns:
        print(f"Skipping mosaic generation: need at least 4 images, got {len(df)}")

    return stats_path
