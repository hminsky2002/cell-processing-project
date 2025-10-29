import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def connected_components_analyze_results(csv_path='results/component_comparison.csv'):
    df = pd.read_csv(csv_path)
    differences = df['difference']

    stats = {
        'mean': differences.mean(),
        'std': differences.std(),
        'min': differences.min(),
        'max': differences.max(),
        'median': differences.median()
    }

    print("\nResults Analysis:")
    print(f"Mean difference: {stats['mean']:.2f}")
    print(f"Std deviation: {stats['std']:.2f}")
    print(f"Min difference: {stats['min']}")
    print(f"Max difference: {stats['max']}")
    print(f"Median difference: {stats['median']:.2f}")

    stats_path = Path('results/component_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Results Analysis:\n")
        f.write(f"Mean difference: {stats['mean']:.2f}\n")
        f.write(f"Std deviation: {stats['std']:.2f}\n")
        f.write(f"Min difference: {stats['min']}\n")
        f.write(f"Max difference: {stats['max']}\n")
        f.write(f"Median difference: {stats['median']:.2f}\n")

    return stats

def process_connected_components(image_path, annotations: pd.DataFrame, image=None):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    intensity_threshold = 0.35 * 255
    binary_img = (image < intensity_threshold).astype('uint8')

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

    min_area = 100
    filtered_labels = np.zeros_like(labels)
    valid_label = 1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if min_area <= area:
            filtered_labels[labels == label] = valid_label
            valid_label += 1

    detected_count = valid_label - 1
    actual_count = len(annotations)

    print(f"Image: {image_path.name}, Detected: {detected_count}, Actual: {actual_count}")

    output_path = results_dir / f"{image_path.stem}_1_result.png"
    cv2.imwrite(str(output_path), filtered_labels)


    original_color = cv2.imread(str(image_path))
    overlay = original_color.copy()
    mask = (filtered_labels > 0).astype('uint8') * 255
    overlay[mask > 0] = [0, 0, 255]

    overlay_path = results_dir / f"{image_path.stem}_2_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    original_path = results_dir / f"{image_path.stem}_3_original.png"
    cv2.imwrite(str(original_path), original_color)

    return {
        'image_name': image_path.name,
        'detected_components': detected_count,
        'actual_components': actual_count,
        'difference': abs(detected_count - actual_count)
    }
