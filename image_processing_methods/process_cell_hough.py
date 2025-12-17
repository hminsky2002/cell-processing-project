import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from project_utils import calculate_detection_metrics

def process_cell_hough(
    image_path,
    annotations: pd.DataFrame,
    image=None,
    min_radius=5,
    max_radius=25,
    param1=50,
    param2=30,
    min_dist=20,
    distance_threshold=100
):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    detected_centroids = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detected_centroids.append(np.array([x, y], dtype=np.float32))

    detected_count = len(detected_centroids)
    actual_count = len(annotations)

    metrics = calculate_detection_metrics(detected_centroids, annotations, distance_threshold)

    print(f"Processing {Path(image_path).name} with Hough Circle Transform")
    print(f"  Detected: {detected_count}, Actual: {actual_count}")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, "
          f"F1: {metrics['f1_score']:.3f}, Accuracy: {metrics['accuracy']:.3f}")

    overlay = image_bgr.copy()

    if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
            cv2.circle(overlay, (x, y), 2, (0, 255, 0), 3)

    if annotations is not None and len(annotations) > 0:
        for _, row in annotations.iterrows():
            cx, cy = int(row['x']), int(row['y'])
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), 2)

    cv2.putText(overlay, f"Hough Circle Transform", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"Detected: {detected_count} | Actual: {actual_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"P: {metrics['precision']:.2f} R: {metrics['recall']:.2f} "
                         f"F1: {metrics['f1_score']:.2f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    overlay_path = results_dir / f"{Path(image_path).stem}_cell_hough_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    blurred_vis = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(results_dir / f"{Path(image_path).stem}_cell_hough_blurred.png"), blurred_vis)

    return {
        'image_name': Path(image_path).name,
        'detected_components': detected_count,
        'actual_components': actual_count,
        'difference': abs(detected_count - actual_count),
        **metrics
    }
