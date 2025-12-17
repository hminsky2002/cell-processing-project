import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from project_utils import calculate_detection_metrics

def process_cell_haar(
    image_path,
    annotations: pd.DataFrame,
    image=None,
    cascade_path=None,
    scale_factor=1.1,
    min_neighbors=3,
    min_size=(10, 10),
    max_size=(50, 50),
    distance_threshold=100
):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        print(f"Warning: Could not load cascade from {cascade_path}")
        return {
            'image_name': Path(image_path).name,
            'detected_components': 0,
            'actual_components': len(annotations),
            'difference': len(annotations),
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(annotations),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0
        }

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    detections = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        maxSize=max_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    detected_centroids = []
    for (x, y, w, h) in detections:
        cx = x + w // 2
        cy = y + h // 2
        detected_centroids.append(np.array([cx, cy], dtype=np.float32))

    detected_count = len(detected_centroids)
    actual_count = len(annotations)

    metrics = calculate_detection_metrics(detected_centroids, annotations, distance_threshold)

    print(f"Processing {Path(image_path).name} with Haar Cascade")
    print(f"  Detected: {detected_count}, Actual: {actual_count}")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, "
          f"F1: {metrics['f1_score']:.3f}, Accuracy: {metrics['accuracy']:.3f}")

    overlay = image_bgr.copy()

    for (x, y, w, h) in detections:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(overlay, (cx, cy), 3, (0, 255, 0), -1)

    if annotations is not None and len(annotations) > 0:
        for _, row in annotations.iterrows():
            cx, cy = int(row['x']), int(row['y'])
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), 2)

    cv2.putText(overlay, f"Haar Cascade Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"Detected: {detected_count} | Actual: {actual_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"P: {metrics['precision']:.2f} R: {metrics['recall']:.2f} "
                         f"F1: {metrics['f1_score']:.2f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    overlay_path = results_dir / f"{Path(image_path).stem}_cell_haar_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    equalized_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(results_dir / f"{Path(image_path).stem}_cell_haar_equalized.png"), equalized_vis)

    return {
        'image_name': Path(image_path).name,
        'detected_components': detected_count,
        'actual_components': actual_count,
        'difference': abs(detected_count - actual_count),
        **metrics
    }
