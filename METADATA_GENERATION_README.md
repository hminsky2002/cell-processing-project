# Metadata Generation for Cell Detection Accuracy

## Overview
This update adds comprehensive metadata generation for all cell detection methods, comparing detected cells against ground truth labels using centroid distance matching.

## What Was Added

### 1. New Utility Function in `project_utils.py`

**`calculate_detection_metrics(detected_centroids, ground_truth_annotations, distance_threshold=100)`**

This function performs intelligent matching between detected and ground truth cells:
- Matches detected centroids to ground truth labels within a configurable distance threshold (default: 100 pixels)
- Uses greedy nearest-neighbor matching to avoid double-counting
- Returns comprehensive metrics dictionary with:
  - `true_positives`: Number of correctly detected cells
  - `false_positives`: Number of incorrect detections
  - `false_negatives`: Number of missed cells
  - `precision`: TP / (TP + FP)
  - `recall`: TP / (TP + FN)
  - `f1_score`: Harmonic mean of precision and recall
  - `accuracy`: TP / (TP + FP + FN)

### 2. Enhanced Analysis Function

**`analyze_cell_results(csv_path)`** now generates detailed statistics including:

- **Component Count Differences** (original metrics)
  - Mean, std, min, max, median differences

- **Detection Accuracy Metrics** (NEW)
  - Mean precision, recall, F1 score, accuracy across all images
  - Total true positives, false positives, false negatives
  - Per-image statistics with min/max/std for each metric

- **Organ-Specific Metrics** (NEW)
  - Breakdown of all metrics by organ type
  - Identifies which organs are easier/harder to detect

### 3. Updated Processing Methods

All three cell detection methods now include metadata generation:

#### `process_cell_binary.py`
- Imports `calculate_detection_metrics` from `project_utils`
- Calculates and prints detection metrics for each image
- Adds precision/recall/F1 overlay text to output images
- Returns extended result dictionary with all metrics

#### `process_cell_color.py`
- Same enhancements as binary method
- Works with organ-specific color parameters
- Includes organ info in results

#### `process_cell_advanced.py`
- Same enhancements as other methods
- Multi-stage detection with comprehensive metrics
- All intermediate visualizations include accuracy info

### 4. Updated Main Processing Pipeline

**`process_ocelot_data.py`** now:
- Automatically generates results CSV for all cell methods (not just binary)
- Automatically runs analysis and generates statistics files
- Names output files by method: `{method_name}_comparison.csv`

## Usage

### Running with a Specific Method

```bash
# Process with binary thresholding
python process_ocelot_data.py test cell --method cell_binary --image-limit 10

# Process with color-based detection
python process_ocelot_data.py test cell --method cell_color --image-limit 10

# Process with advanced multi-stage detection
python process_ocelot_data.py test cell --method cell_advanced --image-limit 10
```

### Customizing Distance Threshold

The default distance threshold is 100 pixels. You can modify it by editing the function calls in the processing methods or by adding a parameter to the CLI (future enhancement).

## Output Files

For each method, you'll get:

1. **CSV Results**: `results/{method_name}_comparison.csv`
   - Contains per-image metrics for all processed images
   - Includes all detection metrics (TP, FP, FN, precision, recall, F1, accuracy)
   - Includes organ type for methods that support it

2. **Statistics Report**: `results/{method_name}_comparison_statistics.txt`
   - Comprehensive analysis of detection performance
   - Overall metrics across all images
   - Organ-specific breakdowns
   - Min/max/std statistics

3. **Accuracy Quartile Mosaics** (NEW):
   - `results/{method_name}_mosaic_q4_best.png` - Top 25% most accurate detections
   - `results/{method_name}_mosaic_q3_good.png` - 50-75% percentile
   - `results/{method_name}_mosaic_q2_fair.png` - 25-50% percentile
   - `results/{method_name}_mosaic_q1_worst.png` - Bottom 25% least accurate detections
   - Each mosaic is a 2x2 grid showing representative samples
   - Automatically generated when processing 4+ images

4. **Individual Visualizations**:
   - Overlay images now include accuracy metrics as text
   - Green circles: detected cells
   - Red circles: ground truth labels
   - Text overlay: detection counts + precision/recall/F1

## Example Console Output

```
Processing image_001.jpg as kidney organ (advanced)
  Organ: kidney, Detected: 45, Actual: 42
  TP: 40, FP: 5, FN: 2
  Precision: 0.889, Recall: 0.952, F1: 0.919, Accuracy: 0.851

Analysis saved to: results/cell_advanced_comparison_statistics.txt

Generating accuracy quartile mosaics for cell_advanced...
  Saved mosaic: cell_advanced_mosaic_q4_best.png
  Saved mosaic: cell_advanced_mosaic_q3_good.png
  Saved mosaic: cell_advanced_mosaic_q2_fair.png
  Saved mosaic: cell_advanced_mosaic_q1_worst.png
```

## Mosaic Layout

Each mosaic is organized as a 2x2 grid:

```
┌─────────────┬─────────────┐
│ Image 1     │ Image 2     │
│ Acc: 0.952  │ Acc: 0.938  │
├─────────────┼─────────────┤
│ Image 3     │ Image 4     │
│ Acc: 0.925  │ Acc: 0.912  │
└─────────────┴─────────────┘
  CELL_ADVANCED - Best (Top 25%)
```

Each tile displays:
- The overlay image with detected (green) and ground truth (red) cells
- Image filename (truncated to 20 chars)
- Accuracy score with color coding (green > 0.8, orange > 0.6, red < 0.6)

## Example Statistics Output

```
============================================================
DETECTION RESULTS ANALYSIS
============================================================

COMPONENT COUNT DIFFERENCES:
  Mean difference: 3.45
  Std deviation: 2.31
  Min difference: 0
  Max difference: 8
  Median difference: 3.00

DETECTION ACCURACY METRICS:
  Mean Precision: 0.8234
  Mean Recall: 0.8756
  Mean F1 Score: 0.8487
  Mean Accuracy: 0.7891

DETECTION COUNTS:
  Total True Positives: 423
  Total False Positives: 91
  Total False Negatives: 64

PER-IMAGE STATISTICS:
  Precision - Min: 0.6500, Max: 0.9800, Std: 0.0823
  Recall    - Min: 0.7200, Max: 1.0000, Std: 0.0654
  F1 Score  - Min: 0.6850, Max: 0.9750, Std: 0.0712
  Accuracy  - Min: 0.5200, Max: 0.9600, Std: 0.1034

ORGAN-SPECIFIC METRICS:

  KIDNEY:
    Images: 4
    Precision: 0.8456
    Recall: 0.8923
    F1 Score: 0.8684
    Accuracy: 0.7956

  BLADDER:
    Images: 3
    Precision: 0.8012
    Recall: 0.8589
    F1 Score: 0.8291
    Accuracy: 0.7823

============================================================
```

## Key Features

1. **Accurate Cell Matching**: Uses distance-based matching (100px threshold) to determine true positives
2. **No Double Counting**: Greedy nearest-neighbor ensures each detection matches at most one ground truth
3. **Standard Metrics**: Precision, recall, F1, accuracy - standard metrics used in detection tasks
4. **Per-Image and Aggregate**: Get both individual image results and overall statistics
5. **Organ Analysis**: Understand which tissue types are easier/harder to detect
6. **Visual Feedback**: Overlay images show metrics directly
7. **Accuracy Quartile Mosaics**: Automatically generated 2x2 grids showing best/worst performing samples
   - Helps identify patterns in successful vs. failed detections
   - Color-coded accuracy labels (green > 0.8, orange > 0.6, red < 0.6)
   - Each tile shows image name and accuracy score

## Future Enhancements

Possible improvements:
- Add distance threshold as CLI parameter
- Support for confidence scores/thresholds
- ROC curves and precision-recall curves
- Confusion matrices for multi-class detection
- Export metrics to JSON for easier parsing
