# Cell Detection Tuning Guide

## Problem: Too Many False Negatives (Missing Real Cells)

If you're getting more **false negatives** than true positives, the algorithm is being too conservative. Follow this guide to tune the parameters.

## Quick Start - Edit These Lines in `process_cell_advanced.py`

Find `DEFAULT_TUNING_PARAMS` (lines 19-43) and modify these key values:

### For High Recall (Reduce False Negatives)

```python
DEFAULT_TUNING_PARAMS = {
    # CRITICAL: Lower this to detect more cells
    'dist_transform_threshold': 0.03,     # Was 0.05, try 0.03-0.04

    # IMPORTANT: Try skipping adaptive threshold
    'use_color_mask_only': True,          # Was False

    # Accept more varied shapes
    'blob_min_circularity': 0.05,         # Was 0.1
    'blob_min_convexity': 0.05,           # Was 0.1
    'blob_min_inertia': 0.05,             # Was 0.1
    'compactness_min': 0.2,               # Was 0.3
    'aspect_ratio_max': 4.0,              # Was 3.0
}
```

## Parameter Impact Ranking

### TIER 1 - Highest Impact (Adjust These First)

#### 1. `dist_transform_threshold` (Default: 0.05)
**What it does:** Controls how aggressively we detect cell centers using distance transform peaks.

**For reducing false negatives:**
- Try values: `0.03` to `0.04`
- Lower = more detections
- Too low (< 0.02) = may detect noise as cells

**Example:**
```python
'dist_transform_threshold': 0.03  # More sensitive
```

#### 2. `use_color_mask_only` (Default: False)
**What it does:** Skips the adaptive thresholding step, using only color filtering.

**When to use True:**
- If adaptive threshold is eliminating real cells
- If cells have varied brightness/contrast
- If you're missing cells in darker regions

**Example:**
```python
'use_color_mask_only': True  # Less restrictive
```

### TIER 2 - Medium Impact

#### 3. Blob Detector Parameters
All three should be lowered together to accept more varied shapes:

```python
'blob_min_circularity': 0.05,  # Accept less circular (was 0.1)
'blob_min_convexity': 0.05,    # Accept less convex (was 0.1)
'blob_min_inertia': 0.05,      # Accept more varied (was 0.1)
```

#### 4. `compactness_min` (Default: 0.3)
**What it does:** Minimum ratio of component area to bounding box area.

```python
'compactness_min': 0.2  # Accept more irregular shapes
```

#### 5. `aspect_ratio_max` (Default: 3.0)
**What it does:** Maximum elongation ratio (width/height or vice versa).

```python
'aspect_ratio_max': 4.0  # Accept more elongated cells
```

### TIER 3 - Fine Tuning

#### 6. `adaptive_block_size` (Default: 31)
**What it does:** Size of local neighborhood for adaptive threshold.

```python
'adaptive_block_size': 21  # Smaller = more local sensitivity
```

#### 7. `morph_kernel_size` (Default: 3)
**What it does:** Size of morphological operations kernel.

```python
'morph_kernel_size': 3  # Keep small to preserve small cells
```

## Recommended Presets

### Preset 1: Maximum Recall (Catch Everything)
Use when you want to minimize false negatives at the cost of some false positives:

```python
DEFAULT_TUNING_PARAMS = {
    'dist_transform_threshold': 0.03,
    'use_color_mask_only': True,
    'morph_kernel_size': 3,
    'morph_open_iterations': 1,
    'morph_close_iterations': 2,
    'adaptive_block_size': 31,
    'adaptive_C': 8,
    'aspect_ratio_max': 5.0,
    'compactness_min': 0.15,
    'blob_detector_enabled': True,
    'blob_min_circularity': 0.05,
    'blob_min_convexity': 0.05,
    'blob_min_inertia': 0.05,
    'blob_merge_distance': 10,
}
```

### Preset 2: Balanced (Default with Better Recall)
Moderate improvement in recall while maintaining reasonable precision:

```python
DEFAULT_TUNING_PARAMS = {
    'dist_transform_threshold': 0.04,
    'use_color_mask_only': False,
    'morph_kernel_size': 3,
    'morph_open_iterations': 1,
    'morph_close_iterations': 2,
    'adaptive_block_size': 31,
    'adaptive_C': 8,
    'aspect_ratio_max': 3.5,
    'compactness_min': 0.25,
    'blob_detector_enabled': True,
    'blob_min_circularity': 0.08,
    'blob_min_convexity': 0.08,
    'blob_min_inertia': 0.08,
    'blob_merge_distance': 10,
}
```

### Preset 3: High Precision (Current Default)
Minimizes false positives but may miss some cells:

```python
DEFAULT_TUNING_PARAMS = {
    'dist_transform_threshold': 0.05,
    'use_color_mask_only': False,
    'morph_kernel_size': 3,
    'morph_open_iterations': 1,
    'morph_close_iterations': 2,
    'adaptive_block_size': 31,
    'adaptive_C': 8,
    'aspect_ratio_max': 3.0,
    'compactness_min': 0.3,
    'blob_detector_enabled': True,
    'blob_min_circularity': 0.1,
    'blob_min_convexity': 0.1,
    'blob_min_inertia': 0.1,
    'blob_merge_distance': 10,
}
```

## Iterative Tuning Process

1. **Start with Preset 1 (Maximum Recall)**
   - Run on a few test images
   - Check your recall and precision metrics

2. **If too many false positives:**
   - Gradually increase `dist_transform_threshold` by 0.01
   - Increase blob detector minimums by 0.01-0.02
   - Increase `compactness_min` by 0.05

3. **If still too many false negatives:**
   - Set `use_color_mask_only = True`
   - Lower `dist_transform_threshold` to 0.02
   - Disable blob detector temporarily to isolate issues

4. **Monitor metrics:**
   - Recall < 0.8 = too many false negatives, lower thresholds
   - Precision < 0.7 = too many false positives, raise thresholds
   - F1 score balances both

## Understanding the Debug Images

The algorithm saves 4 intermediate images:

1. **`_cell_adv_1_color_mask.png`**: HSV color filtering result
   - White = pixels matching cell color
   - If this misses cells, adjust ORGAN_PARAMS HSV ranges

2. **`_cell_adv_2_combined.png`**: Combined color + adaptive threshold
   - Shows where both color AND intensity conditions are met
   - If too restrictive, set `use_color_mask_only = True`

3. **`_cell_adv_3_distance.png`**: Distance transform
   - Brighter = further from edges (likely cell centers)
   - Shows potential cell peaks

4. **`_cell_adv_4_centers.png`**: Thresholded peaks
   - White blobs = detected cell centers before filtering
   - If missing cells here, lower `dist_transform_threshold`

## Common Issues and Solutions

### Issue: Missing cells in dark regions
**Solution:** Set `use_color_mask_only = True`

### Issue: Missing small/irregular cells
**Solution:**
- Lower all blob detector minimums to 0.05
- Lower `compactness_min` to 0.2
- Check `min_area` in ORGAN_PARAMS isn't too high

### Issue: Detecting gaps between cells as cells
**Solution:**
- Check debug images - if color_mask has gaps, adjust HSV ranges
- Increase `morph_close_iterations` to fill gaps

### Issue: Merging adjacent cells into one
**Solution:**
- Lower `morph_close_iterations`
- Increase `dist_transform_threshold` slightly

## Example: Tuning for Your Data

```python
# Edit DEFAULT_TUNING_PARAMS at top of process_cell_advanced.py:

# Step 1: Try maximum recall first
'dist_transform_threshold': 0.03,
'use_color_mask_only': True,

# Step 2: Run and check metrics
# If Recall = 0.9, Precision = 0.6 → Increase threshold to 0.04

# Step 3: Fine-tune
'dist_transform_threshold': 0.04,  # Slightly more conservative
'blob_min_circularity': 0.07,      # Filter out some noise

# Step 4: Iterate until F1 score is optimized
```

## Quick Diagnosis

```
High False Negatives (FN > TP):
├─ Check color_mask debug image
│  ├─ Missing cells? → Adjust HSV ranges in ORGAN_PARAMS
│  └─ Cells present? → Continue
├─ Check combined_mask debug image
│  ├─ Cells disappeared? → Set use_color_mask_only = True
│  └─ Cells present? → Continue
├─ Check centers debug image
│  ├─ No white blobs? → Lower dist_transform_threshold to 0.03
│  └─ Blobs present? → Lower blob detector minimums

High False Positives (FP > TP):
├─ Increase dist_transform_threshold to 0.06-0.08
├─ Increase blob detector minimums to 0.15-0.2
└─ Increase compactness_min to 0.4-0.5
```

## Best Practices

1. **Always check debug images first** to understand where cells are lost
2. **Adjust one parameter at a time** to understand its impact
3. **Use the mosaics** to visually compare results across quartiles
4. **Monitor F1 score** as the primary metric (balances precision and recall)
5. **Start conservative with HSV ranges** - tighten if needed, don't loosen
