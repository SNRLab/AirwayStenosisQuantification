# Airway Stenosis Quantification — v1.0

A 3D Slicer module for estimating airway stenosis index from bronchoscopic images using cGAN-based depth estimation.

## Requirements

- **3D Slicer** 5.0.3
- **Python packages** (install via the Slicer Python console):

```python
slicer.util.pip_install("torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116")
slicer.util.pip_install("opencv-python")
```

## Model Weights

The depth estimation model is currently included.

## Usage

### Ground Truth Stenosis Index

1. Load inspiration and expiration CT volumes into Slicer.
   - Images must be **200x200** pixels and **grayscale**.
2. In the **Ground Truth** section, select the inspiration and expiration CT volumes.
3. Click **Segment CT** — this creates segmentation nodes (`insp_segmentation`, `exp_segmentation`) and segments (`insp_segment`, `exp_segment`), and displays axial views.
4. Open the **Segment Editor** and segment the airway in the slice corresponding to your bronchoscopic images.
5. Return to the Airway Stenosis module and click **Calculate Ground Truth SI** to see the ground truth stenosis index.

### Estimated Stenosis Index

1. In the **Inputs** section, set the file paths for the expiration and inspiration bronchoscopic images.
2. Click **Show Images and Depths** to display the images and their estimated depth maps.
3. In the **Estimating Stenosis Index** section, place a markup on the edge of the obstruction in the expiration image.
4. Select the marker and click **Run** — segmented depth maps will be displayed.
5. Use the **threshold slider** to adjust the depth threshold and refine the segmented airway boundaries.
6. Read the estimated Stenosis Index.

### Resetting

Click **Reload** to reset the module and start with a new dataset.

## Notes

- The stenosis index (SI) is calculated as: `SI = 1 - (expiration_pixels / inspiration_pixels)`, where pixel counts are determined by thresholding the estimated depth maps.
