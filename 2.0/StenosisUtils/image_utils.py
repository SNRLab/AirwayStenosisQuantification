import numpy as np


def toGrayscale2D(array):
    """Convert a single video frame / volume array to a 2-D uint8 grayscale image.

    Accepts the layouts slicer.util.arrayFromVolume can produce for a single frame:
        (H, W)                       2-D image
        (1, H, W)                    scalar volume, one slice along K
        (H, W, 1)                    scalar volume, one slice along I
        (1, H, W, C) / (H, W, 1, C)  vector (e.g. RGB) volume; first 3 components averaged

    For a 3-D array with no singleton axis (a true multi-slice volume), the first
    slice along axis 0 is used.
    """
    a = np.asarray(array)

    if a.ndim == 4:
        components = a.shape[-1]
        a = np.mean(a[..., :3], axis=-1) if components >= 3 else a[..., 0]

    if a.ndim == 3:
        if a.shape[0] == 1:
            a = a[0]
        elif a.shape[-1] == 1:
            a = a[..., 0]
        elif a.shape[1] == 1:
            a = a[:, 0, :]
        else:
            a = a[0]

    if a.ndim != 2:
        raise ValueError(f"Cannot convert array of shape {np.shape(array)} to a 2-D image")

    return a.astype(np.uint8)
