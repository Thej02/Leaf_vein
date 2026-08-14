import numpy as np

def extract_vein_features(skeleton, binary_mask):
    """
    Computes vein density using the skeletonized vein image and the leaf binary mask.
    Vein density is defined as the ratio of vein pixels (255) to leaf mask pixels (255).
    """
    if skeleton is None or binary_mask is None:
        return {"vein_density": 0.0}

    # Sum up the pixels
    vein_pixels = np.sum(skeleton == 255)
    leaf_pixels = np.sum(binary_mask == 255)

    vein_density = float(vein_pixels) / leaf_pixels if leaf_pixels > 0 else 0.0

    return {
        "vein_density": vein_density
    }
