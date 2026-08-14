import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_texture_features(image):
    """
    Extracts Local Binary Patterns (LBP) texture features from a preprocessed image.
    Computes a normalized histogram of LBP codes for pixels inside the leaf silhouette.
    """
    if image is None:
        return None

    # Convert to Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    mask = gray > 10  # Boolean mask of the leaf pixels

    if not np.any(mask):
        return {f"texture_lbp_{i}": 0.0 for i in range(10)}

    # LBP settings: P=8 neighbors, R=1 radius (standard for fine textures)
    radius = 1
    n_points = 8 * radius
    
    # Use 'uniform' method for rotation invariance and reduced dimensionality (10 bins)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Extract LBP values only within the leaf mask
    lbp_masked = lbp[mask]
    
    # Compute histogram (bins range from 0 to P + 1, so 10 bins total)
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)
    
    features = {}
    for i in range(10):
        features[f"texture_lbp_{i}"] = float(hist[i]) if i < len(hist) else 0.0
        
    return features
