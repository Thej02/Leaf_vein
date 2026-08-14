import cv2
import numpy as np
import os
import pandas as pd

# =====================================
# Shape Feature Extraction Function
# =====================================

def extract_shape_features(image):
    """
    Extracts scale-invariant shape features from a preprocessed leaf image.
    Returns a dictionary of features, or None if extraction fails.
    """
    if image is None:
        print("[-] Error: Image is None.")
        return None

    # Convert to Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binary Threshold to isolate the leaf silhouette
    _, thresh = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # Find Contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("[-] No leaf contours found in image.")
        return None

    # Get largest contour (which is the leaf)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if area == 0:
        print("[-] Leaf contour area is zero.")
        return None

    # 1. Aspect Ratio (Width / Height of bounding box)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # 2. Circularity / Compactness (4 * pi * Area / Perimeter^2)
    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # 3. Solidity (Area / Convex Hull Area)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0

    # 4. Extent (Area / Bounding Box Area)
    bbox_area = float(w * h)
    extent = area / bbox_area if bbox_area > 0 else 0.0

    # 5. Eccentricity (using fitted ellipse semi-axes)
    eccentricity = 0.0
    if len(largest_contour) >= 5:
        try:
            (x_ell, y_ell), (MA, ma), angle = cv2.fitEllipse(largest_contour)
            a = max(MA, ma) / 2.0
            b = min(MA, ma) / 2.0
            eccentricity = np.sqrt(a**2 - b**2) / a if a > 0 else 0.0
        except Exception:
            pass

    # Alternative eccentricity calculation from moments if ellipse fitting failed
    if eccentricity == 0.0:
        moments = cv2.moments(largest_contour)
        mu20 = moments['mu20']
        mu02 = moments['mu02']
        mu11 = moments['mu11']
        diff = mu20 - mu02
        sum_mu = mu20 + mu02
        if sum_mu > 0:
            eccentricity = np.sqrt(diff**2 + 4 * mu11**2) / sum_mu

    # 6. Hu Moments (scale, translation, and rotation invariant)
    moments = cv2.moments(largest_contour)
    hu = cv2.HuMoments(moments).flatten()
    
    # Log-transform Hu moments to make them comparable
    hu_log = []
    for val in hu:
        if val != 0:
            hu_log.append(-1.0 * np.sign(val) * np.log10(np.abs(val)))
        else:
            hu_log.append(0.0)

    # Compile shape features dictionary
    features = {
        "shape_aspect_ratio": aspect_ratio,
        "shape_circularity": circularity,
        "shape_solidity": solidity,
        "shape_extent": extent,
        "shape_eccentricity": eccentricity,
        "shape_hu_1": hu_log[0],
        "shape_hu_2": hu_log[1],
        "shape_hu_3": hu_log[2]
    }

    return features

if __name__ == "__main__":
    # Test script on one of the healthy dataset leaves
    test_path = "../../dataset/healthy/leaf1.jpeg"
    if os.path.exists(test_path):
        img = cv2.imread(test_path)
        print("Testing feature extraction on:", test_path)
        feats = extract_shape_features(img)
        print("Extracted Shape Features:")
        for k, v in feats.items():
            print(f"  {k}: {v:.6f}")
    else:
        print(f"Could not find test image at {test_path}")