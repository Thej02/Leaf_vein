import cv2
import numpy as np

def extract_colour_features(image):
    """
    Extracts mean and standard deviation of color channels (HSV, LAB, RGB)
    for the leaf area (ignoring the black background).
    """
    if image is None:
        return None

    # Convert to grayscale to build mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 10  # Boolean mask of the leaf pixels

    if not np.any(mask):
        # Return zeros if leaf is not detected
        return {
            "color_h_mean": 0.0, "color_h_std": 0.0,
            "color_s_mean": 0.0, "color_s_std": 0.0,
            "color_v_mean": 0.0, "color_v_std": 0.0,
            "color_l_mean": 0.0, "color_l_std": 0.0,
            "color_a_mean": 0.0, "color_a_std": 0.0,
            "color_b_mean": 0.0, "color_b_std": 0.0,
            "color_r_mean": 0.0, "color_r_std": 0.0,
            "color_g_mean": 0.0, "color_g_std": 0.0,
            "color_b_rgb_mean": 0.0, "color_b_rgb_std": 0.0
        }

    # Extract HSV features
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_pixels = hsv[:, :, 0][mask]
    s_pixels = hsv[:, :, 1][mask]
    v_pixels = hsv[:, :, 2][mask]

    # Extract LAB features
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_pixels = lab[:, :, 0][mask]
    a_pixels = lab[:, :, 1][mask]
    b_pixels = lab[:, :, 2][mask]

    # Extract RGB features
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_pixels = rgb[:, :, 0][mask]
    g_pixels = rgb[:, :, 1][mask]
    b_rgb_pixels = rgb[:, :, 2][mask]

    features = {
        "color_h_mean": float(np.mean(h_pixels)),
        "color_h_std": float(np.std(h_pixels)),
        "color_s_mean": float(np.mean(s_pixels)),
        "color_s_std": float(np.std(s_pixels)),
        "color_v_mean": float(np.mean(v_pixels)),
        "color_v_std": float(np.std(v_pixels)),
        
        "color_l_mean": float(np.mean(l_pixels)),
        "color_l_std": float(np.std(l_pixels)),
        "color_a_mean": float(np.mean(a_pixels)),
        "color_a_std": float(np.std(a_pixels)),
        "color_b_mean": float(np.mean(b_pixels)),
        "color_b_std": float(np.std(b_pixels)),
        
        "color_r_mean": float(np.mean(r_pixels)),
        "color_r_std": float(np.std(r_pixels)),
        "color_g_mean": float(np.mean(g_pixels)),
        "color_g_std": float(np.std(g_pixels)),
        "color_b_rgb_mean": float(np.mean(b_rgb_pixels)),
        "color_b_rgb_std": float(np.std(b_rgb_pixels))
    }

    return features
