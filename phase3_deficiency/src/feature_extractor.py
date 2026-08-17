import os
import sys
import cv2
import numpy as np

# Add parent directory of parent directory to sys.path to import from root modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import all base features from Phase 2
from phase2_classification.src.feature_extractor import extract_all_features as extract_phase2_features

def extract_deficiency_features(preprocess_res):
    """
    Extracts all Phase 2 features and adds Phase 3 deficiency-specific spatial features:
    - Interveinal color contrast
    - Marginal vs Central leaf color difference
    """
    # 1. Start with Phase 2 features
    features = extract_phase2_features(preprocess_res)
    if features is None:
        return None
        
    original = preprocess_res["original"]
    binary_mask = preprocess_res["binary_mask"]
    skeleton = preprocess_res["skeleton"]
    
    # Ensure image size matches mask size (512x512)
    if original.shape[:2] != binary_mask.shape[:2]:
        original = cv2.resize(original, (binary_mask.shape[1], binary_mask.shape[0]))
        
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # 2. Compute Interveinal Contrast
    # Dilate vein skeleton to represent the "vein region" (5x5 kernel)
    kernel_vein = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vein_region = cv2.dilate((skeleton == 255).astype(np.uint8), kernel_vein)
    
    # Vein pixels within leaf mask
    vein_mask = (vein_region == 1) & (binary_mask == 255)
    # Leaf pixels outside the vein region
    interveinal_mask = (vein_region == 0) & (binary_mask == 255)
    
    if np.any(vein_mask) and np.any(interveinal_mask):
        mean_vein_gray = np.mean(gray[vein_mask])
        mean_interveinal_gray = np.mean(gray[interveinal_mask])
        interveinal_contrast = abs(float(mean_vein_gray) - float(mean_interveinal_gray))
    else:
        interveinal_contrast = 0.0
        
    # 3. Compute Marginal vs Central color differences
    dist_transform = cv2.distanceTransform((binary_mask == 255).astype(np.uint8), cv2.DIST_L2, 5)
    max_dist = np.max(dist_transform)
    
    if max_dist > 0:
        # Central is the inner 30% of max distance from edge
        central_mask = (dist_transform > 0.30 * max_dist) & (binary_mask == 255)
        # Marginal is the outer 30% from the edge
        marginal_mask = (dist_transform <= 0.30 * max_dist) & (binary_mask == 255)
    else:
        central_mask = np.zeros_like(binary_mask, dtype=bool)
        marginal_mask = np.zeros_like(binary_mask, dtype=bool)
        
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]  # Saturation
    v_channel = hsv[:, :, 2]  # Value (Brightness)
    
    if np.any(central_mask) and np.any(marginal_mask):
        mean_central_s = np.mean(s_channel[central_mask])
        mean_marginal_s = np.mean(s_channel[marginal_mask])
        mean_central_v = np.mean(v_channel[central_mask])
        mean_marginal_v = np.mean(v_channel[marginal_mask])
        
        marginal_vs_central_s = float(mean_marginal_s) - float(mean_central_s)
        marginal_vs_central_v = float(mean_marginal_v) - float(mean_central_v)
    else:
        marginal_vs_central_s = 0.0
        marginal_vs_central_v = 0.0
        
    # Append new features
    features["deficiency_interveinal_contrast"] = interveinal_contrast
    features["deficiency_marginal_vs_central_s"] = marginal_vs_central_s
    features["deficiency_marginal_vs_central_v"] = marginal_vs_central_v
    
    return features
