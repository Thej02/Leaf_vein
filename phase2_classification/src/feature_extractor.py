import os
import sys
import cv2
import numpy as np

# Add parent directory of parent directory to sys.path to import from root modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Identification.Features.shape_analysis import extract_shape_features
from Identification.Features.colour_analysis import extract_colour_features
from Identification.Features.texture_analysis import extract_texture_features

def count_branch_points(skeleton):
    """
    Counts the number of branch points in a binary skeleton.
    A branch point is a skeleton pixel that has 3 or more neighbors.
    """
    if skeleton is None or np.sum(skeleton == 255) == 0:
        return 0
    
    # Binary representation: 1 for skeleton, 0 for background
    skel_bin = (skeleton == 255).astype(np.uint8)
    
    # 3x3 kernel to sum 8-neighbors (excluding the center pixel itself)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Convolve to sum neighbors for every pixel
    neighbor_count = cv2.filter2D(skel_bin, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    
    # Only look at neighbors of skeleton pixels
    skel_neighbors = neighbor_count * skel_bin
    
    # A branch point has 3 or more neighbors in skeletonization
    branch_points = np.sum(skel_neighbors >= 3)
    return int(branch_points)

def extract_advanced_vein_features(skeleton, binary_mask, vein_enhancement):
    """
    Extracts advanced vein features including thickness, length, branch points, and spread.
    """
    if skeleton is None or binary_mask is None or vein_enhancement is None:
        return {
            "vein_density": 0.0,
            "vein_thickness": 0.0,
            "vein_length": 0.0,
            "vein_branch_points": 0.0,
            "vein_spread_x": 0.0,
            "vein_spread_y": 0.0
        }
        
    leaf_pixels = np.sum(binary_mask == 255)
    vein_pixels = np.sum(skeleton == 255)
    
    # 1. Vein Density (from Phase 1)
    vein_density = float(vein_pixels) / leaf_pixels if leaf_pixels > 0 else 0.0
    
    # 2. Vein Length (Total skeleton length normalized by leaf area)
    # Since skeleton is single-pixel wide, length = sum of pixels
    vein_length = float(vein_pixels) / leaf_pixels if leaf_pixels > 0 else 0.0
    
    # 3. Vein Thickness (mean skeleton branch width using distance transform)
    # Threshold vein enhancement to get a binary vein map
    _, binary_veins = cv2.threshold(vein_enhancement, 25, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(binary_veins.astype(np.uint8), cv2.DIST_L2, 5)
    
    # The thickness is estimated at the skeleton locations
    skeleton_pts = skeleton == 255
    if np.any(skeleton_pts) and np.any(binary_veins):
        # Mean thickness is twice the distance from skeleton to the background
        vein_thickness = float(2.0 * np.mean(dist_transform[skeleton_pts]))
    else:
        vein_thickness = 0.0
        
    # 4. Vein branching points count
    branch_points = count_branch_points(skeleton)
    
    # 5. Vein Spatial Spread (std dev of coordinates normalized by image size 512)
    y_indices, x_indices = np.where(skeleton == 255)
    if len(x_indices) > 0:
        spread_x = float(np.std(x_indices)) / 512.0
        spread_y = float(np.std(y_indices)) / 512.0
    else:
        spread_x = 0.0
        spread_y = 0.0
        
    return {
        "vein_density": vein_density,
        "vein_thickness": vein_thickness,
        "vein_length": vein_length,
        "vein_branch_points": float(branch_points),
        "vein_spread_x": spread_x,
        "vein_spread_y": spread_y
    }

def extract_full_leaf_colour_features(image, full_leaf_mask):
    """
    Extracts color features (mean and std dev in HSV, LAB, RGB) for the entire leaf
    including chlorotic (yellow) or necrotic (brown) areas.
    """
    if image is None or full_leaf_mask is None:
        return {}
        
    if image.shape[:2] != full_leaf_mask.shape[:2]:
        image = cv2.resize(image, (full_leaf_mask.shape[1], full_leaf_mask.shape[0]))
        
    mask = full_leaf_mask == 255
    if not np.any(mask):
        return {
            "full_color_h_mean": 0.0, "full_color_h_std": 0.0,
            "full_color_s_mean": 0.0, "full_color_s_std": 0.0,
            "full_color_v_mean": 0.0, "full_color_v_std": 0.0,
            "full_color_l_mean": 0.0, "full_color_l_std": 0.0,
            "full_color_a_mean": 0.0, "full_color_a_std": 0.0,
            "full_color_b_mean": 0.0, "full_color_b_std": 0.0,
            "full_color_r_mean": 0.0, "full_color_r_std": 0.0,
            "full_color_g_mean": 0.0, "full_color_g_std": 0.0,
            "full_color_b_rgb_mean": 0.0, "full_color_b_rgb_std": 0.0
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
    
    return {
        "full_color_h_mean": float(np.mean(h_pixels)),
        "full_color_h_std": float(np.std(h_pixels)),
        "full_color_s_mean": float(np.mean(s_pixels)),
        "full_color_s_std": float(np.std(s_pixels)),
        "full_color_v_mean": float(np.mean(v_pixels)),
        "full_color_v_std": float(np.std(v_pixels)),
        
        "full_color_l_mean": float(np.mean(l_pixels)),
        "full_color_l_std": float(np.std(l_pixels)),
        "full_color_a_mean": float(np.mean(a_pixels)),
        "full_color_a_std": float(np.std(a_pixels)),
        "full_color_b_mean": float(np.mean(b_pixels)),
        "full_color_b_std": float(np.std(b_pixels)),
        
        "full_color_r_mean": float(np.mean(r_pixels)),
        "full_color_r_std": float(np.std(r_pixels)),
        "full_color_g_mean": float(np.mean(g_pixels)),
        "full_color_g_std": float(np.std(g_pixels)),
        "full_color_b_rgb_mean": float(np.mean(b_rgb_pixels)),
        "full_color_b_rgb_std": float(np.std(b_rgb_pixels))
    }

def extract_all_features(preprocess_res):
    """
    Extracts all Phase 2 features from preprocessing results.
    """
    if preprocess_res is None or not preprocess_res.get("success", False):
        return None
        
    preprocessed = preprocess_res["preprocessed"]
    binary_mask = preprocess_res["binary_mask"]
    full_leaf_mask = preprocess_res.get("full_leaf_mask", binary_mask)
    skeleton = preprocess_res["skeleton"]
    vein_enhancement = preprocess_res["vein_enhancement"]
    original = preprocess_res["original"]
    
    # 1. Standard Shape Features
    shape_feats = extract_shape_features(preprocessed)
    
    # 2. Refined Color Features (green leaf parts only)
    refined_color_feats = extract_colour_features(preprocessed)
    
    # 3. Full Leaf Color Features (includes yellow/brown parts)
    full_color_feats = extract_full_leaf_colour_features(original, full_leaf_mask)
    
    # 4. Texture Features
    texture_feats = extract_texture_features(preprocessed)
    
    # 5. Advanced Vein Features
    vein_feats = extract_advanced_vein_features(skeleton, binary_mask, vein_enhancement)
    
    # 6. Non-green area ratio
    green_area = np.sum(binary_mask == 255)
    full_area = np.sum(full_leaf_mask == 255)
    
    non_green_percentage = 0.0
    if full_area > 0:
        non_green_percentage = 1.0 - (float(green_area) / full_area)
        # Ensure it doesn't go negative due to float precision
        non_green_percentage = max(0.0, min(1.0, non_green_percentage))
        
    extra_feats = {
        "color_non_green_percentage": non_green_percentage
    }
    
    # Combine everything
    combined = {}
    if shape_feats:
        combined.update(shape_feats)
    if refined_color_feats:
        combined.update(refined_color_feats)
    if full_color_feats:
        combined.update(full_color_feats)
    if texture_feats:
        combined.update(texture_feats)
    if vein_feats:
        combined.update(vein_feats)
    combined.update(extra_feats)
    
    return combined
