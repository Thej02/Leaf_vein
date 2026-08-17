import numpy as np

def compute_severity(features, baseline):
    """
    Computes a quantitative severity score combining leaf discoloration
    and vein feature deviations from healthy baseline values.
    
    Returns:
        dict: containing severity score, severity tier (mild/moderate/severe),
              discoloration score, and vein deviation.
    """
    # 1. Affected-area discoloration score (0 to 1)
    discoloration_score = features.get("color_non_green_percentage", 0.0)
    
    # 2. Vein feature deviation from baseline (z-score distance)
    vein_density = features.get("vein_density", 0.0)
    vein_thickness = features.get("vein_thickness", 0.0)
    
    density_baseline = baseline.get("vein_density", {"mean": 0.07, "std": 0.01})
    thickness_baseline = baseline.get("vein_thickness", {"mean": 4.2, "std": 0.5})
    
    # Calculate z-scores
    z_density = abs(vein_density - density_baseline["mean"]) / density_baseline["std"]
    z_thickness = abs(vein_thickness - thickness_baseline["mean"]) / thickness_baseline["std"]
    
    # Average z-score deviation
    vein_deviation = (z_density + z_thickness) / 2.0
    
    # Normalize z-score to a 0-1 scale (capping at z-score of 3.0 as max deviation)
    vein_deviation_norm = min(1.0, vein_deviation / 3.0)
    
    # 3. Combined weighted severity index
    # Weight: 60% Discoloration, 40% Vein Deviation
    severity_score = 0.6 * discoloration_score + 0.4 * vein_deviation_norm
    
    # Determine severity tier
    if severity_score < 0.15:
        tier = "mild"
    elif severity_score < 0.40:
        tier = "moderate"
    else:
        tier = "severe"
        
    return {
        "score": float(severity_score),
        "tier": tier,
        "discoloration": float(discoloration_score),
        "vein_deviation": float(vein_deviation_norm)
    }
