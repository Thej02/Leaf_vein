import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add root workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def compute_healthy_baseline(features_csv="phase2_classification/data/features.csv", models_dir="phase4_recommendation/models"):
    """
    Computes the mean and std dev of vein features across healthy leaves in Phase 2 features.csv.
    """
    os.makedirs(models_dir, exist_ok=True)
    baseline_path = os.path.join(models_dir, "healthy_baseline.pkl")
    
    # Check if features.csv exists
    if os.path.exists(features_csv):
        try:
            df = pd.read_csv(features_csv)
            # Filter for healthy samples (label 0)
            healthy_df = df[df["label"] == 0]
            
            if len(healthy_df) > 0:
                vein_density_mean = float(healthy_df["vein_density"].mean())
                vein_density_std = float(healthy_df["vein_density"].std())
                
                vein_thickness_mean = float(healthy_df["vein_thickness"].mean())
                vein_thickness_std = float(healthy_df["vein_thickness"].std())
                
                # If std is 0 or NaN (due to single sample or uniform data), set a small positive value
                if np.isnan(vein_density_std) or vein_density_std == 0:
                    vein_density_std = 0.01
                if np.isnan(vein_thickness_std) or vein_thickness_std == 0:
                    vein_thickness_std = 0.5
                    
                baseline = {
                    "vein_density": {"mean": vein_density_mean, "std": vein_density_std},
                    "vein_thickness": {"mean": vein_thickness_mean, "std": vein_thickness_std}
                }
                
                print(f"[OK] Computed healthy baseline from {len(healthy_df)} healthy samples:")
                print(f"  Vein Density   : Mean = {vein_density_mean:.4f}, Std = {vein_density_std:.4f}")
                print(f"  Vein Thickness : Mean = {vein_thickness_mean:.4f}, Std = {vein_thickness_std:.4f}")
                
                with open(baseline_path, "wb") as f:
                    pickle.dump(baseline, f)
                return baseline
        except Exception as e:
            print(f"[-] Warning: Failed to read healthy baseline from CSV ({e}). Falling back to defaults.")
            
    # Default fallbacks representing healthy hibiscus parameters
    baseline = {
        "vein_density": {"mean": 0.070, "std": 0.010},
        "vein_thickness": {"mean": 4.200, "std": 0.500}
    }
    print("[OK] Sourced default healthy baseline fallback parameters.")
    
    with open(baseline_path, "wb") as f:
        pickle.dump(baseline, f)
    return baseline

if __name__ == "__main__":
    compute_healthy_baseline()
