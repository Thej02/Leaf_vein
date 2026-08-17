import os
import sys
import pandas as pd

# Add the workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ImagePreprocessing.preprocessing import preprocess_image
from phase2_classification.src.feature_extractor import extract_all_features

def collect_features_for_phase2(healthy_dir="dataset/healthy", unhealthy_dir="dataset/unhealthy", output_csv="phase2_classification/data/features.csv"):
    """
    Processes all images in healthy and unhealthy directories,
    extracts features using feature_extractor, and saves them to output_csv.
    """
    data = []
    
    # Process healthy leaves (label 0)
    if os.path.exists(healthy_dir):
        print(f"\nProcessing healthy leaves in {healthy_dir}...")
        files = [f for f in os.listdir(healthy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for idx, filename in enumerate(files):
            img_path = os.path.join(healthy_dir, filename)
            print(f"[{idx+1}/{len(files)}] Extracting features from {filename}...")
            
            res = preprocess_image(img_path, save_outputs=False)
            if res is None or not res.get("success", False):
                print(f"  [-] Preprocessing failed for {filename}. Skipping.")
                continue
                
            features = extract_all_features(res)
            if features is None:
                print(f"  [-] Feature extraction failed for {filename}. Skipping.")
                continue
                
            features["filename"] = filename
            features["label"] = 0  # 0 = Healthy
            data.append(features)
    else:
        print(f"[-] Healthy directory not found: {healthy_dir}")
        
    # Process unhealthy leaves (label 1)
    if os.path.exists(unhealthy_dir):
        print(f"\nProcessing unhealthy leaves in {unhealthy_dir}...")
        files = [f for f in os.listdir(unhealthy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for idx, filename in enumerate(files):
            img_path = os.path.join(unhealthy_dir, filename)
            print(f"[{idx+1}/{len(files)}] Extracting features from {filename}...")
            
            res = preprocess_image(img_path, save_outputs=False)
            if res is None or not res.get("success", False):
                print(f"  [-] Preprocessing failed for {filename}. Skipping.")
                continue
                
            features = extract_all_features(res)
            if features is None:
                print(f"  [-] Feature extraction failed for {filename}. Skipping.")
                continue
                
            features["filename"] = filename
            features["label"] = 1  # 1 = Unhealthy (nutrient-stressed)
            data.append(features)
    else:
        print(f"[-] Unhealthy directory not found: {unhealthy_dir}")
        
    if len(data) == 0:
        print("[-] Error: No data collected.")
        return None
        
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Successfully saved {len(df)} feature records to {output_csv}")
    return df

if __name__ == "__main__":
    collect_features_for_phase2()
