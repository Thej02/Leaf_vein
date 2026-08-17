import os
import sys
import pandas as pd

# Add the workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ImagePreprocessing.preprocessing import preprocess_image
from phase3_deficiency.src.feature_extractor import extract_deficiency_features

LABEL_MAP = {
    "chlorosis": 0,
    "necrosis": 1,
    "scorch": 2
}

def collect_features_for_phase3(unhealthy_dir="dataset/unhealthy", output_csv="phase3_deficiency/data/features.csv"):
    """
    Processes unhealthy leaves from subfolders (chlorosis, necrosis, scorch),
    extracts extended features, and saves them to features.csv.
    """
    data = []
    
    for class_name, label in LABEL_MAP.items():
        class_folder = os.path.join(unhealthy_dir, class_name)
        if not os.path.exists(class_folder):
            print(f"[-] Folder not found: {class_folder}")
            continue
            
        print(f"\nProcessing {class_name} leaves in {class_folder} (Label: {label})...")
        files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for idx, filename in enumerate(files):
            img_path = os.path.join(class_folder, filename)
            print(f"[{idx+1}/{len(files)}] Extracting features from {filename}...")
            
            res = preprocess_image(img_path, save_outputs=False)
            if res is None or not res.get("success", False):
                print(f"  [-] Preprocessing failed for {filename}. Skipping.")
                continue
                
            features = extract_deficiency_features(res)
            if features is None:
                print(f"  [-] Feature extraction failed for {filename}. Skipping.")
                continue
                
            features["filename"] = filename
            features["label"] = label
            data.append(features)
            
    if len(data) == 0:
        print("[-] Error: No deficiency features extracted.")
        return None
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Successfully saved {len(df)} feature records to {output_csv}")
    return df

if __name__ == "__main__":
    collect_features_for_phase3()
