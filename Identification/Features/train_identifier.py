import os
import cv2
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

# Import preprocessing and feature extractors
# Add root path to sys.path to allow importing from other directories
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ImagePreprocessing.preprocessing import preprocess_image
from Identification.Features.shape_analysis import extract_shape_features
from Identification.Features.colour_analysis import extract_colour_features
from Identification.Features.texture_analysis import extract_texture_features
from Identification.Features.vein_analysis import extract_vein_features

def collect_features_from_folder(folder_path, label):
    """
    Scans a folder for images, runs preprocessing, extracts features,
    and returns a list of feature dictionaries and corresponding labels.
    """
    data = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"[-] Folder not found: {folder_path}")
        return data, labels

    print(f"\nProcessing folder: {folder_path} (Label: {label})")
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for idx, filename in enumerate(files):
        img_path = os.path.join(folder_path, filename)
        print(f"[{idx+1}/{len(files)}] Preprocessing and extracting features from {filename}...")
        
        # Run preprocessing
        res = preprocess_image(img_path, save_outputs=False)
        if res is None or not res["success"]:
            print(f"  [-] Preprocessing failed for {filename}. Skipping.")
            continue
            
        preprocessed = res["preprocessed"]
        skeleton = res["skeleton"]
        binary_mask = res["binary_mask"]
        
        # Extract features
        shape_feats = extract_shape_features(preprocessed)
        color_feats = extract_colour_features(preprocessed)
        texture_feats = extract_texture_features(preprocessed)
        vein_feats = extract_vein_features(skeleton, binary_mask)
        
        if shape_feats is None or color_feats is None or texture_feats is None or vein_feats is None:
            print(f"  [-] Feature extraction failed for {filename}. Skipping.")
            continue
            
        # Combine all features into a single dictionary
        combined_feats = {}
        combined_feats.update(shape_feats)
        combined_feats.update(color_feats)
        combined_feats.update(texture_feats)
        combined_feats.update(vein_feats)
        combined_feats["filename"] = filename
        
        data.append(combined_feats)
        labels.append(label)
        
    return data, labels

def train_and_evaluate():
    # 1. Collect dataset
    pos_data, pos_labels = collect_features_from_folder("dataset/healthy", 1) # 1 = Hibiscus
    neg_data, neg_labels = collect_features_from_folder("dataset/Non-hibiscus", 0) # 0 = Not Hibiscus
    
    all_data = pos_data + neg_data
    all_labels = pos_labels + neg_labels
    
    if len(all_data) == 0:
        print("[-] Error: No features extracted. Ensure images are present in the dataset directories.")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    filenames = df["filename"]
    X = df.drop(columns=["filename"])
    y = np.array(all_labels)
    
    print("\n--- Extracted Dataset Summary ---")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples (Hibiscus): {sum(y)}")
    print(f"Negative samples (Non-hibiscus): {len(y) - sum(y)}")
    print(f"Feature count: {X.shape[1]}")
    
    # 2. Evaluate with Stratified K-Fold Cross Validation
    # Since dataset is small, K=3 is appropriate
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_accuracies = []
    cv_fars = [] # False Accept Rates
    cv_frrs = [] # False Reject Rates
    
    print(f"\nRunning {n_splits}-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train classifier (Random Forest with regularization)
        clf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # Predict probabilities
        val_probs = clf.predict_proba(X_val_scaled)[:, 1]
        
        # Use default threshold 0.5 for CV evaluation
        val_preds = (val_probs >= 0.5).astype(int)
        
        acc = accuracy_score(y_val, val_preds)
        cv_accuracies.append(acc)
        
        # Confusion matrix to calculate FAR/FRR
        tn, fp, fn, tp = confusion_matrix(y_val, val_preds, labels=[0, 1]).ravel()
        far = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        
        cv_fars.append(far)
        cv_frrs.append(frr)
        
        print(f"Fold {fold+1}: Accuracy={acc:.4f}, FAR={far:.4f}, FRR={frr:.4f}")
        
    print(f"\n=== CV Performance ===")
    print(f"Mean Accuracy : {np.mean(cv_accuracies):.4f}")
    print(f"Mean FAR (False Accept Rate) : {np.mean(cv_fars):.4f} (Target: Low)")
    print(f"Mean FRR (False Reject Rate) : {np.mean(cv_frrs):.4f}")
    
    # 3. Train final model on entire dataset
    print("\nTraining final model on full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    clf.fit(X_scaled, y)
    
    # Estimate feature importances
    importances = clf.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("\nTop 5 Feature Importances:")
    print(feat_importances.head(5))
    
    # 4. Optimize Decision Threshold on final model predictions
    # Goal: Ensure False Accept Rate (FAR) is minimized
    final_probs = clf.predict_proba(X_scaled)[:, 1]
    
    # We want a threshold that keeps false-accepts to 0 (or minimal), even if it increases false-rejects slightly.
    best_threshold = 0.5
    for thresh in np.linspace(0.4, 0.8, 41):
        preds = (final_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        far = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # If False Accept Rate is 0, we can use this threshold
        if far == 0:
            best_threshold = thresh
            break
            
    print(f"\nOptimized Decision Threshold (for low False Accept Rate): {best_threshold:.2f}")
    
    final_preds = (final_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, final_preds, labels=[0, 1]).ravel()
    final_acc = accuracy_score(y, final_preds)
    print(f"Final Model Metrics on Training Set (with threshold {best_threshold:.2f}):")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  False Accept Rate (FAR): {fp / (tn + fp):.4f} (wrongly passed species: {fp}/{tn+fp})")
    print(f"  False Reject Rate (FRR): {fn / (tp + fn):.4f} (wrongly blocked hibiscus: {fn}/{tp+fn})")
    
    # 5. Save model artifacts
    model_dir = "Identification/Features/models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "hibiscus_classifier.pkl")
    model_data = {
        "classifier": clf,
        "scaler": scaler,
        "features": list(X.columns),
        "threshold": best_threshold,
        "metrics": {
            "accuracy": final_acc,
            "cv_accuracy": float(np.mean(cv_accuracies)),
            "far": fp / (tn + fp) if (tn + fp) > 0 else 0.0,
            "frr": fn / (tp + fn) if (tp + fn) > 0 else 0.0
        }
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print(f"\n[OK] Model successfully trained and saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
