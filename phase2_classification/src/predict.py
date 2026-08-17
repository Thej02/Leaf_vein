import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add root workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ImagePreprocessing.preprocessing import preprocess_image
from phase2_classification.src.feature_extractor import extract_all_features

# Cache the loaded model and scaler to speed up multiple inferences
_MODEL_CACHE = {}

def load_model_artifacts(models_dir="phase2_classification/models"):
    """
    Loads and caches model and scaler files.
    """
    global _MODEL_CACHE
    if "model" in _MODEL_CACHE and "scaler" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["scaler"]
        
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Trained classifier or scaler not found in models/ directory. Run train.py first.")
        
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    _MODEL_CACHE["model"] = model_data
    _MODEL_CACHE["scaler"] = scaler
    
    return model_data, scaler

def predict_health(image_path, models_dir="phase2_classification/models"):
    """
    Exposes a single predict_health(image_path) -> {"label": ..., "confidence": ...} function
    for reuse in later pipeline phases or in the main application.
    """
    try:
        # 1. Load model artifacts
        model_data, scaler = load_model_artifacts(models_dir)
        clf = model_data["classifier"]
        feature_names = model_data["features"]
        threshold = model_data["threshold"]
        
        # 2. Run Preprocessing (Phase 0)
        res = preprocess_image(image_path, save_outputs=False)
        if res is None or not res.get("success", False):
            return {
                "success": False,
                "error": "Preprocessing failed: no valid leaf shape detected."
            }
            
        # 3. Extract Features (Phase 1 & 2 enhanced)
        features = extract_all_features(res)
        if features is None:
            return {
                "success": False,
                "error": "Feature extraction failed."
            }
            
        # 4. Standardize / Scale Features
        # Convert dictionary to DataFrame with correct columns order
        feat_df = pd.DataFrame([features])
        X_scaled = scaler.transform(feat_df[feature_names])
        
        # 5. Predict using classifier and decision threshold
        # Get probability of being Unhealthy (Class 1)
        unhealthy_prob = float(clf.predict_proba(X_scaled)[0, 1])
        
        is_unhealthy = unhealthy_prob >= threshold
        label = "Unhealthy" if is_unhealthy else "Healthy"
        
        # Confidence score (relative to the decision boundary or class probability)
        confidence = unhealthy_prob if is_unhealthy else (1.0 - unhealthy_prob)
        
        return {
            "success": True,
            "label": label,
            "confidence": confidence,
            "unhealthy_probability": unhealthy_prob,
            "decision_threshold": threshold,
            "features_extracted": features
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Inference execution failed: {str(e)}"
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Fallback to test image
        img_path = "dataset/test/hibiscus1.jpeg"
        print(f"No image provided. Using default test image: {img_path}")
        
    if os.path.exists(img_path):
        verdict = predict_health(img_path)
        if verdict.get("success", False):
            print(f"\n==============================================")
            print(f"VERDICT: The leaf is {verdict['label']}")
            print(f"Confidence: {verdict['confidence']:.2%}")
            print(f"==============================================")
        else:
            print(f"\n[-] Prediction failed: {verdict.get('error')}")
            
        print("\nPrediction Verdict:")
        import json
        print(json.dumps(verdict, indent=4))
    else:
        print(f"Image not found at {img_path}")
