import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add root workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ImagePreprocessing.preprocessing import preprocess_image
from phase3_deficiency.src.feature_extractor import extract_deficiency_features

LABEL_NAMES = ["chlorosis", "necrosis", "scorch"]
_MODEL_CACHE = {}

def load_deficiency_model(models_dir="phase3_deficiency/models"):
    """
    Loads and caches deficiency classifier and scaler.
    """
    global _MODEL_CACHE
    if "model" in _MODEL_CACHE and "scaler" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["scaler"]
        
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Deficiency classifier or scaler not found. Run train.py first.")
        
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    _MODEL_CACHE["model"] = model_data
    _MODEL_CACHE["scaler"] = scaler
    
    return model_data, scaler

def predict_deficiency(image_path, models_dir="phase3_deficiency/models"):
    """
    Exposes deficiency detection. Runs preprocessing, feature extraction,
    scaling, and prediction. Flags ambiguity if top-2 confidence values are close.
    """
    try:
        # 1. Load model artifacts
        model_data, scaler = load_deficiency_model(models_dir)
        clf = model_data["classifier"]
        feature_names = model_data["features"]
        
        # 2. Run Preprocessing
        res = preprocess_image(image_path, save_outputs=False)
        if res is None or not res.get("success", False):
            return {
                "success": False,
                "error": "Preprocessing failed: no valid leaf shape detected."
            }
            
        # 3. Extract extended features
        features = extract_deficiency_features(res)
        if features is None:
            return {
                "success": False,
                "error": "Feature extraction failed."
            }
            
        # 4. Scale features
        feat_df = pd.DataFrame([features])
        X_scaled = scaler.transform(feat_df[feature_names])
        
        # 5. Predict probabilities
        probs = clf.predict_proba(X_scaled)[0]
        
        # 6. Sort class indices by probability descending
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1]
        
        top_prob = float(probs[top_idx])
        second_prob = float(probs[second_idx])
        
        # 7. Ambiguity Handling: check if margin between top 2 classes is < 0.15
        is_ambiguous = (top_prob - second_prob) < 0.15
        
        all_confidences = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}
        
        return {
            "success": True,
            "predicted_class": LABEL_NAMES[top_idx],
            "confidence": top_prob,
            "is_ambiguous": is_ambiguous,
            "alternative_class": LABEL_NAMES[second_idx] if is_ambiguous else None,
            "all_confidences": all_confidences,
            "features_extracted": features
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Deficiency prediction failed: {str(e)}"
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default test image (necrosis / Apple scab)
        img_path = "dataset/unhealthy/necrosis/necrosis_1.jpg"
        print(f"No image provided. Using default test image: {img_path}")
        
    if os.path.exists(img_path):
        verdict = predict_deficiency(img_path)
        if verdict.get("success", False):
            print(f"\n==============================================")
            if verdict["is_ambiguous"]:
                print(f"VERDICT: Ambiguous Symptoms Detected!")
                print(f"Primary Guess   : {verdict['predicted_class']} ({verdict['confidence']:.2%})")
                print(f"Secondary Guess : {verdict['alternative_class']} ({verdict['all_confidences'][verdict['alternative_class']]:.2%})")
            else:
                print(f"VERDICT: Leaf deficiency identified as '{verdict['predicted_class']}'")
                print(f"Confidence      : {verdict['confidence']:.2%}")
            print(f"==============================================")
        else:
            print(f"\n[-] Prediction failed: {verdict.get('error')}")
            
        print("\nPrediction Details (JSON):")
        import json
        print_verdict = verdict.copy()
        if "features_extracted" in print_verdict:
            del print_verdict["features_extracted"]
        print(json.dumps(print_verdict, indent=4))
    else:
        print(f"Image not found at {img_path}")
