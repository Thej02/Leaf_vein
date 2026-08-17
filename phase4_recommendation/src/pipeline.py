import os
import sys
import datetime
import pickle
import numpy as np
import pandas as pd

# Add the workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ImagePreprocessing.preprocessing import preprocess_image
from Identification.Features.hibiscus_identifier import HibiscusIdentifier
from phase2_classification.src.feature_extractor import extract_all_features as extract_phase2_features
from phase2_classification.src.predict import load_model_artifacts as load_phase2_model
from phase3_deficiency.src.feature_extractor import extract_deficiency_features
from phase3_deficiency.src.predict import load_deficiency_model as load_phase3_model
from phase4_recommendation.src.severity import compute_severity
from phase4_recommendation.src.recommender import get_recommendation

def run_diagnose_pipeline(image_path, workspace_root="."):
    """
    Runs the full plant health diagnostic pipeline:
    Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 0. Preprocessing (Phase 0)
    res = preprocess_image(image_path, save_outputs=False)
    if res is None or not res.get("success", False):
        return {
            "timestamp": timestamp,
            "image_path": image_path,
            "species": "unknown",
            "health": "unknown",
            "deficiency": "none",
            "severity": "none",
            "recommendation": "Error: Image preprocessing failed (no valid leaf shape found).",
            "confidence": {
                "species_confidence": 0.0,
                "health_confidence": 0.0,
                "deficiency_confidence": 0.0,
                "severity_score": 0.0
            }
        }
        
    # 1. Species Identification (Phase 1)
    try:
        identifier = HibiscusIdentifier()
        is_hibiscus, species_confidence, msg = identifier.identify(res)
    except Exception as e:
        is_hibiscus = False
        species_confidence = 0.0
        msg = f"Error running Phase 1 classifier: {e}"
        
    if not is_hibiscus:
        return {
            "timestamp": timestamp,
            "image_path": image_path,
            "species": "not_hibiscus",
            "health": "unknown",
            "deficiency": "none",
            "severity": "none",
            "recommendation": f"Rejection: {msg}",
            "confidence": {
                "species_confidence": float(species_confidence),
                "health_confidence": 0.0,
                "deficiency_confidence": 0.0,
                "severity_score": 0.0
            }
        }
        
    # 2. Health Classification (Phase 2)
    try:
        p2_model, p2_scaler = load_phase2_model(os.path.join(workspace_root, "phase2_classification/models"))
        clf2 = p2_model["classifier"]
        feat2_names = p2_model["features"]
        p2_threshold = p2_model["threshold"]
        
        # Extract features for Phase 2
        features_p2 = extract_phase2_features(res)
        feat2_df = pd.DataFrame([features_p2])
        X2_scaled = p2_scaler.transform(feat2_df[feat2_names])
        
        unhealthy_prob = float(clf2.predict_proba(X2_scaled)[0, 1])
        is_unhealthy = unhealthy_prob >= p2_threshold
        
        health_label = "Unhealthy" if is_unhealthy else "Healthy"
        health_confidence = unhealthy_prob if is_unhealthy else (1.0 - unhealthy_prob)
    except Exception as e:
        return {
            "timestamp": timestamp,
            "image_path": image_path,
            "species": "hibiscus",
            "health": "unknown",
            "deficiency": "none",
            "severity": "none",
            "recommendation": f"Error running Phase 2 classifier: {e}",
            "confidence": {
                "species_confidence": float(species_confidence),
                "health_confidence": 0.0,
                "deficiency_confidence": 0.0,
                "severity_score": 0.0
            }
        }
        
    # Bypassing downstream phases if Healthy
    if health_label == "Healthy":
        rec = get_recommendation("none", "none")
        return {
            "timestamp": timestamp,
            "image_path": image_path,
            "species": "hibiscus",
            "health": "Healthy",
            "deficiency": "none",
            "severity": "none",
            "recommendation": rec,
            "confidence": {
                "species_confidence": float(species_confidence),
                "health_confidence": float(health_confidence),
                "deficiency_confidence": 0.0,
                "severity_score": 0.0
            }
        }
        
    # 3. Deficiency Classification (Phase 3)
    try:
        p3_model, p3_scaler = load_phase3_model(os.path.join(workspace_root, "phase3_deficiency/models"))
        clf3 = p3_model["classifier"]
        feat3_names = p3_model["features"]
        
        # Extract features for Phase 3
        features_p3 = extract_deficiency_features(res)
        feat3_df = pd.DataFrame([features_p3])
        X3_scaled = p3_scaler.transform(feat3_df[feat3_names])
        
        probs3 = clf3.predict_proba(X3_scaled)[0]
        sorted_indices = np.argsort(probs3)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1]
        
        top_prob = float(probs3[top_idx])
        second_prob = float(probs3[second_idx])
        
        label_names = ["chlorosis", "necrosis", "scorch"]
        deficiency_label = label_names[top_idx]
        
        # Ambiguity Flagging
        is_ambiguous = (top_prob - second_prob) < 0.15
        if is_ambiguous:
            def_message = f"{deficiency_label} (ambiguous with {label_names[second_idx]})"
        else:
            def_message = deficiency_label
    except Exception as e:
        return {
            "timestamp": timestamp,
            "image_path": image_path,
            "species": "hibiscus",
            "health": "Unhealthy",
            "deficiency": "unknown",
            "severity": "none",
            "recommendation": f"Error running Phase 3 classifier: {e}",
            "confidence": {
                "species_confidence": float(species_confidence),
                "health_confidence": float(health_confidence),
                "deficiency_confidence": 0.0,
                "severity_score": 0.0
            }
        }
        
    # 4. Severity Grading & Actionable Recommendation (Phase 4)
    try:
        baseline_path = os.path.join(workspace_root, "phase4_recommendation/models/healthy_baseline.pkl")
        if os.path.exists(baseline_path):
            with open(baseline_path, "rb") as f:
                baseline = pickle.load(f)
        else:
            # Fallback default baseline
            baseline = {
                "vein_density": {"mean": 0.070, "std": 0.010},
                "vein_thickness": {"mean": 4.200, "std": 0.500}
            }
            
        severity_info = compute_severity(features_p3, baseline)
        severity_tier = severity_info["tier"]
        severity_score = severity_info["score"]
        
        # Fetch care recommendation
        rec = get_recommendation(deficiency_label, severity_tier)
        
        if is_ambiguous:
            rec = f"[Ambiguity Warning: leaf shows signs of both {deficiency_label} and {label_names[second_idx]}] " + rec
            
    except Exception as e:
        return {
            "timestamp": timestamp,
            "image_path": image_path,
            "species": "hibiscus",
            "health": "Unhealthy",
            "deficiency": def_message,
            "severity": "unknown",
            "recommendation": f"Error computing Phase 4 severity/recommendations: {e}",
            "confidence": {
                "species_confidence": float(species_confidence),
                "health_confidence": float(health_confidence),
                "deficiency_confidence": float(top_prob),
                "severity_score": 0.0
            }
        }
        
    # Assemble complete final structured diagnostic result
    return {
        "timestamp": timestamp,
        "image_path": image_path,
        "species": "hibiscus",
        "health": "Unhealthy",
        "deficiency": def_message,
        "severity": severity_tier,
        "recommendation": rec,
        "confidence": {
            "species_confidence": float(species_confidence),
            "health_confidence": float(health_confidence),
            "deficiency_confidence": float(top_prob),
            "severity_score": float(severity_score)
        }
    }
