import os
import pickle
import numpy as np

# Import feature extractors
from .shape_analysis import extract_shape_features
from .colour_analysis import extract_colour_features
from .texture_analysis import extract_texture_features
from .vein_analysis import extract_vein_features

class HibiscusIdentifier:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default path to model relative to this file
            model_path = os.path.join(os.path.dirname(__file__), "models", "hibiscus_classifier.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_identifier.py first.")
            
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            
        self.clf = model_data["classifier"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["features"]
        self.threshold = model_data["threshold"]

    def identify(self, preprocessed_res):
        """
        Predicts whether the preprocessed image contains a Hibiscus (Rosa-sinensis) leaf.
        
        Parameters:
            preprocessed_res (dict): Preprocessed image dictionary returned by preprocess_image.
            
        Returns:
            is_hibiscus (bool): Pass/Reject flag indicating whether the leaf is verified.
            confidence (float): Probability score of the leaf being a Hibiscus leaf (0.0 to 1.0).
            message (str): Explanatory accept/reject message.
        """
        if preprocessed_res is None or not preprocessed_res.get("success", False):
            return False, 0.0, "Rejection: No valid leaf structure could be detected or preprocessed."
            
        preprocessed = preprocessed_res["preprocessed"]
        skeleton = preprocessed_res["skeleton"]
        binary_mask = preprocessed_res["binary_mask"]
        
        # Extract features
        shape_feats = extract_shape_features(preprocessed)
        color_feats = extract_colour_features(preprocessed)
        texture_feats = extract_texture_features(preprocessed)
        vein_feats = extract_vein_features(skeleton, binary_mask)
        
        if shape_feats is None or color_feats is None or texture_feats is None or vein_feats is None:
            return False, 0.0, "Rejection: Failed to extract features from the leaf contour."
            
        # Combine all features
        combined_feats = {}
        combined_feats.update(shape_feats)
        combined_feats.update(color_feats)
        combined_feats.update(texture_feats)
        combined_feats.update(vein_feats)
        
        # Construct feature vector in correct order
        try:
            row = [combined_feats[feat] for feat in self.feature_names]
        except KeyError as e:
            return False, 0.0, f"Rejection: Feature extraction mismatch. Missing feature: {e}"
            
        import pandas as pd
        X = pd.DataFrame([row], columns=self.feature_names)
        X_scaled = self.scaler.transform(X)
        
        # Predict probability
        prob = self.clf.predict_proba(X_scaled)[0, 1]
        
        # Decision Rule: Compare against optimized threshold
        is_hibiscus = prob >= self.threshold
        
        if is_hibiscus:
            message = f"Accepted: Verified as a Hibiscus leaf with confidence {prob:.2f}."
        else:
            message = f"Rejected: Not verified as a Hibiscus leaf (confidence {prob:.2f} is below threshold {self.threshold:.2f})."
            
        return is_hibiscus, float(prob), message
