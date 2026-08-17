import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Add root workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def train_deficiency_classifier(features_csv="phase3_deficiency/data/features.csv", models_dir="phase3_deficiency/models"):
    """
    Loads features.csv, trains multi-class SVM and Random Forest,
    compares performance, and saves the best model and scaler.
    """
    if not os.path.exists(features_csv):
        print(f"[-] Features CSV not found: {features_csv}. Run data_loader.py first.")
        return None
        
    df = pd.read_csv(features_csv)
    if len(df) == 0:
        print("[-] Error: Features CSV is empty.")
        return None
        
    print(f"\nLoaded {len(df)} samples from {features_csv}")
    
    y = df["label"].values
    X = df.drop(columns=["filename", "label"])
    feature_names = list(X.columns)
    
    # Class distribution
    classes = [0, 1, 2] # 0: chlorosis, 1: necrosis, 2: scorch
    print("Class distribution:")
    print(f"  Chlorosis (0) = {sum(y == 0)}")
    print(f"  Necrosis (1)  = {sum(y == 1)}")
    print(f"  Scorch (2)    = {sum(y == 2)}")
    print(f"Number of features: {X.shape[1]}")
    
    n_splits = min(3, min([sum(y == c) for c in classes]))
    if n_splits < 2:
        print("[-] Error: Not enough samples per class to run Stratified Cross-Validation.")
        return None
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define hyperparameter grid search for multi-class classification
    models = {
        "Random Forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {
                "n_estimators": [20, 50, 100],
                "max_depth": [3, 5, None],
                "min_samples_split": [2, 5]
            }
        },
        "SVM": {
            "model": SVC(probability=True, decision_function_shape='ovr', class_weight='balanced', random_state=42),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }
        }
    }
    
    best_overall_model_name = None
    best_overall_score = -1.0
    best_overall_grid = None
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print("\n--- Tuning Multi-Class Classifiers via Cross-Validation ---")
    for name, config in models.items():
        print(f"\nTuning {name}...")
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=skf,
            scoring="f1_macro",  # Optimize for macro F1-score across all 3 classes
            n_jobs=-1
        )
        grid.fit(X_scaled, y)
        
        best_params = grid.best_params_
        best_score = grid.best_score_
        
        print(f"  Best params: {best_params}")
        print(f"  Best Mean CV F1-Macro: {best_score:.4f}")
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_model_name = name
            best_overall_grid = grid
            
    print(f"\n=== Best Deficiency Model Selected: {best_overall_model_name} (F1-Macro: {best_overall_score:.4f}) ===")
    
    # Fit final model
    final_clf = best_overall_grid.best_estimator_
    final_clf.fit(X_scaled, y)
    
    # Evaluate final model on training set (sanity check)
    final_preds = final_clf.predict(X_scaled)
    print("\nFinal Model Accuracy on Dataset:")
    print(f"  Accuracy: {accuracy_score(y, final_preds):.4f}")
    
    # Extract feature importances
    importance_info = {}
    if best_overall_model_name == "Random Forest":
        importances = final_clf.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print("\nTop 5 Multi-class Feature Importances:")
        print(feat_imp.head(5))
        importance_info = feat_imp.to_dict()
    elif best_overall_model_name == "SVM" and final_clf.kernel == "linear":
        # SVM multiclass coefs are of shape (n_classes * (n_classes - 1) / 2, n_features)
        # Average absolute weights over all binary classifiers
        coefs = np.mean(np.abs(final_clf.coef_), axis=0)
        feat_coef = pd.Series(coefs, index=feature_names).sort_values(ascending=False)
        print("\nTop 5 SVM Coefficient Weights:")
        print(feat_coef.head(5))
        importance_info = feat_coef.to_dict()
    else:
        print("\nNote: Feature importance is not inspectable for non-linear SVM.")
        importance_info = {"message": "Non-linear kernel SVM does not expose direct feature importances."}
        
    # Save the deficiency model artifacts
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    model_data = {
        "model_name": best_overall_model_name,
        "classifier": final_clf,
        "features": feature_names,
        "best_params": best_params,
        "cv_score": best_score,
        "feature_importance": importance_info
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"\n[OK] Deficiency model successfully saved to {model_path}")
    print(f"[OK] Scaler successfully saved to {scaler_path}")
    
    return model_data

if __name__ == "__main__":
    train_deficiency_classifier()
