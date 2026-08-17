import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

# Add root workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def train_and_select_best_model(features_csv="phase2_classification/data/features.csv", models_dir="phase2_classification/models"):
    """
    Loads features.csv, trains SVM, RF, and Logistic Regression classifiers,
    evaluates using cross-validation, and serializes the best performing model.
    """
    if not os.path.exists(features_csv):
        print(f"[-] Features CSV not found: {features_csv}. Please run data_loader.py first.")
        return None
        
    df = pd.read_csv(features_csv)
    if len(df) == 0:
        print("[-] Error: Features CSV is empty.")
        return None
        
    print(f"\nLoaded {len(df)} samples from {features_csv}")
    
    # Separate features and target
    filenames = df["filename"]
    y = df["label"].values
    X = df.drop(columns=["filename", "label"])
    feature_names = list(X.columns)
    
    print(f"Class distribution: Healthy (0) = {sum(y == 0)}, Unhealthy (1) = {sum(y == 1)}")
    print(f"Number of features: {X.shape[1]}")
    
    # If the dataset is too small, we will use Stratified 3-Fold Cross-Validation
    n_splits = min(3, sum(y == 0), sum(y == 1))
    if n_splits < 2:
        print("[-] Error: Not enough samples in each class to perform cross-validation.")
        return None
        
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models and their parameter grids for GridSearchCV
    models = {
        "SVM": {
            "model": SVC(probability=True, class_weight='balanced', random_state=42),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {
                "n_estimators": [20, 50, 100],
                "max_depth": [3, 5, None],
                "min_samples_split": [2, 5]
            }
        },
        "Logistic Regression": {
            "model": LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"]
            }
        }
    }
    
    best_overall_model_name = None
    best_overall_score = -1.0
    best_overall_grid_search = None
    
    results = {}
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print("\n--- Tuning Models and Comparing via Cross-Validation ---")
    for name, config in models.items():
        print(f"\nTuning {name}...")
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=skf,
            scoring="f1",  # Optimize for F1 score since recall/precision balance is important
            n_jobs=-1
        )
        grid.fit(X_scaled, y)
        
        best_params = grid.best_params_
        best_score = grid.best_score_
        
        print(f"  Best params: {best_params}")
        print(f"  Best Mean CV F1-Score: {best_score:.4f}")
        
        results[name] = {
            "best_score": best_score,
            "best_params": best_params,
            "cv_results": grid.cv_results_
        }
        
        # Select best model based on F1-score
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_model_name = name
            best_overall_grid_search = grid
            
    print(f"\n=== Best Model Selected: {best_overall_model_name} (F1 Score: {best_overall_score:.4f}) ===")
    
    # Train the final model with best parameters on the full dataset
    final_clf = best_overall_grid_search.best_estimator_
    final_clf.fit(X_scaled, y)
    
    # Evaluate final model on the training data itself as a sanity check
    final_probs = final_clf.predict_proba(X_scaled)[:, 1]
    final_preds = final_clf.predict(X_scaled)
    
    print("\nFinal Model Metrics on Dataset:")
    print(f"  Accuracy: {accuracy_score(y, final_preds):.4f}")
    print(f"  Precision: {precision_score(y, final_preds):.4f}")
    print(f"  Recall (Sensitivity): {recall_score(y, final_preds):.4f}")
    print(f"  F1-Score: {f1_score(y, final_preds):.4f}")
    
    # Calculate feature importances / coefficients for logging
    importance_info = {}
    if best_overall_model_name == "Random Forest":
        importances = final_clf.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print("\nTop Feature Importances:")
        print(feat_imp.head(10))
        importance_info = feat_imp.to_dict()
    elif best_overall_model_name == "Logistic Regression":
        coefs = final_clf.coef_[0]
        feat_coef = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False)
        print("\nTop Model Coefficients:")
        print(feat_coef.head(10))
        importance_info = feat_coef.to_dict()
    elif best_overall_model_name == "SVM" and final_clf.kernel == "linear":
        coefs = final_clf.coef_[0]
        feat_coef = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False)
        print("\nTop Model Coefficients (Linear SVM):")
        print(feat_coef.head(10))
        importance_info = feat_coef.to_dict()
    else:
        print("\nNote: Feature importance is not directly inspectable for RBF SVM.")
        importance_info = {"message": "RBF Kernel SVM does not expose direct feature importances"}

    # Optimize decision threshold to meet the 90%+ recall requirement on Unhealthy (Class 1)
    # We want to find a threshold that maximizes recall while preserving precision
    best_threshold = 0.5
    high_recall_threshold = 0.5
    max_f1_with_recall_target = -1.0
    
    # Scan possible thresholds
    for thresh in np.linspace(0.1, 0.9, 81):
        preds = (final_probs >= thresh).astype(int)
        rec = recall_score(y, preds, zero_division=0)
        prec = precision_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        
        # We prioritize recall >= 90%
        if rec >= 0.90:
            if f1 > max_f1_with_recall_target:
                max_f1_with_recall_target = f1
                high_recall_threshold = thresh
                
    print(f"\nStandard Decision Threshold: 0.5")
    print(f"Recall at 0.5: {recall_score(y, (final_probs >= 0.5).astype(int)):.4f}")
    print(f"Optimized Decision Threshold for >= 90% Recall: {high_recall_threshold:.2f}")
    print(f"Recall at {high_recall_threshold:.2f}: {recall_score(y, (final_probs >= high_recall_threshold).astype(int)):.4f}")
    
    # Save the model and preprocessing artifacts
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    model_data = {
        "model_name": best_overall_model_name,
        "classifier": final_clf,
        "features": feature_names,
        "threshold": high_recall_threshold,
        "best_params": best_params,
        "cv_score": best_score,
        "feature_importance": importance_info
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"\n[OK] Model successfully trained and saved to {model_path}")
    print(f"[OK] Scaler successfully saved to {scaler_path}")
    
    return model_data

if __name__ == "__main__":
    train_and_select_best_model()
