import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Add root workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def evaluate_classifier(features_csv="phase2_classification/data/features.csv", models_dir="phase2_classification/models", reports_dir="phase2_classification/reports"):
    """
    Evaluates the trained model against the dataset and generates reports, metrics JSON,
    and plots (confusion matrix and feature importances/coefficients).
    """
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    if not os.path.exists(features_csv) or not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("[-] Error: Missing features.csv or model artifacts. Run train.py first.")
        return
        
    # 1. Load data and model
    df = pd.read_csv(features_csv)
    y = df["label"].values
    X = df.drop(columns=["filename", "label"])
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    clf = model_data["classifier"]
    feature_names = model_data["features"]
    threshold = model_data["threshold"]
    model_name = model_data["model_name"]
    
    # Scale features
    X_scaled = scaler.transform(X[feature_names])
    
    # Predict using decision threshold
    probs = clf.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    # 2. Compute metrics
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    
    metrics = {
        "model_name": model_name,
        "optimized_threshold": float(threshold),
        "overall_accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "cv_mean_f1": float(model_data["cv_score"])
    }
    
    print("\n==============================================")
    print("PHASE 2 MODEL EVALUATION REPORT")
    print("==============================================")
    print(f"Model Name            : {model_name}")
    print(f"Decision Threshold    : {threshold:.2f}")
    print(f"Accuracy              : {acc:.4f}")
    print(f"Precision             : {prec:.4f}")
    print(f"Recall (Sensitivity)  : {rec:.4f} (Target: >= 90%)")
    print(f"F1-Score              : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"  Predicted Healthy (0)  |  TN: {tn}  |  FN: {fn}")
    print(f"  Predicted Unhealthy (1)|  FP: {fp}  |  TP: {tp}")
    print("==============================================")
    
    # Save metrics to JSON
    os.makedirs(reports_dir, exist_ok=True)
    metrics_path = os.path.join(reports_dir, "phase2_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Saved metrics to {metrics_path}")
    
    # 3. Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = [[tn, fp], [fn, tp]]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix - Phase 2')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Healthy', 'Unhealthy'])
    plt.yticks(tick_marks, ['Healthy', 'Unhealthy'])
    
    # Label cell values
    thresh_val = (tn + fp + fn + tp) / 2.
    for i, j in np.ndindex(2, 2):
        plt.text(j, i, format(cm[i][j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i][j] > thresh_val else "black")
                 
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    cm_plot_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"[OK] Saved confusion matrix plot to {cm_plot_path}")
    
    # 4. Plot Feature Importance or Coefficients
    importance_info = model_data.get("feature_importance", {})
    if importance_info and "message" not in importance_info:
        # Sort importances
        sorted_feats = sorted(importance_info.items(), key=lambda item: abs(item[1]), reverse=True)
        top_feats = sorted_feats[:10]  # Show top 10
        
        plt.figure(figsize=(10, 6))
        names = [item[0] for item in top_feats]
        values = [item[1] for item in top_feats]
        
        # Determine bar colors based on positive/negative influence if coefficients, or green if importances
        if model_name == "Random Forest":
            colors = 'teal'
            plt.title('Top 10 Feature Importances (Random Forest) - Phase 2')
            plt.xlabel('Relative Importance')
        else:
            colors = ['green' if v >= 0 else 'red' for v in values]
            plt.title(f'Top 10 Feature Coefficients ({model_name}) - Phase 2')
            plt.xlabel('Coefficient Value')
            
        plt.barh(names[::-1], values[::-1], color=colors[::-1] if isinstance(colors, list) else colors)
        plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        plt.tight_layout()
        
        feat_plot_path = os.path.join(reports_dir, "feature_importance.png")
        plt.savefig(feat_plot_path, dpi=150)
        plt.close()
        print(f"[OK] Saved feature importance plot to {feat_plot_path}")
        
if __name__ == "__main__":
    evaluate_classifier()
