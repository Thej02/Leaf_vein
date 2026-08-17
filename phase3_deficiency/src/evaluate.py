import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Add root workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

LABEL_NAMES = ["chlorosis", "necrosis", "scorch"]

def evaluate_deficiency_classifier(features_csv="phase3_deficiency/data/features.csv", models_dir="phase3_deficiency/models", reports_dir="phase3_deficiency/reports"):
    """
    Evaluates Phase 3 multi-class classifier and generates reports/plots.
    """
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    if not os.path.exists(features_csv) or not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("[-] Error: Missing features.csv or model files. Run train.py first.")
        return
        
    # 1. Load data & models
    df = pd.read_csv(features_csv)
    y = df["label"].values
    X = df.drop(columns=["filename", "label"])
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    clf = model_data["classifier"]
    feature_names = model_data["features"]
    model_name = model_data["model_name"]
    
    X_scaled = scaler.transform(X[feature_names])
    
    # Predict
    probs = clf.predict_proba(X_scaled)
    preds = np.argmax(probs, axis=1)
    
    # 2. Calculate metrics
    acc = accuracy_score(y, preds)
    
    # Compute per-class and macro metrics
    prec_per_class = precision_score(y, preds, average=None, labels=[0, 1, 2], zero_division=0)
    rec_per_class = recall_score(y, preds, average=None, labels=[0, 1, 2], zero_division=0)
    f1_per_class = f1_score(y, preds, average=None, labels=[0, 1, 2], zero_division=0)
    
    macro_f1 = f1_score(y, preds, average="macro")
    macro_prec = precision_score(y, preds, average="macro")
    macro_rec = recall_score(y, preds, average="macro")
    
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    
    metrics = {
        "model_name": model_name,
        "overall_accuracy": float(acc),
        "macro_metrics": {
            "f1_score": float(macro_f1),
            "precision": float(macro_prec),
            "recall": float(macro_rec)
        },
        "per_class_metrics": {},
        "confusion_matrix": cm.tolist()
    }
    
    print("\n==============================================")
    print("PHASE 3 MULTI-CLASS EVALUATION REPORT")
    print("==============================================")
    print(f"Classifier Algorithm : {model_name}")
    print(f"Overall Accuracy     : {acc:.4f}")
    print(f"Macro F1-Score       : {macro_f1:.4f}")
    print("\nPer-class Metrics:")
    for idx, name in enumerate(LABEL_NAMES):
        print(f"  Class '{name}':")
        print(f"    Precision : {prec_per_class[idx]:.4f}")
        print(f"    Recall    : {rec_per_class[idx]:.4f}")
        print(f"    F1-Score  : {f1_per_class[idx]:.4f}")
        
        metrics["per_class_metrics"][name] = {
            "precision": float(prec_per_class[idx]),
            "recall": float(rec_per_class[idx]),
            "f1_score": float(f1_per_class[idx])
        }
    print("==============================================")
    
    # Save JSON report
    os.makedirs(reports_dir, exist_ok=True)
    metrics_path = os.path.join(reports_dir, "phase3_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Saved deficiency metrics report to {metrics_path}")
    
    # 3. Plot Confusion Matrix
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Phase 3 Deficiency')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, LABEL_NAMES)
    plt.yticks(tick_marks, LABEL_NAMES)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i][j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i][j] > thresh else "black")
                 
    plt.ylabel('True Deficiency')
    plt.xlabel('Predicted Deficiency')
    plt.tight_layout()
    cm_plot_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"[OK] Saved confusion matrix plot to {cm_plot_path}")
    
    # 4. Plot Feature Importance/Coefficients
    importance_info = model_data.get("feature_importance", {})
    if importance_info and "message" not in importance_info:
        # Sort and select top 10 features
        sorted_feats = sorted(importance_info.items(), key=lambda item: abs(item[1]), reverse=True)
        top_feats = sorted_feats[:10]
        
        plt.figure(figsize=(10, 6))
        names = [item[0] for item in top_feats]
        values = [item[1] for item in top_feats]
        
        if model_name == "Random Forest":
            plt.barh(names[::-1], values[::-1], color='dodgerblue')
            plt.title('Top 10 Feature Importances (Random Forest) - Phase 3')
            plt.xlabel('Importance Value')
        else:
            colors = ['green' if v >= 0 else 'red' for v in values]
            plt.barh(names[::-1], values[::-1], color=colors[::-1])
            plt.title(f'Top 10 Coefficient Weights ({model_name}) - Phase 3')
            plt.xlabel('Coefficient weight')
            plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
            
        plt.tight_layout()
        feat_plot_path = os.path.join(reports_dir, "feature_importance.png")
        plt.savefig(feat_plot_path, dpi=150)
        plt.close()
        print(f"[OK] Saved feature importance plot to {feat_plot_path}")

if __name__ == "__main__":
    evaluate_deficiency_classifier()
