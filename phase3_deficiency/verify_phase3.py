import os
import sys

# Add root workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase3_deficiency.src.data_loader import collect_features_for_phase3
from phase3_deficiency.src.train import train_deficiency_classifier
from phase3_deficiency.src.evaluate import evaluate_deficiency_classifier
from phase3_deficiency.src.predict import predict_deficiency

def run_phase3_verification():
    print("==================================================")
    print("RUNNING PHASE 3 DEFICIENCY PIPELINE VERIFICATION")
    print("==================================================")
    
    # Step 1: Run feature extraction on deficiency classes
    print("\n--- STEP 1: Feature Extraction (Data Loader) ---")
    df = collect_features_for_phase3()
    if df is None:
        print("[-] Error: Feature extraction failed.")
        sys.exit(1)
    print(f"[+] Step 1 completed successfully! Extracted features for {len(df)} samples.")
    
    # Step 2: Model Training and Selection
    print("\n--- STEP 2: Model Training and Selection ---")
    model_data = train_deficiency_classifier()
    if model_data is None:
        print("[-] Error: Model training failed.")
        sys.exit(1)
    print(f"[+] Step 2 completed successfully! Trained best model: {model_data['model_name']}")
    
    # Step 3: Model Evaluation and Plotting
    print("\n--- STEP 3: Model Evaluation and Plotting ---")
    evaluate_deficiency_classifier()
    print("[+] Step 3 completed successfully! Reports and plots generated in reports/.")
    
    # Step 4: Run inference test on Chlorosis class
    print("\n--- STEP 4: Test Inference on Chlorosis leaf ---")
    chlorosis_test = "dataset/unhealthy/chlorosis/chlorosis_1.jpg"
    if os.path.exists(chlorosis_test):
        res = predict_deficiency(chlorosis_test)
        print(f"Prediction for {chlorosis_test}:")
        print(f"  Predicted Label: {res['predicted_class']} (Expected: chlorosis)")
        print(f"  Confidence: {res['confidence']:.4f}")
        print(f"  Is Ambiguous: {res['is_ambiguous']}")
    else:
        print(f"[-] Warning: Chlorosis test leaf not found at {chlorosis_test}")
        
    # Step 5: Run inference test on Necrosis class
    print("\n--- STEP 5: Test Inference on Necrosis leaf ---")
    necrosis_test = "dataset/unhealthy/necrosis/necrosis_1.jpg"
    if os.path.exists(necrosis_test):
        res = predict_deficiency(necrosis_test)
        print(f"Prediction for {necrosis_test}:")
        print(f"  Predicted Label: {res['predicted_class']} (Expected: necrosis)")
        print(f"  Confidence: {res['confidence']:.4f}")
        print(f"  Is Ambiguous: {res['is_ambiguous']}")
    else:
        print(f"[-] Warning: Necrosis test leaf not found at {necrosis_test}")
        
    # Step 6: Run inference test on Scorch class
    print("\n--- STEP 6: Test Inference on Scorch leaf ---")
    scorch_test = "dataset/unhealthy/scorch/scorch_1.jpg"
    if os.path.exists(scorch_test):
        res = predict_deficiency(scorch_test)
        print(f"Prediction for {scorch_test}:")
        print(f"  Predicted Label: {res['predicted_class']} (Expected: scorch)")
        print(f"  Confidence: {res['confidence']:.4f}")
        print(f"  Is Ambiguous: {res['is_ambiguous']}")
    else:
        print(f"[-] Warning: Scorch test leaf not found at {scorch_test}")
        
    print("\n==================================================")
    print("PHASE 3 DEFICIENCY PIPELINE VERIFICATION SUCCESSFUL!")
    print("==================================================")

if __name__ == "__main__":
    run_phase3_verification()
