import os
import sys

# Add root workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase2_classification.src.data_loader import collect_features_for_phase2
from phase2_classification.src.train import train_and_select_best_model
from phase2_classification.src.evaluate import evaluate_classifier
from phase2_classification.src.predict import predict_health

def run_pipeline_verification():
    print("==================================================")
    print("RUNNING PHASE 2 PIPELINE VERIFICATION")
    print("==================================================")
    
    # Step 1: Run feature extraction on datasets
    print("\n--- STEP 1: Feature Extraction (Data Loader) ---")
    df = collect_features_for_phase2()
    if df is None:
        print("[-] Error: Feature extraction failed.")
        sys.exit(1)
    print(f"[+] Step 1 completed successfully! Extracted features for {len(df)} samples.")
    
    # Step 2: Train and select best classifier
    print("\n--- STEP 2: Model Training and Selection ---")
    model_data = train_and_select_best_model()
    if model_data is None:
        print("[-] Error: Model training failed.")
        sys.exit(1)
    print(f"[+] Step 2 completed successfully! Trained best model: {model_data['model_name']}")
    
    # Step 3: Run evaluation and save reports/plots
    print("\n--- STEP 3: Model Evaluation and Plotting ---")
    evaluate_classifier()
    print("[+] Step 3 completed successfully! Reports and plots generated in reports/.")
    
    # Step 4: Run inference test on a healthy sample
    print("\n--- STEP 4: Test Inference on Healthy Leaf ---")
    healthy_test_img = "dataset/healthy/leaf1.jpeg"
    if os.path.exists(healthy_test_img):
        res = predict_health(healthy_test_img)
        print(f"Prediction for {healthy_test_img}:")
        print(f"  Label: {res['label']} (Expected: Healthy)")
        print(f"  Confidence: {res['confidence']:.4f}")
        print(f"  Unhealthy Probability: {res['unhealthy_probability']:.4f}")
    else:
        print(f"[-] Warning: Healthy test image not found at {healthy_test_img}")
        
    # Step 5: Run inference test on an unhealthy sample
    print("\n--- STEP 5: Test Inference on Unhealthy Leaf ---")
    unhealthy_test_img = "dataset/unhealthy/unhealthy_leaf_1.jpg"
    if os.path.exists(unhealthy_test_img):
        res = predict_health(unhealthy_test_img)
        print(f"Prediction for {unhealthy_test_img}:")
        print(f"  Label: {res['label']} (Expected: Unhealthy)")
        print(f"  Confidence: {res['confidence']:.4f}")
        print(f"  Unhealthy Probability: {res['unhealthy_probability']:.4f}")
    else:
        print(f"[-] Warning: Unhealthy test image not found at {unhealthy_test_img}")
        
    print("\n==================================================")
    print("PHASE 2 PIPELINE VERIFICATION SUCCESSFUL!")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline_verification()
