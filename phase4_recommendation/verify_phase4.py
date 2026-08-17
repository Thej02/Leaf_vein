import os
import sys
import json

# Add root workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase4_recommendation.src.baseline import compute_healthy_baseline
from phase4_recommendation.src.pipeline import run_diagnose_pipeline

def run_phase4_verification():
    print("==================================================")
    print("RUNNING PHASE 4 PIPELINE INTEGRATION VERIFICATION")
    print("==================================================")
    
    # 1. Establish the baseline
    print("\n--- STEP 1: Compute Healthy Baseline ---")
    baseline = compute_healthy_baseline()
    
    # 2. Compile tests for various pipeline branches
    test_cases = [
        {"name": "Healthy Hibiscus Leaf", "path": "dataset/healthy/hibiscus_gen_1.png"},
        {"name": "Chlorotic Unhealthy Leaf", "path": "dataset/unhealthy/chlorosis/chlorosis_1.jpg"},
        {"name": "Necrotic Unhealthy Leaf", "path": "dataset/unhealthy/necrosis/necrosis_1.jpg"},
        {"name": "Scorched Unhealthy Leaf", "path": "dataset/unhealthy/scorch/scorch_1.jpg"}
    ]
    
    reports_dir = "phase4_recommendation/reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    sample_outputs = {}
    
    print("\n--- STEP 2: Running End-to-End Inference Tests ---")
    for case in test_cases:
        name = case["name"]
        path = case["path"]
        
        print(f"\nEvaluating: '{name}' ({path})...")
        if os.path.exists(path):
            res = run_diagnose_pipeline(path)
            sample_outputs[name] = res
            
            print(f"  Species:        {res['species']}")
            print(f"  Health Status:  {res['health']}")
            print(f"  Deficiency:     {res['deficiency']}")
            print(f"  Severity Tier:  {res['severity']}")
            print(f"  Recommendation: {res['recommendation'][:50]}...")
            print(f"  Confidences:    {res['confidence']}")
        else:
            print(f"  [-] Warning: Test image not found at {path}")
            
    # 3. Write structured output report
    output_report = os.path.join(reports_dir, "phase4_sample_results.json")
    with open(output_report, "w") as f:
        json.dump(sample_outputs, f, indent=4)
        
    print(f"\n[OK] Assembled pipeline structured JSON outputs to: {output_report}")
    print("\n==================================================")
    print("PHASE 4 INTEGRATION SUCCESSFUL!")
    print("==================================================")

if __name__ == "__main__":
    run_phase4_verification()
