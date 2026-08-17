import os
import sys
import json

# Add root workspace to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase4_recommendation.src.pipeline import run_diagnose_pipeline

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default test image
        img_path = "dataset/unhealthy/chlorosis/chlorosis_1.jpg"
        print(f"No image provided. Using default test image: {img_path}")
        
    if os.path.exists(img_path):
        print(f"\n[System] Executing end-to-end plant health diagnostic pipeline on: {img_path}")
        result = run_diagnose_pipeline(img_path)
        
        print("\n==============================================")
        print("    STEP-BY-STEP PIPELINE DIAGNOSTIC TRACE    ")
        print("==============================================")
        
        # Step 0: Preprocessing
        print("--- PHASE 0: Image Preprocessing ---")
        if result['species'] == "unknown" and result['health'] == "unknown":
            print("[-] Preprocessing Failed: No valid leaf shape detected in the image.")
            print("==============================================")
            sys.exit(1)
        else:
            print("[OK] Leaf successfully segmented and features extracted.")
            
        # Step 1: Species Identification
        print("\n--- PHASE 1: Species Identification ---")
        if result['species'] == "not_hibiscus":
            print(f"[-] REJECTED: Not verified as a Hibiscus leaf.")
            print(f"    Confidence: {result['confidence']['species_confidence']:.2%}")
            print(f"    Message: {result['recommendation']}")
            print("==============================================")
            sys.exit(1)
        else:
            print(f"[OK] VERIFIED: Image identified as a Hibiscus leaf.")
            print(f"    Confidence: {result['confidence']['species_confidence']:.2%}")
            
        # Step 2: Health Classification
        print("\n--- PHASE 2: Health Classification ---")
        print(f"[OK] Health Status: {result['health'].upper()}")
        print(f"    Confidence: {result['confidence']['health_confidence']:.2%}")
        
        if result['health'] == "Healthy":
            print("\n--- PIPELINE HALT: Healthy leaf needs no further diagnostics. ---")
            print("----------------------------------------------")
            print("RECOMMENDATION:")
            print(result['recommendation'])
            print("==============================================")
        else:
            # Step 3: Deficiency Determination
            print("\n--- PHASE 3: Deficiency Determination ---")
            print(f"[OK] Deficiency Class: {result['deficiency'].upper()}")
            print(f"    Deficiency Confidence: {result['confidence']['deficiency_confidence']:.2%}")
            
            # Step 4: Severity & Recommendation
            print("\n--- PHASE 4: Severity Grading & Actionable Recommendation ---")
            print(f"[OK] Severity Tier: {result['severity'].upper()}")
            print(f"    Severity Index: {result['confidence']['severity_score']:.4f}")
            print("----------------------------------------------")
            print("ACTIONABLE RECOMMENDATION:")
            import textwrap
            wrapped_rec = textwrap.fill(result['recommendation'], width=46)
            print(wrapped_rec)
            print("==============================================")
        
        print("\nStructured JSON Output:")
        print(json.dumps(result, indent=4))
    else:
        print(f"[-] Image not found at {img_path}")
