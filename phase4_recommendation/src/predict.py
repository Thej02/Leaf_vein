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
        print("          PLANT DIAGNOSTIC REPORT             ")
        print("==============================================")
        print(f"Timestamp      : {result['timestamp']}")
        print(f"Leaf Species   : {result['species'].upper()}")
        print(f"Health Status  : {result['health'].upper()}")
        
        if result['health'] == "Unhealthy":
            print(f"Deficiency     : {result['deficiency'].upper()}")
            print(f"Severity Level : {result['severity'].upper()}")
            
        print("----------------------------------------------")
        print("RECOMMENDATION:")
        # Wrap recommendation text for clean printing
        import textwrap
        wrapped_rec = textwrap.fill(result['recommendation'], width=46)
        print(wrapped_rec)
        print("==============================================")
        
        print("\nStructured JSON Output:")
        print(json.dumps(result, indent=4))
    else:
        print(f"[-] Image not found at {img_path}")
