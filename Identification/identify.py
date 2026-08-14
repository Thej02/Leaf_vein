import os
import sys
import datetime
import pandas as pd

# Add the workspace root to sys.path so we can import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ImagePreprocessing.preprocessing import preprocess_image
from Identification.Features.hibiscus_identifier import HibiscusIdentifier

def identify_species(image_path, log_file="output/identification_log.csv"):
    """
    Main pipeline gate function. Checks if the image is a Hibiscus leaf.
    Saves a log entry to log_file.
    
    Returns:
        is_hibiscus (bool): True if verified as Hibiscus, False otherwise.
        confidence (float): Probability score.
        message (str): Explanatory message.
    """
    print(f"\n[Phase 1] Evaluating species identity of: {image_path}")
    
    # 1. Run Preprocessing (Phase 0)
    res = preprocess_image(image_path, save_outputs=False)
    
    # 2. Check if preprocessing succeeded
    if res is None or not res.get("success", False):
        is_hibiscus = False
        confidence = 0.0
        message = "Rejection: Image preprocessing failed (no valid leaf shape found)."
    else:
        # 3. Predict using the Hibiscus Classifier
        try:
            identifier = HibiscusIdentifier()
            is_hibiscus, confidence, message = identifier.identify(res)
        except Exception as e:
            is_hibiscus = False
            confidence = 0.0
            message = f"Error: Failed running classifier ({e})"
            
    # 4. Print Verdict
    if is_hibiscus:
        print(f"VERDICT: ACCEPTED (Confidence: {confidence:.2f})")
        print(f"Message: {message}")
        print("[Phase 1 Passed] Proceeding to Phase 2 (Health Classification)...")
    else:
        print(f"VERDICT: REJECTED (Confidence: {confidence:.2f})")
        print(f"Message: {message}")
        print("[Phase 1 Blocked] Pipeline halted for this image.")
        
    # 5. Log Result
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path,
        "is_hibiscus": is_hibiscus,
        "confidence": confidence,
        "message": message
    }
    
    df_new = pd.DataFrame([log_entry])
    if os.path.exists(log_file):
        try:
            df_old = pd.read_csv(log_file)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.to_csv(log_file, index=False)
        except Exception as e:
            print(f"[-] Warning: Failed to append to log file ({e})")
            df_new.to_csv(log_file, index=False)
    else:
        df_new.to_csv(log_file, index=False)
        
    print(f"Logged result to: {log_file}")
    return is_hibiscus, confidence, message

if __name__ == "__main__":
    # Allow running as a command-line script
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Fallback to test image
        img_path = "dataset/test/hibiscus1.jpeg"
        print(f"No image path provided. Running on default test image: {img_path}")
        
    is_hib, conf, msg = identify_species(img_path)
    # Return exit code 0 if accepted, 1 if rejected or error
    sys.exit(0 if is_hib else 1)