import os
import sys

# Add root path to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Identification.identify import identify_species

def run_verification_tests():
    print("==================================================")
    print("RUNNING AUTOMATED PHASE 1 IDENTIFICATION VERIFICATION")
    print("==================================================")
    
    # 1. Test positive sample (Hibiscus leaf)
    pos_image = "dataset/test/hibiscus1.jpeg"
    is_hib, conf, msg = identify_species(pos_image)
    
    print(f"\nPositive Test Result for {pos_image}:")
    print(f"  Passed Species Gate: {is_hib} (Expected: True)")
    print(f"  Confidence Score: {conf:.4f}")
    print(f"  Message: {msg}")
    
    assert is_hib == True, f"Verification failed: {pos_image} was wrongly rejected!"
    print("[+] Positive test passed successfully!")
    
    # 2. Test negative sample (Maple leaf)
    neg_leaf = "dataset/Non-hibiscus/other_leaf_maple.png"
    is_hib, conf, msg = identify_species(neg_leaf)
    
    print(f"\nNegative Test Result for {neg_leaf}:")
    print(f"  Passed Species Gate: {is_hib} (Expected: False)")
    print(f"  Confidence Score: {conf:.4f}")
    print(f"  Message: {msg}")
    
    assert is_hib == False, f"Verification failed: {neg_leaf} was wrongly accepted!"
    print("[+] Negative leaf test passed successfully!")
    
    # 3. Test another negative sample (a clutter image with some contours)
    # Let's find one of the downloaded clutter images in dataset/Non-hibiscus/
    neg_folder = "dataset/Non-hibiscus"
    clutter_files = [f for f in os.listdir(neg_folder) if f.startswith("non_hibiscus_clutter_") and f.endswith(".jpg")]
    
    if len(clutter_files) > 0:
        neg_clutter = os.path.join(neg_folder, clutter_files[0])
        is_hib, conf, msg = identify_species(neg_clutter)
        
        print(f"\nNegative Test Result for {neg_clutter}:")
        print(f"  Passed Species Gate: {is_hib} (Expected: False)")
        print(f"  Confidence Score: {conf:.4f}")
        print(f"  Message: {msg}")
        
        assert is_hib == False, f"Verification failed: {neg_clutter} was wrongly accepted!"
        print("[+] Negative clutter test passed successfully!")
    else:
        print("\n[!] No negative clutter images found for testing. Skipping.")

    print("\n==================================================")
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    try:
        run_verification_tests()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[!] ASSERTION ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] UNEXPECTED ERROR: {e}")
        sys.exit(1)
