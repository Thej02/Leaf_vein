import os
import sys
import requests

CLASSES = {
    "chlorosis": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "necrosis": "Apple___Apple_scab",
    "scorch": "Strawberry___Leaf_scorch"
}

def download_deficiency_dataset(target_dir="dataset/unhealthy", count_per_class=6):
    headers = {
        "User-Agent": "RosaSinensisHealthClassifier/1.0 (contact: canara_project@example.com; student academic project)"
    }
    
    for class_name, github_folder in CLASSES.items():
        class_folder = os.path.join(target_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        print(f"\nSourcing raw files for Class '{class_name}' ({github_folder})...")
        api_url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{github_folder}"
        
        try:
            r = requests.get(api_url, headers=headers, timeout=15)
            if r.status_code != 200:
                print(f"[ERROR] Failed to fetch folder contents from GitHub API for {class_name} (Status code: {r.status_code})")
                continue
                
            files = r.json()
            # Filter for JPG/PNG files
            img_files = [f for f in files if f["name"].lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(img_files) == 0:
                print(f"[-] No image files found in {github_folder}.")
                continue
                
            print(f"Found {len(img_files)} images. Downloading the first {count_per_class}...")
            
            downloaded = 0
            for idx in range(min(count_per_class, len(img_files))):
                file_info = img_files[idx]
                download_url = file_info["download_url"]
                filename = f"{class_name}_{idx + 1}.jpg"
                filepath = os.path.join(class_folder, filename)
                
                if os.path.exists(filepath):
                    print(f"  File {filename} already exists. Skipping.")
                    downloaded += 1
                    continue
                    
                print(f"  Downloading {download_url} to {filepath}...")
                img_r = requests.get(download_url, headers=headers, timeout=15)
                if img_r.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(img_r.content)
                    print(f"  [OK] Saved {filename}")
                    downloaded += 1
                else:
                    print(f"  [ERROR] Failed to download {file_info['name']} (Status: {img_r.status_code})")
                    
            print(f"[OK] Downloaded {downloaded} images for class '{class_name}'")
            
        except Exception as e:
            print(f"[ERROR] Error loading deficiency class {class_name}: {e}")

if __name__ == "__main__":
    print("--- Sourcing Phase 3 Deficiency Dataset ---")
    download_deficiency_dataset()
