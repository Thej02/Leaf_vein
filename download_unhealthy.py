import os
import requests

UNHEALTHY_URLS = [
    "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab%203335.JPG",
    "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___Apple_scab/01a66316-0e98-4d3b-a56f-d78752cd043f___FREC_Scab%203003.JPG",
    "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___Apple_scab/01f3deaa-6143-4b6c-9c22-620a46d8be04___FREC_Scab%203112.JPG",
    "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___Apple_scab/0208f4eb-45a4-4399-904e-989ac2c6257c___FREC_Scab%203037.JPG",
    "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___Apple_scab/023123cb-7b69-4c9f-a521-766d7c8543bb___FREC_Scab%203487.JPG",
    "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/Apple___Apple_scab/0261a6e4-21f8-481a-8827-b674e6955644___FREC_Scab%203055.JPG"
]

def download_images(urls, folder, prefix):
    if not os.path.exists(folder):
        os.makedirs(folder)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    count = 0
    for idx, url in enumerate(urls):
        ext = "jpg"  # standard extension
        filename = f"{prefix}_{idx + 1}.{ext}"
        filepath = os.path.join(folder, filename)
        
        if os.path.exists(filepath):
            print(f"File {filename} already exists. Skipping.")
            count += 1
            continue
            
        print(f"Downloading {url} to {filepath}...")
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(r.content)
                print(f"[OK] Successfully downloaded {filename}")
                count += 1
            else:
                print(f"[ERROR] Failed to download from {url} (Status code: {r.status_code})")
        except Exception as e:
            print(f"[ERROR] Error downloading from {url}: {e}")
            
    return count

if __name__ == "__main__":
    print("--- Starting Unhealthy Dataset Downloads ---")
    unhealthy_count = download_images(UNHEALTHY_URLS, "dataset/unhealthy", "unhealthy_leaf")
    print(f"\nDone! Downloaded {unhealthy_count} unhealthy leaf images.")
