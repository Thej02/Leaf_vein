import os
import requests
import json

def download_picsum_clutter(count=10, folder="dataset/Non-hibiscus"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Fetch list of photos
    url = f"https://picsum.photos/v2/list?page=1&limit={count}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"[ERROR] Failed to fetch image list (Status: {r.status_code})")
            return 0
        
        photos = r.json()
        downloaded = 0
        
        for idx, photo in enumerate(photos):
            # We can download a resized version to save bandwidth and keep size consistent (512x512)
            download_url = f"https://picsum.photos/id/{photo['id']}/512/512"
            filename = f"non_hibiscus_clutter_{photo['id']}.jpg"
            filepath = os.path.join(folder, filename)
            
            print(f"Downloading clutter image {idx+1}/{count} from {download_url}...")
            img_r = requests.get(download_url, headers=headers, timeout=15)
            if img_r.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(img_r.content)
                print(f"[OK] Saved {filename}")
                downloaded += 1
            else:
                print(f"[ERROR] Failed to download photo id {photo['id']} (Status: {img_r.status_code})")
                
        return downloaded
    except Exception as e:
        print(f"[ERROR] Error fetching clutter images: {e}")
        return 0

if __name__ == "__main__":
    print("--- Starting Clutter Image Downloads ---")
    downloaded = download_picsum_clutter(10, "dataset/Non-hibiscus")
    print(f"Done! Downloaded {downloaded} background clutter images.")
