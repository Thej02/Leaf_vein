import os
import requests

# Define positive class URLs (Hibiscus / Rosa-sinensis)
POSITIVE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/4/4b/Starr_080117-1792_Hibiscus_rosa-sinensis.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/87/Hibiscus_rosa-sinensis_leaf_back_20150917.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/c/cb/Hibiscus_rosa-sinensis_leaf_front_20150917.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/a/af/Leaf_of_Hibiscus_rosa-sinensis.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/2/23/Hibiscus_rosa-sinensis_leaves.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/e/ec/Hibiscus_rosa-sinensis_leaf_01.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/45/Hibiscus_rosa-sinensis_leaf_02.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/e/ef/Hibiscus_rosa-sinensis_leaf_03.jpg"
]

# Define negative class URLs (Non-hibiscus leaves + background clutter)
NEGATIVE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e0/Rose_leaves.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/c/c9/Hedera_helix_leaf_clean.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/b/b5/Fagus_sylvatica_leaf_2.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/d3/Quercus_robur_leaf_2.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/9e/Morus_alba_leaf.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/Spearmint_leaves_01.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/90/Centella_asiatica_leaf.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/e/eb/Eucalyptus_globulus_leaf.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/df/Fern_leaves_02.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/c/ce/Taraxacum_officinale_leaf_1.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/52/Lawn_grass.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/b/b2/Dry_soil.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/45/A_small_cup_of_coffee.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/c/c7/Brick_wall_close-up.jpg"
]

def download_images(urls, folder, prefix):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Use a standard web browser user agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    count = 0
    for idx, url in enumerate(urls):
        ext = url.split(".")[-1].lower()
        if ext not in ["jpg", "jpeg", "png"]:
            ext = "jpg"
            
        filename = f"{prefix}_{idx + 1}.{ext}"
        filepath = os.path.join(folder, filename)
        
        # Check if already exists to avoid redundant downloads
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
    print("--- Starting Dataset Downloads ---")
    
    print("\n1. Sourcing Positive Images (additional Hibiscus leaves)...")
    pos_count = download_images(POSITIVE_URLS, "dataset/healthy", "hibiscus_download")
    
    print("\n2. Sourcing Negative Images (other leaves & background clutter)...")
    neg_count = download_images(NEGATIVE_URLS, "dataset/Non-hibiscus", "non_hibiscus")
    
    print(f"\nDone! Downloaded {pos_count} positive and {neg_count} negative images.")
