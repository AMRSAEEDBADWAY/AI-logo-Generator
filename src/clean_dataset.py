import os
from PIL import Image
import imagehash
from tqdm import tqdm
from config import DATA_CONFIG

# Define dataset path
TRAIN_DIR = DATA_CONFIG['train_dir']

def clean_dataset(root_dir):
    print(f"🔍 Scanning dataset at: {root_dir}")
    
    corrupted_count = 0
    duplicate_count = 0
    non_image_count = 0
    total_files = 0
    
    hashes = {}  # Store image hashes to find duplicates
    
    # Walk through all directories
    for root, dirs, files in os.walk(root_dir):
        for file in tqdm(files, desc=f"Checking {os.path.basename(root)}"):
            file_path = os.path.join(root, file)
            total_files += 1
            
            # 1. Check file extension
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                print(f"⚠️ Removing non-image file: {file}")
                os.remove(file_path)
                non_image_count += 1
                continue

            try:
                # 2. Verify Image Integrity
                with Image.open(file_path) as img:
                    img.verify()
                
                # Re-open for hashing (verify closes the file)
                with Image.open(file_path) as img:
                    # 3. Check for Duplicates using Perceptual Hash
                    img_hash = str(imagehash.phash(img))
                    
                    if img_hash in hashes:
                        print(f"♻️ Duplicate found: {file} (Same as {hashes[img_hash]})")
                        os.remove(file_path)
                        duplicate_count += 1
                    else:
                        hashes[img_hash] = file
                    
            except (IOError, SyntaxError, OSError) as e:
                print(f"❌ Corrupted file found: {file_path} - {e}")
                try:
                    os.remove(file_path)
                    corrupted_count += 1
                except:
                    pass

    print("\n" + "="*30)
    print(f"✅ Scanning Complete!")
    print(f"📂 Total Files Scanned: {total_files}")
    print(f"🚫 Corrupted Removed: {corrupted_count}")
    print(f"♻️ Duplicates Removed: {duplicate_count}")
    print(f"⚠️ Non-Images Removed: {non_image_count}")
    print("="*30)

if __name__ == "__main__":
    if os.path.exists(TRAIN_DIR):
        clean_dataset(TRAIN_DIR)
    else:
        print(f"❌ Directory not found: {TRAIN_DIR}")
