import os
import json
import glob
import shutil
from sklearn.model_selection import train_test_split

# --- DIRECTORIES ---
IMAGE_DIR = "data/ThyroidXL/train/images"     # Original training images
LABEL_DIR = "data/labels/train_task_c"     
ID2INFO_PATH = "data/ThyroidXL/stats/id2info_eng.json" # The clinical metadata file

# New YOLO directories
BASE_OUT = "yolo_dataset"
for split in ['train', 'val']:
    os.makedirs(f"{BASE_OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{BASE_OUT}/labels/{split}", exist_ok=True)

# 1. Read the Clinical Data to Map Patient -> Class
with open(ID2INFO_PATH, 'r', encoding='utf-8') as f:
    patient_data = json.load(f)

def get_patient_class(patient_id):
    info = patient_data.get(patient_id)
    if not info: return None
    for nodule_key in ["nodule_1", "nodule_2", "nodule_3", "nodule_4"]:
        nodule = info.get(nodule_key)
        if nodule and nodule.get("Histopathology"):
            if "papillary" in str(nodule["Histopathology"]).lower():
                return 1 # PTC
    return 0 # Non-PTC

# 2. Gather Unique Patients from your generated Task C labels
all_txt_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
patient_dict = {} # { "00000139": 1, ... }

for txt_path in all_txt_files:
    filename = os.path.basename(txt_path)
    patient_id = filename.split('_')[0]
    
    if patient_id not in patient_dict:
        cls = get_patient_class(patient_id)
        if cls is not None:
            patient_dict[patient_id] = cls

# 3. Perform Stratified Patient-Level Split (80% Train, 20% Val)
patients = list(patient_dict.keys())
labels = list(patient_dict.values())

train_patients, val_patients, _, _ = train_test_split(
    patients, labels, 
    test_size=0.20, 
    stratify=labels, # <--- This guarantees the PTC/Non-PTC ratio is identical in both splits
    random_state=42
)

# Convert lists to sets for lightning-fast lookups
train_set = set(train_patients)
val_set = set(val_patients)

# 4. Move Files to the Correct YOLO Folders
print("Copying files to YOLO structure based on patient split...")

train_count, val_count = 0, 0

for txt_path in all_txt_files:
    filename = os.path.basename(txt_path)
    img_name = filename.replace('.txt', '.png')
    patient_id = filename.split('_')[0]
    
    src_img = os.path.join(IMAGE_DIR, img_name)
    src_txt = txt_path
    
    if patient_id in train_set:
        split_dir = "train"
        train_count += 1
    elif patient_id in val_set:
        split_dir = "val"
        val_count += 1
    else:
        continue # Skip if patient wasn't mapped
        
    # Copy Image
    if os.path.exists(src_img):
        shutil.copy(src_img, f"{BASE_OUT}/images/{split_dir}/{img_name}")
    # Copy Label
    shutil.copy(src_txt, f"{BASE_OUT}/labels/{split_dir}/{filename}")

print("-" * 30)
print(f"Data successfully split and balanced (patient-level)!")
print(f"Training files: {train_count}")
print(f"Validation files: {val_count}")