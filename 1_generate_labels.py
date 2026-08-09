import os
import json
import cv2
import glob

# --- DIRECTORIES ---
IMAGE_DIR = "data/ThyroidXL/train/images"             # Folder with .png images
POLYGON_JSON_DIR = "data/ThyroidXL/train/labels_orig" # Folder with polygon JSONs
ID2INFO_PATH = "data/ThyroidXL/stats/id2info_eng.json" # The clinical metadata file
OUTPUT_LABEL_DIR = "data/labels/train_task_c"    # Where the new YOLO txt files go

os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# 1. Load the Clinical Ground Truth
with open(ID2INFO_PATH, 'r', encoding='utf-8') as f:
    patient_data = json.load(f)

# Helper function to check if a patient has PTC
def check_if_ptc(patient_id):
    info = patient_data.get(patient_id)
    if not info:
        return False
    
    for nodule_key in ["nodule_1", "nodule_2", "nodule_3", "nodule_4"]:
        nodule = info.get(nodule_key)
        if nodule and nodule.get("Histopathology"):
            if "papillary" in str(nodule["Histopathology"]).lower():
                return True
    return False

# 2. Process all polygon JSONs
json_files = glob.glob(os.path.join(POLYGON_JSON_DIR, "*.json"))
print(f"Merging data for {len(json_files)} images...")

ptc_count = 0
non_ptc_count = 0

for json_path in json_files:
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    
    # Extract patient ID (usually the first part of the filename before the underscore)
    patient_id = base_name.split('_')[0]
    
    # Determine the Task C Class
    is_ptc = check_if_ptc(patient_id)
    class_id = 1 if is_ptc else 0
    
    if is_ptc: ptc_count += 1
    else: non_ptc_count += 1

    # Load the polygon data
    with open(json_path, 'r', encoding='utf-8') as f:
        poly_data = json.load(f)
        
    points_x = poly_data.get("all_points_x", [])
    points_y = poly_data.get("all_points_y", [])
    
    # Skip if no polygon data
    if not points_x or not points_y:
        continue

    # Get image dimensions for YOLO normalization
    img_path = os.path.join(IMAGE_DIR, f"{base_name}.png")
    if not os.path.exists(img_path):
        continue
        
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Format for YOLO Segmentation
    yolo_line = f"{class_id}"
    for x, y in zip(points_x, points_y):
        yolo_line += f" {x/w:.6f} {y/h:.6f}"
        
    # Save the file
    txt_output_path = os.path.join(OUTPUT_LABEL_DIR, f"{base_name}.txt")
    with open(txt_output_path, 'w') as out_f:
        out_f.write(yolo_line + "\n")

print("-" * 30)
print("Task C Dataset Generation Complete!")
print(f"Total PTC Polygons (Class 1): {ptc_count}")
print(f"Total Non-PTC Polygons (Class 0): {non_ptc_count}")