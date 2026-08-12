import cv2
import json
import os
import shutil

# 1. Setup Original Test Paths
orig_test_masks = 'data/ThyroidXL/test/masks'
orig_test_images = 'data/ThyroidXL/test/images'
test_json_path = 'data/ThyroidXL/test_annotations.json'

# 2. Setup Destination YOLO Paths
yolo_test_images = 'data/yolo_dataset/test/images'
yolo_test_labels = 'data/yolo_dataset/test/labels'

os.makedirs(yolo_test_images, exist_ok=True)
os.makedirs(yolo_test_labels, exist_ok=True)

# 3. Load the Test JSON
with open(test_json_path, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

print("Processing Test Set: Generating YOLO polygons and copying images...")

count = 0
for mask_name in os.listdir(orig_test_masks):
    if not mask_name.endswith('.png'): continue
    
    image_base = mask_name.replace('.png', '')
    
    # Extract label (0 for Benign/non_ptc, 1 for Malignant/ptc)
    img_metadata = annotations.get(image_base, {})
    label_str = img_metadata.get('label', 'lanhtinh') 
    class_id = 1 if label_str == 'ptc' or label_str == 'actinh' else 0
    
    # Read mask and find contours
    mask_path = os.path.join(orig_test_masks, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0: continue
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Normalize coordinates
    polygon = []
    for point in largest_contour:
        x = point[0][0] / W
        y = point[0][1] / H
        polygon.extend([f"{x:.6f}", f"{y:.6f}"])
        
    # Save YOLO .txt label
    txt_filename = os.path.join(yolo_test_labels, f"{image_base}.txt")
    with open(txt_filename, 'w') as f:
        f.write(f"{class_id} " + " ".join(polygon) + "\n")
        
    # Copy the original image to the new YOLO test folder
    src_img = os.path.join(orig_test_images, f"{image_base}.png")
    dst_img = os.path.join(yolo_test_images, f"{image_base}.png")
    if os.path.exists(src_img):
        shutil.copy(src_img, dst_img)
        count += 1

print(f"Test data prep complete! {count} images and labels are ready for the final evaluation.")