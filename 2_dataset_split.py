import os
import shutil
from sklearn.model_selection import train_test_split

# Setup paths
images_dir = 'data/ThyroidXL/train/images'
labels_dir = 'data/ThyroidXL/train/labels'
output_dir = 'data/yolo_dataset'

# YOLO folder structure setup
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# 1. Group images by Patient ID to prevent Data Leakage
patient_data = {}
for label_file in os.listdir(labels_dir):
    if not label_file.endswith('.txt'): continue
    
    # Extract Patient ID (Assuming format: 00000139_217557BB_1.txt -> 00000139)
    # Adjust the split logic if your filenames differ
    patient_id = label_file.split('_')[0] 
    
    # Read the label to see if it's PTC (Class 1) or Non-PTC (Class 0)
    with open(os.path.join(labels_dir, label_file), 'r') as f:
        first_line = f.readline().strip()
        class_id = int(first_line.split(' ')[0])
        
    if patient_id not in patient_data:
        patient_data[patient_id] = {'images': [], 'label': 0}
        
    patient_data[patient_id]['images'].append(label_file.replace('.txt', ''))
    # If any image for this patient is PTC, label the patient as PTC
    if class_id == 1:
        patient_data[patient_id]['label'] = 1

# 2. Extract lists for Scikit-Learn
patients = list(patient_data.keys())
patient_labels = [patient_data[pid]['label'] for pid in patients]

print(f"Total Unique Patients: {len(patients)}")

# 3. Stratified Split (80% Train, 20% Val)
train_patients, val_patients = train_test_split(
    patients, 
    test_size=0.20, 
    stratify=patient_labels, # This forces the 15/85 balance
    random_state=42
)

# 4. Move files to the new YOLO structure
def move_files(patient_list, split_name):
    count = 0
    for pid in patient_list:
        for img_base in patient_data[pid]['images']:
            # Source paths
            src_img = os.path.join(images_dir, f"{img_base}.png")
            src_lbl = os.path.join(labels_dir, f"{img_base}.txt")
            
            # Destination paths
            dst_img = os.path.join(output_dir, split_name, 'images', f"{img_base}.png")
            dst_lbl = os.path.join(output_dir, split_name, 'labels', f"{img_base}.txt")
            
            # Copy files (using copy to keep your original data safe)
            if os.path.exists(src_img) and os.path.exists(src_lbl):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_lbl, dst_lbl)
                count += 1
    return count

print("Copying Training files...")
train_count = move_files(train_patients, 'train')
print(f"Copied {train_count} images to Train set.")

print("Copying Validation files...")
val_count = move_files(val_patients, 'val')
print(f"Copied {val_count} images to Val set.")
print("Dataset successfully split and balanced!")