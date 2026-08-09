import os, glob
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Load your best trained model
model = YOLO('runs/segment/ptc_segmentation_baseline/weights/best.pt')

TEST_DIR = "ThyroidXL/test/images" # Assuming you have a standard test folder
y_true = []
y_pred = []

print("Running inference and saving visual masks...")
# This automatically saves the images with the predicted colored masks drawn on them!
# Check 'runs/segment/predict' when this finishes.
results = model.predict(source=TEST_DIR, save=True, show_labels=True, show_conf=True)

# Calculate Strict Rubric Metrics
for r in results:
    img_path = r.path
    base_name = os.path.basename(img_path)
    
    # 1. Get Ground Truth (You'd look this up from id2info again, assuming test set is there)
    pat_id = base_name.split('_')[0]
    true_class = 1 if is_ptc(pat_id) else 0 # Assuming is_ptc is imported/defined
    y_true.append(true_class)
    
    # 2. Get Model's Prediction
    # If the model found tumors, take the class of the highest-confidence prediction
    if len(r.boxes) > 0:
        pred_class = int(r.boxes.cls[0].item())
    else:
        pred_class = 0 # Default to non-ptc if nothing is detected
    y_pred.append(pred_class)

# 3. Print Official Metrics
print("\n--- FINAL TASK C METRICS ---")
print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
print(f"Macro F1 Score:    {f1_score(y_true, y_pred, average='macro'):.4f}")