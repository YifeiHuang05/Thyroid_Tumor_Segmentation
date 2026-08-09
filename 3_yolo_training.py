from ultralytics import YOLO

# 1. Load the Nano Segmentation Model (it will auto-download the weights)
model = YOLO('yolov26n-seg.pt')

print("Starting training on Apple MPS...")

# 2. Train the model
# device='mps' forces the MacBook's GPU to do the heavy lifting
results = model.train(
    data='thyroid.yaml', # The YAML file we made earlier pointing to your data/ folders
    epochs=50,           # Start with 50 epochs for a quick baseline
    imgsz=640,           # Standard resolution
    batch=16,            # Number of images processed at once (lower this to 8 if your Mac crashes/runs out of memory)
    device='mps',        
    project='Task_C_Runs',
    name='baseline_ptc_model'
)

print("Training complete! Check the 'Task_C_Runs/baseline_ptc_model' folder for your weights and metrics.")