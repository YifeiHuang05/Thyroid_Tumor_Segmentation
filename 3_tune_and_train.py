from ultralytics import YOLO
import os

def main():
    # Set your device: 0 for NVIDIA GPU, 'cpu' for standard processor
    # (Ensure you installed the CUDA version of PyTorch if using device=0!)
    WINDOWS_DEVICE = 0 
    
    # ---------------------------------------------------------
    # PHASE 1: THE "MINI-TUNE" (Find the best parameters)
    # ---------------------------------------------------------
    print("🚀 Phase 1: Starting Hyperparameter Tuning...")
    tuner_model = YOLO('yolo26n-seg.pt') # Start with a fresh Nano model
    
    # Run a short HPO session (10 attempts, 20 epochs each)
    tuner_model.tune(
        data='task_c.yaml',
        epochs=20,          
        iterations=10,      # Only train 10 different times to save time
        optimizer='AdamW',
        device=WINDOWS_DEVICE,
        plots=False,
        save=False,
        name='ptc_tuning_runs'
    )
    
    # ---------------------------------------------------------
    # PHASE 2: THE FINAL TRAIN
    # ---------------------------------------------------------
    print("✅ Phase 1 Complete. Injecting best parameters into Final Training...")
    
    # Load a BRAND NEW fresh model for the final training so weights don't leak
    final_model = YOLO('yolo26n-seg.pt') 
    
    # The tuning phase automatically saved a yaml file with the best settings
    # We pass that file to the 'cfg' argument
    best_params_path = "runs/segment/ptc_tuning_runs/best_hyperparameters.yaml"
    
    if os.path.exists(best_params_path):
        final_model.train(
            data='task_c.yaml',
            epochs=50,          # Train for a full 50 epochs this time
            imgsz=640,
            batch=16,           # Lower to 8 if Windows runs out of memory (OOM error)
            device=WINDOWS_DEVICE,
            cfg=best_params_path, # <--- THIS is where the tuning pays off!
            name='ptc_final_model'
        )
        print("All done! Optimized model is ready.")
    else:
        print("Error: Could not find the tuning parameters file.")

# THIS IS MANDATORY ON WINDOWS
if __name__ == '__main__':
    # On Windows, you might need to set this environment variable to prevent memory fragmentation crashes
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()