from ultralytics import YOLO

# Load the base model
model = YOLO('yolov26n-seg.pt')

print("Starting Hyperparameter Tuning...")

# The .tune() method automatically handles the genetic algorithm
# It will train for 30 epochs, 30 separate times, mutating parameters each time.
model.tune(
    data='thyroid.yaml',
    epochs=30, 
    iterations=30,      # Number of different hyperparameter combinations to try
    optimizer='AdamW',  # AdamW usually performs best for medical datasets
    device='mps', # change this for non-Apple device
    plots=False,        # Save memory by not plotting every single mutation
    save=False,         # Don't save weights for the bad runs
    val=False           # Only validate at the very end of an epoch to save time
)

print("Tuning complete! Ultralytics just saved a 'best_hyperparameters.yaml' file in your runs folder.")