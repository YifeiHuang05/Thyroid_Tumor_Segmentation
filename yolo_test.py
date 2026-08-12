from ultralytics import YOLO

model = YOLO(
    "/data/ThyroidXL/runs/yolo26n_seg_test/weights/best.pt"
)

model.predict(
    source="/data/ThyroidXL/images/val",
    save=True,
    conf=0.25,
)