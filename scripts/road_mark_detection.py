from ultralytics import YOLO

model = YOLO("weights/yolov5n.pt")

results = model.train(
    data="data/data.yaml",
    epochs=1,
    imgsz=50,
    batch=1,
    workers=0,
    device="cpu",
    project="runs/detect",
    name="train",
    exist_ok=False   # mỗi lần là train2, train3...
)

print(f"Train done at: {results.save_dir}")
