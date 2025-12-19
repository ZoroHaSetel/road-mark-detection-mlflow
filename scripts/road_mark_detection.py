import os
import mlflow
from ultralytics import YOLO

# ======================
# MLflow config
# ======================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("road-mark-yolo")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

# ======================
# Train YOLOv5
# ======================
model = YOLO("weights/yolov5n.pt")

with mlflow.start_run(run_name="yolov5_road_mark_v1"):
    results = model.train(
    data="data/data.yaml",
    epochs=3,
    imgsz=320,
    batch=2,
    workers=0,
    device="cpu"
    )
    def clean_metric_name(name: str):
        return (
            name.replace("(", "")
                .replace(")", "")
                .replace("/", "_")
                .replace(" ", "_")
        )
    metrics = results.results_dict
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(clean_metric_name(k), v)

    best_weight = "runs/detect/train/weights/best.pt"
    mlflow.log_artifact(best_weight, artifact_path="model")

    mlflow.set_tag("model", "YOLOv5")
    mlflow.set_tag("task", "road-mark-detection")
