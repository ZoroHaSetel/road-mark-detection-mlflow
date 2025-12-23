import os
import mlflow
from pathlib import Path

# MLflow + MinIO
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("road-mark-yolo")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

# tìm best.pt mới nhất
runs_dir = Path("runs/detect")
latest_train = sorted(runs_dir.glob("train*"), key=lambda x: x.stat().st_mtime)[-1]
best_pt = latest_train / "weights/best.pt"

print(f"Logging model: {best_pt}")

with mlflow.start_run(run_name=latest_train.name):
    mlflow.log_artifact(str(best_pt), artifact_path="model")
    mlflow.set_tag("yolo_run", latest_train.name)

print("Log to MLflow + MinIO DONE")
