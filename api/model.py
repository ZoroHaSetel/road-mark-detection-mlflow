import os
import mlflow
from ultralytics import YOLO

MODEL_NAME = "road-mark-yolo"
STAGE = "Production"

def load_model():
    print("Loading model from MLflow Registry...")

    mlflow.set_tracking_uri("http://mlflow:5000")

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    local_path = mlflow.artifacts.download_artifacts(model_uri)

    weight_path = None
    for root, _, files in os.walk(local_path):
        for f in files:
            if f.endswith(".pt"):
                weight_path = os.path.join(root, f)
                break

    if weight_path is None:
        raise RuntimeError("Model .pt not found")

    print(f"Model loaded: {weight_path}")
    return YOLO(weight_path)
