import os
import mlflow
from ultralytics import YOLO

mlflow.set_tracking_uri("http://mlflow:5000")

os.environ.update({
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
    "AWS_ACCESS_KEY_ID": "minio",
    "AWS_SECRET_ACCESS_KEY": "minio123"
})

MODEL_NAME = "road-mark-yolo"
MODEL_STAGE = "Production"

def load_model():
    """Load YOLO model từ MLflow Model Registry"""
    try:
        # Cách 1: Load bằng mlflow.pyfunc
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"Loading: {model_uri}")
        
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully")
        return pyfunc_model
        
    except Exception as e:
        print(f"Pyfunc load failed: {e}")
        return None

# Test
if __name__ == "__main__":
    model = load_model()
    if model:
        print("Model ready for prediction")