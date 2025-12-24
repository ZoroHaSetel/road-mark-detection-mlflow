from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")

client.delete_model_version(
    name="road-mark-yolo",
    version="9"
)

print("✅ Deleted version 9")
