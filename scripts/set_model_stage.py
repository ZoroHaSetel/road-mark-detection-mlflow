from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")

client.transition_model_version_stage(
    name="road-mark-yolo",
    version="11",
    stage="Production"
)

print("✅ Archived version 8")