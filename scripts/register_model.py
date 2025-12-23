import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

MODEL_NAME = "road-mark-yolo"

client = MlflowClient()

runs = client.search_runs(
    experiment_ids=[client.get_experiment_by_name("road-mark-yolo").experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1
)

run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registered version {result.version}")

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Production"
)

print("Promoted to Production")
