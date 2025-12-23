from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil
from .yolo_predictor import load_model

app = FastAPI(title="Road Mark Detection API")
model = load_model()
if not model:
    print("WARNING: Model failed to load. API will return errors.")
else:
    print("Model loaded successfully")
# predictor = YOLOPredictor()


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         shutil.copyfileobj(file.file, tmp)
#         tmp_path = tmp.name

#     result = predictor.predict(tmp_path)
#     return {"result": result}
