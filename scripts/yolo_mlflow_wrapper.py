import mlflow.pyfunc
from ultralytics import YOLO

class YOLOv5Wrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load YOLO model từ artifact weights
        weights_path = context.artifacts["weights"]
        self.model = YOLO(weights_path)

    def predict(self, context, model_input):
        # model_input có thể là path ảnh hoặc numpy array
        results = self.model(model_input)

        # Trả về bounding boxes dưới dạng list/dict để MLflow dễ log
        output = []
        for r in results:
            boxes = r.boxes.xyxy.tolist()   # [x1, y1, x2, y2, conf, class]
            output.append(boxes)

        return output