from pathlib import Path

from ultralytics import YOLO, RTDETR


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def load_model(self, weights_path=None):
        if weights_path is None:
            weights_path = self.cfg.detector.weights

        model_name = self.cfg.detector.model_name

        if model_name.startswith('yolov8'):
            self.model = YOLO(weights_path)
        elif model_name.startswith('rtdetr'):
            self.model = RTDETR(weights_path)
        else:
            raise ValueError(f"Model {model_name} not supported. Use yolov8* or rtdetr*")

        print(f'Loaded {model_name} from {weights_path}')

    def predict(self, source, save=False, save_dir=None, run_name=None):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        project_dir = Path(save_dir).resolve() if save_dir else None

        results = self.model.predict(
            source=source,
            imgsz=self.cfg.detector.image_size,
            conf=self.cfg.detector.conf_threshold,
            iou=self.cfg.detector.iou_threshold,
            save=save,
            project=str(project_dir) if project_dir else None,
            name=run_name,
            exist_ok=True
        )
        return results

    def export(self, format='onnx'):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.export(format=format)
        print(f'Model exported to {format} format')
