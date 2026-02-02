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

    def predict(self, source, save=False, save_dir=None):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        results = self.model.predict(
            source=source,
            imgsz=self.cfg.detector.image_size,
            conf=self.cfg.detector.conf_threshold,
            iou=self.cfg.detector.iou_threshold,
            save=save,
            project=save_dir
        )
        return results

    def predict_and_crop(self, image_path):
        results = self.predict(source=image_path, save=False)

        crops = []
        for result in results:
            boxes = result.boxes
            img = result.orig_img

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]
                crops.append({
                    'image': crop,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0])
                })

        return crops

    def export(self, format='onnx'):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.export(format=format)
        print(f'Model exported to {format} format')
