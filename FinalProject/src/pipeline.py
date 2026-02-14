from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.detector import Detector
from src.classifier import Classifier
from src.data import DataProcessor


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.classifier = Classifier(cfg)
        self.idx_to_class = None
        self.transform = None

    def setup(self):
        self.detector.load_model()

        self.classifier.build_model()
        self.classifier.load_weights(self.cfg.classifier.checkpoint)
        self.classifier.model.eval()

        self._build_class_mapping()
        self._build_transform()

    def _build_class_mapping(self):
        dp = DataProcessor(self.cfg)
        dp.setup()
        self.idx_to_class = {idx: cls for cls, idx in dp.class_to_idx.items()}

    def _build_transform(self):
        size = self.cfg.data.transforms.image_size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.cfg.data.transforms.mean,
                std=self.cfg.data.transforms.std
            ),
        ])

    def detect(self, image_path: str):
        results = self.detector.predict(source=image_path, save=False)
        if not results or len(results[0].boxes) == 0:
            return None, []

        img = results[0].orig_img
        boxes = results[0].boxes

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'det_confidence': float(box.conf[0]),
                'det_class': int(box.cls[0])
            })

        return img, detections

    def crop_and_classify(self, img: np.ndarray, bbox: tuple):
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            return None, 0.0

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        tensor = self.transform(pil_img).unsqueeze(0).to(self.classifier.device)

        with torch.no_grad():
            logits = self.classifier.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs))
            pred_prob = float(probs[pred_idx])

        pred_name = self.idx_to_class.get(pred_idx, f"class_{pred_idx}")
        return pred_name, pred_prob

    def run(self, image_path: str, save_dir: str = None):
        img, detections = self.detect(image_path)

        if img is None or not detections:
            return {'image_path': image_path, 'detections': []}

        best_det = max(detections, key=lambda d: d['det_confidence'])

        pred_name, pred_prob = self.crop_and_classify(img, best_det['bbox'])

        best_det['class_name'] = pred_name
        best_det['class_confidence'] = pred_prob

        if save_dir:
            annotated = self._draw_annotation(img, best_det)
            out_path = Path(save_dir) / Path(image_path).name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), annotated)

        return {
            'image_path': image_path,
            'detections': [best_det]
        }

    def run_batch(self, image_paths: list, save_dir: str = None):
        results = []
        for img_path in image_paths:
            result = self.run(str(img_path), save_dir)
            results.append(result)
        return results

    def _draw_annotation(self, img: np.ndarray, detection: dict):
        annotated = img.copy()
        x1, y1, x2, y2 = detection['bbox']

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

        label = f"{detection['class_name']} {detection['class_confidence']:.2f}"
        cv2.putText(
            annotated, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        return annotated