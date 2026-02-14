import hydra
from omegaconf import DictConfig
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.detector import Detector
from src.classifier.model import Classifier
from src.data.data_loader import DataProcessor


# def build_idx_to_class(cfg):
#     # This matches your training mapping (combined datasets, sorted class names)
#     dp = DataProcessor(cfg)
#     _, class_to_idx = dp.get_dataset_info()
#     idx_to_class = {v: k for k, v in class_to_idx.items()}
#     return idx_to_class

from src.data import DataProcessor  # or from src.data.data_loader import DataProcessor

def build_idx_to_class(cfg):
    dp = DataProcessor(cfg)
    dp.setup()  # ✅ IMPORTANT: builds class_to_idx + datasets
    class_to_idx = dp.class_to_idx
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return idx_to_class

def build_classifier_transform(cfg):
    size = cfg.data.transforms.image_size
    mean = cfg.data.transforms.mean
    std = cfg.data.transforms.std

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    print("Initializing Detector...")
    detector = Detector(cfg)
    detector.load_model()

    print("Initializing Classifier...")
    classifier = Classifier(cfg)
    classifier.build_model()
    classifier.model.eval()

    ckpt_path = Path(cfg.classifier.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=classifier.device, weights_only=False)
    classifier.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded classifier checkpoint: {ckpt_path}")

    idx_to_class = build_idx_to_class(cfg)
    tfm = build_classifier_transform(cfg)

    test_images = input("\nEnter path to test images (folder or single image): ").strip()
    p = Path(test_images)

    if not p.exists():
        print(f"Path not found: {test_images}")
        return

    # If user enters a dataset root like data/FruitQ, run all subfolders
    subfolders = [d for d in p.iterdir() if d.is_dir()] if p.is_dir() else []
    targets = subfolders if subfolders else [p]

    base_results = Path(cfg.paths.results)
    base_results.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_detections = 0

    for t in targets:
        run_name = t.name if t.is_dir() else t.stem
        out_dir = base_results / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # collect images
        if t.is_dir():
            images = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                images.extend(t.glob(ext))
        else:
            images = [t]

        if not images:
            print(f"\n[{run_name}] No images found, skipping.")
            continue

        print(f"\n[{run_name}] Processing {len(images)} image(s)...")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # detect (do NOT trust detector class names)
            results = detector.predict(source=str(img_path), save=False)
            if not results or len(results[0].boxes) == 0:
                # save original (no detections)
                cv2.imwrite(str(out_dir / img_path.name), img)
                total_images += 1
                continue

            # pick the best detection by confidence (good for single-object images)
            boxes = results[0].boxes
            confs = boxes.conf.detach().cpu().numpy()
            best_i = int(confs.argmax())

            xyxy = boxes.xyxy[best_i].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)

            crop_bgr = img[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                cv2.imwrite(str(out_dir / img_path.name), img)
                total_images += 1
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(crop_rgb)

            x = tfm(pil).unsqueeze(0).to(classifier.device)

            with torch.no_grad():
                logits = classifier.model(x)
                probs = F.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                pred_prob = float(probs[pred_idx].item())

            pred_name = idx_to_class.get(pred_idx, f"class_{pred_idx}")

            # draw bbox + classifier label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"{pred_name} {pred_prob:.2f}"
            cv2.putText(img, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imwrite(str(out_dir / img_path.name), img)

            total_images += 1
            total_detections += 1

        print(f"[{run_name}] Saved to: {out_dir}")

    print("\n" + "=" * 60)
    print("Detection+Classification Results")
    print("=" * 60)
    print(f"Results saved under: {base_results}")
    print(f"Processed images: {total_images}")
    print(f"Classified detections: {total_detections}")


if __name__ == "__main__":
    main()
