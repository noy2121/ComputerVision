import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.detector import Detector


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    print("Initializing Detector...")
    detector = Detector(cfg)

    print(f"\nLoading model: {cfg.detector.model_name}")
    detector.load_model()

    test_images = input("\nEnter path to test images (folder or single image): ").strip()

    if not Path(test_images).exists():
        print(f"Path not found: {test_images}")
        return

    print("\nRunning detection...")
    results = detector.predict(
        source=test_images,
        save=True,
        save_dir=cfg.paths.results
    )

    print("\n" + "="*60)
    print("Detection Results")
    print("="*60)
    print(f"Processed {len(results)} image(s)")
    print(f"Results saved to: {cfg.paths.results}")

    total_detections = sum(len(r.boxes) for r in results)
    print(f"Total detections: {total_detections}")


if __name__ == "__main__":
    main()
