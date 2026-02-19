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
    p = Path(test_images)

    if not p.exists():
        print(f"Path not found: {test_images}")
        return

    base_results = Path(cfg.paths.results).resolve()
    base_results.mkdir(parents=True, exist_ok=True)

    if p.is_dir():
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if subdirs:
            print("\nRunning detection for all subfolders...")
            total_images = 0
            total_detections = 0

            for class_dir in sorted(subdirs):
                print(f"\n--- {class_dir.name} ---")
                results = detector.predict(
                    source=str(class_dir),
                    save=True,
                    save_dir=str(base_results),
                    run_name=class_dir.name
                )
                total_images += len(results)
                total_detections += sum(len(r.boxes) for r in results)

                print("\n" + "="*60)
                print("Detection Results (All Subfolders)")
                print("="*60)
                print(f"Results saved to: {base_results}")
                print(f"Processed {total_images} image(s)")
                print(f"Total detections: {total_detections}")
            return

    run_name = p.name if p.is_dir() else p.stem

    print("\nRunning detection...")
    results = detector.predict(
        source=str(p),
        save=True,
        save_dir=str(base_results),
        run_name=run_name
    )

    print("\n" + "="*60)
    print("Detection Results")
    print("="*60)
    print(f"Processed {len(results)} image(s)")
    print(f"Results saved to: {base_results / run_name}")

    total_detections = sum(len(r.boxes) for r in results)
    print(f"Total detections: {total_detections}")


if __name__ == "__main__":
    main()
