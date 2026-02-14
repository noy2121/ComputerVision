import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig

from pipeline import Pipeline
from utils.utils import collect_images



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    pipeline = Pipeline(cfg)
    pipeline.setup()

    source = input("\nEnter path to test images (folder or single image): ").strip()
    p = Path(source)

    if not p.exists():
        print(f"Path not found: {source}")
        return

    base_results = Path(cfg.paths.results) / "pipeline"
    base_results.mkdir(parents=True, exist_ok=True)

    subdirs = [d for d in p.iterdir() if d.is_dir()] if p.is_dir() else []
    targets = subdirs if subdirs else [p]

    total_images = 0
    total_detections = 0

    for target in sorted(targets):
        run_name = target.name if target.is_dir() else target.stem
        save_dir = str(base_results / run_name)

        images = collect_images(target)
        if not images:
            print(f"\n[{run_name}] No images found, skipping.")
            continue

        print(f"\n[{run_name}] Processing {len(images)} image(s)...")

        results = pipeline.run_batch(images, save_dir=save_dir)

        detected = sum(1 for r in results if r['detections'])
        total_images += len(images)
        total_detections += detected

        print(f"[{run_name}] {detected}/{len(images)} detections saved to: {save_dir}")

    print(f"\n{'='*60}")
    print("Pipeline Results")
    print(f"{'='*60}")
    print(f"Total images: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Results saved to: {base_results}")


if __name__ == "__main__":
    main()