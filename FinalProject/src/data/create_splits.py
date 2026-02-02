from pathlib import Path
import random

import hydra
from omegaconf import DictConfig


def collect_image_paths(root_dir: Path, dataset_names: list) -> list:
    image_paths = []

    for dataset_name in dataset_names:
        dataset_path = root_dir / dataset_name

        for class_dir in sorted(dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue

            for img_path in class_dir.glob('**/*.jpg'):
                rel_path = img_path.relative_to(root_dir)
                image_paths.append(str(rel_path))

            for img_path in class_dir.glob('**/*.png'):
                rel_path = img_path.relative_to(root_dir)
                image_paths.append(str(rel_path))

    return image_paths


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):

    root_dir = Path(cfg.data.root_dir)
    enabled_datasets = [ds.name for ds in cfg.data.datasets if ds.enabled]

    print(f"Collecting images from datasets: {enabled_datasets}")
    all_images = collect_image_paths(root_dir, enabled_datasets)

    print(f"Total images collected: {len(all_images)}")

    random.shuffle(all_images)

    total_size = len(all_images)
    train_size = int(cfg.data.train_split * total_size)
    val_size = int(cfg.data.val_split * total_size)

    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_images)} ({len(train_images)/total_size*100:.1f}%)")
    print(f"  Val:   {len(val_images)} ({len(val_images)/total_size*100:.1f}%)")
    print(f"  Test:  {len(test_images)} ({len(test_images)/total_size*100:.1f}%)")

    train_file = root_dir / 'train_images.txt'
    val_file = root_dir / 'val_images.txt'
    test_file = root_dir / 'test_images.txt'

    with open(train_file, 'w') as f:
        f.write('\n'.join(train_images))

    with open(val_file, 'w') as f:
        f.write('\n'.join(val_images))

    with open(test_file, 'w') as f:
        f.write('\n'.join(test_images))

    print(f"\nSplit files created:")
    print(f"  {train_file}")
    print(f"  {val_file}")
    print(f"  {test_file}")


if __name__ == "__main__":
    main()
