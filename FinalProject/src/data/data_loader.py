from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_paths: list, root_dir: Path, class_to_idx: dict, transform=None):
        self.image_paths = image_paths
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        img_path = self.root_dir / rel_path

        img = Image.open(img_path).convert('RGB')

        class_name = Path(rel_path).parent.name
        label = self.class_to_idx.get(class_name, -1)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_paths)


class DataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_to_idx = None

    def setup(self):
        root_dir = Path(self.cfg.data.root_dir)
        splits_dir = Path(self.cfg.data.splits_dir)

        self.class_to_idx = self._build_class_mapping(root_dir)

        train_paths = self._load_split_file(splits_dir / 'train_images.txt')
        val_paths = self._load_split_file(splits_dir / 'val_images.txt')
        test_paths = self._load_split_file(splits_dir / 'test_images.txt')

        train_transform = self._get_train_transforms()
        val_transform = self._get_val_transforms()

        self.train_dataset = ImageDataset(train_paths, root_dir, self.class_to_idx, train_transform)
        self.val_dataset = ImageDataset(val_paths, root_dir, self.class_to_idx, val_transform)
        self.test_dataset = ImageDataset(test_paths, root_dir, self.class_to_idx, val_transform)

    def _build_class_mapping(self, root_dir: Path) -> dict:
        classes = set()

        for dataset_config in self.cfg.data.datasets:
            if dataset_config.enabled:
                dataset_path = root_dir / dataset_config.name
                for class_dir in dataset_path.iterdir():
                    if class_dir.is_dir():
                        classes.add(class_dir.name)

        sorted_classes = sorted(classes)
        return {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}

    def _load_split_file(self, file_path: Path) -> list:
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")

        with open(file_path, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]

        return paths

    def _get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.cfg.data.transforms.image_size,
                             self.cfg.data.transforms.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.data.transforms.mean,
                               std=self.cfg.data.transforms.std)
        ])

    def _get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.cfg.data.transforms.image_size,
                             self.cfg.data.transforms.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.data.transforms.mean,
                               std=self.cfg.data.transforms.std)
        ])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

    def get_dataset_info(self) -> Dict:
        return {
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'test_size': len(self.test_dataset),
            'num_classes': len(self.class_to_idx),
            'class_to_idx': self.class_to_idx
        }
