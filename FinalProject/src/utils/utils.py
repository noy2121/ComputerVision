from pathlib import Path
import random
import numpy as np
import torch

IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')


def set_seed(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_images(path: Path):
    if path.is_file():
        return [path]

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(path.glob(ext))
    return sorted(images)