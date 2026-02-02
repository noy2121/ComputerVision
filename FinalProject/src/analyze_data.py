from pathlib import Path
from collections import defaultdict

import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image
import plotly.graph_objects as go


def analyze_dataset(dataset_path: Path):
    stats = defaultdict(lambda: {'count': 0, 'sizes': []})
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = list(class_dir.glob('**/*.jpg')) + list(class_dir.glob('**/*.png'))
        stats[class_name]['count'] = len(image_files)

        for img_path in image_files[:50]:
            try:
                with Image.open(img_path) as img:
                    stats[class_name]['sizes'].append(img.size)
            except Exception:
                continue

    return stats


def print_dataset_summary(dataset_name: str, stats: dict):
    total_images = sum(s['count'] for s in stats.values())

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Total Classes: {len(stats)}")
    print(f"Total Images: {total_images}")
    print(f"\nPer-Class Distribution:")
    print(f"{'-'*60}")

    for class_name, data in sorted(stats.items()):
        sizes = data['sizes']
        if sizes:
            avg_w = np.mean([s[0] for s in sizes])
            avg_h = np.mean([s[1] for s in sizes])
            print(f"{class_name:40} | {data['count']:6} images | Avg size: {avg_w:.0f}x{avg_h:.0f}")
        else:
            print(f"{class_name:40} | {data['count']:6} images | Avg size: N/A")


def parse_class_name(class_name: str):
    parts = class_name.split('__')
    if len(parts) == 2:
        return parts[0], parts[1]
    return class_name, 'Unknown'


def create_visualization(dataset_name: str, stats: dict):
    crop_data = defaultdict(lambda: {'Healthy': 0, 'Rotten': 0})

    for class_name, data in stats.items():
        crop, condition = parse_class_name(class_name)
        if condition in ['Healthy', 'Rotten']:
            crop_data[crop][condition] = data['count']

    crops = sorted(crop_data.keys())
    healthy_counts = [crop_data[crop]['Healthy'] for crop in crops]
    rotten_counts = [crop_data[crop]['Rotten'] for crop in crops]

    fig = go.Figure(data=[
        go.Bar(name='Healthy', x=crops, y=healthy_counts, marker_color='green', marker_pattern_shape='/'),
        go.Bar(name='Rotten', x=crops, y=rotten_counts, marker_color='red', marker_pattern_shape='\\')
    ])

    fig.update_layout(
        title=f'{dataset_name} - Healthy vs Rotten by Crop',
        xaxis_title='Crop',
        yaxis_title='Number of Images',
        barmode='group',
        xaxis_tickangle=-45,
        height=500,
        legend=dict(x=1.02, y=1)
    )

    return fig


def create_combined_visualization(all_stats: dict):
    combined_crop_data = defaultdict(lambda: {'Healthy': 0, 'Rotten': 0})

    for dataset_name, stats in all_stats.items():
        for class_name, data in stats.items():
            crop, condition = parse_class_name(class_name)
            if condition in ['Healthy', 'Rotten']:
                combined_crop_data[crop][condition] += data['count']

    crops = sorted(combined_crop_data.keys())
    healthy_counts = [combined_crop_data[crop]['Healthy'] for crop in crops]
    rotten_counts = [combined_crop_data[crop]['Rotten'] for crop in crops]

    fig = go.Figure(data=[
        go.Bar(name='Healthy', x=crops, y=healthy_counts, marker_color='green', marker_pattern_shape='/'),
        go.Bar(name='Rotten', x=crops, y=rotten_counts, marker_color='red', marker_pattern_shape='\\')
    ])

    fig.update_layout(
        title='Combined Datasets - Healthy vs Rotten by Crop',
        xaxis_title='Crop',
        yaxis_title='Number of Images',
        barmode='group',
        xaxis_tickangle=-45,
        height=500,
        legend=dict(x=1.02, y=1)
    )

    return fig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    root_dir = Path(cfg.data.root_dir)

    all_stats = {}
    for dataset in cfg.data.datasets:
        if dataset.enabled:
            dataset_path = root_dir / dataset.name
            stats = analyze_dataset(dataset_path)
            print_dataset_summary(dataset.name, stats)
            all_stats[dataset.name] = stats

    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    total_classes = sum(len(stats) for stats in all_stats.values())
    total_images = sum(sum(s['count'] for s in stats.values()) for stats in all_stats.values())
    print(f"Total Datasets: {len(all_stats)}")
    print(f"Total Classes: {total_classes}")
    print(f"Total Images: {total_images}")

    print(f"\nRecommended splits (based on config):")
    print(f"  Train: {int(total_images * cfg.data.train_split)} images ({cfg.data.train_split*100:.0f}%)")
    print(f"  Val:   {int(total_images * cfg.data.val_split)} images ({cfg.data.val_split*100:.0f}%)")
    print(f"  Test:  {int(total_images * cfg.data.test_split)} images ({cfg.data.test_split*100:.0f}%)")

    for dataset_name, stats in all_stats.items():
        fig = create_visualization(dataset_name, stats)
        output_path = Path(cfg.paths.results) / f"{dataset_name}_distribution.html"
        output_path.parent.mkdir(exist_ok=True)
        fig.write_html(str(output_path))
        print(f"\nVisualization saved: {output_path}")

    combined_fig = create_combined_visualization(all_stats)
    combined_output_path = Path(cfg.paths.results) / "combined_distribution.html"
    combined_fig.write_html(str(combined_output_path))
    print(f"Combined visualization saved: {combined_output_path}")


if __name__ == "__main__":
    main()
