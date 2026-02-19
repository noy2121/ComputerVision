# Crop Freshness Detection System (YOLOv8 Detector + ResNet Classifier + Dash Dashboard)

This repo contains a complete pipeline for crop/produce analysis:
- **Detector** (Ultralytics YOLOv8 / RT-DETR): finds objects in images.
- **Classifier** (torchvision ResNet): classifies detected crops into dataset classes.
- **Evaluation plots**: confusion matrix + per-class metrics + “top confusions” images.
- **Dashboard (Dash GUI)**: web UI for inference, performance, dataset overview, and evaluation plots.

> Large artifacts such as datasets, checkpoints, and runs outputs are usually not committed to git.  
> The project uses **Hydra** configuration (`conf/config.yaml`) to control paths and experiment outputs.


---

## Repository Structure

FinalProject/
├─ conf/
│ └─ config.yaml
├─ data/
│ ├─ FruitQ/
│ ├─ FruitVegetableDiseases/
│ ├─ train_images.txt
│ ├─ val_images.txt
│ └─ test_images.txt
├─ runs/
│ └─ <experiment_name>/
│ ├─ checkpoints/
│ │ └─ best_model.pth
│ ├─ logs/
│ └─ results/
│ └─ classifier_eval_plots/
│ ├─ confusion_matrix_counts.png
│ ├─ confusion_matrix_normalized.png
│ ├─ per_class_precision.png
│ ├─ per_class_recall.png
│ ├─ per_class_f1.png
│ └─ per_class_top_confusions/
│ └─ top_confusions__<class>.png
└─ src/
├─ classifier/
│ ├─ model.py
│ ├─ train.py
│ ├─ evaluate.py
│ └─ eval_plots.py
├─ detector/
│ ├─ model.py
│ └─ evaluate.py
├─ data/
│ ├─ create_splits.py
│ ├─ analyze_data.py
│ └─ data_loader.py
├─ dashboard/
│ ├─ app.py
│ └─ tabs/
│ ├─ inference.py
│ ├─ performance.py
│ ├─ dataset.py
│ └─ evaluation_plots.py
├─ pipeline.py
└─ utils/
└─ gradcam.py


---

## Requirements

### Python
- Recommended: **Python 3.13**
- Works on Windows / macOS / Linux.

### Packages
If you don’t have a `requirements.txt`, install dependencies manually:

```bash
python -m pip install --upgrade pip

# core
pip install hydra-core omegaconf tqdm numpy matplotlib pillow opencv-python

# dashboard + metrics
pip install dash plotly scikit-learn

# detector
pip install ultralytics

# classifier
pip install torch torchvision

# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux:
python -m venv .venv
source .venv/bin/activate

# If split files are missing (train/val/test), generate them:
python -m src.data.create_splits

```

### Outputs are written under:
- runs/${experiment}/checkpoints
- runs/${experiment}/results


--- 

### Commands
> Trains ResNet and saves checkpoints to:
> runs/<experiment>/checkpoints/
> Output example:
> - runs/det_yolov8n_crop_resnet50/checkpoints/best_model.pth

- python -m src.classifier.train

> Evaluates validation and test sets and generates plots under:
> runs/<experiment>/results/classifier_eval_plots/

- python -m src.classifier.evaluate
> Generated plots include:
> Confusion matrix (counts)
> Confusion matrix (normalized)
> Per-class Precision / Recall / F1 bar charts
> Per-class “top confusions” charts:
> - - .../classifier_eval_plots/per_class_top_confusions/top_confusions__<class>.png
> The classifier evaluation plots are stored under:
> - runs/<experiment>/results/classifier_eval_plots/

> Starts the Dash server:
- python -m src.dashboard.app
> Then open the browser at:
> - http://127.0.0.1:8050 (default)
> Dashboard tabs:
> - Live Inference
> - Model Performance
> - Dataset Overview
> - Evaluation Plots (reads images from runs/<experiment>/results/classifier_eval_plots)


::contentReference[oaicite:0]{index=0}






