import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.classifier import Classifier
from src.classifier.eval_plots import save_classifier_eval_plots  # NEW


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    print("Initializing Classifier...")
    classifier = Classifier(cfg)

    print("\nBuilding model...")
    classifier.build_model()

    print("\nSetting up data...")
    classifier.setup_data()

    print("\nLoading weights...")
    classifier.setup_training()
    # classifier.load_weights('best_model.pth')
    classifier.load_weights(cfg.classifier.checkpoint)

    print("\nRunning evaluation on validation set...")
    val_loss, val_acc = classifier.evaluate()
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    print("\nRunning evaluation on test set...")
    test_acc, predictions, labels = classifier.test()
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Build class_names from DataProcessor mapping (index order matters!)
    class_to_idx = classifier.data_processor.class_to_idx
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Save plots
    plots_dir = Path(cfg.paths.results) / "classifier_eval_plots"
    # Save plots (function creates classifier_eval_plots/ inside out_dir)
    save_classifier_eval_plots(
        cfg=cfg,
        labels=labels,
        predictions=predictions,
        class_names=class_names,
        out_dir=str(cfg.paths.results),
    )

    plots_dir = Path(cfg.paths.results) / "classifier_eval_plots"
    print(f"\nSaved evaluation plots to: {plots_dir}")


if __name__ == "__main__":
    main()
