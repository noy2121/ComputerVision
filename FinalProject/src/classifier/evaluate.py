import hydra
from omegaconf import DictConfig

from src.classifier import Classifier


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
    classifier.load_weights('best_model.pth')

    print("\nRunning evaluation on validation set...")
    val_loss, val_acc = classifier.evaluate()
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    print("\nRunning evaluation on test set...")
    test_acc, predictions, labels = classifier.test()
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
