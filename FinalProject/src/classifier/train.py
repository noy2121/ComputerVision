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

    info = classifier.data_processor.get_dataset_info()
    print(f"\nDataset Info:")
    print(f"  Train: {info['train_size']} images")
    print(f"  Val: {info['val_size']} images")
    print(f"  Test: {info['test_size']} images")
    print(f"  Classes: {info['num_classes']}")

    print("\nSetting up training...")
    classifier.setup_training()

    print("\nStarting training...")
    classifier.train()

    print("\n" + "="*60)
    print("Running final test evaluation...")
    print("="*60)
    test_acc, predictions, labels = classifier.test()
    print(f"\nTest Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
