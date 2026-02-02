from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from src.data import DataProcessor


class Classifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.data_processor = None
        self.optimizer = None
        self.criterion = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def build_model(self):
        if self.cfg.model.name == 'resnet50':
            self.model = models.resnet50(pretrained=self.cfg.model.pretrained)
        elif self.cfg.model.name == 'resnet18':
            self.model = models.resnet18(pretrained=self.cfg.model.pretrained)
        else:
            raise ValueError(f"Model {self.cfg.model.name} not supported")

        if self.cfg.model.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.cfg.model.num_classes)

        self.model = self.model.to(self.device)

    def setup_data(self):
        self.data_processor = DataProcessor(self.cfg)
        self.data_processor.setup()

        self.train_loader = self.data_processor.train_dataloader()
        self.val_loader = self.data_processor.val_dataloader()
        self.test_loader = self.data_processor.test_dataloader()

    def setup_training(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100. * correct / total
        return accuracy, all_predictions, all_labels

    def train(self):
        best_val_acc = 0.0
        patience_counter = 0
        patience = self.cfg.training.get('early_stopping_patience', 10)
        min_delta = self.cfg.training.get('early_stopping_min_delta', 0.001)

        for epoch in range(self.cfg.training.epochs):
            print(f'\nEpoch {epoch+1}/{self.cfg.training.epochs}')

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()

            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_weights('best_model.pth')
                print(f'Best model saved with validation accuracy: {val_acc:.2f}%')
            else:
                patience_counter += 1
                print(f'No improvement for {patience_counter} epoch(s)')

            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                print(f'Best validation accuracy: {best_val_acc:.2f}%')
                break

    def save_weights(self, filename):
        checkpoint_dir = Path(self.cfg.paths.checkpoints)
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg
        }, checkpoint_path)

        print(f'Checkpoint saved to {checkpoint_path}')

    def load_weights(self, filename):
        checkpoint_dir = Path(self.cfg.paths.checkpoints)
        checkpoint_path = checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f'Checkpoint loaded from {checkpoint_path}')
