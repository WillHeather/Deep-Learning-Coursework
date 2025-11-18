# Auto-generated from notebook.ipynb

# Cell 0
# Setup & Imports
!pip install timm -q

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import timm
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import copy
from pathlib import Path
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Data paths
DATA_DIR = Path('./data')
TRAIN_DIR = Path('./data/train')
TEST_DIR = Path('./data/test')

# Initialize global variables (will be updated based on actual data)
INPUT_CHANNELS = 3  # Default to RGB
IS_GRAYSCALE = False
BATCH_SIZE = 32

print('Setup complete!')

# Cell 1
# Load Training Data
def load_data(data_dir):
    """Load image paths and labels from directory structure"""
    paths, labels = [], []
    for lbl in [0, 1]:
        folder = data_dir / str(lbl)
        if folder.exists():
            for ext in ['*.jpg', '*.png']:
                for p in folder.glob(ext):
                    paths.append(str(p))
                    labels.append(lbl)
    return paths, labels

# Load training data
if TRAIN_DIR.exists():
    train_paths, train_labels = load_data(TRAIN_DIR)
    train_df = pd.DataFrame({'path': train_paths, 'label': train_labels})
    print(f'Loaded {len(train_df)} training images')
    print('\nClass distribution:')
    print(train_df['label'].value_counts().sort_index())
else:
    train_df = pd.DataFrame()
    print('Training data directory not found!')

# Cell 2
# Visualize Class Distribution
if len(train_df) > 0:
    plt.figure(figsize=(8, 4))
    train_df['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class (0=Female, 1=Male)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

# Cell 3
# Detect Image Properties
if len(train_df) > 0:
    # Check first image to determine if grayscale or RGB
    sample_img = Image.open(train_df.iloc[0]['path'])
    IS_GRAYSCALE = sample_img.mode == 'L'
    INPUT_CHANNELS = 1 if IS_GRAYSCALE else 3
    
    print(f'Image mode: {sample_img.mode}')
    print(f'Input channels: {INPUT_CHANNELS} ({"Grayscale" if IS_GRAYSCALE else "RGB"})')
    print(f'Image size: {sample_img.size}')

# Cell 4
# Define Transforms
if len(train_df) > 0:
    # Normalization parameters
    mean_std = ([0.5], [0.5]) if IS_GRAYSCALE else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])
    
    print('Transforms defined:')

# Cell 5
# Create Dataset Class and DataLoaders
class FootprintDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L' if IS_GRAYSCALE else 'RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

if len(train_df) > 0:
    # Split into train and validation sets (80/20)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_df['path'].tolist(),
        train_df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=train_df['label']
    )
    
    # Create datasets
    train_dataset = FootprintDataset(train_paths, train_labels, train_transform)
    val_dataset = FootprintDataset(val_paths, val_labels, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Batch size: {BATCH_SIZE}')

# Cell 6
# Define Baseline CNN Architecture
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

print('Baseline CNN architecture defined')

# Cell 7
# Training and Evaluation Functions
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

print('Training functions defined')

# Cell 8
# Generic Training Loop Function
def train_model(model, train_loader, val_loader, config, device):
    """
    Generic training function that works with any model and config
    
    config should contain: epochs, lr, optimizer ('sgd' or 'adam'), weight_decay (optional)
    """
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], 
                             momentum=0.9, weight_decay=config.get('weight_decay', 0))
    else:  # adam
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                              weight_decay=config.get('weight_decay', 0))
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    print(f'\nBest validation accuracy: {best_val_acc:.4f}')
    
    return model, history, best_val_acc

print('Generic training loop defined')

# Cell 9
# Train Baseline Model
if len(train_df) > 0:
    # Configuration for baseline
    baseline_config = {
        'epochs': 10,
        'lr': 0.001,
        'optimizer': 'sgd'
    }
    
    # Create and train model
    baseline_model = BaselineCNN(num_classes=2, input_channels=INPUT_CHANNELS).to(device)
    baseline_model, baseline_hist, baseline_acc = train_model(
        baseline_model, train_loader, val_loader, baseline_config, device
    )
    
    # Initialize results tracking
    experiment_results = [{
        'name': 'Baseline',
        'val_accuracy': baseline_acc,
        'val_loss': baseline_hist['val_loss'][-1]
    }]
    
    print(f'\nBaseline model training complete!')
    print(f'Final validation accuracy: {baseline_acc:.4f}')

# Cell 10
# Plot Learning Curves
if len(train_df) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(baseline_hist['train_loss'], 'o-', label='Train')
    ax1.plot(baseline_hist['val_loss'], 's-', label='Validation')
    ax1.set_title('Baseline Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(baseline_hist['train_acc'], 'o-', label='Train')
    ax2.plot(baseline_hist['val_acc'], 's-', label='Validation')
    ax2.set_title('Baseline Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Cell 11
# Helper function to count parameters
def count_params(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Parameter counting function defined')

# Cell 12
# Model 1: ResNet-18
print('='*60)
print('1. ResNet-18')
print('='*60)

r18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Modify first conv layer if grayscale
if INPUT_CHANNELS == 1:
    r18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Replace final layer for binary classification
r18.fc = nn.Linear(r18.fc.in_features, 2)

print(f'Parameters: {count_params(r18):,}')
print(f'Input channels: {INPUT_CHANNELS}')
print(f'Output classes: 2')
print('\nKey features:')
print('- Residual connections (skip connections)')
print('- Deep architecture (18 layers)')
print('- Pre-trained on ImageNet')

# Cell 13
# Model 2: EfficientNet-B0
print('='*60)
print('2. EfficientNet-B0')
print('='*60)

eff = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2, in_chans=INPUT_CHANNELS)

print(f'Parameters: {count_params(eff):,}')
print(f'Input channels: {INPUT_CHANNELS}')
print(f'Output classes: 2')
print('\nKey features:')
print('- Compound scaling (depth, width, resolution)')
print('- Mobile inverted bottleneck convolutions')
print('- Efficient architecture, fewer params than ResNet')
print('- Pre-trained on ImageNet')

# Cell 14
# Model 3: Vision Transformer (ViT)
print('='*60)
print('3. Vision Transformer (ViT-Base/16)')
print('='*60)

vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2, in_chans=INPUT_CHANNELS)

print(f'Parameters: {count_params(vit):,}')
print(f'Input channels: {INPUT_CHANNELS}')
print(f'Output classes: 2')
print('\nKey features:')
print('- Transformer architecture (attention-based)')
print('- Patch-based processing (16x16 patches)')
print('- No convolutions - pure attention')
print('- Pre-trained on ImageNet')
print('- May require more data to train effectively')

# Cell 15
# Define additional model architectures for experiments
class CNN_BatchNorm(nn.Module):
    """CNN with Batch Normalization"""
    def __init__(self, num_classes=2, input_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNN_Dropout(nn.Module):
    """CNN with Dropout regularization"""
    def __init__(self, num_classes=2, input_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

print('Additional model architectures defined')

# Cell 16
# Experiment 1: Adam Optimizer
if len(train_df) > 0:
    print('='*60)
    print('Experiment 1: Adam Optimizer')
    print('='*60)
    
    exp1_config = {
        'epochs': 10,
        'lr': 0.001,
        'optimizer': 'adam'
    }
    
    exp1_model = BaselineCNN(num_classes=2, input_channels=INPUT_CHANNELS).to(device)
    exp1_model, exp1_hist, exp1_acc = train_model(exp1_model, train_loader, val_loader, exp1_config, device)
    
    experiment_results.append({
        'name': 'Exp1: Adam',
        'val_accuracy': exp1_acc,
        'val_loss': exp1_hist['val_loss'][-1]
    })

# Cell 17
# Experiment 2: Batch Normalization
if len(train_df) > 0:
    print('='*60)
    print('Experiment 2: Batch Normalization')
    print('='*60)
    
    exp2_config = {
        'epochs': 10,
        'lr': 0.001,
        'optimizer': 'sgd'
    }
    
    exp2_model = CNN_BatchNorm(num_classes=2, input_channels=INPUT_CHANNELS).to(device)
    exp2_model, exp2_hist, exp2_acc = train_model(exp2_model, train_loader, val_loader, exp2_config, device)
    
    experiment_results.append({
        'name': 'Exp2: BatchNorm',
        'val_accuracy': exp2_acc,
        'val_loss': exp2_hist['val_loss'][-1]
    })

# Cell 18
# Experiments 3-10
if len(train_df) > 0:
    # Exp 3: Dropout
    print('\n' + '='*60)
    print('Experiment 3: Dropout Regularization')
    print('='*60)
    exp3_config = {'epochs': 10, 'lr': 0.001, 'optimizer': 'sgd'}
    exp3_model = CNN_Dropout(num_classes=2, input_channels=INPUT_CHANNELS).to(device)
    exp3_model, exp3_hist, exp3_acc = train_model(exp3_model, train_loader, val_loader, exp3_config, device)
    experiment_results.append({'name': 'Exp3: Dropout', 'val_accuracy': exp3_acc, 'val_loss': exp3_hist['val_loss'][-1]})
    
    # Exp 4: Weight Decay
    print('\n' + '='*60)
    print('Experiment 4: Weight Decay (L2 Regularization)')
    print('='*60)
    exp4_config = {'epochs': 10, 'lr': 0.001, 'optimizer': 'sgd', 'weight_decay': 1e-4}
    exp4_model = CNN_Dropout(num_classes=2, input_channels=INPUT_CHANNELS).to(device)
    exp4_model, exp4_hist, exp4_acc = train_model(exp4_model, train_loader, val_loader, exp4_config, device)
    experiment_results.append({'name': 'Exp4: WeightDecay', 'val_accuracy': exp4_acc, 'val_loss': exp4_hist['val_loss'][-1]})
    
    # Exp 5: Adam + Dropout + Weight Decay
    print('\n' + '='*60)
    print('Experiment 5: Adam + Dropout + Weight Decay')
    print('='*60)
    exp5_config = {'epochs': 10, 'lr': 0.001, 'optimizer': 'adam', 'weight_decay': 1e-4}
    exp5_model = CNN_Dropout(num_classes=2, input_channels=INPUT_CHANNELS).to(device)
    exp5_model, exp5_hist, exp5_acc = train_model(exp5_model, train_loader, val_loader, exp5_config, device)
    experiment_results.append({'name': 'Exp5: Adam+Drop+WD', 'val_accuracy': exp5_acc, 'val_loss': exp5_hist['val_loss'][-1]})
    
    # Exp 6: ResNet-18 Frozen
    print('\n' + '='*60)
    print('Experiment 6: ResNet-18 (Frozen Features)')
    print('='*60)
    exp6_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if INPUT_CHANNELS == 1:
        exp6_model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    for param in exp6_model.parameters():
        param.requires_grad = False
    exp6_model.fc = nn.Linear(exp6_model.fc.in_features, 2)
    exp6_model = exp6_model.to(device)
    exp6_config = {'epochs': 10, 'lr': 0.001, 'optimizer': 'adam'}
    exp6_model, exp6_hist, exp6_acc = train_model(exp6_model, train_loader, val_loader, exp6_config, device)
    experiment_results.append({'name': 'Exp6: ResNet Frozen', 'val_accuracy': exp6_acc, 'val_loss': exp6_hist['val_loss'][-1]})
    
    # Exp 7: ResNet-18 Fine-tuned
    print('\n' + '='*60)
    print('Experiment 7: ResNet-18 (Fine-tuned)')
    print('='*60)
    exp7_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if INPUT_CHANNELS == 1:
        exp7_model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    exp7_model.fc = nn.Linear(exp7_model.fc.in_features, 2)
    exp7_model = exp7_model.to(device)
    exp7_config = {'epochs': 10, 'lr': 0.0001, 'optimizer': 'adam', 'weight_decay': 1e-4}
    exp7_model, exp7_hist, exp7_acc = train_model(exp7_model, train_loader, val_loader, exp7_config, device)
    experiment_results.append({'name': 'Exp7: ResNet Finetuned', 'val_accuracy': exp7_acc, 'val_loss': exp7_hist['val_loss'][-1]})
    
    # Exp 8: EfficientNet-B0 (Save as final model)
    print('\n' + '='*60)
    print('Experiment 8: EfficientNet-B0')
    print('='*60)
    exp8_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2, in_chans=INPUT_CHANNELS).to(device)
    exp8_config = {'epochs': 10, 'lr': 0.0001, 'optimizer': 'adam', 'weight_decay': 1e-4}
    exp8_model, exp8_hist, exp8_acc = train_model(exp8_model, train_loader, val_loader, exp8_config, device)
    experiment_results.append({'name': 'Exp8: EfficientNet', 'val_accuracy': exp8_acc, 'val_loss': exp8_hist['val_loss'][-1]})
    final_model = exp8_model  # Save as final model
    
    # Exp 9: Higher Learning Rate
    print('\n' + '='*60)
    print('Experiment 9: Higher Learning Rate (0.01)')
    print('='*60)
    exp9_config = {'epochs': 10, 'lr': 0.01, 'optimizer': 'sgd'}
    exp9_model = CNN_Dropout(num_classes=2, input_channels=INPUT_CHANNELS).to(device)
    exp9_model, exp9_hist, exp9_acc = train_model(exp9_model, train_loader, val_loader, exp9_config, device)
    experiment_results.append({'name': 'Exp9: HighLR', 'val_accuracy': exp9_acc, 'val_loss': exp9_hist['val_loss'][-1]})
    
    # Exp 10: Longer Training
    print('\n' + '='*60)
    print('Experiment 10: Longer Training (20 epochs)')
    print('='*60)
    exp10_config = {'epochs': 20, 'lr': 0.001, 'optimizer': 'adam'}
    exp10_model = CNN_Dropout(num_classes=2, input_channels=INPUT_CHANNELS).to(device)
    exp10_model, exp10_hist, exp10_acc = train_model(exp10_model, train_loader, val_loader, exp10_config, device)
    experiment_results.append({'name': 'Exp10: LongTrain', 'val_accuracy': exp10_acc, 'val_loss': exp10_hist['val_loss'][-1]})
    
    print('\n' + '='*80)
    print('All experiments complete!')
    print('='*80)

# Cell 19
# Display Experiment Results Summary
if len(train_df) > 0:
    results_df = pd.DataFrame(experiment_results).sort_values('val_accuracy', ascending=False)
    print('\n' + '='*80)
    print('EXPERIMENT RESULTS SUMMARY (Sorted by Validation Accuracy)')
    print('='*80)
    display(results_df)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df_sorted = results_df.sort_values('val_accuracy')
    ax.barh(results_df_sorted['name'], results_df_sorted['val_accuracy'])
    ax.set_xlabel('Validation Accuracy')
    ax.set_title('Experiment Comparison')
    ax.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

# Cell 20
# Identify Best Model
if len(train_df) > 0:
    best_experiment = max(experiment_results, key=lambda x: x['val_accuracy'])
    print('='*80)
    print('BEST MODEL')
    print('='*80)
    print(f"Name: {best_experiment['name']}")
    print(f"Validation Accuracy: {best_experiment['val_accuracy']:.4f}")
    print(f"Validation Loss: {best_experiment['val_loss']:.4f}")
    print('\nUsing this model for final evaluation...')

# Cell 21
# Detailed Metrics on Validation Set
if len(train_df) > 0:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, classification_report
    
    criterion = nn.CrossEntropyLoss()
    final_loss, final_acc, final_preds, final_labels = evaluate(final_model, val_loader, criterion, device)
    
    print('='*80)
    print('DETAILED METRICS')
    print('='*80)
    print(f'Accuracy: {final_acc:.4f}')
    print(f'Loss: {final_loss:.4f}')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(final_labels, final_preds, average=None)
    print('\nPer-Class Metrics:')
    for i in range(2):
        print(f'  Class {i} ({"Female" if i==0 else "Male"}):')
        print(f'    Precision: {precision[i]:.4f}')
        print(f'    Recall: {recall[i]:.4f}')
        print(f'    F1-Score: {f1[i]:.4f}')
        print(f'    Support: {support[i]}')
    
    # Macro averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(final_labels, final_preds, average='macro')
    print(f'\nMacro Averages:')
    print(f'  Precision: {macro_p:.4f}')
    print(f'  Recall: {macro_r:.4f}')
    print(f'  F1-Score: {macro_f1:.4f}')

# Cell 22
# Confusion Matrix
if len(train_df) > 0:
    cm = confusion_matrix(final_labels, final_preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Female', 'Male'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Final Model')
    plt.show()
    
    print('\nConfusion Matrix Analysis:')
    print(f'True Negatives (Correctly classified Female): {cm[0,0]}')
    print(f'False Positives (Female classified as Male): {cm[0,1]}')
    print(f'False Negatives (Male classified as Female): {cm[1,0]}')
    print(f'True Positives (Correctly classified Male): {cm[1,1]}')

# Cell 23
# Generate Kaggle Submission
if len(train_df) > 0 and TEST_DIR.exists():
    # Load test images
    test_paths = sorted(list(TEST_DIR.glob('*.jpg')) + list(TEST_DIR.glob('*.png')))
    
    if len(test_paths) > 0:
        print('='*80)
        print('GENERATING KAGGLE SUBMISSION')
        print('='*80)
        print(f'Found {len(test_paths)} test images')
        
        # Get test image IDs from filenames
        test_ids = [p.stem for p in test_paths]
        test_paths_str = [str(p) for p in test_paths]
        
        # Create test dataset and loader
        test_dataset = FootprintDataset(test_paths_str, [0]*len(test_paths_str), val_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        # Generate predictions
        final_model.eval()
        test_predictions = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(test_loader, desc='Predicting test set'):
                inputs = inputs.to(device)
                outputs = final_model(inputs)
                _, predicted = outputs.max(1)
                test_predictions.extend(predicted.cpu().numpy())
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id': test_ids,
            'label': test_predictions
        })
        
        # Save to CSV
        submission_df.to_csv('submission.csv', index=False)
        
        print(f'\nSubmission file saved: submission.csv')
        print(f'Total predictions: {len(submission_df)}')
        print('\nFirst few predictions:')
        display(submission_df.head(10))
        
        # Show prediction distribution
        print(f'\nPrediction distribution:')
        print(submission_df['label'].value_counts().sort_index())
    else:
        print('No test images found in test directory')
else:
    if len(train_df) == 0:
        print('Training data not loaded - cannot generate submission')
    else:
        print('Test directory not found - cannot generate submission')

