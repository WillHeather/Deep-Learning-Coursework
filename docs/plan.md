🔧 Global Constraints For Claude / The Notebook

Tell Claude something like:

Use PyTorch only (no TensorFlow/Keras).

Assume the notebook runs on Google Colab.

At the top, include a markdown cell where I will later fill in:

Kaggle username

Final private leaderboard score

Number of Kaggle submissions

Keep all code in one notebook.

Use functions and config dictionaries where possible so each experiment only tweaks parameters.

Any long explanations should be outline text / bullet prompts that I’ll rewrite myself later.

📚 Notebook Structure

Below is the structure + what each section should contain (code + markdown).
You can pass this as a structured spec.

0. Setup & Imports

Markdown:

Brief note: “This section sets up libraries, paths and environment for Colab.”

Code:

pip install (if using timm or anything extra).

Imports:

torch, torch.nn, torch.optim, torchvision, torchvision.transforms, etc.

timm (if used)

numpy, pandas, matplotlib.pyplot

sklearn.metrics

Device selection:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Set random seeds:

def set_seed(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)


Define data paths for Kaggle dataset (assume something like train/ and test/ directories; I’ll adjust paths later).

1. Introduction (Markdown Only)

Markdown cells:

Heading: # 1. Introduction

Prompts for me to fill later:

Problem statement (sex classification from footprint images).

Forensic/business context.

Why deep learning vs traditional ML.

Project objectives (baseline, experiments, SOTA model, explainability, Kaggle).

Claude should only put bullet-point prompts, not final polished text.

2. EDA & Preprocessing
2.1 Load Metadata

Code:

Load training CSV if exists (e.g., train.csv with id and label), or build dataframe from filenames + labels folder structure.

Show first few rows with df.head().

2.2 Basic Dataset Statistics

Code:

Inspect:

Number of images per class (value counts).

Plot bar chart of class distribution.

Sample random few image file paths, open with PIL, print:

Image size

Mode (RGB/Grayscale)

2.3 Visualise Sample Images

Code:

Utility function show_grid(images, labels) using matplotlib.

Show:

Some examples from class 0

Some examples from class 1

2.4 Ethical Bias & Data Issues (Markdown)

Markdown:

Prompts for me:

Comment on class imbalance (from bar chart).

Mention potential bias (e.g., dataset demographics, imprint differences).

2.5 Transforms & Datasets

Code:

Define transforms:

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # adjust if RGB
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


Custom FootprintDataset(torch.utils.data.Dataset):

Reads image path + label.

Converts to tensor with given transform.

Create train/validation split (e.g., train_test_split or manual 80/20).

Create DataLoaders:

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

3. Baseline Model (From Scratch)
3.1 Define Baseline CNN

Code:

Simple CNN model class:

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1 or 3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (224//8) * (224//8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


(I’ll adjust input channels depending on grayscale/RGB.)

3.2 Generic Training & Evaluation Functions

Code:

train_one_epoch(model, loader, criterion, optimizer, device)

evaluate(model, loader, criterion, device) returning loss + accuracy.

train_model(model, train_loader, val_loader, config):

Loop over epochs

Track history (train/val loss & accuracy)

Save best model weights based on val accuracy

Return model, history

3.3 Train Baseline

Code:

Define config dict:

baseline_config = {
    "epochs": 10,
    "lr": 0.001,
    "batch_size": 32,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0,
}


Create model, optimizer, loss (nn.CrossEntropyLoss()).

Train baseline model.

3.4 Plot Learning Curves & Record Baseline

Code:

Plot train/val accuracy and loss curves.

Save baseline metrics for comparison (e.g., list/dict).

Markdown:

Short placeholder for me to later write:

Description of baseline performance

Comments on under/overfitting.

4. State-of-the-Art Model Analysis (Theory + Light Code)
4.1 Choose 3 Architectures

Suggested architectures:

ResNet-18 (torchvision)

EfficientNet-B0 (timm)

ViT-B/16 (timm)

4.2 Model Summary Code

Code:

Small helper to count parameters:

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


Instantiate each model (with pretrained=True or weights= for torchvision).

Print:

Model name

Number of parameters

Expected input size

Rough note on FLOPs if using library or comment placeholder.

Markdown:

For each model, create bullets for:

Key features (residual connections, depthwise convs, attention, etc.).

Pros/cons for this task.

Leave as prompts for me to expand.

5. Systematic Experiments (10 Distinct Experiments)

Design notebook so that each experiment reuses the same training function but with a different config and/or model.

Add a section:

# 5. Systematic Experimentation for Model Improvement

For each experiment:

Subheading: ## Experiment X – <Title>

Markdown:

Hypothesis: placeholder sentence.

Change: what is being modified vs baseline.

Code:

Define config or model change.

Train model (maybe fewer epochs for quick runs).

Evaluate on validation set.

Store results in a global results list or dataframe.

Suggested experiment list:

Optimizer: SGD → Adam

Same baseline CNN, change optimizer.

Compare accuracy vs baseline.

Add Batch Normalization

Modify CNN to include nn.BatchNorm2d after conv layers.

Add Dropout in Fully Connected Layers

Insert nn.Dropout(p=0.5) before final Linear layers.

Increase Network Depth

Add an extra Conv block (Conv→ReLU→Pool).

Stronger Data Augmentation

Add stronger transforms (RandomAffine, RandomPerspective, maybe ColorJitter if RGB).

Keep model same as best from Experiments 2–4.

Learning Rate Scheduler

Use StepLR or ReduceLROnPlateau.

Weight Decay (L2 Regularisation)

Add weight_decay in optimizer.

Pretrained ResNet18 – Frozen Feature Extractor

Replace CNN with resnet18(pretrained=True), replace final FC.

Freeze backbone, train classifier head only.

Pretrained ResNet18 – Fine-tuning Deeper Layers

Unfreeze last block or all layers; train with smaller LR.

Higher Input Resolution / Different Pretrained Model (e.g., EfficientNet-B0)

Change input size to 256×256 or 288×288.

Use EfficientNet from timm with transfer learning.

For each experiment, Claude should:

Log:

Best validation accuracy

Validation loss

Optional: F1-score if cheap to compute.

Append to a results list:

experiment_results.append({
    "name": "Experiment X - ...",
    "val_accuracy": best_acc,
    "val_loss": best_loss,
    "notes": "short code-level description"
})


At the end of Section 5:

Code: create a pandas DataFrame from experiment_results and display it, sorted by accuracy.

Markdown: Placeholder for my comparative analysis.

6. Final Model Evaluation & Explainability (XAI)
6.1 Select Best Model

Code:

Pick experiment with highest val accuracy from experiment_results.

Rebuild that model configuration, load best weights (or retrain with full settings if needed).

6.2 Detailed Metrics

Code:

Run on val_loader to collect all predictions and true labels.

Compute:

Accuracy

Precision, Recall, F1 (macro and per-class) using sklearn.metrics.

Plot:

Confusion matrix using ConfusionMatrixDisplay.

6.3 XAI – Grad-CAM

Code:

Implement Grad-CAM for the final model:

Target last conv layer or relevant feature map.

Choose top N misclassified or high-loss samples.

Generate heatmaps and overlay on original images.

Show images + heatmaps in a grid.

Markdown:

Prompts for me to interpret:

Where is the model focusing?

Is it actually looking at the footprint?

Any suspicious patterns?

6.4 Reliability & Forensic Use

Markdown:

Bullet prompts:

Risks of false positives/negatives.

Potential bias and fairness issues.

Why model should be advisory only, not sole evidence.

Need for calibration, human oversight, and external validation.

7. Conclusion & Reflection

Markdown:

Prompts:

Summarise performance improvements from baseline → final.

Which experiment gave the biggest gain?

Limitations (data size, overfitting, generalisation).

Would I deploy? Under what conditions?

Future work: e.g., more data, cross-validation, robust testing, better XAI, calibration.

8. Kaggle Submission Generation

Code:

Build DataLoader for test images with val_test_transform.

Use final model to predict class probabilities/logits.

Take argmax as predicted label.

Create a DataFrame:

submission = pd.DataFrame({
    "id": test_ids,
    "label": preds
})
submission.to_csv("submission.csv", index=False)


Brief markdown note: “This CSV can be uploaded to Kaggle.”

9. Top-of-Notebook Summary Cell

Markdown at very top:

# Deep Learning Kaggle Competition: Footprint Image Classification

- **Student Name:** [TO FILL]
- **Student ID:** [TO FILL]
- **Kaggle Username:** [TO FILL]
- **Final Private Leaderboard Score:** [TO FILL]
- **Total Number of Submissions:** [TO FILL]


(I’ll fill this in after the competition ends.)