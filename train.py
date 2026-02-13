# train.py
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------
# Configuration / Hyperparameters
# -------------------------------
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams (tweak as needed)
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
IMG_SIZE = 224
TRAIN_VAL_SPLIT = 0.8  # 0.8 train, 0.2 val

DATA_ROOT = Path("dataset")
ALL_DIR = DATA_ROOT / "all"     # should contain `healthy/` and `diseased/`
TRAIN_DIR = DATA_ROOT / "train" # created automatically if missing
VAL_DIR = DATA_ROOT / "val"     # created automatically if missing

BEST_MODEL_PATH = "best_model.pth"
FINAL_MODEL_PATH = "model_weights.pth"
HISTORY_CSV = "training_history.csv"

# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behavior (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# -------------------------------
# Prepare train/val folders (if needed)
# -------------------------------
def prepare_split(all_dir=ALL_DIR, train_dir=TRAIN_DIR, val_dir=VAL_DIR, split=TRAIN_VAL_SPLIT):
    """
    If train/val folders already exist, do nothing.
    Otherwise, reads images from `all_dir/healthy` and `all_dir/diseased`
    and splits them into train/ and val/ preserving class folders.
    """
    if train_dir.exists() and val_dir.exists():
        print("Train/Val directories already exist — skipping split.")
        return

    if not all_dir.exists():
        raise FileNotFoundError(f"Expected dataset/all with 'healthy' and 'diseased' subfolders. Not found: {all_dir}")

    classes = [p.name for p in all_dir.iterdir() if p.is_dir()]
    if not classes:
        raise FileNotFoundError(f"No class subfolders found in {all_dir}. Expected 'healthy' and 'diseased'.")

    print(f"Creating train/val split from {all_dir} with classes: {classes}")

    # create target dirs
    for d in [train_dir, val_dir]:
        for cl in classes:
            (d / cl).mkdir(parents=True, exist_ok=True)

    # split per class
    for cl in classes:
        imgs = list((all_dir / cl).glob("*"))
        imgs = [p for p in imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        random.shuffle(imgs)
        n_train = int(len(imgs) * split)
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:]

        for src in train_imgs:
            dst = train_dir / cl / src.name
            if not dst.exists():
                shutil.copy(src, dst)

        for src in val_imgs:
            dst = val_dir / cl / src.name
            if not dst.exists():
                shutil.copy(src, dst)

    print("Split complete. Train/Val folders created.")

#prepare_split()

# -------------------------------
# Transforms and Dataloaders
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(8),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.02),  # small chance, optional
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_data = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Classes: {train_data.classes}")
print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

# -------------------------------
# Model, Loss, Optimizer, Scheduler
# -------------------------------
model = models.resnet34(pretrained=True)
# If you want to freeze feature extractor at start:
# for param in model.parameters():
#     param.requires_grad = False
# Then unfreeze final layer(s). Here we fine-tune entire network:
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  #reduce LR on schedule

# Mixed precision scaler (optional, speeds up on modern GPUs)
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# -------------------------------
# Training loop with metrics & save best
# -------------------------------
best_val_acc = 0.0
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Train", leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_data)
    epoch_acc = 100.0 * running_corrects / len(train_data)

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_running_corrects += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = val_running_loss / len(val_data)
    val_acc = 100.0 * val_running_corrects / len(val_data)

    # Metrics
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=train_data.class_to_idx.get('diseased', 1))
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=train_data.class_to_idx.get('diseased', 1))
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=train_data.class_to_idx.get('diseased', 1))
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | Precision: {precision:.3f} Recall: {recall:.3f} F1: {f1:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    history.append({
        "epoch": epoch,
        "train_loss": epoch_loss,
        "train_acc": epoch_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc
        }, BEST_MODEL_PATH)
        print(f"--> New best model saved (val_acc: {val_acc:.2f}%)")

    scheduler.step()

# Save final model weights
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"Final model saved to: {FINAL_MODEL_PATH}")

# Save history to CSV
try:
    import csv
    keys = history[0].keys()
    with open(HISTORY_CSV, "w", newline="") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(history)
    print(f"Training history saved to {HISTORY_CSV}")
except Exception as e:
    print("Could not save history:", e)

print("Training complete.")
