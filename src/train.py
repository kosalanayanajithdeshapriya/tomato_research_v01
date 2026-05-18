import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR       = "data/"
OUTPUT_DIR     = "outputs/"
MODEL_DIR      = os.path.join(OUTPUT_DIR, "models")
METRICS_PATH   = os.path.join(OUTPUT_DIR, "metrics.json")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_classifier.pth")
PLOTS_DIR      = os.path.join(OUTPUT_DIR, "plots")

IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
EPOCHS_PHASE1  = 30
EPOCHS_PHASE2  = 20
LR_PHASE1      = 0.0005
LR_PHASE2      = 1e-5
VAL_SPLIT      = 0.10
IS_CI          = os.getenv("CI", "false").lower() == "true"
MIN_ACCURACY   = 0.30 if IS_CI else 0.55
CLASS_NAMES    = ["developing", "flowering", "fruiting", "seeding"]
NUM_CLASSES    = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Transforms ────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, shear=20),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_data():
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    val_size     = max(1, int(len(full_dataset) * VAL_SPLIT))  # FIX: ensure at least 1 sample
    train_size   = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

    # FIX: num_workers=0 for CI compatibility (GitHub Actions can fail with num_workers>0)
    num_workers = 2 if DEVICE.type == "cuda" else 0

    train_loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, train_size),
                              shuffle=True, num_workers=num_workers,
                              pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=min(BATCH_SIZE, val_size),
                              shuffle=False, num_workers=num_workers,
                              pin_memory=(DEVICE.type == "cuda"))

    print(f"[INFO] Train: {train_size} | Val: {val_size}")
    print(f"[INFO] Classes: {full_dataset.classes}")
    return train_loader, val_loader, full_dataset.classes


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    print("[INFO] Building ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    return model.to(DEVICE)


# ── Train One Epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


# ── Validate ──────────────────────────────────────────────────────────────────
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels


# ── Main Training Loop ────────────────────────────────────────────────────────
def train():
    train_loader, val_loader, classes = load_data()
    model     = build_model()
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch   = 0
    history      = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    no_improve   = 0
    patience     = 7

    # ── Phase 1: Head only ────────────────────────────────────────────────
    print("\n[INFO] Phase 1 - Training classification head...")
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_PHASE1)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)  # FIX: removed deprecated verbose=True

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _ = validate(model, val_loader, criterion)
        scheduler.step(vl_loss)

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        print(f"Epoch {epoch}/{EPOCHS_PHASE1} - "
              f"loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
              f"val_loss: {vl_loss:.4f} val_acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), CNN_MODEL_PATH)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    # ── Phase 2: Fine-tune top ResNet layers ──────────────────────────────
    print("\n[INFO] Phase 2 - Fine-tuning top ResNet50 layers...")
    for name, param in model.named_parameters():
        if "layer4" in name or "layer3" in name or "fc" in name:
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2
    )
    scheduler  = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)  # FIX: removed deprecated verbose=True
    no_improve = 0
    patience   = 5

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, preds, labels = validate(model, val_loader, criterion)
        scheduler.step(vl_loss)

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        print(f"Epoch {epoch}/{EPOCHS_PHASE2} - "
              f"loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
              f"val_loss: {vl_loss:.4f} val_acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), CNN_MODEL_PATH)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    # FIX: weights_only=True to suppress security warning
    model.load_state_dict(torch.load(CNN_MODEL_PATH, weights_only=True))
    _, final_val_acc, final_preds, final_labels = validate(model, val_loader, criterion)

    return model, history, final_preds, final_labels, final_val_acc, classes


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import sys
    model, history, preds, labels, val_acc, classes = train()

    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall    = recall_score(labels, preds, average="weighted", zero_division=0)
    f1        = f1_score(labels, preds, average="weighted", zero_division=0)

    # FIX: labels=list(range(NUM_CLASSES)) ensures all 4 classes always reported
    #      even when CI val set has fewer than 4 classes present
    print("\n" + classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    ))

    metrics = {
        "train_accuracy": round(float(history["train_acc"][-1]), 4),
        "val_accuracy":   round(float(val_acc), 4),
        "precision":      round(float(precision), 4),
        "recall":         round(float(recall), 4),
        "f1_score":       round(float(f1), 4),
        "total_epochs":   len(history["train_acc"]),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n── RESULTS ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    status = "PASS" if val_acc >= MIN_ACCURACY else "FAIL"
    print(f"\n[{status}] val_accuracy {val_acc:.4f} "
          f"{'≥' if val_acc >= MIN_ACCURACY else '<'} threshold {MIN_ACCURACY}")
    print("[INFO] Training complete.")

    if val_acc < MIN_ACCURACY:
        sys.exit(1)


if __name__ == "__main__":
    main()