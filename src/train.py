import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR       = "data/"
OUTPUT_DIR     = "outputs/"
MODEL_DIR      = os.path.join(OUTPUT_DIR, "models")
METRICS_PATH   = os.path.join(OUTPUT_DIR, "metrics.json")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_classifier.pth")
PLOTS_DIR      = os.path.join(OUTPUT_DIR, "plots")

IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
EPOCHS_PHASE1   = 30
EPOCHS_PHASE2   = 20
LR_PHASE1       = 0.0005
LR_PHASE2       = 1e-5
MIN_ACCURACY    = 0.35
CLASS_NAMES     = ["developing", "flowering", "fruiting", "seeding"]
NUM_CLASSES     = len(CLASS_NAMES)

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

# ── Data Loader (Stratified 70 | 20 | 10) ────────────────────────────────────
def load_data():
    print("[INFO] Loading data with stratified 70/20/10 split...")

    full_train_ds = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    full_val_ds   = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

    labels  = full_train_ds.targets        # class index for every image
    indices = list(range(len(full_train_ds)))

    # Step 1: Split 10% test from 90% train+val (stratified)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.10,
        stratify=labels,
        random_state=42
    )

    # Step 2: Split remaining 90% → 70% train / 20% val (stratified)
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.222,           # 0.222 × 90% ≈ 20% of total
        stratify=train_val_labels,
        random_state=42
    )

    # Create subsets — train gets augmentation, val/test do not
    train_ds = Subset(full_train_ds, train_idx)
    val_ds   = Subset(full_val_ds,   val_idx)
    test_ds  = Subset(full_val_ds,   test_idx)

    # Print per-class distribution for verification
    print(f"\n{'Split':<8} {'Total':>6}  ", end="")
    for cls in full_train_ds.classes:
        print(f"{cls:>12}", end="")
    print()
    print("-" * 60)
    for split_name, split_idx in [("Train", train_idx),
                                   ("Val",   val_idx),
                                   ("Test",  test_idx)]:
        split_labels = [labels[i] for i in split_idx]
        print(f"{split_name:<8} {len(split_idx):>6}  ", end="")
        for cls_idx in range(NUM_CLASSES):
            count = split_labels.count(cls_idx)
            print(f"{count:>12}", end="")
        print()
    print()

    # Save split indices so evaluate.py can reuse exact same test set
    split_info = {
        "train_idx": train_idx,
        "val_idx":   val_idx,
        "test_idx":  test_idx
    }
    with open(os.path.join(OUTPUT_DIR, "split_indices.json"), "w") as f:
        json.dump(split_info, f)
    print("[INFO] Split indices saved → outputs/split_indices.json")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, full_train_ds.classes


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
        total   += labels.size(0)
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
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels


# ── Main Training Loop ────────────────────────────────────────────────────────
def train():
    train_loader, val_loader, test_loader, classes = load_data()
    model     = build_model()
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history      = {"train_acc": [], "val_acc": [],
                    "train_loss": [], "val_loss": []}

    # ── Phase 1: Head only ────────────────────────────────────────────────
    print("[INFO] Phase 1 - Training classification head...")
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_PHASE1)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
    no_improve, patience = 0, 7

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        tr_loss, tr_acc          = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _    = validate(model, val_loader, criterion)
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
            torch.save(model.state_dict(), CNN_MODEL_PATH)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    # ── Phase 2: Fine-tune layer3 + layer4 ───────────────────────────────
    print("\n[INFO] Phase 2 - Fine-tuning top ResNet50 layers...")
    for name, param in model.named_parameters():
        if "layer4" in name or "layer3" in name or "fc" in name:
            param.requires_grad = True

    optimizer  = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2
    )
    scheduler  = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
    no_improve, patience = 0, 5

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        tr_loss, tr_acc                   = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, preds, lbls      = validate(model, val_loader, criterion)
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
            torch.save(model.state_dict(), CNN_MODEL_PATH)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    # Load best weights for final val evaluation
    model.load_state_dict(torch.load(CNN_MODEL_PATH))
    _, final_val_acc, final_preds, final_labels = validate(
        model, val_loader, criterion
    )

    return model, history, final_preds, final_labels, final_val_acc, \
           test_loader, classes


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import sys
    model, history, preds, labels, val_acc, test_loader, classes = train()

    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall    = recall_score(labels, preds,    average="weighted", zero_division=0)
    f1        = f1_score(labels, preds,        average="weighted", zero_division=0)

    print("\n── Validation Classification Report ─────")
    print(classification_report(labels, preds,
                                 target_names=classes, zero_division=0))

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
    print("[INFO] Run python src/evaluate.py for final TEST set results.")

    if val_acc < MIN_ACCURACY:
        sys.exit(1)


if __name__ == "__main__":
    main()