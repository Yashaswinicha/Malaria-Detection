from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt

from malaria_utils import (
    PROJECT_DIR, OUTPUT_DIR, CLASS_NAMES, SEED, set_seed,
    prepare_dataset, collect_image_records, split_records,
    make_loader, MalariaCNN, get_device, save_checkpoint
)

# -------------------- Settings you can change --------------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 4
# ---------------------------------------------------------------


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = (logits >= 0).float()
        total_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    data_dir = prepare_dataset()
    records = collect_image_records(data_dir)
    train_records, val_records, test_records = split_records(records)

    print(f"Total images: {len(records)}")
    print(f"Training: {len(train_records)} | Validation: {len(val_records)} | Test: {len(test_records)}")

    train_loader = make_loader(train_records, IMG_SIZE, BATCH_SIZE, train=True)
    val_loader = make_loader(val_records, IMG_SIZE, BATCH_SIZE, train=False)
    test_loader = make_loader(test_records, IMG_SIZE, BATCH_SIZE, train=False)

    model = MalariaCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "malaria_cnn_best.pth"

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, model, IMG_SIZE)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping: validation loss did not improve.")
                break

    print(f"\nTraining time: {(time.time() - start_time) / 60:.1f} minutes")

    # Reload best validation checkpoint before final test evaluation
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    y_true, y_prob = [], []
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.numpy().astype(int).tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nTEST SET RESULTS")
    print("Accuracy :", f"{accuracy_score(y_true, y_pred):.4f}")
    print("Precision:", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
    print("Recall   :", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
    print("F1-score :", f"{f1_score(y_true, y_pred, zero_division=0):.4f}")
    try:
        print("ROC-AUC  :", f"{roc_auc_score(y_true, y_prob):.4f}")
    except ValueError:
        print("ROC-AUC  : not available for this test split")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix:\n", cm)

    # Save training curves and test results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Test Confusion Matrix")
    plt.xticks([0, 1], CLASS_NAMES, rotation=15)
    plt.yticks([0, 1], CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.show()

    results = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_images": int(len(y_true)),
        "device": str(device),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nBest model saved to: {checkpoint_path}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
