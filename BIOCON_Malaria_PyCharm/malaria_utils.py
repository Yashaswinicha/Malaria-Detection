
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

CLASS_NAMES = ["Uninfected", "Parasitized"]

SEED = 42

OUTPUT_DIR = PROJECT_DIR / "outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "mps"):
        torch.mps.manual_seed(seed)


# ============================================================
# DEVICE SELECTION
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_dataset():

    possible_directories = [
        PROJECT_DIR,
        PROJECT_DIR / "cell_images",
    ]

    for directory in possible_directories:

        if all(
            (directory / class_name).is_dir()
            for class_name in CLASS_NAMES
        ):
            print(f"Dataset found at: {directory}")
            return directory

    raise FileNotFoundError(
        "Dataset folders were not found. Expected folders:\n"
        "Uninfected/\n"
        "Parasitized/"
    )


# ============================================================
# COLLECT IMAGE RECORDS
# ============================================================

def collect_image_records(data_dir):

    records = []

    for label, class_name in enumerate(CLASS_NAMES):

        class_dir = Path(data_dir) / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing dataset folder: {class_dir}"
            )

        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            records.append((str(image_path), label))

    random.Random(SEED).shuffle(records)

    if not records:
        raise ValueError("No images found in the dataset.")

    return records


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_records(records):

    labels = [label for _, label in records]

    train_records, remaining_records = train_test_split(
        records,
        test_size=0.30,
        random_state=SEED,
        stratify=labels
    )

    remaining_labels = [
        label for _, label in remaining_records
    ]

    val_records, test_records = train_test_split(
        remaining_records,
        test_size=0.50,
        random_state=SEED,
        stratify=remaining_labels
    )

    return train_records, val_records, test_records


# ============================================================
# IMAGE TRANSFORMATIONS
# ============================================================

def get_transform(train=False, img_size=128):

    if train:

        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


# ============================================================
# CUSTOM PYTORCH DATASET
# ============================================================

class CellDataset(Dataset):

    def __init__(self, records, transform=None):

        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        image_path, label = self.records[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(
            float(label),
            dtype=torch.float32
        )

        return image, label


# ============================================================
# DATALOADER
# ============================================================

def make_loader(
    records,
    img_size=128,
    batch_size=32,
    train=False,
    num_workers=0,
    shuffle=None
):

    if shuffle is None:
        shuffle = train

    dataset = CellDataset(
        records,
        transform=get_transform(
            train=train,
            img_size=img_size
        )
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )


# ============================================================
# CNN MODEL
# ============================================================

class MalariaCNN(nn.Module):

    def __init__(self):

        super(MalariaCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(256, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# SAVE MODEL CHECKPOINT
# ============================================================

def save_checkpoint(path, model, img_size=128):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "img_size": img_size,
        "class_names": CLASS_NAMES,
        "seed": SEED
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved: {path}")


# ============================================================
# LOAD MODEL CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path,
    model=None,
    device=None
):

    if device is None:
        device = get_device()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if model is None:
        model = MalariaCNN()

    model.load_state_dict(checkpoint["state_dict"])

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

def predict_image(
    image_path,
    model=None,
    checkpoint_path=None,
    device=None
):

    if device is None:
        device = get_device()

    if model is None:

        if checkpoint_path is None:
            raise ValueError(
                "Provide either a trained model "
                "or a checkpoint_path."
            )

        model = load_checkpoint(
            checkpoint_path,
            device=device
        )

    model = model.to(device)
    model.eval()

    image_transform = get_transform(
        train=False,
        img_size=128
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")

    image = image_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.inference_mode():

        logits = model(image)

        probability = torch.sigmoid(logits).item()

    predicted_label = (
        1 if probability >= 0.5 else 0
    )

    return {
        "predicted_class": CLASS_NAMES[predicted_label],
        "probability": probability,
        "class_index": predicted_label
    }
