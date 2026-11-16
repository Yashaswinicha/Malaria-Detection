"""Predict one image using the trained custom CNN.

Usage:
    python predict.py path/to/cell_image.png
"""
import argparse
from pathlib import Path

from PIL import Image

from malaria_utils import OUTPUT_DIR, get_device, load_checkpoint, predict_image, CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description="BIOCON malaria image prediction")
    parser.add_argument("image", help="Path to a cell image")
    parser.add_argument("--checkpoint", default=str(OUTPUT_DIR / "malaria_cnn_best.pth"))
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit("Trained model not found. Run train.py first.")

    device = get_device()
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    with Image.open(image_path) as image:
        probability = predict_image(model, image, checkpoint["img_size"], device)

    threshold = checkpoint.get("threshold", 0.5)
    label = CLASS_NAMES[1] if probability >= threshold else CLASS_NAMES[0]
    confidence = probability if label == CLASS_NAMES[1] else 1 - probability

    print(f"Device: {device}")
    print(f"Prediction: {label}")
    print(f"Predicted-class score: {confidence:.2%}")
    print(f"P(Parasitized): {probability:.2%}")


if __name__ == "__main__":
    main()
