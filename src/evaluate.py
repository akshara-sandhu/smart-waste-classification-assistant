from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.config import CLASS_NAMES_PATH, DATA_DIR, MODEL_PATH
from src.data import create_dataloaders
from src.model import build_model


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run training first."
        )

    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    _, _, test_loader, _ = create_dataloaders(
        DATA_DIR,
        batch_size=32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()

            true_labels.extend(labels.numpy())
            predicted_labels.extend(predictions.numpy())

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    matrix = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(xticks_rotation=45)
    plt.tight_layout()
    output_path = MODEL_PATH.parent / "confusion_matrix.png"
    plt.savefig(output_path, dpi=160)
    print(f"Saved confusion matrix to {output_path}")


if __name__ == "__main__":
    main()
