from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.config import (
    CLASS_NAMES_PATH,
    DATA_DIR,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_SEED,
)
from src.data import create_dataloaders
from src.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    model.train(training)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names = create_dataloaders(
        DATA_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    best_val_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
        )

        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            training=False,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f} | "
            f"val acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(class_names),
                    "class_names": class_names,
                },
                MODEL_PATH,
            )

            CLASS_NAMES_PATH.write_text(
                json.dumps(class_names, indent=2),
                encoding="utf-8",
            )

            print(f"Saved best model to {MODEL_PATH}")

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
