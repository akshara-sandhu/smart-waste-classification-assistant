from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from src.config import CLASS_NAMES_PATH, MODEL_PATH
from src.data import build_transforms
from src.model import build_model


class WastePredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                "Trained model not found. Run the training script first."
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.class_names = json.loads(
            CLASS_NAMES_PATH.read_text(encoding="utf-8")
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = build_model(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        _, self.transform = build_transforms()

    def predict(self, image: Image.Image) -> tuple[str, float, dict[str, float]]:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        confidence, index = torch.max(probabilities, dim=0)

        all_scores = {
            class_name: float(probabilities[i].item())
            for i, class_name in enumerate(self.class_names)
        }

        return (
            self.class_names[index.item()],
            float(confidence.item()),
            all_scores,
        )
