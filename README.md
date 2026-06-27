# Smart Waste Classification Assistant

A beginner-friendly PyTorch computer-vision project that classifies waste images and provides confidence scores through a Streamlit web app.

## Features

- Transfer learning with ResNet18
- Training and validation loops
- Accuracy, precision, recall, and F1-score
- Confusion matrix generation
- Saved model checkpoint
- Streamlit image upload interface
- Low-confidence warning

## Waste classes

Create one folder per class inside each dataset split.

Example:

```text
data/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── val/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── test/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```


## Automatic dataset setup

This project can automatically download and prepare the original TrashNet dataset.

Run:

```bash
python prepare_data.py
```

The script will:

1. Download TrashNet from its original GitHub repository.
2. Extract the six waste categories.
3. Create reproducible 70% training, 15% validation, and 15% test splits.
4. Place the prepared images inside the `data/` folder.

Then train the model:

```bash
python -m src.train --epochs 10 --batch-size 32
```

Training time depends on your computer. A GPU is faster, but the code also works on CPU.


## Train the model

```bash
python -m src.train --epochs 10 --batch-size 32
```

The trained model will be saved to:

```text
models/waste_classifier.pt
```

## Evaluate the model

```bash
python -m src.evaluate
```

This generates:

- test metrics
- confusion matrix image
- classification report

## Run the web app

```bash
streamlit run app.py
```

## Important

This starter project does not include a dataset or a trained model. Add your own legally usable dataset and train the model before claiming results on a resume.
