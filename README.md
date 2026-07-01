## Smart Waste Classification Assistant

This project is a waste image classifier built using Python, PyTorch, and Streamlit.

A user can upload an image, and the model predicts one of six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash
  
# Main Features

- Uses transfer learning with ResNet18
- Trains and validates the model
- Calculates accuracy, precision, recall, and F1-score
- Generates a confusion matrix
- Shows prediction confidence
- Displays a warning when confidence is low
- Includes a Streamlit web interface
- 
# Dataset

The project uses the TrashNet dataset.

Run:

python prepare_data.py

This downloads the dataset and divides it into:

- 70% training data
- 15% validation data
- 15% testing data
  
# Install Requirements

pip install -r requirements.txt

# Train the Model

python -m src.train --epochs 10 --batch-size 32

The trained model is saved as:

models/waste_classifier.pt

# Evaluate the Model

python -m src.evaluate

This creates:

- Test results
- Classification report
- Confusion matrix
  
# Run the App

streamlit run app.py

# Limitations

The model may give incorrect predictions when images have unusual backgrounds, lighting, angles, or multiple objects.

This is a learning project and should not be used as official waste-disposal guidance.
