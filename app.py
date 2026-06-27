from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from src.predict import WastePredictor


st.set_page_config(
    page_title="Smart Waste Classifier",

    layout="centered",
)



st.title(" Smart Waste Classification Assistant")

st.write(
    "Hello! This is Akshara Sandhu. I created this AI waste-classification assistant "
    "using Python, PyTorch, and Streamlit."
)

st.write(
    "Upload an image of waste below, and the model will predict its category :)"
)

try:
    predictor = WastePredictor()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("Add a dataset and run the training script before using this app.")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    label, confidence, scores = predictor.predict(image)

    if confidence < 0.60:
        st.warning(
            f"The model is unsure. Best guess: {label} "
            f"({confidence:.1%} confidence)"
        )
    else:
        st.success(f"Prediction: {label}")
        st.metric("Confidence", f"{confidence:.1%}")

    score_frame = pd.DataFrame(
        {
            "Category": list(scores.keys()),
            "Confidence": list(scores.values()),
        }
    ).sort_values("Confidence", ascending=False)

    st.subheader("All predictions")
    st.bar_chart(
        score_frame.set_index("Category")
    )

st.caption(
    "Predictions may be incorrect. Do not rely on this tool for official "
    "waste-disposal guidance."
)
