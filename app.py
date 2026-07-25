import warnings

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

MODEL_PATH = "covid19(2).h5"
IMG_SIZE = 224
CLASS_MAPPING = {
    0: "Covid-19",
    1: "Normal",
    2: "Pneumonia",
}


@st.cache_resource(show_spinner=False)
def load_classification_model(path: str):
    """Load the trained Keras model from disk."""
    return load_model(path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize the uploaded chest X-ray image."""
    image = image.convert("RGB")
    img = np.asarray(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(image: Image.Image) -> tuple[str, float, np.ndarray]:
    """Run inference and return the predicted label, confidence, and raw probabilities."""
    model = load_classification_model(MODEL_PATH)
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    probabilities = prediction[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    return CLASS_MAPPING[predicted_class], confidence, probabilities


def main() -> None:
    st.set_page_config(
        page_title="Covid-19 X-ray Classifier",
        page_icon="🩺",
        layout="centered",
    )

    st.sidebar.header("About")
    st.sidebar.write(
        "This demo app uses a trained deep learning model to classify chest X-ray images as **Covid-19**, **Normal**, or **Pneumonia**."
    )
    st.sidebar.markdown(
        "**Disclaimer:** This tool is for education only and is not a medical diagnosis system."
    )

    st.title("🩺 Covid-19 X-ray Classifier")
    st.write(
        "Upload a clear chest X-ray scan in JPG/PNG format, then click **Predict** to evaluate the image."
    )

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload a chest X-ray image above to begin.")
        return

    try:
        image = Image.open(uploaded_file)
    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a JPG or PNG chest X-ray image.")
        return

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Evaluating image..."):
            label, confidence, probabilities = predict_image(image)

        st.success(f"Prediction: **{label}**")
        st.info(f"Model Confidence: **{confidence * 100:.2f}%**")
        st.progress(confidence)

        st.markdown("### Prediction probabilities")
        probability_data = {
            CLASS_MAPPING[index]: f"{prob * 100:.2f}%"
            for index, prob in enumerate(probabilities)
        }
        st.write(probability_data)

        st.markdown("---")
        st.write(
            "**Note:** This model is intended for demonstration purposes only. "
            "It should not be used as a substitute for professional medical advice or diagnosis."
        )


if __name__ == "__main__":
    main()
