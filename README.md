# Covid-19 X-ray Classifier

A Streamlit application for classifying chest X-ray images into three categories: **Covid-19**, **Normal**, or **Pneumonia**.

## Project Overview

This repository contains a demonstration app that loads a pre-trained Keras model from `covid19(2).h5` and performs inference on uploaded X-ray images. The goal is to provide a polished interface for exploring model behavior and comparing prediction confidence across the three classes.

## Features

- Upload chest X-ray images in JPG/PNG format
- Display the uploaded image inside the app
- Predict one of: Covid-19, Normal, Pneumonia
- Show model confidence and probability breakdown
- Informational sidebar and usage guidance

## Installation

1. Create a Python virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app from the project folder:

```powershell
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

## Project Structure

- `app.py` - Streamlit app entry point
- `covid19(2).h5` - Trained Keras classification model
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Notes

- This app is for educational and demonstration purposes only.
- It is not a substitute for clinical diagnosis.
- Use high-quality frontal chest X-rays for best results.

## Dependencies

The project uses the following Python packages:

- `tensorflow`
- `numpy`
- `streamlit`
- `opencv-python`
- `Pillow`

## License

This project is released under the MIT License.
