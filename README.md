# ProjectDL
# Transformer Prototype

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/) 
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A small, **toy** Transformer-based project demonstrating two simple tasks:

1. **Calculator Prototype**: Performs basic addition on digit sequences, using the letter `a` to represent the plus sign.  
2. **Translation Prototype**: Converts letters to uppercase, transforms digits `x` into `9 - x`, and reverses the string—acting as a toy "translation" model.

This project highlights how Transformers can be adapted for small-scale, demonstration purposes. While not meant for production, it serves as an educational tool to explore sequence-to-sequence learning, data preparation, and a minimal web interface with Streamlit.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Loading Pretrained Models](#loading-pretrained-models)
  - [Random Data Generation](#random-data-generation)
  - [Online Prediction Demo](#online-prediction-demo)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Two Prototype Modes**  
  - **Calculator**:  
    - Handles basic addition tasks where sequences of digits are concatenated with `'a'` as a plus sign.  
    - Demonstrates how Transformers can learn simple arithmetic patterns.
  - **Translation**:  
    - Transforms input by uppercasing letters, mapping digits `x -> 9 - x`, and reversing the entire string.  
    - Illustrates a toy example of how Transformers can be used for string manipulation and sequence transformation.

- **Streamlit UI**  
  - An interactive web interface that lets you:
    - Load a pretrained model (for both calculator or translation tasks).
    - Generate random data consistent with the training distribution.
    - Provide custom input for on-the-fly predictions.

- **Modular Code Structure**  
  - Clearly separated data loaders (`data.py`, `data_calc.py`), model definitions (`model.py`, `calculator.py`), and main training script (`main.py`).  
  - An `app.py` file leveraging Streamlit for user interaction.

---

## Project Structure

```
.
├── app.py                # Streamlit-based web application
├── calculator.py         # Transformer model for calculator tasks
├── data.py               # Data loading and dictionary for translation tasks
├── data_calc.py          # Data loading and dictionary for calculator tasks
├── main.py               # Main training & testing script (handles both modes)
├── mask.py               # Mask functions for translation
├── mask_calc.py          # Mask functions for calculator
├── model.py              # Transformer model for translation tasks
├── README.md             # You are here!
└── requirements.txt      # Python dependencies (optional)
```

- **`main.py`**: Orchestrates training for either the translation or calculator prototype.  
- **`app.py`**: Streamlit UI allowing users to:
  - Select a prototype mode.
  - Load pretrained weights.
  - Optionally train (though not recommended, as it can be time-consuming).
  - Perform random data generation and online prediction.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/transformer-prototype.git
   cd transformer-prototype
   ```

2. **Install Dependencies**  
   We recommend using a virtual environment (e.g., `venv` or Conda). Then install:
   ```bash
   pip install -r requirements.txt
   ```
   Make sure you have Python 3.8 or higher.

---

## Usage

### Training

> **Note:** Training from scratch can be **very time-consuming** for the translation prototype. We do not recommend it unless you want to experiment with hyperparameters or see how the model learns.

You can train a selected prototype directly by calling:
```bash
python main.py --mode calculator --num_epochs 10
```
or
```bash
python main.py --mode translation --num_epochs 1
```
This will generate a `trained_model.pth` (for translation) or `trained_calculator.pth` (for calculator) depending on the mode.

### Loading Pretrained Models

- **Calculator**:  
  - Default checkpoint file: `trained_calculator.pth`
- **Translation**:  
  - Default checkpoint file: `trained_model.pth`

In the Streamlit UI, you can click **“Load Pretrained Model”** after selecting the corresponding prototype. This loads the appropriate checkpoint and readies the model for predictions.

### Random Data Generation

When using the Streamlit app, you can click **“Generate Random Sample”** to create a data point from the same distribution used during training:
- Calculator: A random addition problem with `a` as plus sign.
- Translation: A random string of letters and digits to be uppercased, digit-transformed, and reversed.

### Online Prediction Demo

In **Streamlit**, you can also manually enter an input string (digits for calculator, letters & digits for translation) and press **“Predict”** to see the model’s output.  
- For **Calculator**: The model interprets `'a'` as a plus sign.  
- For **Translation**: The model uppercases letters, converts digits via `x -> 9 - x`, and reverses the string.


