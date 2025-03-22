import streamlit as st
import torch

# Import core functions from main.py:
#    - run_main(): The main() function in main.py used for training and testing.
#    - predict(): The online prediction function which requires additional parameters like dictionaries and masks.
from main import main as run_main, predict

# Import dictionaries and model resources for both the Translation and Calculator tasks.
#    These are defined in data.py / data_calc.py as dict_y, dict_yr, dict_xr, etc.
from data import (
    dict_y as dict_y_trans,
    dict_yr as dict_yr_trans,
    dict_xr as dict_xr_trans,
)
from data_calc import dict_y_calc, dict_yr_calc, dict_xr_calc

# Import mask functions for both Translation and Calculator tasks.
from mask import mask_pad_trans, mask_tril_trans
from mask_calc import mask_pad_calc, mask_tril_calc

# Import the models for Translation and Calculator tasks.
from model import Transformer  # Translation task model
from calculator import Transformer_calc  # Calculator task model


st.title("Transformer Web Application")

# Sidebar: Select Prototype
st.sidebar.header("Select Prototype")
prototype = st.sidebar.radio(
    "Please choose a prototype", ("Calculator Prototype", "Translation Prototype")
)

# Set parameters based on selected prototype
if prototype == "Translation Prototype":
    st.sidebar.info("Using Translation Prototype")
    num_epochs = st.sidebar.number_input(
        "Number of Training Epochs", min_value=1, value=1, step=1
    )
    mode = "translation"

    # Resources for translation mode
    ModelClass = Transformer
    dict_xr = dict_xr_trans  # Reverse dictionary (index -> token)
    zidian_y = dict_y_trans  # Output token -> index
    zidian_yr = dict_yr_trans  # Output index -> token
    mask_pad = mask_pad_trans
    mask_tril = mask_tril_trans
    pretrained_path = (
        "trained_model.pth"  # Default pretrained weights for translation task
    )
    vocab_size = 39

else:
    st.sidebar.info("Using Calculator Prototype")
    num_epochs = st.sidebar.number_input(
        "Number of Training Epochs", min_value=1, value=10, step=1
    )
    st.sidebar.info("The calculator prototype needs at least 10 epochs to train.")
    mode = "calculator"

    # Resources for calculator mode
    ModelClass = Transformer_calc
    dict_xr = dict_xr_calc
    zidian_y = dict_y_calc
    zidian_yr = dict_yr_calc
    mask_pad = mask_pad_calc
    mask_tril = mask_tril_calc
    pretrained_path = (
        "trained_calculator.pth"  # Default pretrained weights for calculator task
    )
    vocab_size = 14

st.write(f"**Selected Prototype:** {prototype}")
st.write(f"**Number of Training Epochs:** {num_epochs}")

# Add a short introduction for each prototype
if mode == "calculator":
    st.markdown(
        """
        **Calculator Model Introduction**:
        This is a simple toy calculator model that performs basic addition.
        We use the letter 'a' to represent the plus sign.
        Note that this model only handles relatively short sequences of digits
        and is not a full-fledged calculator.
        """
    )
else:
    st.markdown(
        """
        **Translation Model Introduction**:
        This is a toy "translation" model that, instead of performing real language translation,
        transforms the input text by:
        1. Converting letters to uppercase and duplicate the last letter.
        2. Converting digits `x` to `9 - x`.
        3. Reversing the string.
        This demonstration is not a real translator; it only shows how a small
        Transformer might be used for sequence-to-sequence tasks with limited data.
        """
    )


# Construct an input dictionary (token -> index) for encoding user input.
# Since data.py / data_calc.py only provides the reverse dictionary (dict_xr),
# we construct a forward mapping here.
zidian_x = {token: i for i, token in enumerate(dict_xr)}

# Session State: Store the loaded model (if any) and random data (if any).
if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None

if "random_input" not in st.session_state:
    st.session_state.random_input = ""
if "random_target" not in st.session_state:
    st.session_state.random_target = ""
if "random_prediction" not in st.session_state:
    st.session_state.random_prediction = ""

# ------------------------------------------------------------------------------
# Place "Load Pretrained Model" and "Start Training" on the same row
# ------------------------------------------------------------------------------
col_load, col_train = st.columns([1, 1])

with col_load:
    load_clicked = st.button("Load Pretrained Model", key="load_button")
    st.write("You can use our pretrained model.")

with col_train:
    train_clicked = st.button("Start Training", key="train_button")
    st.write("Not recommended, **very** time-consuming.")

# Handle the logic *outside* the columns, so text is full-width
if load_clicked:
    try:
        if mode == "calculator":
            st.write("Loading pretrained calculator model...")
            model = ModelClass()
            model.load_state_dict(
                torch.load("trained_calculator.pth", map_location=torch.device("cpu"))
            )
            model.eval()
            st.session_state.loaded_model = model
            st.write("Pretrained calculator model loaded successfully!")
        else:
            st.write("Loading pretrained translation model...")
            model = ModelClass()
            model.load_state_dict(
                torch.load("trained_model.pth", map_location=torch.device("cpu"))
            )
            model.eval()
            st.session_state.loaded_model = model
            st.write("Pretrained translation model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading pretrained model: {e}")

if train_clicked:
    st.write("Initializing model for training...")
    run_main(mode=mode, num_epochs=num_epochs, lr=2e-3)
    st.write("Training complete!")
    st.write("Please check the terminal output for test results.")


def encode_user_input(s, max_len=50):
    """
    Encode a user input string into a tensor of token indices with a fixed length.

    This function converts the input string `s` into a list of tokens (assuming each character
    is a token), adds the <SOS> and <EOS> tokens, and then truncates or pads the sequence to `max_len`.
    Finally, it encodes the tokens using the forward dictionary (zidian_x).

    Args:
        s (str): The input string.
        max_len (int, optional): The fixed sequence length. Defaults to 50.

    Returns:
        torch.LongTensor: A tensor of shape [1, max_len] containing the token indices.
    """
    tokens = list(s)
    tokens = ["<SOS>"] + tokens + ["<EOS>"]
    if len(tokens) < max_len:
        tokens = tokens + ["<PAD>"] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    encoded = [
        zidian_x[token] if token in zidian_x else zidian_x["<PAD>"] for token in tokens
    ]
    return torch.LongTensor(encoded).unsqueeze(0)  # shape: [1, max_len]


def predict_online(model, x):
    """
    Generate a prediction for the given input tensor using the provided model.

    Args:
        model (torch.nn.Module): The Transformer model.
        x (torch.Tensor): Input tensor of shape [1, seq_len].

    Returns:
        torch.Tensor: Predicted output tensor of shape [1, seq_len].
    """
    return predict(model, x, zidian_y, mask_pad, mask_tril)


def tokens_to_clean_str(indices, idx2token):
    """
    Convert a list of token indices into a cleaned string.

    This function:
      - Converts indices to tokens using idx2token.
      - Trims everything after the first <EOS>.
      - Removes <SOS> if it appears at the start.
      - Removes any <PAD> tokens.
    """
    tokens = [idx2token[idx] for idx in indices]
    # Find first <EOS>
    try:
        eos_idx = tokens.index("<EOS>")
    except ValueError:
        eos_idx = len(tokens)
    # Skip leading <SOS>
    start_idx = 0
    if tokens and tokens[0] == "<SOS>":
        start_idx = 1
    # Slice up to <EOS>
    trimmed_tokens = tokens[start_idx:eos_idx]
    # Remove <PAD> from the middle
    trimmed_tokens = [t for t in trimmed_tokens if t != "<PAD>"]
    return "".join(trimmed_tokens)


# ------------------------------------------------------------------------------
# Random Data Generation Demo
# ------------------------------------------------------------------------------
st.header("Random Data Generation Demo")


def generate_random_sample():
    """
    Generate a random sample from the training distribution and run a prediction.

    This function calls the same get_data() used during training to ensure the
    generated sample follows the same distribution. The input and target sequences
    are displayed, and the model prediction is shown for comparison. The results
    are stored in session state so they persist after user interactions.
    """
    try:
        # Dynamically import the appropriate get_data() based on the mode
        if mode == "calculator":
            from data_calc import get_data
        else:
            from data import get_data_translation as get_data

        # Generate a random (x, y) sample
        x_sample, y_sample = get_data()  # e.g. shape [50], [51]

        # Convert x_sample and y_sample to clean strings
        input_str = tokens_to_clean_str(x_sample.tolist(), dict_xr)
        target_str = tokens_to_clean_str(y_sample.tolist(), zidian_yr)

        # Prepare a model for prediction
        if mode == "calculator" and st.session_state.loaded_model is not None:
            model = st.session_state.loaded_model
        else:
            model = ModelClass()
            model.load_state_dict(
                torch.load(pretrained_path, map_location=torch.device("cpu"))
            )
            model.eval()

        # Predict
        pred_out = predict_online(model, x_sample.unsqueeze(0))
        pred_str = tokens_to_clean_str(pred_out[0].tolist(), zidian_yr)

        # Store results in session state
        st.session_state.random_input = input_str
        st.session_state.random_target = target_str
        st.session_state.random_prediction = pred_str

    except Exception as e:
        st.error(f"Error generating random sample or predicting: {e}")


# Button to generate a new random sample
if st.button("Generate Random Sample"):
    generate_random_sample()

# If random data exists in session state, display it
if st.session_state.random_input:
    st.write("Random Input (x):", st.session_state.random_input)
if st.session_state.random_target:
    st.write("Random Target (y):", st.session_state.random_target)

# ------------------------------------------------------------------------------
# Online Prediction Demo
# ------------------------------------------------------------------------------
st.header("Online Prediction Demo")
user_input = st.text_input(
    "Please enter a text input (input will be padded if too short)", value=""
)

if st.button("Predict"):
    try:
        # Convert user input to a tensor
        input_tensor = encode_user_input(user_input, max_len=50)

        # For Calculator mode, if a pretrained model is loaded, use the model from session_state.
        if mode == "calculator" and st.session_state.loaded_model is not None:
            model = st.session_state.loaded_model
        else:
            # Otherwise, load the model from the corresponding pretrained file.
            model = ModelClass()
            model.load_state_dict(
                torch.load(pretrained_path, map_location=torch.device("cpu"))
            )
            model.eval()

        # Call the predict_online() wrapper to get the predicted sequence.
        pred_out = predict_online(model, input_tensor)

        # Convert the predicted token indices back to a clean string.
        pred_str = tokens_to_clean_str(pred_out[0].tolist(), zidian_yr)

        st.write("Prediction result:", pred_str)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
