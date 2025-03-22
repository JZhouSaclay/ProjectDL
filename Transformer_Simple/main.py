import torch

# Import dictionaries for translation and calculator tasks respectively
from data import (
    dict_y as dict_y_trans,
    dict_yr as dict_yr_trans,
    dict_xr as dict_xr_trans,
)
from data_calc import dict_y_calc, dict_yr_calc, dict_xr_calc

# Import mask functions for translation and calculator tasks respectively
from mask import mask_pad_trans, mask_tril_trans
from mask_calc import mask_pad_calc, mask_tril_calc

# Import translation and calculator models (if using the same model class, one import would suffice)
from model import Transformer
from calculator import Transformer_calc


def predict(model, x, zidian_y, mask_pad, mask_tril):
    """Predict the output sequence for a given input using the Transformer model.

    This function uses the provided Transformer model to generate an output sequence step-by-step.
    It requires the output dictionary (zidian_y) and functions for generating padding and triangular
    masks (mask_pad and mask_tril). The predicted output sequence is returned as a tensor.

    Args:
        model (torch.nn.Module): The Transformer model.
        x (torch.Tensor): Input tensor of shape [1, seq_len] containing token indices.
        zidian_y (dict): Dictionary mapping output tokens to indices.
        mask_pad (function): Function to generate the padding mask for the input.
        mask_tril (function): Function to generate the lower-triangular mask for decoder self-attention.

    Returns:
        torch.Tensor: Predicted output tensor of shape [1, seq_len].
    """
    model.eval()
    with torch.no_grad():

        # Generate the pad mask for the input sequence x
        mask_pad_x = mask_pad(x)

        # Initialize the target sequence with <SOS> followed by <PAD> tokens
        target = [zidian_y["<SOS>"]] + [zidian_y["<PAD>"]] * 49
        target = torch.LongTensor(target).unsqueeze(0)  # Shape: [1, 50]

        # Encode the input sequence: add positional embeddings and pass through the encoder
        x_embed = model.embed_x(x)  # [1, seq_len] -> [1, seq_len, 32]
        x_enc = model.encoder(x_embed, mask_pad_x)

        # Decode step-by-step to generate the output sequence
        for i in range(49):
            y = target
            mask_tril_y = mask_tril(
                y
            )  # Generate the lower-triangular mask for the target
            y_embed = model.embed_y(y)  # [1, seq_len] -> [1, seq_len, 32]
            y_dec = model.decoder(x_enc, y_embed, mask_pad_x, mask_tril_y)
            out = model.fc_out(
                y_dec
            )  # Project decoder output to vocabulary space: [1, seq_len, vocab_size]
            # For position i, select the token with the highest probability as the next token
            next_token = out[:, i, :].argmax(dim=1).detach()
            target[:, i + 1] = next_token

    return target


def train(model, loader, zidian_y, vocab_size, num_epochs=1, lr=2e-3):
    """Train the Transformer model.

    This function trains the provided Transformer model using the specified DataLoader.
    It uses cross-entropy loss to optimize the model to predict the next token in the sequence.
    The output dictionary (zidian_y) and vocabulary size (vocab_size) are provided externally
    to avoid hardcoding.

    Args:
        model (torch.nn.Module): The Transformer model to train.
        loader (torch.utils.data.DataLoader): DataLoader providing training batches.
        zidian_y (dict): Dictionary mapping output tokens to indices.
        vocab_size (int): The size of the output vocabulary.
        num_epochs (int, optional): Number of training epochs. Defaults to 1.
        lr (float, optional): Initial learning rate. Defaults to 2e-3.
    """
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
    model.train()

    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(loader):
            # Use the first n-1 tokens of y to predict the subsequent n-1 tokens
            pred = model(x, y[:, :-1])
            # Reshape predictions to [batch * seq_len, vocab_size]
            pred = pred.reshape(-1, vocab_size)
            # True labels: shift y by one token: [batch * seq_len]
            y_true = y[:, 1:].reshape(-1)

            # Ignore <PAD> tokens when computing the loss
            select = y_true != zidian_y["<PAD>"]
            pred = pred[select]
            y_true = y_true[select]

            loss = loss_func(pred, y_true)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 200 == 0:
                pred_labels = pred.argmax(dim=1)
                correct = (pred_labels == y_true).sum().item()
                accuracy = correct / len(pred)
                current_lr = optim.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch}, Batch {i}, LR {current_lr:.5f}, Loss {loss.item():.4f}, Acc {accuracy:.4f}"
                )
        sched.step()


def test_model(model, loader, zidian_xr, zidian_yr, zidian_y, mask_pad, mask_tril):
    """Test the Transformer model by printing sample predictions.

    This function retrieves a single batch from the provided DataLoader and prints sample inputs,
    targets, and predictions generated by the model. It uses the provided dictionaries and mask functions
    to decode and display the sequences.

    Args:
        model (torch.nn.Module): The Transformer model.
        loader (torch.utils.data.DataLoader): DataLoader for testing.
        zidian_xr (list): List mapping input token indices to their corresponding tokens (for display).
        zidian_yr (list): List mapping output token indices to their corresponding tokens (for display).
        zidian_y (dict): Dictionary mapping output tokens to indices.
        mask_pad (function): Function to generate the padding mask.
        mask_tril (function): Function to generate the lower-triangular mask.
    """
    # Retrieve the first batch for testing
    for x, y in loader:
        break

    for i in range(8):
        input_str = "".join([zidian_xr[idx] for idx in x[i].tolist()])
        target_str = "".join([zidian_yr[idx] for idx in y[i].tolist()])

        # Call the predict function with the required parameters
        pred_out = predict(model, x[i].unsqueeze(0), zidian_y, mask_pad, mask_tril)
        pred_str = "".join([zidian_yr[idx] for idx in pred_out[0].tolist()])

        print(f"Sample {i}:")
        print("Input:", input_str)
        print("Target:", target_str)
        print("Prediction:", pred_str)
        print("-" * 40)


def main(mode="translation", num_epochs=1, lr=2e-3):
    """Main function to train and test the Transformer model for different tasks.

    This function selects the appropriate DataLoader, model, mask functions, and dictionaries based
    on the specified mode ('translation' or 'calculator'). It then trains the model, tests it, and saves
    the trained model weights to disk.

    Args:
        mode (str, optional): The task mode, either 'translation' or 'calculator'. Defaults to "translation".
        num_epochs (int, optional): Number of training epochs. Defaults to 1.
        lr (float, optional): Initial learning rate. Defaults to 2e-3.
    """
    # Delay import to avoid circular dependencies
    from data import loader_trans, loader_calc

    if mode == "translation":
        loader = loader_trans
        model = Transformer()
        mask_pad = mask_pad_trans
        mask_tril = mask_tril_trans
        zidian_y = dict_y_trans
        zidian_yr = dict_yr_trans
        zidian_xr = dict_xr_trans
        vocab_size = 39  # Vocabulary size for translation task
    elif mode == "calculator":
        loader = loader_calc
        model = Transformer_calc()
        mask_pad = mask_pad_calc
        mask_tril = mask_tril_calc
        zidian_y = dict_y_calc
        zidian_yr = dict_yr_calc
        zidian_xr = dict_xr_calc
        vocab_size = 14  # Vocabulary size for calculator task
    else:
        raise ValueError("mode must be 'translation' or 'calculator'")

    # Train the model
    train(
        model,
        loader,
        zidian_y=zidian_y,
        vocab_size=vocab_size,
        num_epochs=num_epochs,
        lr=lr,
    )

    # Test the model by printing sample predictions
    test_model(model, loader, zidian_xr, zidian_yr, zidian_y, mask_pad, mask_tril)

    # Save the trained model weights to a file
    torch.save(model.state_dict(), "trained_model.pth")
    print("Model weights have been saved to trained_model.pth")


if __name__ == "__main__":
    main()
