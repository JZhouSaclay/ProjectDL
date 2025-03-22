import torch

from data_calc import dict_x_calc, dict_y_calc


def mask_pad_calc(data):
    """Generate a padding mask for the input data.

    This function creates a boolean mask indicating positions of the <PAD> token for the encoder input.
    The input is expected to be a tensor of shape [batch, seq_length] with token indices before embedding.
    The mask is then reshaped and expanded to match the attention matrix dimensions.

    Args:
        data (torch.Tensor): Input tensor of shape [batch, seq_length].

    Returns:
        torch.Tensor: A boolean mask tensor of shape [batch, 1, seq_length, seq_length] where True
                      indicates that the token is a <PAD> token. In the attention computation, this mask
                      will cause any attention from any token to the <PAD> tokens to be zero.
    """
    # data: [batch, 50]; determine which tokens are <PAD>
    mask = data == dict_x_calc["<PAD>"]

    # Reshape mask from [batch, 50] to [batch, 1, 1, 50]
    mask = mask.reshape(-1, 1, 1, 50)

    # Expand the mask to cover the attention matrix dimension: [batch, 1, 50, 50]
    mask = mask.expand(-1, 1, 50, 50)

    return mask


def mask_tril_calc(data):
    """Generate a lower-triangular mask for decoder self-attention.

    This function creates a mask that prevents each token in the target sequence from attending to
    tokens ahead of it (i.e., only the token itself and previous tokens are visible). It also ensures that
    positions corresponding to the <PAD> token in the target sequence are masked out.

    Args:
        data (torch.Tensor): Input tensor of token indices of shape [batch, seq_length] for the decoder.

    Returns:
        torch.Tensor: A boolean mask tensor of shape [batch, 1, seq_length, seq_length] where True indicates
                      that the token should not be attended to.
    """
    # Create a 50x50 matrix where the upper triangular part (excluding the diagonal) is 1.
    # This indicates positions that should be masked (i.e., future tokens are not visible).
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long))

    # Determine which tokens in the target sequence are <PAD> tokens.
    # data: [batch, 50]
    mask = data == dict_y_calc["<PAD>"]

    # Unsqueeze to shape [batch, 1, 50] and convert to long for addition.
    mask = mask.unsqueeze(1).long()

    # Combine the padding mask with the lower-triangular mask.
    # The result will have shape [batch, 50, 50] indicating positions to be masked.
    mask = mask + tril

    # Convert the mask to boolean type: values > 0 are True.
    mask = mask > 0

    # Ensure the mask has an extra dimension for compatibility with subsequent computations: [batch, 1, 50, 50].
    mask = (mask == 1).unsqueeze(dim=1)

    return mask
