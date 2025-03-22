import torch

from mask_calc import mask_pad_calc, mask_tril_calc
from util_calc import MultiHead, PositionEmbedding, FullyConnectedOutput


class EncoderLayer(torch.nn.Module):
    """Encoder layer that applies self-attention and a fully connected output layer.

    This layer processes the input tensor with multi-head self-attention followed by a fully connected layer.
    Both operations maintain the same dimensions as the input.

    Attributes:
        mh (MultiHead): Multi-head self-attention module.
        fc (FullyConnectedOutput): Fully connected output module.
    """

    def __init__(self):
        """Initialize the EncoderLayer."""
        super().__init__()
        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self, x, mask):
        """Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor with shape [batch, seq_length, dim].
            mask (torch.Tensor): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor with the same shape as input [batch, seq_length, dim].
        """
        # Compute self-attention; dimensions remain unchanged
        # [b, 50, 32] -> [b, 50, 32]
        score = self.mh(x, x, x, mask)

        # Apply fully connected output layer; dimensions remain unchanged
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(score)

        return out


class Encoder(torch.nn.Module):
    """Stacked encoder composed of multiple encoder layers.

    Attributes:
        layer_1 (EncoderLayer): First encoder layer.
        layer_2 (EncoderLayer): Second encoder layer.
        layer_3 (EncoderLayer): Third encoder layer.
    """

    def __init__(self):
        """Initialize the Encoder with three EncoderLayer modules."""
        super().__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()

    def forward(self, x, mask):
        """Forward pass of the encoder stack.

        Args:
            x (torch.Tensor): Input tensor with shape [batch, seq_length, dim].
            mask (torch.Tensor): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor with shape [batch, seq_length, dim].
        """
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x


class DecoderLayer(torch.nn.Module):
    """Decoder layer that applies self-attention, encoder-decoder attention, and a fully connected layer.

    This layer first applies self-attention on the target sequence, then applies attention combining
    the encoder and target sequence, and finally processes the result through a fully connected layer.

    Attributes:
        mh1 (MultiHead): Multi-head self-attention module for target sequence.
        mh2 (MultiHead): Multi-head attention module for encoder-decoder interaction.
        fc (FullyConnectedOutput): Fully connected output module.
    """

    def __init__(self):
        """Initialize the DecoderLayer."""
        super().__init__()
        self.mh1 = MultiHead()
        self.mh2 = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        """Forward pass of the decoder layer.

        Args:
            x (torch.Tensor): Encoder output tensor with shape [batch, seq_length, dim].
            y (torch.Tensor): Decoder input tensor with shape [batch, seq_length, dim].
            mask_pad_x (torch.Tensor): Padding mask for encoder input.
            mask_tril_y (torch.Tensor): Lower-triangular mask for decoder self-attention.

        Returns:
            torch.Tensor: Decoder output tensor with shape [batch, seq_length, dim].
        """
        # First compute self-attention for y; dimensions remain unchanged
        # [b, 50, 32] -> [b, 50, 32]
        y = self.mh1(y, y, y, mask_tril_y)

        # Compute attention combining x and y; dimensions remain unchanged
        # [b, 50, 32], [b, 50, 32] -> [b, 50, 32]
        y = self.mh2(y, x, x, mask_pad_x)

        # Apply fully connected output layer; dimensions remain unchanged
        # [b, 50, 32] -> [b, 50, 32]
        y = self.fc(y)

        return y


class Decoder(torch.nn.Module):
    """Stacked decoder composed of multiple decoder layers.

    Attributes:
        layer_1 (DecoderLayer): First decoder layer.
        layer_2 (DecoderLayer): Second decoder layer.
        layer_3 (DecoderLayer): Third decoder layer.
    """

    def __init__(self):
        """Initialize the Decoder with three DecoderLayer modules."""
        super().__init__()
        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        """Forward pass of the decoder stack.

        Args:
            x (torch.Tensor): Encoder output tensor with shape [batch, seq_length, dim].
            y (torch.Tensor): Decoder input tensor with shape [batch, seq_length, dim].
            mask_pad_x (torch.Tensor): Padding mask for encoder input.
            mask_tril_y (torch.Tensor): Lower-triangular mask for decoder self-attention.

        Returns:
            torch.Tensor: Decoder output tensor with shape [batch, seq_length, dim].
        """
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        return y


class Transformer_calc(torch.nn.Module):
    """Transformer model that integrates an encoder and a decoder with positional embeddings.

    The Transformer encodes the input sequence, applies a decoder to generate an output sequence,
    and then projects the decoder output to the final output space.

    Attributes:
        embed_x (PositionEmbedding): Positional embedding for encoder input.
        embed_y (PositionEmbedding): Positional embedding for decoder input.
        encoder (Encoder): Encoder stack.
        decoder (Decoder): Decoder stack.
        fc_out (torch.nn.Linear): Final linear projection layer.
    """

    def __init__(self):
        """Initialize the Transformer model."""
        super().__init__()
        self.embed_x = PositionEmbedding()
        self.embed_y = PositionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_out = torch.nn.Linear(32, 14)

    def forward(self, x, y):
        """Forward pass of the Transformer model.

        Args:
            x (torch.Tensor): Input tensor (source sequence) with shape [batch, seq_length].
            y (torch.Tensor): Input tensor (target sequence) with shape [batch, seq_length].

        Returns:
            torch.Tensor: Output tensor with shape [batch, seq_length, output_dim].
        """
        # Generate masks:
        # mask_pad_x: Padding mask for encoder input, shape [b, 1, 50, 50]
        # mask_tril_y: Lower-triangular mask for decoder self-attention
        mask_pad_x = mask_pad_calc(x)
        mask_tril_y = mask_tril_calc(y)

        # Apply positional embeddings to inputs
        # x: [b, 50] -> [b, 50, 32]
        # y: [b, 50] -> [b, 50, 32]
        x, y = self.embed_x(x), self.embed_y(y)

        # Process encoder: [b, 50, 32] -> [b, 50, 32]
        x = self.encoder(x, mask_pad_x)

        # Process decoder: combine encoder and decoder inputs, [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)

        # Final linear projection: [b, 50, 32] -> [b, 50, 14]
        y = self.fc_out(y)

        return y
