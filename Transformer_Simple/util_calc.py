import math

import torch


def attention(Q, K, V, mask):
    """Compute the scaled dot-product attention.

    This function performs the attention operation on the input query (Q), key (K), and value (V)
    tensors with the given attention mask.

    Args:
        Q (torch.Tensor): Query tensor of shape [batch, heads, seq_length, head_dim].
        K (torch.Tensor): Key tensor of shape [batch, heads, seq_length, head_dim].
        V (torch.Tensor): Value tensor of shape [batch, heads, seq_length, head_dim].
        mask (torch.Tensor): Boolean mask tensor of shape [batch, 1, seq_length, seq_length].
                             Positions with True are masked (set to -inf).

    Returns:
        torch.Tensor: Output tensor of shape [batch, seq_length, d_model] after merging all heads.
    """
    # Q, K, V have shape [b, 4, 50, 8] (4 heads, each head dimension=8)
    # Compute attention scores by matrix multiplication of Q and K^T
    # [b, 4, 50, 8] x [b, 4, 8, 50] -> [b, 4, 50, 50]
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # Scale scores by the square root of head dimension (sqrt(8)) for numerical stability
    score /= 8**0.5

    # Apply the mask: replace masked positions with -inf so that softmax outputs zero probability there.
    score = score.masked_fill_(mask, -float("inf"))
    score = torch.softmax(score, dim=-1)

    # Multiply the attention weights by V to obtain the attention output
    # [b, 4, 50, 50] x [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)

    # Merge the heads: rearrange and reshape from [b, 4, 50, 8] to [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32)

    return score


class MultiHead(torch.nn.Module):
    """Multi-head attention layer with residual connection, layer normalization, and dropout.

    This module applies multi-head self-attention on the input and adds a residual connection from the input.
    It splits the input into multiple heads, performs linear transformations, computes attention, and then
    merges the heads back together.
    """

    def __init__(self):
        """Initialize the MultiHead attention module."""
        super().__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)

        self.out_fc = torch.nn.Linear(32, 32)

        # Use LayerNorm to normalize across channels; elementwise_affine=True applies an additional linear mapping after normalization.
        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

        # Dropout layer with probability 0.1 for regularization
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        """Compute multi-head attention with residual connection.

        Args:
            Q (torch.Tensor): Query tensor of shape [batch, seq_length, 32].
            K (torch.Tensor): Key tensor of shape [batch, seq_length, 32].
            V (torch.Tensor): Value tensor of shape [batch, seq_length, 32].
            mask (torch.Tensor): Attention mask tensor of shape [batch, 1, seq_length, seq_length].

        Returns:
            torch.Tensor: Output tensor of shape [batch, seq_length, 32].
        """
        # Number of sentences (batch size)
        b = Q.shape[0]

        # Keep a copy of the original Q for the residual connection
        clone_Q = Q.clone()

        # Apply layer normalization on Q, K, and V
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # Apply linear transformations; dimensions remain unchanged [b, 50, 32]
        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(V)

        # Split into multiple heads:
        # Reshape from [b, 50, 32] to [b, 50, 4, 8] and then permute to [b, 4, 50, 8]
        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)

        # Compute attention output across heads; result shape will be [b, 50, 32]
        score = attention(Q, K, V, mask)

        # Apply final linear projection and dropout; dimensions remain [b, 50, 32]
        score = self.dropout(self.out_fc(score))

        # Add the residual connection from the original input
        score = clone_Q + score
        return score


class PositionEmbedding(torch.nn.Module):
    """Positional embedding layer with token embeddings.

    This module creates fixed positional encodings using sine and cosine functions and combines them
    with learned token embeddings.
    """

    def __init__(self):
        """Initialize the PositionEmbedding module."""
        super().__init__()

        def get_pe(pos, i, d_model):
            """Calculate the positional encoding for a given position and dimension.

            Args:
                pos (int): Position index.
                i (int): Dimension index.
                d_model (int): Total embedding dimension.

            Returns:
                float: Positional encoding value.
            """
            denominator = 1e4 ** (i / d_model)
            pe = pos / denominator

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        # Initialize positional encoding matrix with shape [50, 32]
        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        pe = pe.unsqueeze(0)  # Shape becomes [1, 50, 32]

        # Register the positional encoding matrix as a buffer (non-trainable)
        self.register_buffer("pe", pe)

        # Token embedding layer: maps tokens (with vocabulary size 14) to 32-dimensional vectors
        self.embed = torch.nn.Embedding(14, 32)
        # Initialize token embeddings with a normal distribution (mean=0, std=0.1)
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        """Apply token embeddings and add positional encodings.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape [batch, seq_length].

        Returns:
            torch.Tensor: Combined embedding tensor with shape [batch, seq_length, 32].
        """
        # Obtain token embeddings: [batch, seq_length] -> [batch, seq_length, 32]
        embed = self.embed(x)

        # Add the fixed positional encoding to token embeddings: broadcasting over the batch dimension
        embed = embed + self.pe
        return embed


class FullyConnectedOutput(torch.nn.Module):
    """Fully connected output module with a residual connection and layer normalization.

    This module applies a two-layer MLP with ReLU activation and dropout, normalizes the input,
    and then adds a residual connection.
    """

    def __init__(self):
        """Initialize the FullyConnectedOutput module."""
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1),
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

    def forward(self, x):
        """Forward pass of the fully connected output module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_length, 32].

        Returns:
            torch.Tensor: Output tensor of shape [batch, seq_length, 32].
        """
        # Keep a copy of the original input for the residual connection
        clone_x = x.clone()

        # Normalize the input
        x = self.norm(x)

        # Process through the MLP; output shape remains [batch, seq_length, 32]
        out = self.fc(x)

        # Add the original input as a residual connection
        out = clone_x + out

        return out
