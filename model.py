import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):

    def __init__(self, d_model, vocab_size, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# positional encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # we add dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)

        # create a matrix of the same size as the input (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # we create a vector of positions
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        # the formula with exponentials-log for calculation stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # we need to apply the sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # every line, even indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # we add a batch dimension
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # we register the buffer so that it is saved in # the state_dict
        self.register_buffer("pe", pe)

    def forward(self, x):
        # we add the positional encoding to the input
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # we don't want to learn this
        return self.dropout(x)


class LayerNoemalization(nn.Module):

    # we add a small epsilon to avoid division by zero
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(
            torch.ones(1)
        )  # learnable parameter for multiplication
        self.beta = nn.Parameter(torch.zeros(1))  # added bias

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # -1 for the last dimension
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # W2 and b2, we don't need to define b2
        # because it is included in the linear layer

    def forward(self, x):
        # Shape transformation:
        # 1. batch_size, seq_len, d_model
        # 2. batch_size, seq_len, d_ff
        # 3. batch_size, seq_len, d_model
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h

        # we need to make sure that the number of heads divides the model dimension
        assert d_model % h == 0
        self.d_k = d_model // h  # dimension of each head

        # we define the linear layers for Q, K, V and the output
        self.w_q = nn.Linear(d_model, d_model)  # Query transformation
        self.w_k = nn.Linear(d_model, d_model)  # Key transformation
        self.w_v = nn.Linear(d_model, d_model)  # Value transformation
        self.w_o = nn.Linear(d_model, d_model)  # Output transformation

        self.dropout = nn.Dropout(dropout)

    # we define the scaled dot product attention
    @staticmethod
    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape(-1)

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # before attention mechanism, we need to apply the mask
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = torch.softmax(
            attention_scores, dim=-1
        )  # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(
            q
        )  # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(k)  # same
        value = self.w_v(v)  # same

        # Transform dimensions in steps:
        # 1. batch_size, seq_len, d_model
        # 2. batch_size, seq_len, h, d_k
        # 3. batch_size, h, seq_len, d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Transform dimensions back:
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) ->
        # (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.norm = LayerNoemalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):  # sublayer is the previous layer
        # a little different from the original paper in the order of the operations
        # we normalize first and then apply the sublayer and the dropout
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # we have 2 skip connections in the encoder block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        # we don't want the padding to be considered in the attention mechanism
        # we are calling the function from MultiHeadAttentionBlock
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


# we have a stack of encoder blocks
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNoemalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # for the decoder, we have 3 skip connections like in the paper
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


# and now we stack the decoder blocks
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNoemalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# one last thing for the Transformer model
# we need to affect the output to the dictionnary - vocabulary, by linear layer
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


# and now we can define the final Transformer model
class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection_layer

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        return self.decoder(
            self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask
        )

    def project(self, x):
        return self.projection(x)


# we need to build the final function to combined all the blocks
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size, src_seq_len)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size, tgt_seq_len)

    # create positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len)

    # create the encoder and decoder
    encoder_block = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention, feed_forward_block, dropout
        )
        encoder_block.append(encoder_block)

    decoder_block = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention, decoder_cross_attention, feed_forward_block, dropout
        )
        decoder_block.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_block))
    decoder = Decoder(nn.ModuleList(decoder_block))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # initialize the parameters to avoid a total random initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
