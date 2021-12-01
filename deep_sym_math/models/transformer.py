from torch import nn
from deep_sym_math.models.transformer_utils import MultiHeadAttention, TransformerFFN, get_masks
import torch
import torch.nn.functional as F

N_MAX_POSITIONS = 4096  # maximum input sequence length


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class TransformerModel(nn.Module):
    """
    Transformer model.
    """

    def __init__(self, data_config, model_config={}):
        super().__init__()

        # Dictionary
        self.n_words = data_config['n_words']
        self.eos_index = data_config['eos_index']
        self.pad_index = data_config['pad_index']
        self.id2word = data_config['id2word']

        assert len(self.id2word) == self.n_words

        # Model parameters
        self.dim = model_config.get('dim', 512)
        self.hidden_dim = model_config.get('hidden_dim', self.dim * 4)
        self.n_heads = model_config.get('n_heads', 8)
        self.n_enc_layers = model_config.get('n_enc_layers', 6)
        self.n_dec_layers = model_config.get('n_dec_layers', 6)
        self.dropout = model_config.get('dropout', 0.4)
        self.attention_dropout = model_config.get('attention_dropout', 0.1)

        assert self.dim % self.n_heads == 0, 'Transformer dim must be a multiple of n_heads'

        # Embeddings
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, self.pad_index)

        # Enocder layers
        self.enc_layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)
        self.enc_attentions = nn.ModuleList()
        self.enc_layer_norm1 = nn.ModuleList()
        self.enc_ffns = nn.ModuleList()
        self.enc_layer_norm2 = nn.ModuleList()

        # Decoder layers
        self.dec_layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)
        self.dec_attentions = nn.ModuleList()
        self.dec_layer_norm1 = nn.ModuleList()
        self.dec_ffns = nn.ModuleList()
        self.dec_layer_norm2 = nn.ModuleList()
        self.dec_layer_norm15 = nn.ModuleList()
        self.dec_encoder_attn = nn.ModuleList()

        # Encoder
        for layer_id in range(self.n_enc_layers):
            self.enc_attentions.append(
                MultiHeadAttention(self.n_heads,
                                   self.dim,
                                   dropout=self.attention_dropout))
            self.enc_layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            self.enc_ffns.append(
                TransformerFFN(self.dim,
                               self.hidden_dim,
                               self.dim,
                               dropout=self.dropout))
            self.enc_layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        # Decoder
        for _ in range(self.n_dec_layers):
            self.dec_attentions.append(
                MultiHeadAttention(self.n_heads,
                                   self.dim,
                                   dropout=self.attention_dropout))
            self.dec_layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            self.dec_layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
            self.dec_encoder_attn.append(
                MultiHeadAttention(self.n_heads,
                                   self.dim,
                                   dropout=self.attention_dropout))
            self.dec_ffns.append(
                TransformerFFN(self.dim,
                               self.hidden_dim,
                               self.dim,
                               dropout=self.dropout))
            self.dec_layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))
        self.dec_proj = nn.Linear(self.dim, self.n_words, bias=True)
        self.dec_proj.weight = self.embeddings.weight

    def encode(self, x, len_x):
        slen, bs = x.size()
        assert len_x.size(0) == bs
        assert len_x.max().item() <= slen
        x = x.transpose(0, 1)  # Batch size as dimension 0

        # Generate masks
        mask, attn_mask = get_masks(slen, len_x, False)

        # Positions
        positions = x.new(slen).long()
        positions = torch.arange(slen, out=positions).unsqueeze(0)

        # Embeddings
        tensor = self.embeddings(x)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.enc_layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # Transformer layers
        for i in range(self.n_enc_layers):
            # Self attention
            attn = self.enc_attentions[i](tensor, attn_mask)
            attn = F.dropout(attn, p=self.dropout)
            tensor = tensor + attn
            tensor = self.enc_layer_norm1[i](tensor)

            # FFN
            tensor = tensor + self.enc_ffns[i](tensor)
            tensor = self.enc_layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # Move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)
        return tensor

    def decode(self, x, len_x, y, len_y):
        slen, bs = y.size()
        assert len_y.size(0) == bs
        assert len_y.max().item() <= slen
        y = y.transpose(0, 1)

        x.size(0) == bs

        # Generate masks
        mask, attn_mask = get_masks(slen, len_y, True)
        src_mask = torch.arange(len_x.max(), dtype=torch.long) < len_x[:, None]

        # Positions
        positions = y.new(slen).long()
        positions = torch.arange(slen, out=positions).unsqueeze(0)

        # Embeddings
        tensor = self.embeddings(y)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dec_layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # Transformer layers
        for i in range(self.n_dec_layers):
            # Self attention
            attn = self.dec_attentions[i](tensor, attn_mask)
            attn = F.dropout(attn, p=self.dropout)
            tensor = tensor + attn
            tensor = self.dec_layer_norm1[i](tensor)

            # Encoder attention (for decoder only)
            attn = self.dec_encoder_attn[i](tensor, src_mask, kv=x)
            attn = F.dropout(attn, p=self.dropout)
            tensor = tensor + attn
            tensor = self.dec_layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.dec_ffns[i](tensor)
            tensor = self.dec_layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # Move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)
        return tensor

    def forward(self, x, len_x, y, len_y):
        x = self.encode(x, len_x)
        output = self.decode(x.transpose(0, 1), len_x, y, len_y)
        return output
