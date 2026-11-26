"""Four sequence classifiers for ListOps.

Base comparison (brief): RNN, LSTM, Transformer, Linear Attention.
Extension (ours): Looped Transformer (one shared layer, number of loops can change at test time).

Every model: token embedding -> sequence model (reads left to right) ->
representation at the final EOS position -> linear 10-class classifier.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from data import MAX_SOURCE_TOKENS, NUM_CLASSES

MAX_POS = MAX_SOURCE_TOKENS + 1  # source tokens + EOS


# ---------------------------------------------------------------- recurrent models
class RecurrentClassifier(nn.Module):
    """RNN or LSTM. Uses the final *valid* hidden state (packed sequences)."""

    def __init__(self, vocab_size, cell, d_embed, d_hidden, layers, dropout, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(d_embed, d_hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.cell = cell
        self.head = nn.Linear(d_hidden, NUM_CLASSES)

    def forward(self, ids, lengths):
        x = self.embed(ids)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.cell == "lstm":
            h = h[0]
        return self.head(h[-1])          # last layer's state at each sequence's EOS

    def features(self, ids, lengths):
        """Last-layer hidden state at every position (B, T, H), for probing.
        The RNN reads left to right, so positions before the padding are exact."""
        out, _ = self.rnn(self.embed(ids))
        return out


# ---------------------------------------------------------------- positional information
class PositionalEncoding(nn.Module):
    """'learned', 'sinusoidal' or 'none' (used in the positional-encoding ablation)."""

    def __init__(self, kind, d_model):
        super().__init__()
        self.kind = kind
        if kind == "learned":
            self.pos = nn.Embedding(MAX_POS, d_model)
        elif kind == "sinusoidal":
            pe = torch.zeros(MAX_POS, d_model)
            position = torch.arange(MAX_POS).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe)
        elif kind != "none":
            raise ValueError(kind)

    def forward(self, x):
        T = x.size(1)
        if self.kind == "learned":
            return x + self.pos(torch.arange(T, device=x.device))
        if self.kind == "sinusoidal":
            return x + self.pe[:T]
        return x


def gather_eos(h, lengths):
    """Pick the hidden vector at the EOS position (index length-1) for each sequence."""
    idx = (lengths - 1).to(h.device).view(-1, 1, 1).expand(-1, 1, h.size(-1))
    return h.gather(1, idx).squeeze(1)


# ---------------------------------------------------------------- causal Transformer
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, layers, heads, d_ff, dropout, pad_id, pos="learned"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(pos, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_ff, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, NUM_CLASSES)
        self.pad_id = pad_id

    def forward(self, ids, lengths):
        T = ids.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
        padding = ids.eq(self.pad_id)
        x = self.pos(self.embed(ids))
        h = self.encoder(x, mask=causal, src_key_padding_mask=padding)
        return self.head(self.norm(gather_eos(h, lengths)))

    def features(self, ids, lengths):
        h, _ = self.run_with_attention(ids, self.encoder.layers)
        return self.norm(h)

    def run_with_attention(self, ids, layers):
        """Manual pass through pre-LN layers that also returns attention maps
        (list of (B, heads, T, T)). Call in eval mode."""
        T = ids.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
        padding = ids.eq(self.pad_id)
        x = self.pos(self.embed(ids))
        maps = []
        for layer in layers:
            y = layer.norm1(x)
            a, w = layer.self_attn(y, y, y, attn_mask=causal, key_padding_mask=padding,
                                   need_weights=True, average_attn_weights=False)
            x = x + layer.dropout1(a)
            x = x + layer._ff_block(layer.norm2(x))
            maps.append(w)
        return x, maps


# ---------------------------------------------------------------- causal kernel Linear Attention
class CausalLinearAttention(nn.Module):
    """Katharopoulos et al. (2020): phi(x) = elu(x) + 1, causal prefix sums.

    S_t = sum_{i<=t} phi(k_i) v_i^T,  z_t = sum_{i<=t} phi(k_i)
    out_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + eps)
    Feature dimension = head dimension (d_model / heads).
    """

    def __init__(self, d_model, heads, eps=1e-6):
        super().__init__()
        assert d_model % heads == 0
        self.h, self.dh, self.eps = heads, d_model // heads, eps
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, padding):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        keep = (~padding).to(x.dtype).view(B, T, 1, 1)
        k, v = k * keep, v * keep                      # padded keys/values contribute nothing
        S = torch.einsum("bthd,bthe->bthde", k, v).cumsum(1)   # key-value state
        z = k.cumsum(1)                                         # normaliser state
        num = torch.einsum("bthd,bthde->bthe", q, S)
        den = torch.einsum("bthd,bthd->bth", q, z).unsqueeze(-1) + self.eps
        return self.out((num / den).reshape(B, T, D))


class LinearAttentionBlock(nn.Module):
    """Pre-LN block, same layout as the Transformer layer but with linear attention."""

    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalLinearAttention(d_model, heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, padding):
        x = x + self.drop(self.attn(self.ln1(x), padding))
        return x + self.drop(self.ff(self.ln2(x)))


class LinearAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, layers, heads, d_ff, dropout, pad_id, pos="learned"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(pos, d_model)
        self.blocks = nn.ModuleList(LinearAttentionBlock(d_model, heads, d_ff, dropout)
                                    for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, NUM_CLASSES)
        self.pad_id = pad_id

    def forward(self, ids, lengths):
        padding = ids.eq(self.pad_id)
        x = self.pos(self.embed(ids))
        for blk in self.blocks:
            x = blk(x, padding)
        return self.head(self.norm(gather_eos(x, lengths)))

    def features(self, ids, lengths):
        padding = ids.eq(self.pad_id)
        x = self.pos(self.embed(ids))
        for blk in self.blocks:
            x = blk(x, padding)
        return self.norm(x)


# ---------------------------------------------------------------- Looped Transformer (our extension)
class LoopedTransformerClassifier(TransformerClassifier):
    """ONE Transformer layer applied `loops` times with shared weights
    (a looped / universal Transformer). Because the weights are shared, the
    number of loops can be changed at test time: more loops = more steps of
    computation for the same parameters. A learned loop-step embedding tells
    the layer which iteration it is in.
    """

    def __init__(self, vocab_size, d_model, heads, d_ff, dropout, pad_id, loops, pos="learned",
                 max_loops=32):
        super().__init__(vocab_size, d_model, 1, heads, d_ff, dropout, pad_id, pos)
        self.loops = loops
        self.step_embed = nn.Embedding(max_loops, d_model)

    def _layers(self, loops):
        return [self.encoder.layers[0]] * loops

    def _run(self, ids, loops, want_maps=False):
        T = ids.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
        padding = ids.eq(self.pad_id)
        x = self.pos(self.embed(ids))
        layer = self.encoder.layers[0]
        maps = []
        for i in range(loops):
            x = x + self.step_embed.weight[min(i, self.step_embed.num_embeddings - 1)]
            if want_maps:
                y = layer.norm1(x)
                a, w = layer.self_attn(y, y, y, attn_mask=causal, key_padding_mask=padding,
                                       need_weights=True, average_attn_weights=False)
                x = x + layer.dropout1(a)
                x = x + layer._ff_block(layer.norm2(x))
                maps.append(w)
            else:
                x = layer(x, src_mask=causal, src_key_padding_mask=padding)
        return x, maps

    def forward(self, ids, lengths, loops=None):
        h, _ = self._run(ids, loops or self.loops)
        return self.head(self.norm(gather_eos(h, lengths)))

    def features(self, ids, lengths, loops=None):
        h, _ = self._run(ids, loops or self.loops)
        return self.norm(h)

    def run_with_attention(self, ids, layers=None, loops=None):
        return self._run(ids, loops or self.loops, want_maps=True)


# ---------------------------------------------------------------- factory
# Default sizes chosen so all four models have roughly 0.23-0.27M parameters.
DEFAULTS = {
    "rnn":         dict(d_embed=128, d_hidden=256, layers=2),
    "lstm":        dict(d_embed=128, d_hidden=128, layers=2),
    "transformer": dict(d_model=128, layers=2, heads=4, d_ff=256),
    "linear":      dict(d_model=128, layers=2, heads=4, d_ff=256),
    # one shared layer looped 4 times in training; made wider so its parameter
    # count matches the others (~0.28M), so any gain is not just "more parameters"
    "looped":      dict(d_model=176, layers=1, heads=4, d_ff=352, loops=4),
}


def build_model(name, vocab_size, pad_id, dropout=0.1, pos="learned", **overrides):
    cfg = {**DEFAULTS[name], **{k: v for k, v in overrides.items() if v is not None}}
    if name in ("rnn", "lstm"):
        model = RecurrentClassifier(vocab_size, name, cfg["d_embed"], cfg["d_hidden"],
                                    cfg["layers"], dropout, pad_id)
    elif name == "transformer":
        model = TransformerClassifier(vocab_size, cfg["d_model"], cfg["layers"], cfg["heads"],
                                      cfg["d_ff"], dropout, pad_id, pos)
    elif name == "linear":
        model = LinearAttentionClassifier(vocab_size, cfg["d_model"], cfg["layers"], cfg["heads"],
                                          cfg["d_ff"], dropout, pad_id, pos)
    elif name == "looped":
        model = LoopedTransformerClassifier(vocab_size, cfg["d_model"], cfg["heads"], cfg["d_ff"],
                                            dropout, pad_id, cfg["loops"], pos)
    else:
        raise ValueError(name)
    return model, cfg


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
