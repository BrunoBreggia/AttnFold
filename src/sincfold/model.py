from torch import nn
from torch.nn.functional import cross_entropy, interpolate, pad
import torch.nn.functional as F
import torch as tr
from tqdm import tqdm
import pandas as pd
import math
import numpy as np
import os, shutil
from pathlib import Path

from sincfold.metrics import contact_f1
from sincfold.utils import mat2bp, postprocessing
from sincfold._version import __version__

def sincfold(pretrained=False, weights=None, attention_only=True, **kwargs):
    """ 
    SincFold: a deep learning-based model for RNA secondary structure prediction
    pretrained (bool): Use pretrained weights (from attention matrix)
    attention_only (bool): If True, create only AttentionMatrix. If False, create AttentionMatrix + Conv2DBlock
    weights (str): Path to weights file to load
    **kwargs: Model hyperparameters
    """
    model = SincFold(attention_only=attention_only, **kwargs)
    if pretrained:
        print("Funcionalidad todavia no implementada")
    else:
        if weights is not None:
            print(f"Load weights from {weights}")
            model.load_state_dict(tr.load(weights, map_location=tr.device(model.device)))
        else:
            print("No weights provided, using random initialization")
        
    return model


class SincFold(nn.Module):

    def __init__(
                    self,
                    device="cpu",
                    attention_only=True,
                    pos_enc="absolute", # or "rope"
                    negative_weight=0.1,
                    emb_dim=32,
                    lr=1e-4,
                    lr_conv=1e-4,
                    verbose=True,
                    save_dir=None,
                    conv_k=3,
                    **kwargs
                ):
        super().__init__()

        self.device = device
        self.attention_only = attention_only
        self.pos_enc = pos_enc.lower()
        self.class_weight = tr.tensor([negative_weight, 1.0]).float().to(device)
        self.verbose = verbose
        self.output_th = 0.5

        # Unpacking command-line arguments
        self.config = kwargs
        self.force_symmetry = self.config.get('force_symmetry', False)
        self.enc_base = self.config.get('enc_base', 1_000)
        self.mask = self.config.get('mask', False)
        self.k_bias = self.config.get('k_bias', False)

        # bandera para guardar las matrices de operacion internas
        self.save_dir = Path(save_dir) / "matrices" if save_dir else None
        self.save_flag = False

        # Choose type of attention
        if self.pos_enc in ["absolute", "abs"]:
            self.msg("Training with ABSOLUTE positional encoding")
            self.attention = AttentionMatrix(d_model=emb_dim, device=device,
                                            enc_base=self.enc_base,
                                            mask=self.mask,
                                            k_bias=self.k_bias,
                                            )
        elif self.pos_enc in ["rope", "rotary"]:
            self.msg("Training with ROTARY positional encoding")
            self.attention = RoPEAttnLayer(d_model=emb_dim, device=device,
                                            )
        
        if self.force_symmetry:
            self.msg("Symmetry activated")

        print(f"Embedding dimension: {emb_dim}")

        self.optimizer_attention = tr.optim.Adam(self.attention.parameters(), lr=lr)
        self.scheduler = tr.optim.lr_scheduler.StepLR(self.optimizer_attention, step_size=100, gamma=0.1)
        
        if not attention_only:
            self.conv = Conv2DBlock(k=conv_k, device=device)
            self.optimizer_conv = tr.optim.Adam(self.conv.parameters(), lr=lr_conv)
        else:
            self.conv = None
            self.optimizer_conv = None

        self.to(device)
    
    def msg(self, message):
        print(message) if self.verbose else None            

    def forward(self, batch):
        x = batch["embedding"].to(self.device)
        L = max(batch["length"])

        # Apply attention layer
        y0 = self.attention(x) # size of y? (L, N, E)?
        
        # checkpoint
        self.save_routine(y0, "compressed", batch['id'][0])
        
        # Remove k bias
        length = y0.shape[-2]
        if y0.shape[-1] > length:
            y0 = y0[:, :, :length]

        if len(y0.shape) < 4:
            y0 = tr.unsqueeze(y0, dim=1)

        # Unpooling
        k = 3
        target_size = y0.shape[-1]*k
        expanded = interpolate(y0, size=(target_size,target_size), mode='nearest')
        if target_size < L:
            padding = L - target_size
            expanded = pad(expanded, (0 ,padding, 0, padding), mode="constant", value=0)
        
        # checkpoint
        self.save_routine(expanded, "expanded", batch['id'][0])
        
        # Aplicacion de convolucion
        if not self.attention_only and self.conv is not None:
            expanded = self.conv(expanded)
        
        # simetria forzada
        if self.force_symmetry:
            expanded_t = tr.transpose(expanded, -1, -2)
            expanded = (expanded + expanded_t) / 2
        
        expanded = expanded.squeeze(1)

        # checkpoint
        self.save_routine(expanded, "final", batch['id'][0])
        
        return expanded

    def save_routine(self, matrix, mat_stage, id):
        if self.save_flag: # and len(batch['id']) == 1:
            np_matrix = np.squeeze(matrix.detach().cpu().numpy())
            np.savetxt(f"{self.save_dir}/{mat_stage}_{id}.csv", np_matrix, delimiter=',')
        
    def loss_func(self, yhat, y):
        """yhat and y are [N, M, M]"""
        if self.mask:
            y = tr.tril(y)
        y = y.view(y.shape[0], -1)
        yhat = yhat.view(yhat.shape[0], -1)
        yhat = yhat.unsqueeze(1)
        yhat = tr.cat((-yhat, yhat), dim=1)
        error_loss = cross_entropy(yhat, y, ignore_index=-1, weight=self.class_weight, label_smoothing=0.1) # smoothing added
        loss = error_loss
        return loss

    def test(self, loader):
        """
        Returns a metrics dictionary with loss and F1 performance
        """
        self.eval()

        # If there is a save directory, will save all matrices
        # calculated during procedure
        if self.save_dir:
            self.save_flag = True
            # If the matrix folder exists (matrices), erase it and start over
            if os.path.isdir(self.save_dir):
                shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir, exist_ok=True)

        metrics = {"loss": 0, "f1": 0}

        if self.verbose:
            loader = tqdm(loader)

        with tr.no_grad():
            for batch in loader:
                y = batch["contact"].to(self.device)
                batch.pop("contact")
                lengths = batch["length"]
                
                y_pred = self(batch)
                loss = self.loss_func(y_pred, y)
                metrics["loss"] += loss.item()
                
                f1 = contact_f1(y.cpu(), y_pred.cpu(), lengths, th=self.output_th, reduce=True, method="triangular")
                metrics["f1"] += f1

        for k in metrics:
            metrics[k] /= len(loader)
        
        self.save_flag = False
        return metrics

    def freeze_attention(self):
        for p in self.attention.parameters():
            p.requires_grad = False

    def unfreeze_attention(self):
        for p in self.attention.parameters():
            p.requires_grad = True

    def freeze_conv(self):
        if self.conv is not None:
            for p in self.conv.parameters():
                p.requires_grad = False

    def unfreeze_conv(self):
        if self.conv is not None:
            for p in self.conv.parameters():
                p.requires_grad = True

    def save_attention(self, path):
        tr.save(self.attention.state_dict(), path)
        #print(f"Attention weights saved to {path}")

    def load_attention(self, path):
        self.attention.load_state_dict(tr.load(path, map_location=self.device))
        #print(f"Attention weights loaded from {path}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, enc_base=1000.0, max_len=5000):
        super().__init__()
        pe = tr.zeros(max_len, d_model)
        position = tr.arange(0, max_len, dtype=tr.float).unsqueeze(1)
        div_term = tr.exp(tr.arange(0, d_model, 2).float() * (-math.log(enc_base) / d_model))
        pe[:, 0::2] = tr.sin(position * div_term)
        pe[:, 1::2] = tr.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class AttentionMatrix(nn.Module):
    def __init__(
        self,
        d_model=32,
        mask=False,
        k_bias=False,
        enc_base=1000,
        device="cpu"
    ):
        super().__init__()
        self.device = device
        self.mask = mask
        self.k_bias = k_bias
        self.embedding = nn.Embedding(65, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, enc_base=enc_base, max_len=5000)
        self.multiHead_1 = nn.MultiheadAttention(d_model, 
                                                 num_heads=1,
                                                 batch_first=True,
                                                 add_bias_kv=self.k_bias
                                                 )
        if self.mask:
            print("Using MASK")
        if self.k_bias:
            print("Using k BIAS")
  
    def forward(self, batch):
        y = self.embedding(batch.int())

        N, L, E = y.shape
        y = self.pos_encoding(y) # adds the positional encoding to the embedded input

        causal_mask = None
        if self.mask:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(self.device)
        
        _, attn_matrix = self.multiHead_1(query=y, key=y, value=y,  # shape [batch_size, seq_len, seq_len]
                                            need_weights=True,
                                            attn_mask=causal_mask,
                                            is_causal=self.mask,
                                        )

        return attn_matrix


class RoPEAttnLayer(nn.Module):
    """
    A version of an Attention-Encoder Layer with Rotary Positional Encodings
    (RoPE) as described in [1].
    Extracted (and modified) from https://github.com/JannisZeller/rope-multi-head-attention/blob/main/src/rope_attention_layer.py

    References:
    [1] https://arxiv.org/pdf/2104.09864.pdf - RoPE Paper
    [2] https://github.com/karpathy/nanoGPT/blob/master/model.py - Guidance for
            loop-free implementation of multi-head architecture.
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 1,
        enc_base: int = 100,
        max_pos_enc_len: int = 5000,
        dropout: float = 0.0,
        bias: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.inv_sqrt_d_head = 1.0 / tr.sqrt(tr.tensor(self.d_head) + 1e-8)

        self.embedding = nn.Embedding(65, d_model)
        
        self.multi_head_in_projection = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.multi_head_out_projection = nn.Linear(d_model, d_model, bias=bias)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.d_model = d_model
        self._construct_rope_matrices(enc_base, max_pos_enc_len)
        self.to(device)

    def _construct_rope_matrices(self, enc_base, max_pos_enc_len):
        """Constructs rotary embedding matrices for additive version
        [1, p. 7, eq. (34)]. Configured for x beeing of shape
        (batch_size, seqlen, d_model).
        """
        assert self.d_head % 2 == 0
        # [t1, t1, t2, t2, t3, t3, ...]
        thetas = enc_base ** (
            -2.0 * tr.arange(1, self.d_head / 2 + 1) / self.d_head
        ).repeat_interleave(2)
        positions = tr.arange(1, max_pos_enc_len + 1).float()
        # [ [1t1, 1t1, 1t2, 1t2, ...],
        #   [2t1, 2t1, 2t2, 2t2, ...],
        #   [3t1, 3t1, 3t2, 3t2, ...],
        #   ...                       ]
        args = positions.reshape(-1, 1) @ thetas.reshape(1, -1)
        self.register_buffer("rope_sin", tr.sin(args))
        self.register_buffer("rope_cos", tr.cos(args))

    def _reorder_for_rope_sin(self, x):
        """Reorders the inputs according to [1, p. 7, eq. (34)] for the
        multiplication with the sinus-part of the RoPE. Configured for x beeing
        having d_head as last dimension, should be of shape
        (batch_size, n_heads, seqlen, d_head).
        """
        # [x1, x3, x5, ...]
        x_odd = x[..., ::2]
        # [x2, x4, x6, ...]
        x_even = x[..., 1::2]
        # [[-x2, x1], [-x4, x3], [-x6, x5], ...]
        x_stacked = tr.stack([-x_even, x_odd], dim=-1)
        # [-x2, x1, -x4, x3, ...]
        return x_stacked.flatten(start_dim=-2)

    def _apply_rope(self, x):
        """Applies RoPE the inputs according to [1, p. 7, eq. (34)].
        Configured for x being of shape (batch_size, n_heads, seqlen, d_head).
        """
        T = x.shape[2]
        x_sin = self._reorder_for_rope_sin(x)
        x_rope = x * self.rope_cos[:T, :] + x_sin * self.rope_sin[:T, :]
        return x_rope

    def forward(self, x, return_attn=True):
        y = self.embedding(x.int())
        B, T, C = y.size()  # batch_size, seqlen, d_model

        # apply key, query, value projections
        q, k, v = self.multi_head_in_projection(y).split(self.d_model, dim=2)

        # separate heads (batch_size, n_heads, seqlen, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # apply RoPE transformation [1, p. 7]
        q_rope = self._apply_rope(q)
        k_rope = self._apply_rope(k)

        # RoPE self attention:
        #   (batch_size, n_heads, seqlen, d_head) x
        #   (batch_size, n_heads, d_head, seqlen)
        #       -> (batch_size, n_heads, seqlen, seqlen)
        #  This is the place, where the rotations get "inserted" into the
        #  attention mechanism as presented in [1, p. 6, eq. 19]. I stick to
        #  the basic `exp` as non-negativities.
        
        # Use softmax
        #att = F.softmax((q_rope @ k_rope.transpose(-2, -1)) * self.inv_sqrt_d_head, dim=-1)

        # Use original implementation with exp (without normalization)
        att_numerator = tr.exp(
            (q_rope @ k_rope.transpose(-2, -1)) * self.inv_sqrt_d_head
        )
        att_denominator = tr.exp((q @ k.transpose(-2, -1)) * self.inv_sqrt_d_head)
        att_denominator = tr.sum(att_denominator, dim=-1, keepdim=True)
        att = att_numerator / att_denominator
        if return_attn:
            return att  # Your L x L matrix
        # (batch_size, n_heads, seqlen, seqlen) x
        #   (batch_size, n_heads, seqlen, d_head)
        # -> (batch_size, n_heads, seqlen, d_head)
        y = att @ v
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.multi_head_out_projection(y)

        # skip-connection and regularization
        y = self.layer_norm(y + x)
        y = self.dropout(y)
        return y

class Conv2DBlock(nn.Module):
    def __init__(self, k=3, device="cpu"):
        super().__init__()
        self.device = device
        self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(k, k), padding="same")

    def forward(self, x):
        x.to(self.device)
        y = self.convolution(x)
        return y
