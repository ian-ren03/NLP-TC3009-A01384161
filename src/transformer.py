import math
import torch
import torch.nn as nn


# Configuraciones oficiales del proyecto (Sección 4 del enunciado)
MODEL_B_CONFIG = {
    "name": "modelo_b_transformer_compacto",
    "d_model": 128,
    "num_heads": 4,
    "d_ff": 512,
    "num_layers": 2,
    "dropout": 0.1,
    "max_seq_len": 256,
}

MODEL_C_CONFIG = {
    "name": "modelo_c_transformer_base",
    "d_model": 512,
    "num_heads": 8,
    "d_ff": 2048,
    "num_layers": 6,
    "dropout": 0.1,
    "max_seq_len": 256,
}


class PositionalEmbedding(nn.Module):
    """Codificación posicional sinusoidal compatible con batch_first=True.

    Corrección obligatoria de dimensiones:
    La matriz posicional `pe` tiene forma exacta (1, max_seq_len, d_model) y se
    suma a lo largo de la dimensión temporal `x.size(1)` para hacer broadcasting
    correcto sobre el tamaño de batch (dimensión 0).
    """

    def __init__(self, d_model: int, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Forma obligatoria: (1, max_seq_len, d_model) para tensores (B, T, D)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Capa del Encoder Transformer (Pre-LN) con Multi-Head Self-Attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x


class DecoderLayer(nn.Module):
    """Capa del Decoder Transformer (Pre-LN) con Self-Attention causal y Cross-Attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 1. Masked Self-Attention
        x_norm = self.norm1(x)
        self_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(self_out)

        # 2. Cross-Attention (query=decoder, key/value=encoder memory)
        x_norm = self.norm2(x)
        cross_out, attn_weights = self.cross_attn(
            x_norm,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        x = x + self.dropout2(cross_out)

        # 3. Feed-Forward
        ffn_out = self.ffn(self.norm3(x))
        x = x + self.dropout3(ffn_out)
        return x, attn_weights


class TransformerNMT(nn.Module):
    """Arquitectura Encoder-Decoder Transformer desde cero para NMT (EN -> ES).

    Sirve tanto para el Modelo B (Compacto) como para el Modelo C (Base)
    variando d_model, num_heads, d_ff y num_layers.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.scale = math.sqrt(d_model)

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEmbedding(d_model, max_seq_len, dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Máscara booleana triangular superior: True indica posiciones futuras bloqueadas."""
        return torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        )

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pos_encoding(self.src_embedding(src) * self.scale)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.pos_encoding(self.tgt_embedding(tgt_in) * self.scale)
        last_attn = None
        for idx, layer in enumerate(self.decoder_layers):
            want_attn = return_attn and (idx == len(self.decoder_layers) - 1)
            x, attn_weights = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                return_attn=want_attn,
            )
            if want_attn:
                last_attn = attn_weights
        logits = self.output_proj(self.decoder_norm(x))
        return logits, last_attn

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        return_attn: bool = False,
    ):
        src_key_padding_mask = src == self.pad_idx
        tgt_key_padding_mask = tgt_in == self.pad_idx
        tgt_mask = self.generate_causal_mask(tgt_in.size(1), tgt_in.device)

        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        logits, attn_weights = self.decode(
            tgt_in,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            return_attn=return_attn,
        )
        if return_attn:
            return logits, attn_weights
        return logits

    @torch.no_grad()
    def generate_greedy(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 64,
        return_attn: bool = False,
    ):
        """Decodificación autorregresiva greedy (soporta lotes y extracción de atención)."""
        self.eval()
        batch_size = src.size(0)
        device = src.device

        src_key_padding_mask = src == self.pad_idx
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        last_attn = None

        for _ in range(max_len):
            tgt_mask = self.generate_causal_mask(ys.size(1), device)
            logits, attn_weights = self.decode(
                ys,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_key_padding_mask,
                return_attn=return_attn,
            )
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, eos_idx), next_token
            )
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_idx)
            if return_attn:
                last_attn = attn_weights
            if finished.all():
                break

        if return_attn:
            return ys, last_attn
        return ys


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
