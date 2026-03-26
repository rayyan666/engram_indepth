import numpy as np
from typing import List, Tuple, Optional
from engram.core import EngramModule, EngramConfig


# ---------------------------------------------------------------------------
# Tiny helpers (real impl would use PyTorch / JAX)
# ---------------------------------------------------------------------------

def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True) + eps
    return (x - mean) / std


def fake_attention(h: np.ndarray) -> np.ndarray:
    """
    Simplified self-attention proxy (identity + small noise).
    In a real model this would be multi-head scaled dot-product attention.
    """
    noise = np.random.default_rng(0).normal(0, 0.01, h.shape).astype(np.float32)
    return h + noise


def fake_ffn(h: np.ndarray) -> np.ndarray:
    """
    Simplified FFN proxy (element-wise GELU-like activation).
    In a real model this might be an MoE FFN (DeepSeek-MoE style).
    """
    return h * (1 / (1 + np.exp(-h)))  # sigmoid-scaled (approx GELU)


# ---------------------------------------------------------------------------
# Transformer Block with optional Engram injection
# ---------------------------------------------------------------------------

class TransformerBlockWithEngram:
    """
    One Transformer block.  If `use_engram=True` the Engram delta is added
    to the residual stream after the FFN, exactly as in the paper.
    """

    def __init__(self, layer_idx: int, config: EngramConfig):
        self.layer_idx = layer_idx
        self.config = config
        self.use_engram = (layer_idx in config.layer_positions)
        self.engram = EngramModule(config) if self.use_engram else None

    def forward(
        self,
        h: np.ndarray,         # (seq_len, hidden_dim)
        token_ids: List[int],
    ) -> Tuple[np.ndarray, dict]:
        """Returns updated hidden states and a trace dict."""
        trace = {"layer": self.layer_idx, "engram_active": self.use_engram}

        # --- Attention sub-layer ---
        h = h + fake_attention(layer_norm(h))

        # --- FFN sub-layer ---
        h_ffn = fake_ffn(layer_norm(h))

        engram_trace = None
        if self.use_engram:
            engram_delta, engram_trace = self.engram.forward(token_ids, h, 0)
            h = h + h_ffn + engram_delta          # FFN + memory lookup
            trace["engram_delta_norm"] = float(
                np.linalg.norm(engram_delta, axis=-1).mean()
            )
        else:
            h = h + h_ffn                          # FFN only (baseline MoE)

        trace["hidden_norm_after"] = float(np.linalg.norm(h, axis=-1).mean())
        if engram_trace:
            trace["engram_steps"] = engram_trace["steps"]

        return h, trace


# ---------------------------------------------------------------------------
# Mini Model: stacks N transformer blocks
# ---------------------------------------------------------------------------

class MiniModelWithEngram:
    """
    Stacks `num_layers` blocks.  Engram only fires at the configured positions.
    This lets us observe *exactly* how the residual stream changes at each layer.
    """

    def __init__(self, config: EngramConfig, num_layers: int = 30):
        self.config = config
        self.blocks = [
            TransformerBlockWithEngram(i, config)
            for i in range(num_layers)
        ]

    def forward(
        self,
        token_ids: List[int],
        hidden_dim: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        if hidden_dim is None:
            hidden_dim = self.config.hidden_dim

        seq_len = len(token_ids)
        rng = np.random.default_rng(1)
        h = rng.normal(0, 0.02, (seq_len, hidden_dim)).astype(np.float32)

        all_traces = []
        for block in self.blocks:
            h, trace = block.forward(h, token_ids)
            all_traces.append(trace)

        return h, all_traces



