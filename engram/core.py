import math
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# 1. Tokenizer Compression  (paper §3.1)

class TokenizerCompressor:
    """
    subjective mapping  P : V → V′
    Collapses token IDs that differ only in case / leading-space so that
    "Apple", "apple", " apple" all hash to the same canonical slot
    This maximises *semantec density* of the N-gram tables.
    """

    def __init__(self, vocab_size: int = 128_000):
        self.vocab_size = vocab_size
        # Precompute a simple deterministic canonical map
        # real impl uses tokenizer metadata; here we use mod-arithmetic
        self._canon_map = self._build_canon_map()

    def _build_canon_map(self) -> np.ndarray:
        """Map every token → its canonical token-id (strip case / spacing)."""
        canon = np.arange(self.vocab_size, dtype=np.int32)
        # simulate: tokens that are "shifted" variants point to their base form
        # e.g. token 32 and 32+96 represent same word with orwithout leading space
        shift = 96
        for i in range(shift, self.vocab_size):
            canon[i] = i - shift  # collapse to base
        return canon

    def compress(self, token_ids: List[int]) -> List[int]:
        """Return canonical ids for a token sequence."""
        return [int(self._canon_map[t % self.vocab_size]) for t in token_ids]


# ---------------------------------------------------------------------------
# 2. Multi-Head Hashing  (paper §3.2)
# ---------------------------------------------------------------------------

class MultiHeadHasher:
    """
    maps an N-gram (tuple of canonical token-ids) → H bucket indices,
    one per head.  Uses deterministic SHA-256 seeded hashing so there are
    NO learnable parameters here — only the embedding values are learned.

    why multiple heads?
      Collisions in a single hash table would cause catastrophic interference
      H independent hash functions spread load and let the model average out
      collision noise, similar to multihead attention.
    """

    def __init__(self, num_heads: int = 8, table_size: int = 65_536):
        self.num_heads = num_heads
        self.table_size = table_size
        # Each head gets a different salt so hashes are independent
        self.salts = [f"engram_head_{i}".encode() for i in range(num_heads)]

    def hash_ngram(self, ngram: Tuple[int, ...]) -> List[int]:
        """
        Return one bucket index per head for the given N-gram.
        O(1) — no iteration over the table.
        """
        key = "_".join(str(t) for t in ngram).encode()
        indices = []
        for salt in self.salts:
            digest = hashlib.sha256(salt + key).digest()
            idx = int.from_bytes(digest[:4], "big") % self.table_size
            indices.append(idx)
        return indices

# 3. Engram Memory Table  (paper §3.3)

class EngramMemoryTable:
    """
    The parametric embedding store: shape [H, table_size, dim].

    During training the table is updated via back-prop exactly like any
    embedding layer.  During inference it is READ-ONLY — deterministic
    addressing lets the runtime prefetch rows from DRAM with <3% overhead
    (paper §5.4), even for 100B-parameter tables.

    Here we initialise with small random values to simulate a trained table.
    """

    def __init__(self, num_heads: int = 8, table_size: int = 65_536,
                 embed_dim: int = 256, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Shape: (H, table_size, dim)
        scale = 1.0 / math.sqrt(embed_dim)
        self.table = rng.normal(0, scale,
                                (num_heads, table_size, embed_dim)).astype(np.float32)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def lookup(self, head_indices: List[int]) -> np.ndarray:
        """
        Given one bucket index per head, return the average embedding.
        Shape: (embed_dim,)
        """
        rows = [self.table[h, idx] for h, idx in enumerate(head_indices)]
        return np.mean(rows, axis=0)  # multi-head average

# 4. Context-Aware Gate  (paper §3.4)

class ContextGate:
    """
    α_t = sigmoid( W_g · h_t + b_g )

    The gate asks: "does the current hidden state h_t actually need this
    memory retrieval?"  If the answer is no (e.g. we're in the middle of a
    complex reasoning chain), α ≈ 0 and the retrieved embedding is suppressed.
    This is the 'conditional' in Conditional Memory.

    Parameters W_g and b_g are learned during training.
    Here we initialise randomly to illustrate the mechanism.
    """

    def __init__(self, hidden_dim: int, embed_dim: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(hidden_dim)
        self.W_g = rng.normal(0, scale, (embed_dim, hidden_dim)).astype(np.float32)
        self.b_g = np.zeros(embed_dim, dtype=np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def gate(self, hidden_state: np.ndarray, memory_vec: np.ndarray) -> np.ndarray:
        """
        Compute gated memory contribution.
        hidden_state: (hidden_dim,)
        memory_vec:   (embed_dim,)
        Returns:      (embed_dim,)  — element-wise gated memory
        """
        alpha = self._sigmoid(self.W_g @ hidden_state + self.b_g)  # (embed_dim,)
        return alpha * memory_vec

# 5. EngramModule  (full forward pass, paper §3)

@dataclass
class EngramConfig:
    max_ngram_size: int = 3       # paper uses N=3
    num_heads: int = 8            # paper uses H=8
    table_size: int = 65_536      # per-head rows
    embed_dim: int = 256          # memory vector dim (project to hidden_dim)
    hidden_dim: int = 512         # transformer hidden dim
    vocab_size: int = 128_000
    layer_positions: Tuple = (2, 15)  # paper inserts at layers 2 and 15


class EngramModule:
    """
    Full Engram forward pass — meant to be called INSIDE a Transformer block
    at the residual connection point.

    residual_stream += engram_module(token_ids, hidden_state, position)
    """

    def __init__(self, config: EngramConfig):
        self.config = config
        self.compressor = TokenizerCompressor(config.vocab_size)
        self.hasher = MultiHeadHasher(config.num_heads, config.table_size)
        self.memory = EngramMemoryTable(config.num_heads, config.table_size,
                                        config.embed_dim)
        self.gate = ContextGate(config.hidden_dim, config.embed_dim)
        # Projection: memory dim → hidden dim
        rng = np.random.default_rng(99)
        self.proj = rng.normal(
            0, 1.0 / math.sqrt(config.embed_dim),
            (config.hidden_dim, config.embed_dim)
        ).astype(np.float32)

    def forward(
        self,
        token_ids: List[int],
        hidden_states: np.ndarray,   # shape (seq_len, hidden_dim)
        position: int,               # current token position
    ) -> Tuple[np.ndarray, dict]:
        """
        Returns:
          delta   – (seq_len, hidden_dim) to ADD to the residual stream
          trace   – dict of intermediate values for visualisation
        """
        seq_len = hidden_states.shape[0]
        delta = np.zeros_like(hidden_states)
        trace_list = []

        canon_ids = self.compressor.compress(token_ids)

        for t in range(seq_len):
            # Build all N-grams ending at position t
            best_memory = None
            best_gate_strength = -1.0
            chosen_ngram = None

            for n in range(1, self.config.max_ngram_size + 1):
                start = t - n + 1
                if start < 0:
                    continue
                ngram = tuple(canon_ids[start: t + 1])

                # O(1) lookup
                head_indices = self.hasher.hash_ngram(ngram)
                mem_vec = self.memory.lookup(head_indices)

                # Gate against current hidden state
                h_t = hidden_states[t]
                gated = self.gate.gate(h_t, mem_vec)
                gate_strength = float(np.linalg.norm(gated))

                if gate_strength > best_gate_strength:
                    best_gate_strength = gate_strength
                    best_memory = gated
                    chosen_ngram = ngram

            if best_memory is not None:
                # Project memory dim → hidden dim and add to residual
                delta[t] = self.proj @ best_memory

            trace_list.append({
                "position": t,
                "token_id": token_ids[t],
                "canon_id": canon_ids[t],
                "chosen_ngram": chosen_ngram,
                "gate_strength": best_gate_strength,
                "memory_norm": float(np.linalg.norm(delta[t])),
            })

        return delta, {"steps": trace_list}
