import numpy as np
import sys
import os

# Make sure local package is importable
sys.path.insert(0, os.path.dirname(__file__))

from engram.core import (
    TokenizerCompressor,
    MultiHeadHasher,
    EngramMemoryTable,
    ContextGate,
    EngramModule,
    EngramConfig,
)
from engram.transformer import MiniModelWithEngram


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def hr(char="─", width=70):
    print(char * width)

def section(title: str):
    print()
    hr()
    print(f"  {title}")
    hr()

def fmt_vec(v: np.ndarray, max_elems: int = 6) -> str:
    elems = v.flat[:max_elems]
    return "[" + ", ".join(f"{x:+.4f}" for x in elems) + " ...]"


# ─────────────────────────────────────────────────────────────────────────────
# Toy vocabulary (simulating DeepSeek-V3's 128k tokenizer)
# ─────────────────────────────────────────────────────────────────────────────

# A sentence with both "static" (entity) and "dynamic" (reasoning) tokens
# "Paris is the capital of France and its Eiffel Tower is famous"
SENTENCE = "Paris is the capital of France and its Eiffel Tower is famous"
TOKENS   = SENTENCE.split()

# Simulate token ids (real tokenizer would subword-encode these)
VOCAB = {w.lower().strip(): i + 100 for i, w in enumerate(sorted(set(TOKENS)))}
token_ids = [VOCAB[t.lower().strip()] for t in TOKENS]

# Simulate "case-shifted" duplicates: "Paris" vs "paris" → should canonicalize
shifted_ids = [t + 96 if i % 3 == 0 else t for i, t in enumerate(token_ids)]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Tokenizer Compression
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 1 · Tokenizer Compression  (Paper §3.1)")
print("""
Goal: Collapse semantically identical tokens that differ only in
case/spacing into a single canonical ID so that N-gram lookups
are not diluted by surface-form variation.

  'Apple' (id=132) and 'apple' (id=228) → both map to id=132
""")

compressor = TokenizerCompressor(vocab_size=128_000)
original  = shifted_ids[:8]
canonical = compressor.compress(original)

print(f"  Original  ids : {original}")
print(f"  Canonical ids : {canonical}")
collisions = sum(1 for o, c in zip(original, canonical) if o != c)
print(f"\n  ✓  {collisions}/{len(original)} tokens normalised to canonical form")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Multi-Head Hashing
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 2 · Multi-Head Hashing  (Paper §3.2)")
print("""
Goal: Map any N-gram (up to N=3) to H independent bucket indices
in O(1) without maintaining a massive dense table.

  hash_head_0("Paris is the") → bucket 41823
  hash_head_1("Paris is the") → bucket 17209
  ...
""")

hasher = MultiHeadHasher(num_heads=8, table_size=65_536)

test_ngrams = [
    ("Paris",),
    ("Paris", "is"),
    ("Paris", "is", "the"),
    ("Eiffel", "Tower"),
]

print(f"  {'N-gram':<30} {'Head buckets (H=8)'}")
print(f"  {'──────':<30} {'──────────────────'}")
for ng in test_ngrams:
    buckets = hasher.hash_ngram(ng)
    print(f"  {str(ng):<30} {buckets}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Memory Table Lookup
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 3 · Memory Table Lookup  (Paper §3.3)")
print("""
Goal: Retrieve a pre-trained embedding for the N-gram by averaging
the H rows fetched from the embedding table.
Shape: table[H, table_size, dim]  →  lookup → (dim,)

This is the *static knowledge* that would otherwise require the
FFN to reconstruct through multiple attention layers.
""")

mem_table = EngramMemoryTable(num_heads=8, table_size=65_536, embed_dim=256)

for ng in test_ngrams:
    buckets = hasher.hash_ngram(ng)
    vec = mem_table.lookup(buckets)
    print(f"  {str(ng):<30} norm={np.linalg.norm(vec):.4f}  {fmt_vec(vec)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Context-Aware Gate
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 4 · Context-Aware Gate  (Paper §3.4)")
print("""
Goal: Conditionally suppress retrieved memory if it conflicts with
the current hidden state (i.e. we're in a complex reasoning context
where static lookup would add noise, not signal).

  α_t = sigmoid( W_g · h_t + b_g )
  output = α_t ⊙ memory_vec

α close to 1 → memory is useful here (static-pattern token)
α close to 0 → memory is suppressed (dynamic-reasoning token)
""")

gate = ContextGate(hidden_dim=512, embed_dim=256)
rng  = np.random.default_rng(42)

scenarios = [
    ("Static entity  (e.g. 'Paris')",    rng.normal(0, 0.1, 512).astype(np.float32)),
    ("Dynamic reason (e.g. 'therefore')", rng.normal(0, 2.0, 512).astype(np.float32)),
]

buckets = hasher.hash_ngram(("Paris",))
mem_vec = mem_table.lookup(buckets)

print(f"  {'Scenario':<40} {'gate α mean':>12}  {'gated norm':>12}")
print(f"  {'────────':<40} {'──────────':>12}  {'──────────':>12}")
for label, h in scenarios:
    gated  = gate.gate(h, mem_vec)
    alpha  = 1 / (1 + np.exp(-(gate.W_g @ h + gate.b_g)))
    print(f"  {label:<40} {alpha.mean():>12.4f}  {np.linalg.norm(gated):>12.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Full EngramModule forward pass
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 5 · Full EngramModule Forward Pass")
print(f"""
Sentence : "{SENTENCE}"
Tokens   : {TOKENS}
Token IDs: {token_ids}

For each token position, Engram:
  1. Tries all N-grams ending at that position (N = 1,2,3)
  2. Picks the N-gram whose gated memory vector has the highest norm
  3. Projects it to hidden_dim and returns a residual delta
""")

config = EngramConfig(
    max_ngram_size=3,
    num_heads=8,
    table_size=65_536,
    embed_dim=256,
    hidden_dim=512,
    vocab_size=128_000,
    layer_positions=(2, 15),
)

engram_mod = EngramModule(config)
seq_len    = len(token_ids)
hidden_states = rng.normal(0, 0.1, (seq_len, config.hidden_dim)).astype(np.float32)

delta, trace = engram_mod.forward(token_ids, hidden_states, position=0)

print(f"  {'Pos':<4} {'Token':<12} {'Best N-gram':<30} {'Gate':>8}  {'Δ norm':>8}")
print(f"  {'───':<4} {'─────':<12} {'──────────':<30} {'────':>8}  {'──────':>8}")
for step in trace["steps"]:
    t   = step["position"]
    tok = TOKENS[t] if t < len(TOKENS) else "?"
    ng  = str(step["chosen_ngram"]) if step["chosen_ngram"] else "—"
    print(
        f"  {t:<4} {tok:<12} {ng:<30} "
        f"{step['gate_strength']:>8.4f}  {step['memory_norm']:>8.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Full Transformer pass: baseline vs Engram
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 6 · Transformer Pass — Baseline MoE vs Engram")
print("""
The paper uses 30 Transformer blocks.  Engram fires at layers 2 & 15.
Below we show the mean hidden-state norm after each layer.

Expected behaviour from the paper:
  • At Engram layers, the norm jump is larger (memory delta added)
  • Early layers converge faster ("prediction-ready" sooner)
  • This effectively *deepens* the network for complex reasoning
""")

# We run a short model (10 layers) for clarity
short_config = EngramConfig(
    max_ngram_size=3,
    num_heads=8,
    table_size=65_536,
    embed_dim=64,
    hidden_dim=128,
    vocab_size=128_000,
    layer_positions=(2, 5),   # scaled down
)

model = MiniModelWithEngram(short_config, num_layers=10)
_, all_traces = model.forward(token_ids[:6], hidden_dim=128)

print(f"\n  {'Layer':<8} {'Engram?':<10} {'Δ norm (Engram)':<20} {'Hidden norm':<15}")
print(f"  {'─────':<8} {'───────':<10} {'───────────────':<20} {'───────────':<15}")
for t in all_traces:
    engram_tag  = "✓ YES" if t["engram_active"] else "—"
    engram_norm = f"{t.get('engram_delta_norm', 0.0):.6f}" if t["engram_active"] else "n/a"
    print(
        f"  {t['layer']:<8} {engram_tag:<10} {engram_norm:<20} {t['hidden_norm_after']:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

section("SUMMARY — What Engram Solves")
print("""
  Problem (Standard Transformer / MoE)
  ──────────────────────────────────────
  • Model spends GPU cycles "computing" static facts from scratch
    (e.g. reconstructing "Paris → capital → France" every time)
  • This wastes "effective depth" on trivial pattern matching
  • Long contexts suffer because attention is saturated with locals

  Engram Solution
  ───────────────
  • Insert a conditional memory module at 2 key layers
  • N-gram → O(1) hash lookup → pre-trained embedding
  • Context gate suppresses lookup when reasoning is complex
  • Result: early layers "finish" faster, freeing depth for reasoning

  Paper Results (27B model, strict iso-FLOPs vs MoE baseline)
  ─────────────────────────────────────────────────────────────
  Knowledge   :  MMLU +3.4 · CMMLU +4.0 · MMLU-Pro +1.8
  Reasoning   :  BBH  +5.0 · ARC-Challenge +3.7 · DROP +3.3
  Code / Math :  HumanEval +3.0 · MATH +2.4 · GSM8K +2.2
  Long-context:  Multi-Query NIAH 84.2 → 97.0
""")

hr("═")
print("  DeepSeek Engram demo complete.  See engram/core.py for full source.")
hr("═")
