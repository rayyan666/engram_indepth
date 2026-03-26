# DeepSeek Engram — Python Workflow Demo

A clean, self-contained Python project that walks through every component of
**DeepSeek's Engram architecture** from the paper:

> *"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"*  
> url : [text](https://arxiv.org/abs/2601.07372)
> Xin Cheng et al., DeepSeek AI & Peking University, January 2026.

---

## The Core Idea

Standard Transformers (and MoE models) **compute** everything — including
trivial static patterns like *"Paris → capital → France"* — wasting GPU
cycles on facts that could be retrieved from a lookup table in O(1).

Engram adds a **Conditional Memory module** at two Transformer layers.
It intercepts tokens, looks up a pre-trained embedding via multi-head
hashing, gates it against the current hidden state, and injects the result
into the residual stream.



## Project Layout

```
engram/
├── engram/
│   ├── __init__.py
│   ├── core.py            TokenizerCompressor, MultiHeadHasher, EngramMemoryTable, ContextGate, EngramModule
│   └── transformer.py     TransformerBlockWithEngram, MiniModelWithEngram
├── demo.py               ←end-to-end walkthrough (run this)
└── README.md
```

---

## Quick Start

```bash
# No external dependencies — pure NumPy
pip install numpy        # if not already installed

python demo.py
```

---

## What the Demo Shows
 
1 TokenizerCompressor -  Case/spacing variants collapsed to canonical IDs 
2 MultiHeadHasher -  N-gram → 8 independent bucket indices (O(1)) 
3 EngramMemoryTable -  Embedding retrieved by averaging H table rows 
4 ContextGate -  Gate strength for "static entity" vs "dynamic reasoning" tokens 
5 EngramModule -  Per-token best N-gram selection + residual delta 
6 MiniModelWithEngram -  Layer-by-layer norm comparison: baseline vs Engram layers 

---

## Key Paper Results (Engram-27B vs MoE-27B, iso-FLOPs)
 -
| Domain | Benchmark | Gain |
|--------|-----------|------|
| Knowledge | MMLU | +3.4 |
| Knowledge | CMMLU | +4.0 |
| Reasoning | BBH | +5.0 |
| Reasoning | ARC-Challenge | +3.7 |
| Code/Math | HumanEval | +3.0 |
| Code/Math | MATH | +2.4 |
| Long-context | Multi-Query NIAH | 84.2 → **97.0** |

The reasoning/math gains (larger than pure knowledge gains) are explained
by the **compute-liberation** effect: offloading static patterns frees early
layers to act as *deeper* reasoning layers.
