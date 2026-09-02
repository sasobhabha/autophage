# How to interact with the trained models

Everything you need to load, run, evaluate, and retrain the Evo-style
genomic language model (and to close the loop with the verifier).

## 1. The artifacts

All live in `SciAgent/checkpoints/` (gitignored — regenerate via `train` or
attach to a GitHub Release for sharing).

| File | Size | What it is | Load with |
|---|---|---|---|
| `phage_lm_daphnia.pt` | 49.7 MB | **Best**: trained on the full 215.8 Mb dataset (46 phages + 31,317 *D. magna* CDS + whole genome), fp32 | `torch.load` / `load_lm` |
| `phage_lm_daphnia_bf16.pt` | 28.0 MB | Same weights in bfloat16 (half size; recommended for sharing) | `load_lm` |
| `phage_lm_daphnia.pkl` | 56.0 MB | Same object stored with plain `pickle` (not `torch.save`) | `pickle.load` only |
| `phage_lm_full.pt` | 49.7 MB | Earlier run, 54 Mb dataset (phages + CDS, no whole genome) | `torch.load` / `load_lm` |
| `phage_lm_big.pt` | 49.7 MB | Early interrupted run (keep as a milestone) | `torch.load` / `load_lm` |

All are the same architecture: **12,212,352 parameters**, config
`{"d_model": 384, "n_layers": 6, "n_heads": 6, "seq_len": 512,
"vocab_size": 4099}` (4096 six-mers + BOS/EOS/PAD). They differ only in
learned weights.

> ⚠️ **`.pkl` vs `.pt`**: `torch.save` writes a zip container, so
> `torch.load()` will *not* read the `.pkl` (you'll get "Invalid magic
> number"). Use `pickle.load()` for the `.pkl`, `torch.load()` for the
> `.pt`. Content is equivalent.

## 2. Quick start

```bash
# python 3.10+; torch + numpy in a venv
python3 -m venv .venv && .venv/bin/pip install torch numpy

# generate a 50 kb genome from the trained model (MPS/CUDA/CPU auto-detected)
.venv/bin/python SciAgent/phage_lm.py generate \
    --model SciAgent/checkpoints/phage_lm_daphnia_bf16.pt \
    --length 50000 --temperature 0.75 --out /tmp/phage_lm.fasta

# verify what it produced (phage-biology checks + protein production)
.venv/bin/python SciAgent/phage_genome.py validate --input /tmp/phage_lm.fasta
.venv/bin/python SciAgent/phage_genome.py proteins --input /tmp/phage_lm.fasta

# screen it against real superbug genomes (downloads/caches from NCBI)
.venv/bin/python SciAgent/phage_genome.py compat --input /tmp/phage_lm.fasta \
    --cache SciAgent/data/hosts
```

## 3. Interact from Python

### 3a. Recommended: `load_lm` (handles any `.pt`, fp32 or bf16)

```python
import sys
sys.path.insert(0, "SciAgent")
from phage_lm import load_lm, PhageTokenizer

model, tok = load_lm("SciAgent/checkpoints/phage_lm_daphnia_bf16.pt")
# model is a decoder-only Transformer (GPT-style, rotary embeddings)

# sample a genome: prompt = token ids (6-mers), length = token count
prompt = tok.encode("ATGCGT") or [0]          # needs >= 6 bases
ids = model.generate(prompt, length=5_000, temperature=0.8, top_k=50)
seq = tok.decode(ids)                          # 6-mer ids -> DNA string
print(len(seq), "bp")
```

Sampling knobs: `temperature` (lower = more conservative, 0.6–0.8 is a good
range for DNA), `top_k` (nucleus over k = 50).

### 3b. Plain pickle (.pkl) — no `torch.load` required

```python
import pickle, torch
from phage_lm import PhageGenomeLM, PhageTokenizer

with open("SciAgent/checkpoints/phage_lm_daphnia.pkl", "rb") as f:
    data = pickle.load(f)                       # {"model": state_dict, "config": {...}}

tok = PhageTokenizer()
model = PhageGenomeLM(
    vocab_size=data["config"]["vocab_size"],
    d_model=data["config"]["d_model"],
    n_layers=data["config"]["n_layers"],
    n_heads=data["config"]["n_heads"],
    max_seq_len=data["config"]["seq_len"] + 1,
)
model.load_state_dict(data["model"])
model.eval()
```

### 3c. Measure perplexity on any genome (e.g. "does it understand this DNA?")

```python
import math, random, torch
seq = "AAAA..."                                # any DNA string
toks = tok.encode(seq)
losses = []
with torch.no_grad():
    for _ in range(100):
        i = random.randrange(0, len(toks) - 511)
        t = torch.tensor([toks[i:i + 512]], dtype=torch.long,
                         device=next(model.parameters()).device)
        losses.append(model(t[:, :-1], targets=t[:, 1:])[1].item())
loss = sum(losses) / len(losses)
print(f"loss {loss:.3f} | perplexity {math.exp(loss):.1f}")
```

Reference numbers for the full-dataset model (lower = better understood):
Daphnia genome **3.9** perplexity · *Pasteuria ramosa* **3.9** · foreign
*E. coli* **4.2** · shuffled control **4.2** · uniform random **6.2**.
Ready-made version: `SciAgent/eval_lm_hosts.py`.

## 4. Retrain or fine-tune

```bash
# from scratch on the full dataset (reproduces phage_lm_daphnia.pt;
# ~9 min on Apple MPS, ~2600 steps, gradient accumulation + bf16)
.venv/bin/python SciAgent/phage_lm.py train \
    --data SciAgent/data/train_full --out SciAgent/checkpoints/mymodel.pt \
    --d-model 384 --n-layers 6 --n-heads 6 --seq-len 512 \
    --batch 8 --grad-accum 2 --max-steps 2600

# smaller/faster sanity run
.venv/bin/python SciAgent/phage_lm.py train --data SciAgent/data/train_full \
    --out /tmp/small.pt --max-steps 200
```

Training loop reads `<dir>/*.fasta`, splits 90/10 train/val per file,
evaluates on a validation sample every 100 steps, and keeps the best
checkpoint. Memory accounting for any config is printed at startup
(`[mem]`) — a 2B-parameter model needs ~40+ GB on a GPU cluster.

## 5. Design scripts (the deterministic loop — no checkpoint needed)

The *verified-phage* deliverables come from the symbolically-grounded
builder, which needs no trained weights:

```bash
# any host genome -> complete annotated phage (Daphnia by default)
.venv/bin/python SciAgent/build_daphnia_phage.py

# ... or L. acidophilus FSI4 (KEGG T03681), or any other genome
.venv/bin/python SciAgent/build_daphnia_phage.py \
    --host-name Lactobacillus_acidophilus_FSI4 \
    --genome-fasta SciAgent/data/hosts/Lactobacillus_acidophilus_FSI4.fasta \
    --gtf SciAgent/data/hosts/Lactobacillus_acidophilus_FSI4.gff \
    --prefix lacto_phage_1 --seed 7 --no-pathogen-screen

# batch: every FASTA in a directory -> validated phage (no annotation needed)
.venv/bin/python SciAgent/batch_any_genome.py --dir /tmp/host_cache \
    --out SciAgent/outputs/any_genome_batch.json
```

## 6. Device & precision notes

- Device auto-detect: MPS (Apple Silicon) → CUDA → CPU. Override with
  `device=` in `load_lm(...)` / `train_lm(...)`.
- bf16 weights load and run fine; generation is fastest on MPS/CUDA.
- The `.pt` checkpoints contain **fp32** weights by default (training uses
  bf16 autocast internally but checkpoints are stored in fp32 for
  portability; `phage_lm_daphnia_bf16.pt` is the halved variant).
- Sampled output is *proposal*, not validated biology: pass any generated
  FASTA through `validate` / `proteins` / `compat` (§2) — that loop is the
  point of the pipeline.