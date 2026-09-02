"""Evaluate the trained phage LM on real host genomes (perplexity + CI).

Tests whether the model genuinely learned DNA statistics: Daphnia magna
(host the model was trained toward via CDS; the genome itself was never
seen) vs E. coli (a foreign superbug genome never in training).
"""
import math
import random
import sys
import time

import torch

sys.path.insert(0, "Autophage")
from phage_lm import load_lm                       # noqa: E402
from phage_genome import parse_fasta               # noqa: E402

CKPT = "Autophage/checkpoints/phage_lm_big.pt"

model, tok = load_lm(CKPT)
device = next(model.parameters()).device
model.eval()
print(f"model: {sum(p.numel() for p in model.parameters()):,} params on {device}\n")


def eval_seq(seq, n_chunks=300, seed=7):
    """Per-chunk loss list over random non-overlapping 512-token windows."""
    rng = random.Random(seed)
    seq = "".join(c for c in seq.upper() if c in "ACGT")
    toks = tok.encode(seq)
    if len(toks) < 600:
        return []
    losses = []
    with torch.no_grad():
        for _ in range(n_chunks):
            i = rng.randrange(0, len(toks) - 511)
            t = torch.tensor([toks[i:i + 512]], dtype=torch.long, device=device)
            x, y = t[:, :-1], t[:, 1:]
            losses.append(model(x, targets=y)[1].item())
    return losses


def report(name, seq, chunks=300, seed=7):
    t0 = time.time()
    losses = eval_seq(seq, n_chunks=chunks, seed=seed)
    n = len(losses)
    mean = sum(losses) / n
    var = sum((l - mean) ** 2 for l in losses) / max(1, n - 1)
    se = math.sqrt(var / n)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    print(f"{name}:")
    print(f"  {n} windows x 511 tokens | loss {mean:.3f} "
          f"(95% CI [{lo:.3f}, {hi:.3f}])")
    print(f"  perplexity {math.exp(mean):.1f} | {mean/tok.k:.3f} bits/base "
          f"| {time.time()-t0:.0f}s\n")


print("=== Reference 1: Daphnia magna genome (never in training; 161 Mb) ===")
daph = parse_fasta(open(
    "Autophage/data/Daphnia_magna_NIES/Daphnia_magna_NIES_genome.fa").read())[1]
report("Daphnia genome (40 Mb sampled)", daph[:40_000_000], seed=7)

print("=== Reference 2: E. coli K-12 (foreign superbug, never in training) ===")
ecoli = parse_fasta(open("/tmp/host_cache/GCF_000005845.2.fasta").read())[1]
report("E. coli K-12 genome", ecoli, seed=11)

print("=== Reference 3: Daphnia CDS (mostly training distribution) ===")
cds = parse_fasta(open(
    "Autophage/data/Daphnia_magna_NIES/Daphnia_magna_NIES_cds.fa").read())[1]
report("Daphnia CDS", cds, seed=3)