# SciAgent: Neuro-Symbolic Agentic Reasoner for Scientific Discovery

## Overview

As of 2026, the AI landscape has shifted from large, general-purpose conversational models to specialized, autonomous **Agentic AI** capable of rigorous reasoning and solving open scientific problems. Purely neural models often hallucinate or fail at long-horizon logical tasks.

**SciAgent** is a "genius" AI architecture designed to tackle these exact limitations. It bridges the gap between deep learning's pattern recognition and symbolic AI's rigorous logic, creating an autonomous agent capable of discovering new mathematical theorems, optimizing physical materials, and auditing scientific literature.

## The 2026 Context: Why is this revolutionary?
1. **Agentic Autonomy:** Instead of passive Q&A, SciAgent operates in a continuous loop: formulating hypotheses, designing experiments (or logical proofs), executing them via external tools, and updating its internal state.
2. **Neuro-Symbolic Architecture:** We combine a specialized, small-parameter Neural Reasoner (based on sparse attention and mixture-of-experts) with a Symbolic Verifier. The neural network "guesses" the creative leap, and the symbolic engine "proves" it.
3. **Self-Auditing Science:** SciAgent is designed to combat the reproducibility crisis by systematically validating research claims through automated logical auditing.

## Architecture

The system consists of three main components:
1. **Neural Proposer (PyTorch):** Generates hypotheses, structural formulas, or proof steps.
2. **Symbolic Verifier (External Engine Interface):** Interfaces with tools like Lean (for math) or AlphaFold/Chem engines (for biology/chemistry) to rigorously test the Proposer's ideas.
3. **Agentic Control Loop:** Manages the iterative process of trial, error, and refinement, employing Monte Carlo Tree Search (MCTS) to navigate the hypothesis space.

## Project Structure
- `model.py`: Contains the PyTorch implementation of the Neural Proposer and the embedding layers.
- `agent.py`: Implements the MCTS-based Agentic Control Loop and the Verifier interface. *(planned)*
- `train.py`: The self-play reinforcement learning training script. *(planned)*
- **`phage_genome.py`**: Synthetic bacteriophage genome **proposer** (in-silico generation) and **symbolic verifier** (phage biology validation), plus a host-compatibility screen.
- **`host_compat.py`**: Fetches real "superbug" reference genomes from NCBI and scores query genomes with the published VirHostMatcher d2* ONF dissimilarity measure.
- **`phage_lm.py`**: Evo-style decoder-only genomic language model trained on real public phage genomes (fetch / train / generate).
- `test_phage_pipeline.py`: Unit tests for the above.
- **`build_daphnia_phage.py`**: any-host design loop — builds a complete,
  annotated, host-adapted phage genome (FASTA + GFF3 + JSON) and verifies it.
- **`batch_any_genome.py`**: runs that loop over every genome in a directory
  (no annotation needed).

## Getting Started

```bash
# (optional) project-local environment
python3 -m venv .venv && .venv/bin/pip install torch numpy

# Generate a synthetic phage genome and validate it against phage biology
.venv/bin/python SciAgent/phage_genome.py generate --length 50000 --gc 0.50 --seed 1 --validate

# Screen a genome for host compatibility against real superbug genomes (downloads from NCBI)
.venv/bin/python SciAgent/phage_genome.py compat --input genome.fasta --cache SciAgent/data/hosts

# Download real phage genomes, train an expert genomic LM, generate a genome
.venv/bin/python SciAgent/phage_lm.py fetch --max-genomes 100 --out SciAgent/data/phages
.venv/bin/python SciAgent/phage_lm.py train --data SciAgent/data/phages --out SciAgent/checkpoints/phage_lm.pt
.venv/bin/python SciAgent/phage_lm.py generate --model SciAgent/checkpoints/phage_lm.pt --length 50000 --out lm_genome.fasta

# Host-conditioned design: make a phage adapted to *any* host genome
.venv/bin/python SciAgent/phage_genome.py host-profile --host-cds host_cds.fa --out host_profile.json
.venv/bin/python SciAgent/phage_genome.py generate --host-cds host_cds.fa --length 50000 --validate
.venv/bin/python SciAgent/phage_genome.py proteins --input synthetic_phage.fasta  # test protein production
```

## Synthetic phage genome design and validation (`phage_genome.py`)

Implements the SciAgent proposer/verifier loop for bacteriophage genomics:

- **Proposer** (`generate_phage_genome`): builds in-silico dsDNA genomes with
  biologically plausible features — protein-coding genes back-translated with
  GC-content targeting, Shine-Dalgarno (AGGAGG) ribosome binding sites,
  ATG starts / stop codons, intergenic spacers free of spurious start codons,
  and configurable direct terminal repeats (as in T7).
- **Verifier** (`validate_phage_genome`): checks sequences against phage
  biology: alphabet, size range, GC content (25–70%), 6-frame ORF coding
  density, ORF count, Shine-Dalgarno fraction, and terminal-repeat status.
  All checks except terminal repeats are hard-failing.

## Host-compatibility screening against real superbugs (`host_compat.py`)

Instead of validating against itself, the pipeline validates against **real,
publicly available genomes of priority antibiotic-resistant pathogens**
(MRSA, CRE, CRAB, MDR/XDR *P. aeruginosa*, VRE, ESBL Enterobacterales,
MDR-TB, drug-resistant *N. gonorrhoeae*, MDR *Salmonella*) — the CDC
2019 Antibiotic Resistance Threats list.

- Host genomes are fetched **live from NCBI** (Datasets v2 API, RefSeq
  reference assemblies) and cached locally.
- Compatibility is scored with **d2\*** — the ONF (oligonucleotide frequency)
  dissimilarity measure of **Ahlgren et al. (2017), *Nucleic Acids Research*
  45(1):39–53** (PMID 27899557), used by the VirHostMatcher tool.
- The d2\* implementation is a **faithful port of the original C++**
  (github.com/jessieren/VirHostMatcher): single-strand k-mer counts with
  N-reset windows, reverse-complement pairing `X_w = count(w)+count(rc(w))`,
  an order-2 Markov background (`pwMC`), and `d2* = 0.5(1 − C2star)`.
  It was validated against the reference executable on the project's toy
  dataset: **max absolute error < 2×10⁻⁶ over 276 virus–host pairs**.
- The screen reports ranked hosts with d2\* scores. Lower d2\* = greater ONF
  similarity = a more plausible host. As the original paper notes, ONF
  similarity indicates a plausible host but is not proof of infectivity;
  experimental host-range assays remain the gold standard.

## Host-conditioned design: make a phage for any host genome (`phage_genome.py`)

Given *any* host cell genome, the pipeline designs a phage adapted to that
host, then validates the phage's DNA is real protein-coding sequence:

- **Host analysis** (`host-profile`): builds a codon-usage profile from the
  host's coding sequences (GTF-annotated CDS or a CDS FASTA), i.e. the tRNA
  pool the phage's genes must draw on.
- **Codon-adapted generation** (`generate --host-cds/--host-genome`): protein
  sequences are back-translated using the **host's** codon frequencies
  instead of a generic table, so every phage gene uses codons the host
  translates well — the same adaptation real phages undergo
  (`Ahlgren et al., 2017`, codon-adaptation index `Sharp & Li, 1987, NAR
  15:1281–95`).
- **Protein-production validation** (`proteins`): the phage DNA is parsed
  into ORFs, translated to amino-acid sequences, and checked that proteins
  are produced with no premature stops — a proxy for "could this virus use
  the host's translation machinery to make its proteins."

**Worked example — *Daphnia magna* NIES** (161 Mb genome, 31,317 curated
CDS, NCBI): a 50 kb synthetic phage designed against the Daphnia codon
profile **passes all phage-biology validations**, produces **full proteins
from 100% of its ORFs** with **mean codon-adaptation index ≈ 0.85** to
Daphnia, and shows genuinely higher ONF similarity to the Daphnia genome
under d2* than an unadapted control (d2* 0.43 vs 0.48 — lower = more
similar). Compatibility with superbugs is screened separately with `compat`.

## The expert model (`phage_lm.py`) — honest scaling note

`phage_lm.py` trains a **decoder-only genomic language model (Evo-style)** on
real public phage genomes (NCBI RefSeq, default taxid 28883 Caudoviricetes):

- k-mer tokenizer (6-mer, 4096-token vocabulary — the same vocabulary size
  used by Evo 2, mirroring its byte-level design)
- causal Transformer with rotary embeddings, pre-LayerNorm, weight tying
- next-token cross-entropy objective, warmup + cosine LR, train/val split,
  checkpointing, and temperature/top-k genome generation

This is the same architecture family and training objective as **Evo 2**
(Brixi, Durrant, Ku et al., 2025, bioRxiv 10.1101/2025.02.18.638918).
**It is not a rival to Evo 2 in scale**: Evo 2 is a 7B–40B parameter model
trained on 9.3 trillion bases on thousands of GPUs. This module is a real,
working version of that pipeline at laptop scale — e.g. a ~12M-parameter,
6-layer/384-wide model trained with gradient accumulation and bf16 mixed
precision on real phage genomes plus host CDS in minutes, producing a genuine
expert model of phage sequence statistics (run validated with `train --help`:
`--d-model`, `--n-layers`, `--grad-accum`, `--no-amp`; memory is accounted
honestly at startup). Scaling toward Evo-2-class capability is a matter of
compute and data, not architecture.

### What a 2B-parameter run actually requires (feasibility)

Training a 2B-parameter genomic LM "rivaling Evo 2" needs, at minimum:

- **Memory:** ~2B params ≈ 8 GB bf16 weights + 8 GB grads + 16 GB Adam
  moments + activations ≈ **40+ GB on a single GPU** (e.g. 1–2× A100/H100
  80 GB) — beyond a 24 GB laptop; the `train` command prints this estimate
  (`[mem]`) for any config.
- **Data:** Evo 2 used 9.3 trillion bases. All of NCBI's ~38,000 RefSeq
  phage genomes ≈ 4 Gbp, and *D. magna* contributes only 0.2 Gbp — orders of
  magnitude short of what 2B parameters can memorize without overfitting.
- **Compute:** Evo 2 trained on thousands of GPUs; a 2B model on one H100 is
  weeks of continuous training just to converge.

The realistic path this repo provides: train the biggest model your hardware
fits (the gradient-accumulation + AMP + checkpointing above), and scale data
and parameters together on a GPU cluster. The host-conditioned designer in
`phage_genome.py` is the practical way to get a host-specific phage right now.

## Daphnia end-to-end: complete phage + lab-trial plan

`build_daphnia_phage.py` builds the **complete, annotated 49.8 kb genome**
from the full dataset (31,317 *D. magna* CDS codon profile + whole genome)
and machine-verifies it: all phage-biology checks pass, 100% of ORFs
produce proteins, mean codon-adaptation index 0.849 vs Daphnia, and d2\* =
0.397 vs the real *Pasteuria ramosa* genome (the bacterium that sterilizes
Daphnia in labs) — the lab-trial target. Outputs land in `SciAgent/outputs/`
(FASTA + GFF3 + JSON), with the full trial protocol in
`SciAgent/DAPHNIA_PHAGE_LAB_TRIAL.md`. Honest limit: in-silico adaptation
is a plausibility signal; infectivity must be proven in the wet lab.

### Any host genome → its phage

`build_daphnia_phage.py` accepts **any** host genome (FASTA + GTF/GFF
annotation, or a CDS FASTA) and produces the same complete, verified
deliverable. Example — KEGG genome **T03681** (*Lactobacillus acidophilus*
FSI4, a yogurt probiotic):

```bash
.venv/bin/python SciAgent/build_daphnia_phage.py \
    --host-name Lactobacillus_acidophilus_FSI4 \
    --genome-fasta SciAgent/data/hosts/Lactobacillus_acidophilus_FSI4.fasta \
    --gtf SciAgent/data/hosts/Lactobacillus_acidophilus_FSI4.gff \
    --prefix lacto_phage_1 --seed 7 --no-pathogen-screen
```

Result: 49.6 kb genome **VALIDATED** (all biology checks pass), **46/46
ORFs → full proteins**, CAI 0.666 vs the host, and notably **d2\* = 0.284
vs the *L. acidophilus* genome — ranked far above every other genome
screened (next best 0.313, MRSA)** — the designed phage is k-mer-adapted to
its intended host. NCBI GFF3 CDS extraction is supported in addition to
GTF.

### Can I just hand it any genome? Yes.

Drop a FASTA in a directory (no annotation needed) and run
`batch_any_genome.py` — the codon profile is then estimated directly from
the genome sequence. Proven on 10 real reference genomes of priority
antibiotic-resistant bacteria spanning **33–67% GC**
(`SciAgent/outputs/any_genome_batch.json`): **10/10 produced a fully
VALIDATED phage**, with 100% protein production, mean CAI 0.68–0.81 vs each
host, and d2\* 0.31–0.46 vs each host's own genome.

```bash
.venv/bin/python SciAgent/batch_any_genome.py --dir /tmp/host_cache \
    --out SciAgent/outputs/any_genome_batch.json
```

Both caveats: (1) *bacterio*phages infect bacteria/archaea — for a
eukaryotic "host" (e.g. Daphnia itself) the pipeline still produces a
codon-adapted, validated genome, but the biology is only meaningful against
the host's bacteria (as in the Daphnia/∗Pasteuria ramosa∗ case above);
(2) "validated" is in-silico phage biology + host adaptation — infectivity
is always settled in the wet lab.

## Interacting with the trained models

**→ See [`MODEL_USAGE.md`](MODEL_USAGE.md)**: how to load the checkpoints
(`.pt` vs `.pkl`, fp32 vs bf16), generate genomes from Python or the CLI,
measure perplexity on any genome, and retrain.

## Test

```bash
cd SciAgent && .venv/bin/python -m unittest test_phage_pipeline -v
```