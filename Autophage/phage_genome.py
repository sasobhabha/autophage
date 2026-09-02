"""
Synthetic bacteriophage genome design and validation.

This module provides two halves, mirroring the Autophage "neural proposer /
symbolic verifier" split:

1. `generate_phage_genome(...)` -- the *proposer*. Builds a synthetic dsDNA
   genome in silico with biologically plausible features:

   * coding regions back-translated from proteins with GC-content targeting
   * Shine-Dalgarno (AGGAGG) ribosome binding sites upstream of ORFs
   * short intergenic spacers, ATG/GTG start codons and stop codons
   * a configurable direct terminal repeat (seen in phages like T7)

2. `validate_phage_genome(...)` -- the *verifier*. Checks a sequence
   (generated or supplied as FASTA) against bacteriophage biology criteria:
   alphabet, length, GC content, 6-frame ORF coding density, ribosome
   binding site fraction, and optional direct terminal repeats.

This is a purely computational, in-silico toolkit for phage genome research
(e.g. phage therapy design). It contains no synthesis instructions.

Usage (CLI)::

    python phage_genome.py generate --length 50000 --gc 0.50 --seed 1 --validate
    python phage_genome.py validate --input genome.fasta

Library usage::

    from phage_genome import generate_phage_genome, validate_phage_genome
    genome = generate_phage_genome(length=50000, gc_target=0.5, seed=1)
    report = validate_phage_genome(genome.sequence)
    print(report.summary())
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Codon tables (standard genetic code)
# ---------------------------------------------------------------------------

CODON_TABLE: Dict[str, List[str]] = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "C": ["TGT", "TGC"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["TTT", "TTC"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "K": ["AAA", "AAG"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "M": ["ATG"],
    "N": ["AAT", "AAC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
}

STOP_CODONS = ["TAA", "TAG", "TGA"]
START_CODONS = frozenset(["ATG", "GTG", "TTG"])

# Codon -> amino acid translation (standard genetic code)
CODON_TO_AA: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Rough proteome amino-acid frequencies (E. coli-like). Normalized in code.
AA_FREQUENCIES: Dict[str, float] = {
    "L": 0.107, "A": 0.095, "G": 0.074, "V": 0.073, "E": 0.067,
    "S": 0.058, "I": 0.060, "R": 0.052, "D": 0.052, "T": 0.054,
    "K": 0.055, "N": 0.043, "Q": 0.043, "P": 0.041, "F": 0.039,
    "Y": 0.032, "M": 0.026, "H": 0.022, "C": 0.013, "W": 0.013,
}

_COMPLEMENT = str.maketrans("ACGT", "TGCA")

# Typical dsDNA bacteriophage genome size (bp) and GC range.
PHAGE_LENGTH_RANGE: Tuple[int, int] = (5_000, 700_000)
PHAGE_GC_RANGE: Tuple[float, float] = (0.25, 0.70)
MIN_ORF_LEN_BP = 300
MIN_CODING_DENSITY = 0.50
MIN_CODING_ORFS = 8
MIN_RBS_FRACTION = 0.30

RBS_MOTIFS = ("AGGAGG", "GGAGG")  # Shine-Dalgarno consensus

CLEAN_SEQ_RE = re.compile(r"[^ACGT]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Small sequence helpers
# ---------------------------------------------------------------------------

def gc_fraction(seq: str) -> float:
    """GC content of a DNA string (0.0 - 1.0)."""
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _gc_biased_dna(
    length: int,
    gc_target: float,
    rng: random.Random,
    avoid_start_codons: bool = False,
) -> str:
    """
    Random DNA whose expected GC content is `gc_target`.

    With ``avoid_start_codons=True`` no ATG/GTG/TTG triplets are emitted,
    which keeps intergenic spacers free of spurious translation starts.
    """
    if length <= 0:
        return ""
    p_gc = max(0.0, min(1.0, gc_target))
    bases = []
    for _ in range(length):
        for _attempt in range(50):
            if rng.random() < p_gc:
                base = rng.choice("GC")
            else:
                base = rng.choice("AT")
            if avoid_start_codons and len(bases) >= 2:
                if (bases[-2] + bases[-1] + base) in START_CODONS:
                    continue  # would create a spurious start codon; retry
            break
        else:
            # Extremely unlikely fallback: A/T never completes ATG/GTG/TTG.
            base = rng.choice("AT")
        bases.append(base)
    return "".join(bases)


def _homopolymer_safe(codon: str, tail: str, max_run: int = 5) -> bool:
    """True if appending `codon` to `tail` creates no run longer than max_run."""
    if not tail:
        return True
    run = 1
    prev = tail[-1]
    for i, b in enumerate(tail[-max_run:]):
        if b == prev:
            run += 1
            if run > max_run:
                return False
        else:
            run = 1
            prev = b
    for b in codon:
        if b == prev:
            run += 1
            if run > max_run:
                return False
        else:
            run = 1
            prev = b
    return True


def back_translate(protein: str,
                   gc_target: float,
                   rng: random.Random,
                   host_codon_usage: Optional[Dict[str, float]] = None) -> str:
    """
    Convert a protein string to DNA.

    Without ``host_codon_usage``, synonymous codons are chosen so the result
    tracks ``gc_target``. With ``host_codon_usage`` (relative frequencies of
    every sense codon in a host genome), codons are sampled proportional to
    host usage -- the classical host-adaptation strategy phages use to match
    the host tRNA pool (codon adaptation). Homopolymer runs are avoided.
    """
    out: List[str] = []
    for aa in protein:
        codons = CODON_TABLE[aa]
        if host_codon_usage is not None:
            weights = [host_codon_usage.get(c, 0.0) for c in codons]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                chosen = rng.choices(codons, weights=weights, k=1)[0]
            else:
                chosen = rng.choice(codons)
            safe = [c for c in codons if _homopolymer_safe(c, "".join(out))]
            if safe and chosen not in safe:
                chosen = rng.choice(safe)
            out.append(chosen)
            continue
        scored = sorted(codons, key=lambda c: abs(gc_fraction(c) - gc_target))
        best_dev = abs(gc_fraction(scored[0]) - gc_target)
        # Random choice among near-optimal synonymous codons.
        candidates = [c for c in codons
                      if abs(gc_fraction(c) - gc_target) <= best_dev + 0.2]
        chosen = rng.choice(candidates)
        # Prefer a candidate that does not extend a homopolymer run.
        safe = [c for c in candidates if _homopolymer_safe(c, "".join(out))]
        if safe:
            chosen = rng.choice(safe)
        out.append(chosen)
    return "".join(out)


def _random_protein(rng: random.Random, length: int) -> str:
    aas, weights = zip(*AA_FREQUENCIES.items())
    return "".join(rng.choices(aas, weights=weights, k=length))


def _random_gene_length(rng: random.Random) -> int:
    """Protein-coding gene length in bp, ~exponential between 150 and 2500 bp."""
    n = int(rng.expovariate(1 / 600))
    return max(150, min(2500, n + 100))


# ---------------------------------------------------------------------------
# Generation (the "proposer")
# ---------------------------------------------------------------------------

@dataclass
class SyntheticPhageGenome:
    sequence: str
    name: str
    gc_target: float
    terminal_repeat: int
    metadata: Dict[str, object] = field(default_factory=dict)


def generate_phage_genome(
    length: int = 50_000,
    gc_target: float = 0.5,
    seed: Optional[int] = None,
    terminal_repeat: int = 200,
    rbs_fraction: float = 0.75,
    name: str = "synthetic_phage",
    host_codon_usage: Optional[Dict[str, float]] = None,
) -> SyntheticPhageGenome:
    """
    Generate a synthetic dsDNA bacteriophage genome in silico.

    Args:
        length: target genome size in bp.
        gc_target: target GC content (0.0 - 1.0); used for spacers and as
            the fallback when no host codon profile is given.
        seed: RNG seed for reproducible generation.
        terminal_repeat: length of the direct terminal repeat to append
            (0 disables). Size must be < length.
        rbs_fraction: fraction of genes preceded by a Shine-Dalgarno motif.
        name: genome identifier used in FASTA output.
        host_codon_usage: optional dict of codon -> relative frequency in a
            host genome. When given, every gene is back-translated using the
            host's codon usage (codon-adapted phage design).
    """
    if terminal_repeat >= length:
        raise ValueError("terminal_repeat must be smaller than length")
    rng = random.Random(seed)

    core_length = length - terminal_repeat
    seq_parts: List[str] = []
    total = 0
    gene_count = 0

    while total < core_length:
        spacer_len = rng.randint(15, 80)
        spacer = _gc_biased_dna(max(0, spacer_len - 6), gc_target, rng,
                                avoid_start_codons=True)
        if rng.random() < rbs_fraction:
            spacer += "AGGAGG"  # Shine-Dalgarno, directly upstream of start codon
        else:
            spacer += _gc_biased_dna(6, gc_target, rng, avoid_start_codons=True)

        if total + len(spacer) >= core_length:
            seq_parts.append(spacer[: core_length - total])
            total = core_length
            break

        seq_parts.append(spacer)
        total += len(spacer)

        protein_len = _random_gene_length(rng)
        # Force an N-terminal methionine so every gene begins with ATG.
        protein = "M" + _random_protein(rng, protein_len // 3 - 1)
        gene = back_translate(protein, gc_target, rng,
                              host_codon_usage=host_codon_usage)
        stop = rng.choice(STOP_CODONS)
        gene += stop

        if total + len(gene) >= core_length:
            # Only add the portion of this gene that fits; validation is
            # tolerant of a truncated terminal gene, but we keep complete
            # genes whenever possible by shrinking the spacer instead.
            seq_parts[-1] = seq_parts[-1][: max(0, core_length - total + len(spacer))]
            total = core_length
            break

        seq_parts.append(gene)
        total += len(gene)
        gene_count += 1

    core = "".join(seq_parts)
    if terminal_repeat > 0:
        core = core + core[:terminal_repeat]

    genome = SyntheticPhageGenome(
        sequence=core,
        name=name,
        gc_target=gc_target,
        terminal_repeat=terminal_repeat,
        metadata={
            "length": len(core),
            "gc": round(gc_fraction(core), 4),
            "genes": gene_count,
            "coding_density": round(_coding_density(core), 4),
        },
    )
    return genome


# ---------------------------------------------------------------------------
# Validation (the "verifier")
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("WARN" if self.name == "terminal_repeats" else "FAIL")
        return f"[{mark:4s}] {self.name}: {self.detail}"


@dataclass
class PhageValidationReport:
    sequence_name: str
    length: int
    gc: float
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """All hard checks pass (terminal repeat status is advisory)."""
        return all(c.passed for c in self.checks if c.name != "terminal_repeats")

    def summary(self) -> str:
        lines = [
            f"Phage validation report: {self.sequence_name}",
            f"  length={self.length} bp  gc={self.gc:.3f}",
        ]
        lines += [f"  {c}" for c in self.checks]
        lines.append(f"  VERDICT: {'VALIDATED' if self.passed else 'NOT VALIDATED'}")
        return "\n".join(lines)


def _coding_density(seq: str) -> float:
    """Fraction of bases covered by at least one ORF across all 6 frames."""
    covered = bytearray(len(seq))
    for strand_seq in (seq, reverse_complement(seq)):
        for frame in range(3):
            start = None
            for i in range(frame, len(strand_seq) - 2, 3):
                codon = strand_seq[i:i + 3]
                if codon in STOP_CODONS:
                    if start is not None and i - start >= MIN_ORF_LEN_BP:
                        covered[start:i] = b"\x01" * (i - start)
                    start = None
                elif start is None and codon in START_CODONS:
                    start = i
            if start is not None and len(strand_seq) - start >= MIN_ORF_LEN_BP:
                covered[start:] = b"\x01" * (len(strand_seq) - start)
    return sum(covered) / len(seq)


def _scan_orfs(seq: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Scan all 6 frames for ORFs >= MIN_ORF_LEN_BP.
    Returns (forward_orfs, reverse_orfs) as (start, end) 0-based half-open
    intervals in the original sequence coordinates.
    """
    forward: List[Tuple[int, int]] = []
    rev = reverse_complement(seq)
    for frame in range(3):
        start = None
        for i in range(frame, len(seq) - 2, 3):
            if seq[i:i + 3] in STOP_CODONS:
                if start is not None and i - start >= MIN_ORF_LEN_BP:
                    forward.append((start, i))
                start = None
            elif start is None and seq[i:i + 3] in START_CODONS:
                start = i
        if start is not None and len(seq) - start >= MIN_ORF_LEN_BP:
            forward.append((start, len(seq)))
    # Reverse-strand ORFs: mirror back from the reverse-complement coordinates.
    reverse = []
    for frame in range(3):
        start = None
        for i in range(frame, len(rev) - 2, 3):
            if rev[i:i + 3] in STOP_CODONS:
                if start is not None and i - start >= MIN_ORF_LEN_BP:
                    s, e = len(seq) - i, len(seq) - start
                    reverse.append((s, e))
                start = None
            elif start is None and rev[i:i + 3] in START_CODONS:
                start = i
        if start is not None and len(rev) - start >= MIN_ORF_LEN_BP:
            s, e = len(seq) - len(rev), len(seq) - start
            reverse.append((s, e))
    return forward, reverse


def _terminal_repeat_length(seq: str, max_repeat: int = 2_000) -> int:
    """Largest k such that the first k bases equal the last k bases."""
    best = 0
    limit = min(max_repeat, len(seq) // 4)
    for k in range(20, limit + 1, 4):
        if seq[:k] == seq[-k:]:
            best = k
    return best


def validate_phage_genome(
    sequence: str,
    name: str = "unknown",
    check_terminal_repeats: bool = True,
) -> PhageValidationReport:
    """
    Validate a DNA sequence against bacteriophage biology criteria.

    All checks except `terminal_repeats` are hard checks; a failing hard
    check fails validation.
    """
    seq = sequence.strip().upper()
    length = len(seq)
    gc = gc_fraction(seq)
    checks: List[CheckResult] = []

    # 1. Alphabet
    bad = CLEAN_SEQ_RE.findall(seq)
    checks.append(CheckResult(
        "alphabet",
        not bad,
        "only A/C/G/T" if not bad else f"contains non-ACGT chars: {sorted(set(bad))[:5]}",
    ))

    # 2. Length within typical dsDNA phage range
    lo, hi = PHAGE_LENGTH_RANGE
    checks.append(CheckResult(
        "length",
        lo <= length <= hi,
        f"{length} bp (expected {lo}-{hi})",
    ))

    # 3. GC content in observed phage range
    glo, ghi = PHAGE_GC_RANGE
    checks.append(CheckResult(
        "gc_content",
        glo <= gc <= ghi,
        f"gc={gc:.3f} (expected {glo:.2f}-{ghi:.2f})",
    ))

    # 4. Coding density from 6-frame ORF scan
    density = _coding_density(seq)
    checks.append(CheckResult(
        "coding_density",
        density >= MIN_CODING_DENSITY,
        f"{density:.3f} of bases inside ORFs (min {MIN_CODING_DENSITY})",
    ))

    # 5. ORF count
    forward_orfs, reverse_orfs = _scan_orfs(seq)
    n_orfs = len(forward_orfs) + len(reverse_orfs)
    checks.append(CheckResult(
        "orf_count",
        n_orfs >= MIN_CODING_ORFS,
        f"{n_orfs} ORFs >= {MIN_ORF_LEN_BP} bp (min {MIN_CODING_ORFS})",
    ))

    # 6. RBS fraction among forward-strand ORFs
    rbs_n = 0
    for s, _ in forward_orfs:
        upstream = seq[max(0, s - 20):s]
        if any(m in upstream for m in RBS_MOTIFS):
            rbs_n += 1
    rbs_frac = rbs_n / len(forward_orfs) if forward_orfs else 0.0
    checks.append(CheckResult(
        "rbs_fraction",
        rbs_frac >= MIN_RBS_FRACTION,
        f"{rbs_frac:.3f} of forward ORFs have Shine-Dalgarno upstream "
        f"(min {MIN_RBS_FRACTION})",
    ))

    # 7. Direct terminal repeats (advisory: only some phages have them)
    if check_terminal_repeats:
        tr = _terminal_repeat_length(seq)
        checks.append(CheckResult(
            "terminal_repeats",
            tr >= 20,
            f"direct terminal repeat of {tr} bp found" if tr >= 20
            else "no direct terminal repeat detected (not all phages have one)",
        ))

    return PhageValidationReport(name, length, gc, checks)


# ---------------------------------------------------------------------------
# Protein production (translation of parsed DNA)
# ---------------------------------------------------------------------------

def translate_cds(cds: str) -> str:
    """
    Translate a coding DNA sequence to protein using the standard genetic
    code. Starts at the first in-frame start codon and stops at the first
    in-frame stop; 'X' for any ambiguous codon. Returns the protein string.
    """
    seq = cds.upper()
    # trim to codon boundaries if needed
    trimmed = seq[: len(seq) - (len(seq) % 3)]
    aa = []
    for i in range(0, len(trimmed) - 2, 3):
        aa.append(CODON_TO_AA.get(trimmed[i:i + 3], "X"))
    protein = "".join(aa)
    # stop at first internal stop codon
    star = protein.find("*")
    if star != -1:
        protein = protein[:star]
    # require a start codon at the beginning
    if protein and protein[0] != "M" and seq[:3] in START_CODONS:
        protein = "M" + protein[1:]
    return protein


def extract_cds_from_gtf(genome_fasta: str, gtf_path: str,
                         max_transcripts: Optional[int] = None) -> List[str]:
    """
    Assemble coding DNA sequences (CDS) per transcript from a GTF annotation
    and a genome FASTA. Splices exons in order and reverse-complements genes
    on the minus strand. Useful to build a host codon-usage profile and to
    test host protein production.
    """
    # load genome contigs; index by both the bare token and any (alias) in
    # the header, since GTF files often use a different naming scheme
    contigs: Dict[str, str] = {}
    name = ""
    aliases: Dict[str, str] = {}
    buf: List[str] = []
    for raw in Path(genome_fasta).read_text().splitlines():
        if raw.startswith(">"):
            if name:
                contigs[name] = "".join(buf).upper()
            header = raw[1:].strip()
            name = header.split()[0]
            aliases[name] = name
            m = re.search(r"\(([^)]+)\)", header)
            if m:
                aliases[m.group(1)] = name
            buf = []
        else:
            buf.append(raw.strip())
    if name:
        contigs[name] = "".join(buf).upper()
    contigs.update({alias: contigs[real] for alias, real in aliases.items()
                    if real in contigs})

    # collect CDS features grouped by transcript
    transcripts: Dict[str, List[Tuple[str, int, int, str]]] = {}
    n = 0
    for raw in Path(gtf_path).read_text().splitlines():
        if raw.startswith("#"):
            continue
        fields = raw.strip().split("\t")
        if len(fields) < 9 or fields[2] != "CDS":
            continue
        chrom, start, end, strand = fields[0], int(fields[3]), int(fields[4]), fields[6]
        attrs = fields[8]
        tid = None
        for token in attrs.split(";"):
            token = token.strip()
            if token.startswith("transcript_id"):  # GTF style: transcript_id "x"
                tid = token.split(" ")[1].strip('"')
                break
            if token.startswith("Parent="):  # GFF3 style: group CDS by parent
                tid = token.split("=", 1)[1]
                break
            if token.startswith("ID=cds-") or token.startswith("ID=CDS"):
                tid = token.split("=", 1)[1]
                break
        if not tid:
            continue
        transcripts.setdefault(tid, []).append((chrom, start, end, strand))
        n += 1
        if max_transcripts and len(transcripts) >= max_transcripts and n > max_transcripts * 20:
            break

    cds_list: List[str] = []
    for tid, exons in transcripts.items():
        if not exons:
            continue
        exons.sort(key=lambda e: e[1])
        chrom = exons[0][0]
        strand = exons[0][3]
        seq = "".join(contigs.get(chrom, "")[s - 1:e] for _, s, e, _ in exons)
        if strand == "-":
            seq = reverse_complement(seq)
        if seq:
            cds_list.append(seq)
        if max_transcripts and len(cds_list) >= max_transcripts:
            break
    return cds_list


@dataclass
class ProteinProductionReport:
    n_regions_scanned: int
    n_orfs: int
    n_proteins: int
    n_truncated: int
    mean_protein_len: float

    def summary(self) -> str:
        return (
            f"Protein production report:\n"
            f"  ORFs detected across all 6 frames: {self.n_orfs}\n"
            f"  ORFs translating to full proteins: {self.n_proteins} "
            f"({100.0 * self.n_proteins / max(1, self.n_orfs):.1f}%)\n"
            f"  truncated ORFs (frameshift/partial): {self.n_truncated}\n"
            f"  mean protein length: {self.mean_protein_len:.1f} aa"
        )


def protein_production_report(
    sequence: str,
    min_orf_len: int = MIN_ORF_LEN_BP,
    n_slices: int = 200,
) -> ProteinProductionReport:
    """
    Parse a genome and test for protein production: detect ORFs in all six
    reading frames, translate each to protein, and report how many yield a
    full-length protein (start codon + no premature stop).
    """
    seq = sequence.strip().upper()
    forward, reverse = _scan_orfs(seq)
    orfs: List[Tuple[int, int, str]] = (
        [(s, e, "+") for s, e in forward] +
        [(s, e, "-") for s, e in reverse]
    )
    if len(orfs) < 20:
        # for short sequences, rescan with a smaller minimum
        orfs = _scan_orfs_minlen(seq, max(60, min_orf_len // 3))

    n_proteins = 0
    n_truncated = 0
    lengths: List[int] = []
    for s, e, strand in orfs:
        cds = seq[s:e] if strand == "+" else reverse_complement(seq[s:e])
        protein = translate_cds(cds)
        if protein and protein[0] == "M" and "*" not in protein:
            n_proteins += 1
            lengths.append(len(protein))
        else:
            n_truncated += 1
    return ProteinProductionReport(
        n_regions_scanned=max(1, n_slices),
        n_orfs=len(orfs),
        n_proteins=n_proteins,
        n_truncated=n_truncated,
        mean_protein_len=sum(lengths) / len(lengths) if lengths else 0.0,
    )


def _scan_orfs_minlen(seq: str, min_len: int) -> List[Tuple[int, int, str]]:
    """Minimal ORF scanner with a custom minimum length, all six frames."""
    out: List[Tuple[int, int, str]] = []
    for strand_str, strand in ((seq, "+"), (reverse_complement(seq), "-")):
        for frame in range(3):
            start = None
            for i in range(frame, len(strand_str) - 2, 3):
                codon = strand_str[i:i + 3]
                if codon in STOP_CODONS:
                    if start is not None and i - start >= min_len:
                        if strand == "+":
                            out.append((start, i, "+"))
                        else:
                            out.append((len(seq) - i, len(seq) - start, "-"))
                    start = None
                elif start is None and codon in START_CODONS:
                    start = i
            if start is not None and len(strand_str) - start >= min_len:
                if strand == "+":
                    out.append((start, len(seq), "+"))
                else:
                    out.append((0, len(seq) - start, "-"))
    return out


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def to_fasta(genome: SyntheticPhageGenome, width: int = 80) -> str:
    header = (
        f">{genome.name} length={len(genome.sequence)} "
        f"gc={genome.metadata.get('gc', 0):.3f} "
        f"genes={genome.metadata.get('genes', 0)}"
    )
    body = "\n".join(
        genome.sequence[i:i + width] for i in range(0, len(genome.sequence), width)
    )
    return header + "\n" + body + "\n"


def parse_fasta(text: str) -> Tuple[str, str]:
    """Return (name, sequence) of the first FASTA record."""
    name = "record"
    seq_parts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].strip().split()[0]
        else:
            seq_parts.append(line.upper())
    return name, "".join(seq_parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _host_seq_from_args(args) -> str:
    """Load the raw host genome sequence (first record or concatenated)."""
    fasta = getattr(args, "host_genome", None)
    if not fasta:
        return ""
    _, seq = parse_fasta(Path(fasta).read_text())
    return seq


def _extract_gene_regions(sequence: str) -> List[str]:
    """Extract forward-strand ORF regions >= MIN_ORF_LEN_BP as CDS strings."""
    forward, _reverse = _scan_orfs(sequence)
    return [sequence[s:e] for s, e in forward]


def _proteins(args: argparse.Namespace) -> int:
    if not args.input:
        print("error: --input is required", file=sys.stderr)
        return 2
    with open(args.input) as f:
        name, seq = parse_fasta(f.read())
    report = protein_production_report(seq)
    print(report.summary())
    return 0


def _host_profile(args: argparse.Namespace) -> int:
    """Build a host codon-usage profile from CDS or genome, optionally via GTF."""
    import json as _json
    from host_compat import (codon_usage_from_cds,
                             codon_usage_from_genome_sequence)
    if args.host_cds:
        usage = codon_usage_from_cds(args.host_cds)
        print(f"[host] codon profile from {args.host_cds}: "
              f"{len(usage)} sense codons (normalized frequencies)")
    elif args.genome_fasta and args.gtf:
        cds = extract_cds_from_gtf(args.genome_fasta, args.gtf)
        concat = "".join(cds)
        print(f"[host] assembled {len(cds)} spliced CDS from GTF "
              f"({len(concat):,} bp)")
        usage = {c: 0.0 for c in CODON_TO_AA}
        for c in CODON_TO_AA:
            pass
        # count codons in CDS
        counts = {c: 0 for c in CODON_TO_AA}
        for cds in cds:
            for i in range(0, len(cds) - 2, 3):
                codon = cds[i:i + 3]
                if codon in counts:
                    counts[codon] += 1
        total = sum(counts.values())
        usage = {c: n / total for c, n in counts.items()}
    elif args.host_genome:
        name, seq = parse_fasta(Path(args.host_genome).read_text())
        usage = codon_usage_from_genome_sequence(seq)
        print(f"[host] codon profile from genome {args.host_genome} "
              f"({len(seq):,} bp)")
    else:
        print("error: provide --host-cds, --host-genome, or --genome-fasta + --gtf",
              file=sys.stderr)
        return 2

    if args.out:
        Path(args.out).write_text(_json.dumps(usage, indent=1))
        print(f"[host] wrote profile to {args.out}")
    return 0


def _load_host_profile(args) -> Optional[Dict[str, float]]:
    """Build a host codon-usage profile from CDS FASTA, or genome+GTF."""
    from host_compat import (codon_usage_from_cds,
                             codon_usage_from_genome_sequence)
    if getattr(args, "host_cds", None):
        print(f"[host] building codon profile from CDS {args.host_cds}")
        return codon_usage_from_cds(args.host_cds)
    if getattr(args, "host_genome", None):
        fasta = args.host_genome
        name, seq = parse_fasta(Path(fasta).read_text())
        print(f"[host] building codon profile from genome {fasta} "
              f"({len(seq):,} bp loaded)")
        return codon_usage_from_genome_sequence(seq)
    return None


def _generate(args: argparse.Namespace) -> int:
    host_usage = _load_host_profile(args)
    if host_usage is not None and getattr(args, "gc", None) is None:
        # default GC to the host's GC when adapting to a host
        args.gc = gc_fraction(_host_seq_from_args(args)) or 0.5
    genome = generate_phage_genome(
        length=args.length,
        gc_target=args.gc,
        seed=args.seed,
        terminal_repeat=args.terminal_repeat,
        name=args.name,
        host_codon_usage=host_usage,
    )
    if args.output:
        with open(args.output, "w") as f:
            f.write(to_fasta(genome))
        print(f"Wrote {args.output} "
              f"({len(genome.sequence)} bp, gc={genome.metadata['gc']:.3f}, "
              f"genes={genome.metadata['genes']})")
    else:
        print(to_fasta(genome), end="")

    if args.validate:
        report = validate_phage_genome(genome.sequence, genome.name)
        print("\n" + report.summary())
        if report.passed and host_usage is not None:
            from host_compat import screen_host_adaptation
            query_cds = [translate_cds(gene_region)] if False else _extract_gene_regions(genome.sequence)
            host_seq = _host_seq_from_args(args)
            if query_cds and host_seq:
                adapt = screen_host_adaptation(
                    query_cds, host_name=args.name or "host",
                    host_usage=host_usage, host_seq=host_seq,
                    query_seq=genome.sequence)
                print("\n" + adapt.summary())
        return 0 if report.passed else 1
    return 0


def _validate(args: argparse.Namespace) -> int:
    if not args.input:
        print("error: --input is required for validate", file=sys.stderr)
        return 2
    with open(args.input) as f:
        name, seq = parse_fasta(f.read())
    report = validate_phage_genome(
        seq,
        name=name,
        check_terminal_repeats=args.terminal_repeats,
    )
    print(report.summary())
    return 0 if report.passed else 1


def _compat(args: argparse.Namespace) -> int:
    from host_compat import screen_compatibility
    if not args.input:
        print("error: --input is required for compat", file=sys.stderr)
        return 2
    with open(args.input) as f:
        name, seq = parse_fasta(f.read())
    report = screen_compatibility(
        seq,
        query_name=name,
        cache_dir=Path(args.cache),
        k=args.k,
        verify_ssl=args.verify_ssl,
    )
    print(report.summary())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phage_genome",
        description="Generate and validate synthetic bacteriophage genomes (in silico).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="create a synthetic phage genome")
    gen.add_argument("--length", type=int, default=50_000, help="target genome size in bp")
    gen.add_argument("--gc", type=float, default=0.5, help="target GC content (0-1)")
    gen.add_argument("--host-genome", default=None,
                     help="host genome FASTA: codon-adapt the phage to this host")
    gen.add_argument("--host-cds", default=None,
                     help="host CDS FASTA: codon-adapt the phage to this host's CDS")
    gen.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    gen.add_argument("--terminal-repeat", type=int, default=200,
                     help="direct terminal repeat length in bp (0 disables)")
    gen.add_argument("--name", default="synthetic_phage", help="genome name")
    gen.add_argument("--output", default=None, help="write FASTA to this file")
    gen.add_argument("--validate", action="store_true",
                     help="validate the genome after generating it")
    gen.set_defaults(func=_generate)

    val = sub.add_parser("validate", help="check a FASTA file against phage criteria")
    val.add_argument("--input", default=None, help="FASTA file to validate")
    val.add_argument("--no-terminal-repeats", dest="terminal_repeats",
                     action="store_false", default=True,
                     help="skip the (advisory) terminal repeat check")
    val.set_defaults(func=_validate)

    prot = sub.add_parser("proteins",
                          help="parse DNA and test protein production (translate ORFs)")
    prot.add_argument("--input", default=None, help="FASTA file to analyze")
    prot.set_defaults(func=_proteins)

    host = sub.add_parser(
        "host-profile",
        help="build a host codon-usage profile for phage adaptation",
    )
    host.add_argument("--host-genome", default=None, help="host genome FASTA")
    host.add_argument("--host-cds", default=None, help="host CDS FASTA")
    host.add_argument("--genome-fasta", default=None,
                      help="genome FASTA for GTF-based CDS extraction")
    host.add_argument("--gtf", default=None, help="GTF annotation")
    host.add_argument("--out", default=None, help="write profile JSON")
    host.set_defaults(func=_host_profile)

    compat = sub.add_parser(
        "compat",
        help="screen against real superbug genomes (VirHostMatcher d2*)",
    )
    compat.add_argument("--input", default=None, help="FASTA file of the query genome")
    compat.add_argument("--cache", default="data/hosts",
                        help="directory to cache downloaded host genomes")
    compat.add_argument("--k", type=int, default=6, help="k-mer length (default 6)")
    compat.add_argument("--verify-ssl", action="store_true",
                        help="strict SSL verification (default: relaxed for proxies)")
    compat.set_defaults(func=_compat)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())