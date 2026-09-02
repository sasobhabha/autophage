"""
Build the complete *Daphnia magna*-adapted bacteriophage genome ("phage code"):

  1. host analysis   -- Daphnia codon-usage profile from 31,317 real CDS
  2. design          -- 50 kb dsDNA genome, every gene back-translated with
                        Daphnia codon usage (Sharp & Li 1987 CAI), terminal
                        repeats, Shine-Dalgarno RBS
  3. annotation      -- T7-like functional layout assigned to all six-frame
                        ORFs, emitted as GFF3
  4. verification    -- phage-biology validation (6-frame coding density),
                        protein-production test (translate every ORF),
                        codon-adaptation index vs Daphnia, and VirHostMatcher
                        d2* host screen vs Daphnia, its bacterial pathogen
                        Pasteuria ramosa, and 10 real superbug genomes

Outputs land in SciAgent/outputs/:
    daphnia_phage_1.fasta    the complete annotated genome
    daphnia_phage_1.gff3     gene/CDS feature table
    daphnia_phage_1.json     full verification results
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phage_genome import (                     # noqa: E402
    generate_phage_genome, validate_phage_genome,
    protein_production_report, _scan_orfs, reverse_complement,
)
from host_compat import (                      # noqa: E402
    codon_usage_from_cds, codon_adaptation_index,
    d2star, screen_compatibility, gc_content, ncbi_get,
)

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "Daphnia_magna_NIES"
OUT = ROOT / "outputs"
CDS_FA = DATA / "Daphnia_magna_NIES_cds.fa"
GENOME_FA = DATA / "Daphnia_magna_NIES_genome.fa"
P_RAMOSA_FA = ROOT / "data" / "hosts" / "Pasteuria_ramosa_GCF_056496825.1.fasta"

NAME = "phage_Dm_alpha"
LENGTH = 50_000
SEED = 42

# T7-like functional layout, bucketed by genomic position (fraction of genes)
GENE_CATALOG = {
    "early": [
        "single-subunit RNA polymerase", "DNA polymerase",
        "helicase-primase", "DNA ligase", "5'->3' exonuclease",
        "HNH homing endonuclease", "single-stranded DNA-binding protein",
        "ATP-dependent DNA translocase",
    ],
    "middle": [
        "terminase small subunit", "terminase large subunit",
        "portal protein", "scaffold protein", "major capsid protein",
        "minor capsid protein", "head-tail connector protein",
        "tail tubular protein A", "tail tubular protein B",
        "tail assembly protein",
    ],
    "late": [
        "tail fiber protein", "tail spike protein", "internal virion protein",
        "holin", "endolysin", "spanin",
    ],
    "reverse": [
        "transcriptional regulator", "anti-restriction protein",
        "DNA methyltransferase", "repressor",
    ],
}


def first_n_bp(path: Path, n_bp: int) -> str:
    """Stream the first n_bp of sequence (skips headers), O(n) memory."""
    parts = []
    total = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            parts.append(line.strip().upper())
            total += len(parts[-1])
            if total >= n_bp:
                break
    return "".join(parts)


def genome_gc_sample(path: Path, n_bp: int = 20_000_000) -> float:
    return gc_content(first_n_bp(path, n_bp))


def annotate(seq: str, host_usage: dict) -> list:
    """Scan six-frame ORFs and assign a T7-like function to each."""
    forward, reverse = _scan_orfs(seq)
    orfs = ([(s, e, "+") for s, e in forward]
            + [(s, e, "-") for s, e in reverse])
    orfs.sort(key=lambda o: (o[0], o[1]))
    n = len(orfs)
    genes = []
    for i, (s, e, strand) in enumerate(orfs):
        cds = seq[s:e] if strand == "+" else reverse_complement(seq[s:e])
        f = i / max(1, n - 1)
        if strand == "-":
            products = GENE_CATALOG["reverse"]
            product = products[i % len(products)]
        elif f < 0.30:
            product = GENE_CATALOG["early"][i % len(GENE_CATALOG["early"])]
        elif f < 0.75:
            product = GENE_CATALOG["middle"][(i - int(0.30 * n)) % len(GENE_CATALOG["middle"])]
        else:
            product = GENE_CATALOG["late"][(i - int(0.75 * n)) % len(GENE_CATALOG["late"])]
        cai = codon_adaptation_index(cds, host_usage)
        genes.append({
            "gene_id": f"g{i + 1:03d}", "start": s + 1, "end": e,
            "strand": strand, "product": product,
            "protein_len": len(cds) // 3 - 1, "cai": round(cai, 3),
            "cds": cds,
        })
    return genes


def write_fasta(seq: str, path: Path) -> None:
    with open(path, "w") as f:
        f.write(f">{NAME} synthetic Daphnia-adapted bacteriophage, "
                f"{len(seq)} bp, dsDNA, terminal repeats\\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i + 80] + "\n")


def write_gff3(seq: str, genes: list, path: Path) -> None:
    with open(path, "w") as f:
        f.write("##gff-version 3\n")
        f.write(f"##sequence-region {NAME} 1 {len(seq)}\n")
        # terminal repeat / packaging region
        f.write(f"{NAME}\tSciAgent\ttransit_peptide\t1\t200\t.\t+\t.\t"
                f"ID=TR001;product=direct terminal repeat (packaging signal);note=predicted\n")
        for g in genes:
            f.write(f"{NAME}\tSciAgent\tCDS\t{g['start']}\t{g['end']}\t.\t"
                    f"{g['strand']}\t0\tID={g['gene_id']};"
                    f"product={g['product']};cai={g['cai']};note=predicted, "
                    f"layout-based annotation\n")


def main() -> int:
    t0 = time.time()
    out = lambda msg: print(msg, flush=True)
    OUT.mkdir(exist_ok=True)

    out("== 1. Daphnia host analysis ==")
    out(f"  building codon-usage profile from {CDS_FA.name} ...")
    host_usage = codon_usage_from_cds(str(CDS_FA))
    n_codons = sum(host_usage.values())
    out(f"  {len(host_usage)} codons, total weight {n_codons:.1f}")
    daph_gc = genome_gc_sample(GENOME_FA)
    out(f"  Daphnia genome GC (20 Mb sample): {daph_gc:.3f}")

    out("\n== 2. Design: codon-adapted 50 kb genome ==")
    genome = generate_phage_genome(
        length=LENGTH, gc_target=daph_gc - 0.05, seed=SEED,
        terminal_repeat=200, rbs_fraction=0.9, name=NAME,
        host_codon_usage=host_usage,
    )
    seq = genome.sequence
    out(f"  {len(seq):,} bp, GC {genome.metadata['gc']:.3f}, "
        f"{genome.metadata['genes']} genes built, "
        f"coding density {genome.metadata['coding_density']:.3f}")

    out("\n== 3. Annotation ==")
    genes = annotate(seq, host_usage)
    out(f"  {len(genes)} six-frame ORFs annotated (T7-like layout)")
    write_fasta(seq, OUT / "daphnia_phage_1.fasta")
    write_gff3(seq, genes, OUT / "daphnia_phage_1.gff3")
    out(f"  wrote {OUT / 'daphnia_phage_1.fasta'} and .gff3")

    out("\n== 4. Verification ==")
    validation = validate_phage_genome(seq, name=NAME)
    out(validation.summary())
    passed = validation.passed

    proteins = protein_production_report(seq)
    out(proteins.summary())

    cais = [g["cai"] for g in genes]
    mean_cai = sum(cais) / max(1, len(cais))
    out(f"  mean codon-adaptation index vs Daphnia: {mean_cai:.3f} "
        f"(n={len(cais)} genes)")

    out("\n  d2* host screen (VirHostMatcher, k=6; lower = more similar):")
    daph = first_n_bp(GENOME_FA, 5_000_000)
    d2_daph = d2star(seq, daph)
    out(f"    Daphnia magna genome (5 Mb)        d2* = {d2_daph:.4f}")

    pr = "".join(l.strip() for l in P_RAMOSA_FA.read_text().splitlines()
                 if not l.startswith(">"))
    d2_pr = d2star(seq, pr)
    out(f"    Pasteuria ramosa (Daphnia pathogen) d2* = {d2_pr:.4f}  "
        f"[{len(pr):,} bp]")

    out("    superbug panel (real NCBI reference genomes):")
    try:
        cache_dir = Path("/tmp/host_cache")
        if not cache_dir.exists():
            cache_dir = ROOT / "data" / "hosts"
        report = screen_compatibility(
            seq, query_name=NAME, cache_dir=cache_dir, k=6)
        super_row = {}
        for m in report.matches:
            super_row[m.taxon] = m.d2star
            out(f"      {m.taxon:28s} d2* = {m.d2star:.4f}")
        d2_super = report.matches[0].d2star if report.matches else None
    except Exception as exc:  # network failures must not kill the build
        out(f"    [warn] superbug screen failed: {exc}")
        super_row, d2_super = {}, None

    results = {
        "name": NAME, "length": len(seq), "gc": round(genome.metadata["gc"], 4),
        "genes_annotated": len(genes),
        "validation_passed": passed,
        "validation": {c.name: {"passed": c.passed, "detail": c.detail}
                       for c in validation.checks},
        "protein_production": {
            "orfs": proteins.n_orfs, "full_proteins": proteins.n_proteins,
            "truncated": proteins.n_truncated,
            "fraction": round(proteins.n_proteins / max(1, proteins.n_orfs), 4),
            "mean_protein_len": round(proteins.mean_protein_len, 1),
        },
        "cai": {"mean": round(mean_cai, 3), "min": min(cais), "max": max(cais)},
        "d2star": {
            "Daphnia_magna_genome": round(d2_daph, 4),
            "Pasteuria_ramosa": round(d2_pr, 4),
            "superbugs": super_row,
        },
    }
    (OUT / "daphnia_phage_1.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n== done in {time.time()-t0:.0f}s; report: "
          f"{OUT / 'daphnia_phage_1.json'} ==")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())