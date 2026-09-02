"""
Build a complete, host-adapted bacteriophage genome for ANY host ("if I give
any cell genome, make a phage for it"):

  1. host analysis  -- codon-usage profile from the host's CDS (extracted
                       from a GTF/GFF annotation, or a CDS FASTA)
  2. design         -- 40-50 kb dsDNA genome, every gene back-translated with
                       the host's codon usage (Sharp & Li 1987 CAI), terminal
                       repeats, Shine-Dalgarno RBS
  3. annotation     -- T7-like functional layout assigned to all six-frame
                       ORFs, emitted as GFF3
  4. verification   -- phage-biology validation, protein-production test,
                       codon-adaptation index vs the host, and VirHostMatcher
                       d2* screen vs the host genome, an optional extra
                       bacterium, and 10 real superbug reference genomes

Outputs land in SciAgent/outputs/<prefix>.<fasta|gff3|json>

Examples:
    # Daphnia magna (default; KEGG/ncbi dataset in data/Daphnia_magna_NIES)
    .venv/bin/python build_daphnia_phage.py

    # Lactobacillus acidophilus FSI4 (KEGG T03681, yogurt probiotic)
    .venv/bin/python build_daphnia_phage.py \
        --host-name Lactobacillus_acidophilus_FSI4 \
        --genome-fasta data/hosts/Lactobacillus_acidophilus_FSI4.fasta \
        --gtf data/hosts/Lactobacillus_acidophilus_FSI4.gff \
        --prefix lacto_phage_1 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phage_genome import (                     # noqa: E402
    generate_phage_genome, validate_phage_genome,
    protein_production_report, _scan_orfs, reverse_complement,
    extract_cds_from_gtf,
)
from host_compat import (                      # noqa: E402
    codon_usage_from_cds, codon_adaptation_index,
    d2star, screen_compatibility, gc_content,
)

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "Daphnia_magna_NIES"
OUT = ROOT / "outputs"

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


def build_codon_usage(args) -> dict:
    """Codon-usage profile from GTF-extracted CDS, or a CDS FASTA directly."""
    if args.gtf:
        cds_list = extract_cds_from_gtf(args.genome_fasta, args.gtf)
        tmp = OUT / f"__cds_{args.prefix}.fasta"
        tmp.write_text("".join(f">cds{i}\n{s}\n" for i, s in enumerate(cds_list)))
        out(f"  extracted {len(cds_list)} spliced CDS from {args.gtf}")
        usage = codon_usage_from_cds(str(tmp))
        tmp.unlink(missing_ok=True)
        return usage
    out(f"  using pre-built CDS FASTA {args.cds_fasta}")
    return codon_usage_from_cds(str(args.cds_fasta))


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
            idx = (i - int(0.30 * n)) % len(GENE_CATALOG["middle"])
            product = GENE_CATALOG["middle"][idx]
        else:
            idx = (i - int(0.75 * n)) % len(GENE_CATALOG["late"])
            product = GENE_CATALOG["late"][idx]
        genes.append({
            "gene_id": f"g{i + 1:03d}", "start": s + 1, "end": e,
            "strand": strand, "product": product,
            "protein_len": len(cds) // 3 - 1,
            "cai": round(codon_adaptation_index(cds, host_usage), 3),
        })
    return genes


def write_fasta(seq: str, name: str, args, path: Path) -> None:
    with open(path, "w") as f:
        f.write(f">{name} synthetic {args.host_name}-adapted bacteriophage, "
                f"{len(seq)} bp, dsDNA, terminal repeats\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i + 80] + "\n")


def write_gff3(seq: str, name: str, genes: list, path: Path) -> None:
    with open(path, "w") as f:
        f.write("##gff-version 3\n")
        f.write(f"##sequence-region {name} 1 {len(seq)}\n")
        f.write(f"{name}\tSciAgent\ttransit_peptide\t1\t200\t.\t+\t.\t"
                f"ID=TR001;product=direct terminal repeat (packaging signal)"
                f";note=predicted\n")
        for g in genes:
            f.write(f"{name}\tSciAgent\tCDS\t{g['start']}\t{g['end']}\t.\t"
                    f"{g['strand']}\t0\tID={g['gene_id']};"
                    f"product={g['product']};cai={g['cai']};"
                    f"note=predicted, layout-based annotation\n")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_daphnia_phage",
        description="Design and verify a host-adapted phage genome for "
                    "any host genome (Daphnia magna by default).")
    p.add_argument("--host-name", default="Daphnia_magna",
                   help="host species/isolate name (used in genome name)")
    p.add_argument("--genome-fasta", default=str(DATA / "Daphnia_magna_NIES_genome.fa"))
    p.add_argument("--gtf", default=None,
                   help="GTF/GFF annotation to extract CDS from")
    p.add_argument("--cds-fasta", default=str(DATA / "Daphnia_magna_NIES_cds.fa"),
                   help="pre-built CDS FASTA (used when --gtf is not given)")
    p.add_argument("--pathogen-fasta",
                   default=str(ROOT / "data" / "hosts"
                               / "Pasteuria_ramosa_GCF_056496825.1.fasta"),
                   help="optional extra genome to screen (e.g. a pathogen)")
    p.add_argument("--no-pathogen-screen", action="store_true")
    p.add_argument("--superbug-cache", default="/tmp/host_cache",
                   help="cache dir holding real superbug genomes")
    p.add_argument("--length", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", default="daphnia_phage_1")
    return p.parse_args(argv)


def main(argv=None) -> int:
    global out
    args = parse_args(argv)
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    out = lambda msg: print(msg, flush=True)

    genome_name = f"phage_{args.host_name}"

    out("== 1. Host analysis ==")
    host_usage = build_codon_usage(args)
    out(f"  {len(host_usage)} codons in profile (sum {sum(host_usage.values()):.2f})")
    host_gc = genome_gc_sample(Path(args.genome_fasta))
    out(f"  host genome GC (20 Mb sample): {host_gc:.3f}")

    out("\n== 2. Design: host-adapted DNA ==")
    genome = generate_phage_genome(
        length=args.length, gc_target=max(0.25, host_gc - 0.05),
        seed=args.seed, terminal_repeat=200, rbs_fraction=0.9,
        name=genome_name, host_codon_usage=host_usage,
    )
    seq = genome.sequence
    out(f"  {len(seq):,} bp, GC {genome.metadata['gc']:.3f}, "
        f"{genome.metadata['genes']} genes built, "
        f"coding density {genome.metadata['coding_density']:.3f}")

    out("\n== 3. Annotation ==")
    genes = annotate(seq, host_usage)
    out(f"  {len(genes)} six-frame ORFs annotated (T7-like layout)")
    fa, gff = OUT / f"{args.prefix}.fasta", OUT / f"{args.prefix}.gff3"
    write_fasta(seq, genome_name, args, fa)
    write_gff3(seq, genome_name, genes, gff)
    out(f"  wrote {fa} and {gff}")

    out("\n== 4. Verification ==")
    validation = validate_phage_genome(seq, name=genome_name)
    out(validation.summary())
    passed = validation.passed

    proteins = protein_production_report(seq)
    out(proteins.summary())

    cais = [g["cai"] for g in genes]
    mean_cai = sum(cais) / max(1, len(cais))
    out(f"  mean codon-adaptation index vs host: {mean_cai:.3f} "
        f"(n={len(cais)} genes)")

    out("\n  d2* host screen (VirHostMatcher, k=6; lower = more similar):")
    host_sample = first_n_bp(Path(args.genome_fasta), 5_000_000)
    d2_host = d2star(seq, host_sample)
    out(f"    host genome ({args.host_name}, 5 Mb)     d2* = {d2_host:.4f}")

    d2_pathogen, pathogen_name = None, None
    if not args.no_pathogen_screen and Path(args.pathogen_fasta).exists():
        pathogen_name = Path(args.pathogen_fasta).stem
        pseq = first_n_bp(Path(args.pathogen_fasta), 50_000_000)
        d2_pathogen = d2star(seq, pseq)
        out(f"    pathogen/extra genome ({pathogen_name}) d2* = {d2_pathogen:.4f} "
            f"[{len(pseq):,} bp]")

    super_row, d2_super = {}, None
    out("    superbug panel (real NCBI reference genomes):")
    try:
        cache = Path(args.superbug_cache)
        if not cache.exists():
            cache = ROOT / "data" / "hosts"
        report = screen_compatibility(
            seq, query_name=genome_name, cache_dir=cache, k=6)
        for m in report.matches:
            super_row[m.taxon] = m.d2star
            out(f"      {m.taxon:28s} d2* = {m.d2star:.4f}")
        if report.matches:
            d2_super = report.matches[0].d2star
    except Exception as exc:
        out(f"    [warn] superbug screen failed: {exc}")

    results = {
        "host": args.host_name, "name": genome_name,
        "length": len(seq), "gc": round(genome.metadata["gc"], 4),
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
            "host_genome": round(d2_host, 4),
            "pathogen_or_extra": (round(d2_pathogen, 4) if d2_pathogen is not None
                                  else None),
            "superbugs": super_row,
        },
    }
    jp = OUT / f"{args.prefix}.json"
    jp.write_text(json.dumps(results, indent=2))
    out(f"\n== done in {time.time()-t0:.0f}s; report: {jp} ==")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())