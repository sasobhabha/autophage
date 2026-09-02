"""
Batch test: "if I put ANY genome, can it make a phage?"

Runs the complete pipeline (host codon profile from the bare genome
sequence -> host-adapted 50 kb phage design -> phage-biology validation ->
protein production -> CAI -> d2* vs the host) over every FASTA in a
directory, with no annotation required.

Usage:
    .venv/bin/python batch_any_genome.py --dir data/hosts --out outputs/any_genome_batch.json

Example (real reference genomes of 10 priority antibiotic-resistant
bacteria spanning 33-66% GC):
    .venv/bin/python batch_any_genome.py --dir /tmp/host_cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import build_daphnia_phage as bd                  # noqa: E402
from phage_genome import (                        # noqa: E402
    generate_phage_genome, validate_phage_genome, protein_production_report,
    parse_fasta,
)
from host_compat import (                         # noqa: E402
    codon_usage_from_genome_sequence, codon_adaptation_index, d2star,
)

LENGTH = 50_000


def run_one(fasta: Path, seed: int, out) -> dict:
    name, seq = parse_fasta(fasta.read_text())
    host_gc = bd.genome_gc_sample(fasta)
    usage = codon_usage_from_genome_sequence(seq[:5_000_000])
    gc_target = min(0.70, max(0.25, host_gc - 0.05))
    genome = generate_phage_genome(
        length=LENGTH, gc_target=gc_target, seed=seed,
        terminal_repeat=200, rbs_fraction=0.9,
        name=f"phage_{name}", host_codon_usage=usage)
    gseq = genome.sequence
    validation = validate_phage_genome(gseq, name=f"phage_{name}")
    proteins = protein_production_report(gseq)
    genes = bd.annotate(gseq, usage)
    cais = [g["cai"] for g in genes]
    d2_host = d2star(gseq, seq[:5_000_000])
    failed = [c.name for c in validation.checks
              if not c.passed and c.name != "terminal_repeats"]
    row = {
        "host": fasta.stem, "name": name,
        "genome_bp": len(seq), "host_gc": round(host_gc, 3),
        "phage_bp": len(gseq), "phage_gc": round(genome.metadata["gc"], 3),
        "validated": validation.passed,
        "failed_checks": failed,
        "orfs": proteins.n_orfs,
        "full_proteins": proteins.n_proteins,
        "protein_fraction": round(proteins.n_proteins
                                  / max(1, proteins.n_orfs), 3),
        "mean_cai": round(sum(cais) / max(1, len(cais)), 3),
        "d2star_vs_host": round(d2_host, 4),
    }
    out(f"  {row['host']:32s} GC {row['host_gc']:.2f} -> "
        f"phage GC {row['phage_gc']:.2f} | "
        f"{'VALIDATED' if row['validated'] else 'FAILED ' + str(failed)} | "
        f"proteins {row['protein_fraction']:.0%} | "
        f"CAI {row['mean_cai']:.2f} | d2* {row['d2star_vs_host']:.3f}")
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, help="directory of host FASTA files")
    p.add_argument("--out", default="outputs/any_genome_batch.json")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    t0 = time.time()
    out = lambda msg: print(msg, flush=True)
    files = sorted(Path(args.dir).glob("*.fasta"))
    if args.limit:
        files = files[:args.limit]
    out(f"batch: {len(files)} host genomes from {args.dir}")
    rows = []
    for i, f in enumerate(files):
        try:
            rows.append(run_one(f, args.seed + i, out))
        except Exception as exc:
            out(f"  {f.stem:32s} ERROR: {exc}")
            rows.append({"host": f.stem, "error": str(exc)})
    n_ok = sum(1 for r in rows if r.get("validated"))
    n_tot = sum(1 for r in rows if "error" not in r)
    out(f"\n== {n_ok}/{n_tot} genomes yielded a validated phage "
        f"({time.time()-t0:.0f}s) ==")
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n_validated": n_ok, "n_total": n_tot, "results": rows}, indent=2))
    out(f"results: {args.out}")
    return 0 if n_ok == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())