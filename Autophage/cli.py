"""
Autophage interactive CLI.

Read ANY genetic dataset -- FASTA, FASTQ, GenBank (.gb/.gbk/.gbff), a ZIP
archive, a directory of files, a raw pasted DNA string, a gzipped file, or
an NCBI accession -- parse it, and output a complete, host-adapted,
verifiably phage-like genome string.

Usage:
    ./autophage.sh                  # interactive session (paste/type anything)
    ./autophage.sh make --input <X> # one-shot; prints the phage FASTA string
    # (note: macOS case-insensitive FS requires .sh to avoid the Autophage/ dir)

Examples of <X>:
    genome.fasta  reads.fastq  genome.gbk  dataset.zip  some/folder/
    GCA_000934625.1   NC_001604.1   (NCBI accessions, fetched over the network)
    ATGC... (paste >=60 bp of DNA directly)
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import ssl
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phage_genome import (                      # noqa: E402
    generate_phage_genome, validate_phage_genome,
    protein_production_report, extract_cds_from_gtf,
)
from host_compat import (                       # noqa: E402
    codon_usage_from_cds, codon_usage_from_genome_sequence,
    codon_adaptation_index, d2star, gc_content,
)
from build_daphnia_phage import (               # noqa: E402
    first_n_bp, genome_gc_sample, annotate, GENE_CATALOG,
)

OUT_DIR = Path(__file__).parent / "outputs"
BUDGET_BP = 25_000_000  # max bases read from a dataset (huge files are sampled)
ACCESSION_RE = re.compile(r"^(GCF_|GCA_|NC_|NZ_|NT_|NW_|CP\d{6})(\.\d+)?")
DNA_RE = re.compile(r"^[ACGTNacgtnueryswkmbdhvUERYSWKMBDHV]+$")

GENOME_EXTS = {".fna", ".fa", ".fasta", ".fas"}
ANNOT_EXTS = {".gtf", ".gff", ".gff3"}
GB_EXTS = {".gb", ".gbk", ".gbff"}
FASTQ_EXTS = {".fastq", ".fq"}
CDS_HINTS = ("cds", "_cds", "cds.", "orfs")
NON_CDS_HINTS = ("pep", "protein", "rna", "trna", "grna", "translated")


def out(s: str = "") -> None:
    print(s, flush=True)


# ---------------------------------------------------------------------------
# Loaders -> InputDataset (everything normalized to files in a temp dir)
# ---------------------------------------------------------------------------

@dataclass
class InputDataset:
    name: str
    kind: str
    genome_fasta: Path          # contig-level FASTA of the host genome
    gtf_path: Path | None = None
    cds_fasta: Path | None = None
    tmpdir: Path | None = None

    def cleanup(self) -> None:
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None


def _write_fasta(path: Path, records) -> None:
    with open(path, "w") as f:
        for name, seq in records:
            f.write(f">{name}\n{seq}\n")


def read_fasta_records(text: str):
    """Yield (name, seq) for every record in FASTA text."""
    name, buf = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name:
                yield name, "".join(buf)
            name, buf = line[1:].strip().split()[0], []
        else:
            buf.append(line.upper())
    if name:
        yield name, "".join(buf)


def read_fastq_records(path: Path, max_bp: int = 0):
    """Stream FASTQ records (4-line blocks) without loading the file.

    Stops once `max_bp` bases have been yielded (0 = unlimited). Keeps
    memory flat even for multi-GB FASTQ files.
    """
    seen = 0
    with open(path) as f:
        while True:
            hdr = f.readline()
            seq = f.readline()
            f.readline()  # '+'
            f.readline()  # quality
            if not hdr or not seq:
                break
            if hdr.startswith("@"):
                yield hdr[1:].split()[0], seq.strip().upper()
                seen += len(seq) - 1
                if max_bp and seen >= max_bp:
                    break


def parse_gb_location(loc: str):
    """Minimal GenBank location parser -> (coords[(start,end)...], strand)."""
    strand = -1 if "complement(" in loc else 1
    s = loc.replace("complement(", "").replace("join(", "").replace("order(", "")
    s = s.rstrip(")")
    coords = []
    for part in s.split(","):
        part = part.strip().lstrip("<>")
        if ".." in part:
            a, b = part.split("..")
            coords.append((int(a), int(b)))
    return coords, strand


def parse_genbank(path: Path) -> InputDataset:
    """Parse a GenBank file: ORIGIN sequence + CDS features -> cds FASTA."""
    text = path.read_text(errors="replace")
    name = "genbank"
    m = re.search(r"LOCUS\s+(\S+)", text)
    if m:
        name = m.group(1)
    m = re.search(r"ORIGIN\n(.*?)\n//", text, re.S)
    if not m:
        raise ValueError(f"no ORIGIN sequence in {path}")
    seq = re.sub(r"[^A-Za-z]", "", m.group(1)).upper()

    # CDS features:  CDS  <loc>  (location after 21+ spaces)
    cds_list, raw_cds = [], []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s{5}CDS\s+", line):
            loc = line.split("CDS", 1)[1].strip()
            while ")" in loc and loc.count("(") > loc.count(")"):
                i += 1
                loc += lines[i].strip()
            raw_cds.append((loc, name))
        i += 1
    for loc, _ in raw_cds:
        try:
            coords, strand = parse_gb_location(loc)
        except Exception:
            continue
        parts = [seq[a - 1:b] for a, b in coords if 0 < a <= len(seq)]
        if not parts:
            continue
        cds = "".join(parts)
        if strand == -1:
            cds = "".join("ACGT"["TGCA".index(b)] for b in reversed(cds))
        cds_list.append(cds)

    tmp = Path(tempfile.mkdtemp(prefix="autophage_"))
    genome_fa = tmp / "genome.fasta"
    _write_fasta(genome_fa, [(name, seq)])
    cds_fa = tmp / "cds.fasta"
    _write_fasta(cds_fa, [(f"cds{i}", s) for i, s in enumerate(cds_list) if s])
    return InputDataset(name=name, kind="genbank", genome_fasta=genome_fa,
                        cds_fasta=cds_fa if cds_list else None, tmpdir=tmp)


def discover_dir(directory: Path, tmp: Path | None = None) -> InputDataset:
    """Scan a directory (or extracted archive) for genome + annotation."""
    files = [p for p in directory.rglob("*") if p.is_file()]
    genome_candidates, cds_candidates, annots = [], [], []
    for p in files:
        ext = p.suffix.lower()
        name = p.name.lower()
        if ext in GB_EXTS:
            return parse_genbank(p)
        if ext in GENOME_EXTS or ext == ".gz":
            if any(h in name for h in CDS_HINTS) and not any(h in name for h in NON_CDS_HINTS):
                cds_candidates.append(p)
            elif not any(h in name for h in NON_CDS_HINTS):
                genome_candidates.append(p)
        if ext in ANNOT_EXTS:
            annots.append(p)
    if not genome_candidates:
        fastqs = [p for p in files if p.suffix.lower() in FASTQ_EXTS]
        if fastqs:
            fq = max(fastqs, key=lambda p: p.stat().st_size)
            out(f"  [dir] no genome FASTA; sampling reads from {fq.name}")
            return load_fastq(fq, BUDGET_BP)
        raise ValueError(f"no genome FASTA found in {directory}")
    genome = max(genome_candidates, key=lambda p: p.stat().st_size)
    gtf = next((a for a in annots if a.stem.split(".")[0] == genome.stem.split(".")[0]), None)
    if gtf is None and annots:
        gtf = annots[0]
    cds_fa = max(cds_candidates, key=lambda p: p.stat().st_size) if cds_candidates else None
    return InputDataset(name=genome.stem.split(".")[0], kind="dir",
                        genome_fasta=genome, gtf_path=gtf,
                        cds_fasta=cds_fa, tmpdir=None)


def load_zip(path: Path, budget_bp: int) -> InputDataset:
    """Extract only what's needed from a ZIP (streamed), so a huge archive
    costs O(needed files) disk + O(budget) memory."""
    tmp = Path(tempfile.mkdtemp(prefix="autophage_"))
    want = GENOME_EXTS | ANNOT_EXTS | GB_EXTS
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            ext = Path(member).suffix.lower()
            if ext in want:
                dest = tmp / Path(member).name
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
    # genome+annotation found? use them. otherwise fall back to reads.
    fastqs = [m for m in zf.namelist()
              if Path(m).suffix.lower() in FASTQ_EXTS]
    try:
        ds = discover_dir(tmp)
    except ValueError:
        if not fastqs:
            raise ValueError(f"no genome/annotation/reads found in {path}")
        fq = max(fastqs, key=lambda m: zf.getinfo(m).file_size)
        out(f"  [zip] no genome found; sampling reads from {Path(fq).name}")
        with zipfile.ZipFile(path) as zf:
            with zf.open(fq) as src:
                tmp_fq = tmp / "reads.fastq"
                with open(tmp_fq, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
        ds = load_fastq(tmp_fq, budget_bp)
    ds.kind = "zip"
    ds.tmpdir = tmp  # take ownership for cleanup
    return ds


def load_fastq(path: Path, budget_bp: int) -> InputDataset:
    """Stream a FASTQ into a capped genome FASTA (O(budget) memory), so a
    60 GB reads file costs only the profile budget of RAM."""
    tmp = Path(tempfile.mkdtemp(prefix="autophage_"))
    genome_fa = tmp / "genome.fasta"
    n_reads = total = 0
    truncated = False
    with open(genome_fa, "w") as g:
        for name, seq in read_fastq_records(path, max_bp=budget_bp):
            g.write(f">{name}\n{seq}\n")
            n_reads += 1
            total += len(seq)
        if budget_bp and total >= budget_bp:
            truncated = True
    note = f" (first {total:,} bp used; file is larger)" if truncated else ""
    out(f"  [fastq] {n_reads} reads, {total:,} bp sampled{note}")
    return InputDataset(name=path.stem, kind="fastq",
                        genome_fasta=genome_fa, tmpdir=tmp)


def _stream_fasta_to(path: Path, dest: Path, budget_bp: int):
    """Stream a (optionally gzipped) FASTA into dest, capped at budget bp.
    Returns (first_record_name, records, truncated)."""
    raw = gzip.open(path, "rt") if path.suffix.lower() == ".gz" else open(path)
    name, buf = None, []
    n_recs = written = 0
    truncated = False
    with raw, open(dest, "w") as g:
        for line in raw:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    g.write(f">{name}\n{''.join(buf)}\n")
                    n_recs += 1
                    written += len(buf)
                    if written >= budget_bp:
                        truncated = True
                        break
                name, buf = line[1:].strip().split()[0], []
            else:
                buf.append(line.upper())
        else:
            if name:
                g.write(f">{name}\n{''.join(buf)}\n")
                n_recs += 1
    return (name or "record", n_recs, truncated)


def load_plain(path: Path, budget_bp: int) -> InputDataset:
    tmp = Path(tempfile.mkdtemp(prefix="autophage_"))
    genome_fa = tmp / "genome.fasta"
    try:
        name, n_recs, truncated = _stream_fasta_to(path, genome_fa, budget_bp)
    except Exception:
        raise ValueError(f"could not parse {path} as FASTA")
    if truncated:
        out(f"  [fasta] {n_recs} record(s), first {budget_bp:,} bp retained "
            f"(file is larger — sampling is sufficient for a codon profile)")
    # sibling annotation with the same stem?
    gtf = None
    for ext in ANNOT_EXTS:
        cand = path.with_suffix(ext)
        if cand.exists():
            gtf = cand
            break
    return InputDataset(name=name, kind="fasta", genome_fasta=genome_fa,
                        gtf_path=gtf, tmpdir=tmp)


def load_accession(acc: str) -> InputDataset:
    """Fetch a genome from NCBI (datasets API) by accession."""
    out(f"  [fetch] downloading {acc} from NCBI ...")
    ctx = ssl._create_unverified_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    url = (f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/"
           f"{acc}/download?include_annotation_type=GENOME_FASTA,GENOME_GFF")
    req = urllib.request.Request(url, headers={"User-Agent": "autophage/1.0"})
    with opener.open(req, timeout=180) as resp:
        blob = resp.read()
    tmp = Path(tempfile.mkdtemp(prefix="autophage_"))
    with zipfile.ZipFile(__import__("io").BytesIO(blob)) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            ext = Path(member).suffix.lower()
            if ext in GENOME_EXTS | ANNOT_EXTS:
                dest = tmp / Path(member).name
                dest.write_bytes(zf.read(member))
    ds = discover_dir(tmp)
    ds.tmpdir = tmp
    ds.kind = "accession"
    return ds


def looks_like_dna(text: str) -> bool:
    t = text.strip()
    if len(t) < 60:
        return False
    clean = re.sub(r"\s+", "", t)
    return len(clean) >= 60 and len(DNA_RE.sub("", clean)) <= len(clean) * 0.02


def load_dataset(raw: str, budget_bp: int = BUDGET_BP) -> InputDataset:
    p = Path(raw)
    if p.exists():
        if p.is_dir():
            return discover_dir(p)
        ext = p.suffix.lower()
        if ext == ".zip" or zipfile.is_zipfile(p):
            return load_zip(p, budget_bp)
        if ext in GB_EXTS:
            return parse_genbank(p)
        if ext in FASTQ_EXTS:
            return load_fastq(p, budget_bp)
        return load_plain(p, budget_bp)
    if ACCESSION_RE.match(raw.strip()):
        return load_accession(raw.strip())
    if looks_like_dna(raw):
        tmp = Path(tempfile.mkdtemp(prefix="autophage_"))
        seq = re.sub(r"\s+", "", raw).upper()
        genome_fa = tmp / "genome.fasta"
        genome_fa.write_text(f">pasted_dna\n{seq}\n")
        return InputDataset(name="pasted_dna", kind="raw-dna",
                            genome_fasta=genome_fa, tmpdir=tmp)
    raise ValueError(
        f"cannot read '{raw}': not a file/dir/zip, not an NCBI accession, "
        f"and not a DNA string. Give a path (FASTA/FASTQ/GenBank/ZIP), a "
        f"directory, an accession like GCA_000934625.1, or paste DNA.")


# ---------------------------------------------------------------------------
# The phage factory (shared by REPL and one-shot)
# ---------------------------------------------------------------------------

def build_phage(ds: InputDataset, length: int = 50_000, seed: int = 7,
                verbose: bool = True) -> dict:
    """Build + verify a host-adapted phage for the loaded dataset."""
    # 1. host codon profile (prefer curated CDS > GTF extraction > genome seq)
    if ds.cds_fasta:
        usage = codon_usage_from_cds(str(ds.cds_fasta))
        src = f"CDS FASTA ({ds.cds_fasta.name})"
    elif ds.gtf_path:
        cds_list = extract_cds_from_gtf(str(ds.genome_fasta), str(ds.gtf_path))
        if not cds_list:
            raise ValueError(f"no CDS could be extracted from {ds.gtf_path}")
        tmp = ds.tmpdir or Path(tempfile.mkdtemp(prefix="autophage_"))
        cds_fa = tmp / "cds.fasta"
        _write_fasta(cds_fa, [(f"cds{i}", s) for i, s in enumerate(cds_list)])
        usage = codon_usage_from_cds(str(cds_fa))
        src = f"GTF/GFF CDS ({len(cds_list)} genes from {ds.gtf_path.name})"
    else:
        usage = codon_usage_from_genome_sequence(first_n_bp(ds.genome_fasta, 5_000_000))
        src = "genome sequence (no annotation found)"

    host_gc = genome_gc_sample(ds.genome_fasta)
    gc_target = min(0.70, max(0.25, host_gc - 0.05))
    if verbose:
        out(f"  host profile : {src}")
        out(f"  host GC      : {host_gc:.3f}")

    genome = generate_phage_genome(
        length=length, gc_target=gc_target, seed=seed,
        terminal_repeat=200, rbs_fraction=0.9,
        name=f"phage_{ds.name}", host_codon_usage=usage)
    seq = genome.sequence

    validation = validate_phage_genome(seq, name=f"phage_{ds.name}")
    proteins = protein_production_report(seq)
    genes = annotate(seq, usage)
    cais = [g["cai"] for g in genes]
    d2_host = d2star(seq, first_n_bp(ds.genome_fasta, 5_000_000))
    if verbose:
        out(validation.summary())
        out(proteins.summary())
        out(f"  annotated genes : {len(genes)} (mean CAI vs host "
            f"{sum(cais)/max(1,len(cais)):.3f})")
        out(f"  d2* vs host     : {d2_host:.4f}  (lower = more host-like)")
        out("  d2* reference   : real lytic phages typically score 0.2-0.5 "
            "against their host")
    return {
        "name": genome.name, "kind": ds.kind, "source": ds.name,
        "length": len(seq), "gc": round(genome.metadata["gc"], 4),
        "validated": validation.passed,
        "failed_checks": [c.name for c in validation.checks
                          if not c.passed and c.name != "terminal_repeats"],
        "orfs": proteins.n_orfs, "full_proteins": proteins.n_proteins,
        "protein_fraction": round(proteins.n_proteins / max(1, proteins.n_orfs), 3),
        "mean_cai": round(sum(cais) / max(1, len(cais)), 3),
        "d2star_vs_host": round(d2_host, 4),
        "sequence": seq,
        "genes": genes,
    }


def phage_fasta_string(result: dict) -> str:
    seq = result["sequence"]
    lines = [f">{result['name']} | Autophage synthetic {result['kind']}-adapted "
             f"phage | {len(seq)} bp | validated={result['validated']}"]
    for i in range(0, len(seq), 80):
        lines.append(seq[i:i + 80])
    return "\n".join(lines)


def write_outputs(result: dict, prefix: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{prefix}.fasta").write_text(phage_fasta_string(result) + "\n")
    with open(OUT_DIR / f"{prefix}.gff3", "w") as f:
        f.write("##gff-version 3\n")
        f.write(f"##sequence-region {result['name']} 1 {result['length']}\n")
        f.write(f"{result['name']}\tAutophage\ttransit_peptide\t1\t200\t.\t+\t.\t"
                f"ID=TR001;product=direct terminal repeat (packaging signal)\n")
        for g in result["genes"]:
            f.write(f"{result['name']}\tAutophage\tCDS\t{g['start']}\t{g['end']}\t.\t"
                    f"{g['strand']}\t0\tID={g['gene_id']};product={g['product']};"
                    f"cai={g['cai']}\n")
    summary = {k: v for k, v in result.items() if k not in ("sequence", "genes")}
    (OUT_DIR / f"{prefix}.json").write_text(json.dumps(summary, indent=2))
    out(f"  wrote {OUT_DIR / (prefix + '.fasta')} (+ .gff3 + .json)")


# ---------------------------------------------------------------------------
# Interactive REPL + CLI
# ---------------------------------------------------------------------------

HELP = """Autophage: give me any genome, I'll write you a phage.
  - a path:  genome.fasta  reads.fastq  genome.gbk  data.zip  some/folder/
  - an NCBI accession:  GCA_000934625.1  NC_001604.1
  - paste 60+ bp of DNA directly
Type "exit" to quit, "help" for this text.
"""


def interactive(budget_bp: int = BUDGET_BP) -> int:
    out("==================================================================")
    out("   Autophage — read any genetic dataset, output a phage")
    out("==================================================================")
    out(HELP)
    while True:
        try:
            raw = input("\nautophage> ").strip()
        except (EOFError, KeyboardInterrupt):
            out("\nbye")
            return 0
        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "q"):
            out("bye")
            return 0
        if raw.lower() in ("help", "h", "?"):
            out(HELP)
            continue
        ds = None
        try:
            t0 = time.time()
            ds = load_dataset(raw, budget_bp)
            result = build_phage(ds)
            out(f"\n===== PHAGE OUTPUT STRING ({result['length']:,} bp, "
                f"validated={result['validated']}) [{time.time()-t0:.0f}s] =====")
            out(phage_fasta_string(result))
        except Exception as exc:
            out(f"  ! {exc}")
        finally:
            if ds:
                ds.cleanup()


def make_cmd(args: argparse.Namespace) -> int:
    ds = None
    try:
        t0 = time.time()
        ds = load_dataset(args.input, budget_bp=int(args.budget_mb * 1e6))
        result = build_phage(ds, length=args.length, seed=args.seed)
        out("\n===== PHAGE OUTPUT STRING =====")
        out(phage_fasta_string(result))
        if args.out_prefix:
            write_outputs(result, args.out_prefix)
        out(f"\n[{time.time()-t0:.0f}s] validation_passed={result['validated']} "
            f"| full_proteins={result['full_proteins']}/{result['orfs']} | "
            f"mean_cai={result['mean_cai']} | d2*_vs_host={result['d2star_vs_host']}")
        return 0 if result["validated"] else 1
    except Exception as exc:
        out(f"error: {exc}")
        return 2
    finally:
        if ds:
            ds.cleanup()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="autophage",
        description="Read any genetic dataset and output a host-adapted "
                    "bacteriophage genome string.")
    sub = p.add_subparsers(dest="command")
    m = sub.add_parser("make", help="one-shot: dataset in, phage string out")
    m.add_argument("--input", required=True)
    m.add_argument("--length", type=int, default=50_000)
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--budget-mb", type=int, default=25,
                   help="max MB of a large dataset to read for the codon "
                        "profile (60 GB files are sampled, not loaded)")
    m.add_argument("--out-prefix", default=None,
                   help="also write <prefix>.fasta/.gff3/.json into outputs/")
    m.set_defaults(func=make_cmd)
    p.add_argument("--budget-mb", type=int, default=25,
                   help="max MB read from a dataset (interactive mode)")

    args = p.parse_args(argv)
    if args.command == "make":
        return args.func(args)
    return interactive(budget_bp=int(args.budget_mb * 1e6))


if __name__ == "__main__":
    sys.exit(main())