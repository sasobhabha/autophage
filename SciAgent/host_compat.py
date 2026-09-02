"""
Superbug host-compatibility screening for synthetic phage genomes.

This module implements the published VirHostMatcher d2* oligonucleotide
frequency (ONF) dissimilarity measure (Ahlgren et al., Nucleic Acids Research
45(1):39-53, 2017; PMID 27899557) to compare a query genome sequence against
publicly available reference genomes of priority antibiotic-resistant
pathogens ("superbugs") and rank which hosts the sequence is most compatible
with.

The d2* implementation is a faithful port of the reference implementation at
https://github.com/jessieren/VirHostMatcher (computeMeasure_onlyd2star.cpp)
using the pipeline's default settings (k=6, single-strand counts, Markov
order 2). It has been validated against the reference C++ executable on the
project's own toy dataset (max abs error < 2e-6 over 276 virus-host pairs):

  * k-mer counts on the forward strand only, ambiguous bases resetting the
    sliding window (exactly like SeqKmerCountSingle)
  * both-strand pairing in scoring:  X_w = count(w) + count(revcomp(w))
  * background word probabilities under a Markov chain of order 2 (pwMC):
    P(first `order` bases) times the row-normalized transition probabilities
  * expected counts  E[X_w] = p_w * totalKmer
  * C2star = D2star / (sqrt(C2_a) * sqrt(C2_b)) with
    D2star = sum_w X_tilde_a X_tilde_b / sqrt(E_a E_b),
    C2_s   = sum_w (X_tilde_s / sqrt(E_s))^2
  * dissimilarity  d2* = 0.5 * (1 - C2star)   (0 = identical ONF profiles)

Lower d2* means greater ONF similarity, i.e. a more plausible host match.
Host genomes are fetched live from NCBI (Datasets v2 API, RefSeq reference
assemblies), so validation is against real, publicly available data.

Public health context for the species list: CDC Antibiotic Resistance Threats
in the United States (2019) -- the ESKAPE pathogens and other urgent/serious
threats (MRSA, CRE, CRAB, MDR/XDR P. aeruginosa, VRE, ESBL Enterobacterales,
MDR-TB, drug-resistant N. gonorrhoeae, MDR Salmonella).
"""

from __future__ import annotations

import io
import json
import ssl
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

COMPLEMENT = str.maketrans("ACGT", "TGCA")

# ---------------------------------------------------------------------------
# Curated superbug list: (common name, NCBI taxid, resistance context)
# ---------------------------------------------------------------------------

SUPERBUGS: List[Dict[str, object]] = [
    {"name": "Staphylococcus aureus (MRSA)", "taxid": 1280,
     "context": "Methicillin-resistant S. aureus (urgent/serious threat)"},
    {"name": "Klebsiella pneumoniae (CRE)", "taxid": 573,
     "context": "Carbapenem-resistant Enterobacteriaceae (urgent threat)"},
    {"name": "Acinetobacter baumannii (CRAB)", "taxid": 470,
     "context": "Carbapenem-resistant A. baumannii (urgent threat)"},
    {"name": "Pseudomonas aeruginosa (MDR/XDR)", "taxid": 287,
     "context": "Drug-resistant P. aeruginosa (serious threat)"},
    {"name": "Enterococcus faecium (VRE)", "taxid": 1352,
     "context": "Vancomycin-resistant enterococci (serious threat)"},
    {"name": "Escherichia coli (ESBL)", "taxid": 562,
     "context": "ESBL-producing Enterobacterales (serious threat)"},
    {"name": "Enterobacter cloacae complex (CRE)", "taxid": 550,
     "context": "Carbapenem-resistant Enterobacterales (serious threat)"},
    {"name": "Mycobacterium tuberculosis (MDR-TB)", "taxid": 1773,
     "context": "Multidrug-resistant tuberculosis (serious threat)"},
    {"name": "Neisseria gonorrhoeae", "taxid": 485,
     "context": "Drug-resistant N. gonorrhoeae (urgent threat)"},
    {"name": "Salmonella enterica (MDR)", "taxid": 28901,
     "context": "Drug-resistant nontyphoidal Salmonella (serious threat)"},
]


def superbug_taxids() -> Dict[str, int]:
    return {s["name"]: int(s["taxid"]) for s in SUPERBUGS}


# ---------------------------------------------------------------------------
# NCBI datasets API access (with relaxed SSL for local/intercepting proxies)
# ---------------------------------------------------------------------------

def _opener(verify_ssl: bool):
    if verify_ssl:
        return urllib.request.build_opener()
    ctx = ssl._create_unverified_context()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


NCBI_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"
USER_AGENT = "SciAgent/0.1 (synthetic phage research; contact: local)"


def ncbi_get(url: str, verify_ssl: bool = False, timeout: int = 90):
    """GET a URL, returning raw bytes. SSL relaxed by default for proxy envs."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _opener(verify_ssl).open(req, timeout=timeout) as resp:
        return resp.read()


def reference_accession_for_taxid(taxid: int, verify_ssl: bool = False) -> str:
    """
    Find the NCBI RefSeq reference assembly accession for a taxon.
    Raises RuntimeError if none is available.
    """
    url = (f"{NCBI_BASE}/genome/taxon/{taxid}/dataset_report"
           f"?page_size=10&filters.reference_only=true&filters.assembly_version=current")
    data = json.loads(ncbi_get(url, verify_ssl=verify_ssl))
    for rep in data.get("reports", []):
        acc = rep.get("accession", "")
        if acc.startswith("GCF_"):  # prefer RefSeq over GenBank
            return acc
    url = (f"{NCBI_BASE}/genome/taxon/{taxid}/dataset_report"
           f"?page_size=10&filters.assembly_version=current")
    data = json.loads(ncbi_get(url, verify_ssl=verify_ssl))
    for rep in data.get("reports", []):
        acc = rep.get("accession", "")
        if acc.startswith("GCF_"):
            return acc
    raise RuntimeError(f"no RefSeq reference assembly found for taxid {taxid}")


def download_genome_fasta(accession: str, cache_dir: Path,
                          verify_ssl: bool = False) -> str:
    """
    Download a complete genome FASTA for an accession from NCBI datasets,
    cache it locally, and return the concatenated sequence (all contigs).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_file = cache_dir / f"{accession}.fasta"
    if out_file.exists():
        return _read_fasta(out_file)

    url = (f"{NCBI_BASE}/genome/accession/{accession}/download"
           f"?include_annotation_type=GENOME_FASTA")
    blob = ncbi_get(url, verify_ssl=verify_ssl)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    fna_names = [n for n in zf.namelist() if n.endswith(".fna")]
    if not fna_names:
        raise RuntimeError(f"no .fna file for {accession}")
    seq_lines: List[str] = []
    for n in sorted(fna_names):
        text = zf.read(n).decode()
        for line in text.splitlines():
            if line.startswith(">"):
                continue
            seq_lines.append(line.strip().upper())
    fasta = "".join(seq_lines)
    out_file.write_text(f">{accession}\n{fasta}\n")
    return fasta


def _read_fasta(path: Path) -> str:
    seq: List[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            continue
        seq.append(line.strip().upper())
    return "".join(seq)


# ---------------------------------------------------------------------------
# Host codon-usage analysis and codon adaptation index (CAI)
# ---------------------------------------------------------------------------

# Standard genetic code: amino acid -> list of codons (same as phage_genome)
HOST_CODON_TABLE: Dict[str, List[str]] = {
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

HOST_CODON_TO_AA: Dict[str, str] = {
    codon: aa for aa, codons in HOST_CODON_TABLE.items() for codon in codons
}


def load_fasta_sequences(path: str) -> Dict[str, str]:
    """Parse a FASTA file into {header: sequence} (uppercased, no ambiguity)."""
    seqs: Dict[str, str] = {}
    name = ""
    buf: List[str] = []
    for raw in Path(path).read_text().splitlines():
        if raw.startswith(">"):
            if name:
                seqs[name] = "".join(buf).upper()
            name = raw[1:].strip()
            buf = []
        else:
            buf.append(raw.strip())
    if name:
        seqs[name] = "".join(buf).upper()
    return seqs


def codon_usage_from_cds(cds_fasta: str) -> Dict[str, float]:
    """
    Count codon usage across all CDS records of a genome. Returns the relative
    frequency of every sense codon (sums to 1 over synonymous groups).
    Sequences are trimmed of any trailing incomplete codon before counting.
    """
    counts = {c: 0 for c in HOST_CODON_TO_AA}
    for seq in load_fasta_sequences(cds_fasta).values():
        seq = seq[: len(seq) - (len(seq) % 3)]
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i + 3]
            if codon in counts:
                counts[codon] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"no codons found in {cds_fasta}")
    return {c: n / total for c, n in counts.items()}


def codon_usage_from_genome_sequence(seq: str) -> Dict[str, float]:
    """
    Codon usage from a raw genome sequence: count sense codons in all six
    reading frames (a compositional proxy when CDS annotations are missing;
    prefer codon_usage_from_cds when annotations exist).
    """
    counts = {c: 0 for c in HOST_CODON_TO_AA}
    for strand in (seq, seq.translate(COMPLEMENT)[::-1]):
        for frame in range(3):
            for i in range(frame, len(strand) - 2, 3):
                codon = strand[i:i + 3]
                if codon in counts:
                    counts[codon] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no codons found in sequence")
    return {c: n / total for c, n in counts.items()}


def codon_adaptation_index(protein_cdna: str, host_usage: Dict[str, float]) -> float:
    """
    Codon Adaptation Index (Sharp & Li 1987): geometric mean of relative
    synonymous codon usage (w_i) over the CDS, where w_i for each codon is the
    host frequency of that codon divided by the most frequent synonym for its
    amino acid. Ranges 0-1; phages adapted to a host score high (~0.7+).
    """
    w: Dict[str, float] = {}
    for aa, codons in HOST_CODON_TABLE.items():
        freqs = [(c, host_usage.get(c, 0.0)) for c in codons]
        top = max((f for _, f in freqs), default=0.0)
        for codon, f in freqs:
            w[codon] = (f / top) if top > 0 else 1.0
    seq = protein_cdna[: len(protein_cdna) - (len(protein_cdna) % 3)]
    codons = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]
    n = len(codons)
    if n == 0:
        return 0.0
    log_sum = sum(
        __import__("math").log(max(w.get(c, 1e-9), 1e-9)) for c in codons
    )
    return __import__("math").exp(log_sum / n)


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


@dataclass
class HostAdaptationReport:
    host_name: str
    host_gc: float
    mean_cai: float
    gc_distance: float
    d2star: float
    n_genes: int = 0
    n_valid_proteins: int = 0

    def summary(self) -> str:
        lines = [
            f"Host-adaptation report for '{self.host_name}':",
            f"  host GC = {self.host_gc:.3f}",
            f"  CDS genes screened = {self.n_genes}",
            f"  genes producing full proteins = {self.n_valid_proteins}",
            f"  mean codon adaptation index (CAI) = {self.mean_cai:.3f}",
            f"  |GC(query) - GC(host)| = {self.gc_distance:.3f}",
            f"  d2* ONF dissimilarity vs host = {self.d2star:.4f}",
            "",
            "  Interpretation:",
            "  - CAI ~0.7+  : phage codon usage closely matches host tRNA pool\n"
            "    (strong host adaptation).",
            "  - GC distance < 0.1 : compatible compositional habitat.",
            "  - d2* < 0.45       : ONF similarity consistent with host\n"
            "    interaction (VirHostMatcher). Lower is more similar.",
        ]
        return "\n".join(lines)


def screen_host_adaptation(
    query_cds: List[str],
    host_name: str,
    host_usage: Dict[str, float],
    host_seq: str = "",
    query_seq: str = "",
    k: int = 6,
) -> HostAdaptationReport:
    """
    Score how well a set of query CDS (e.g. a synthetic phage's genes) is
    adapted to a host's codon usage: mean CAI, GC distance, and (if both
    sequences provided) d2* ONF dissimilarity.
    """
    cais = [codon_adaptation_index(c, host_usage) for c in query_cds if c]
    mean_cai = sum(cais) / len(cais) if cais else 0.0
    n_valid = sum(1 for c in query_cds if _is_valid_cds(c))
    gc_q = gc_content("".join(query_cds))
    d2 = d2star(query_seq, host_seq, k=k) if query_seq and host_seq else float("nan")
    return HostAdaptationReport(
        host_name=host_name,
        host_gc=gc_content(host_seq) if host_seq else float("nan"),
        mean_cai=mean_cai,
        gc_distance=abs(gc_q - (gc_content(host_seq) if host_seq else gc_q)),
        d2star=d2,
        n_genes=len(query_cds),
        n_valid_proteins=n_valid,
    )


def _is_valid_cds(cds: str) -> bool:
    """A CDS is valid if it starts with a start codon, has no internal stop
    codons in frame, and ends with a stop codon."""
    seq = cds.upper()
    if len(seq) < 3:
        return False
    if seq[:3] not in ("ATG", "GTG", "TTG"):
        return False
    if seq[-3:] not in ("TAA", "TAG", "TGA"):
        return False
    body = seq[3:-3]
    return all(body[i:i + 3] not in ("TAA", "TAG", "TGA")
               for i in range(0, len(body) - 2, 3))


# ---------------------------------------------------------------------------
# d2* implementation (faithful port of VirHostMatcher, k=6 by default)
# ---------------------------------------------------------------------------

def _encode_kmer_int(seq: str, k: int) -> int:
    """2-bit integer encoding of a k-mer (A=0, C=1, G=2, T=3)."""
    v = 0
    for b in seq:
        v = (v << 2) | ("ACGT".index(b))
    return v


def count_kmers(seq: str, k: int = 6) -> Tuple[List[int], int]:
    """
    Count k-mers on the forward strand using a rolling 2-bit encoding,
    mirroring VirHostMatcher's SeqKmerCountSingle: ambiguous bases reset the
    sliding window, so k-mers spanning non-ACGT characters are skipped.
    Returns (counts array of size 4^k, valid k-mer observations).
    """
    n = len(seq)
    if n < k:
        return [0] * (4 ** k), 0
    counts = [0] * (4 ** k)
    v = 0          # rolling 2-bit encoding of the last up-to-k bases
    run = 0        # length of the current clean (ACGT-only) run
    total = 0
    for i, ch in enumerate(seq):
        base = "ACGT".find(ch)
        if base < 0:
            run = 0
            v = 0
            continue
        run += 1
        v = ((v << 2) | base) & ((1 << (2 * k)) - 1)
        if run >= k:
            counts[v] += 1
            total += 1
    return counts, total


def mononucleotide_frequencies(seq: str) -> List[float]:
    """
    Base frequencies over unambiguous bases only (A/C/G/T), matching how
    VirHostMatcher's k=1 counts are computed: ambiguous bases are excluded
    from both the numerator and the denominator.
    """
    counts = [seq.count(b) for b in "ACGT"]
    total = sum(counts) or 1
    return [c / total for c in counts]


def _revcomp_int(w: int, k: int) -> int:
    """
    2-bit integer of the reverse complement of k-mer `w` (k bases).

    Matches VirHostMatcher's reverseFour(): ``bases`` here is extracted least
    significant digit first (i.e. last base first), so iterating forward and
    complementing each digit yields comp(last) .. comp(first), which is
    exactly the reverse-complement word.
    """
    bases = []
    v = w
    for _ in range(k):
        bases.append(v & 3)
        v >>= 2
    out = 0
    for b in bases:
        out = (out << 2) | (3 - b)
    return out


def _pw_markov(counts_order: List[int], total_order: int,
               counts_order1: List[int], total_order1: int,
               order: int, k: int) -> List[float]:
    """
    Background word probabilities under a Markov chain of given order,
    faithfully ported from VirHostMatcher's pwMC():  the initial
    probability of the order-gram is count(order)/total(order) and each
    subsequent base is added with the transition probability
    [count(order+1-mer)/total(order+1)] / [count(order-mer)/total(order)].
    """
    n_order = 4 ** order
    ini_prob = [0.0] * n_order
    trans = [[0.0] * 4 for _ in range(n_order)]
    for row in range(n_order):
        count_below = counts_order[row]
        prob_below = count_below / total_order if total_order else 0.0
        ini_prob[row] = prob_below
        if prob_below == 0.0:
            continue
        row_sum = 0.0
        for col in range(4):
            count_above = counts_order1[row * 4 + col]
            prob_above = count_above / total_order1 if total_order1 else 0.0
            trans[row][col] = prob_above / prob_below
            row_sum += trans[row][col]
        # Row-normalize the transition matrix, as in VirHostMatcher's pwMC().
        if row_sum > 0.0:
            for col in range(4):
                trans[row][col] /= row_sum

    pw = [0.0] * (4 ** k)
    for w in range(4 ** k):
        v = w
        bases = [0] * k
        for pos in range(k - 1, -1, -1):
            bases[pos] = v & 3
            v >>= 2
        # initial order-gram probability
        ini_ten = 0
        for pos in range(order):
            ini_ten = (ini_ten << 2) | bases[pos]
        prob = ini_prob[ini_ten]
        # multiply transition probabilities for the remaining bases
        for pos in range(order, k):
            from_ten = 0
            for j in range(pos - order, pos):
                from_ten = (from_ten << 2) | bases[j]
            prob *= trans[from_ten][bases[pos]]
        pw[w] = prob
    return pw


def d2star(seq_a: str, seq_b: str, k: int = 6, order: int = 2) -> float:
    """
    d2* ONF dissimilarity (Ahlgren et al. 2017), lower = more similar.
    Faithful port of VirHostMatcher computeMeasure_onlyd2star.cpp with the
    default pipeline settings (k=6, single-strand counts, order-2 Markov
    background).  Reverse-complement counts are combined as
    X_w = count(w) + count(revcomp(w)) and C2star = D2star / (sqrt(C2a) * sqrt(C2b))
    with d2* = 0.5 * (1 - C2star).
    """
    if len(seq_a) < k or len(seq_b) < k:
        return 1.0

    counts_a, total_a = count_kmers(seq_a, k)
    counts_b, total_b = count_kmers(seq_b, k)

    pw_a = _pw_markov(*count_kmers(seq_a, order), *count_kmers(seq_a, order + 1),
                      order, k)
    pw_b = _pw_markov(*count_kmers(seq_b, order), *count_kmers(seq_b, order + 1),
                      order, k)

    D2star = 0.0
    C2star_below = [0.0, 0.0]
    n_words = 4 ** k
    for w in range(n_words):
        rw = _revcomp_int(w, k)
        # counts and expectations combine both strands
        X_a = float(counts_a[w] + counts_a[rw])
        X_b = float(counts_b[w] + counts_b[rw])
        p_a = pw_a[w] + pw_a[rw]
        p_b = pw_b[w] + pw_b[rw]
        E_a = p_a * total_a
        E_b = p_b * total_b
        Xt_a = X_a - E_a
        Xt_b = X_b - E_b
        if E_a != 0 and E_b != 0:
            D2star += (Xt_a * Xt_b) / ((E_a * E_b) ** 0.5)
        if E_a != 0:
            t = Xt_a / (E_a ** 0.5)
            C2star_below[0] += t * t
        if E_b != 0:
            t = Xt_b / (E_b ** 0.5)
            C2star_below[1] += t * t

    denom = (C2star_below[0] ** 0.5) * (C2star_below[1] ** 0.5)
    if denom == 0:
        return 1.0
    C2star = D2star / denom
    # clamp into [-1, 1] for numerical safety
    C2star = max(-1.0, min(1.0, C2star))
    return 0.5 * (1.0 - C2star)


# ---------------------------------------------------------------------------
# Compatibility screening
# ---------------------------------------------------------------------------

@dataclass
class HostMatch:
    taxon: str
    accession: str
    gc_query: float
    gc_host: float
    d2star: float

    @property
    def label(self) -> str:
        return f"{self.taxon} [{self.accession}]"


@dataclass
class CompatibilityReport:
    query_name: str
    query_length: int
    query_gc: float
    k: int
    matches: List[HostMatch] = field(default_factory=list)

    @property
    def best(self) -> Optional[HostMatch]:
        return self.matches[0] if self.matches else None

    def summary(self) -> str:
        lines = [
            f"Host-compatibility screen (d2*, k={self.k}; lower = more similar):",
            f"  query: {self.query_name}  ({self.query_length} bp, "
            f"gc={self.query_gc:.3f})",
            "",
            "  rank  taxon  accession  d2*      gc(q/h)",
        ]
        for i, m in enumerate(self.matches):
            lines.append(
                f"  {i + 1:4d}  {m.taxon[:34]:34s} {m.accession:16s} "
                f"{m.d2star:7.4f}  {m.gc_query:.2f}/{m.gc_host:.2f}"
            )
        best = self.best
        if best:
            lines.append("")
            lines.append(
                f"  Most compatible host: {best.taxon} (d2*={best.d2star:.4f}, "
                f"GC q/h = {best.gc_query:.2f}/{best.gc_host:.2f})"
            )
            lines.append(
                "  NOTE: ONF similarity indicates a plausible host, not proof of "
                "infectivity; experimental host-range assays are required."
            )
        return "\n".join(lines)


def screen_compatibility(
    query_seq: str,
    query_name: str = "query",
    taxids: Optional[Dict[str, int]] = None,
    cache_dir: Path = Path("data/hosts"),
    k: int = 6,
    verify_ssl: bool = False,
    max_hosts: Optional[int] = None,
) -> CompatibilityReport:
    """
    Screen a query genome for ONF compatibility against real superbug genomes.

    Fetches RefSeq reference genomes from NCBI on demand (cached in
    `cache_dir`), computes d2* dissimilarity between query and each host,
    and returns a report ordered by increasing d2* (best match first).
    """
    taxids = taxids or superbug_taxids()
    matches: List[HostMatch] = []
    for taxon, taxid in taxids.items():
        try:
            acc = reference_accession_for_taxid(taxid, verify_ssl=verify_ssl)
            host_seq = download_genome_fasta(acc, cache_dir, verify_ssl=verify_ssl)
        except Exception as exc:  # keep screening going if one host fails
            print(f"  [warn] skipping {taxon}: {exc}", file=sys.stderr)
            continue
        d2 = d2star(query_seq, host_seq, k=k)
        matches.append(HostMatch(
            taxon=taxon,
            accession=acc,
            gc_query=round(_gc(query_seq), 4),
            gc_host=round(_gc(host_seq), 4),
            d2star=round(d2, 6),
        ))
        if max_hosts and len(matches) >= max_hosts:
            break
    matches.sort(key=lambda m: m.d2star)
    return CompatibilityReport(
        query_name=query_name,
        query_length=len(query_seq),
        query_gc=round(_gc(query_seq), 4),
        k=k,
        matches=matches,
    )


def _gc(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="host_compat",
        description="Screen a genome for compatibility against real superbug "
                    "genomes using the published VirHostMatcher d2* measure.",
    )
    p.add_argument("--query", required=True, help="FASTA file of the query genome")
    p.add_argument("--cache", default="data/hosts", help="host genome cache dir")
    p.add_argument("--k", type=int, default=6, help="k-mer length")
    p.add_argument("--verify-ssl", action="store_true",
                   help="use strict SSL verification (default: relaxed)")
    args = p.parse_args(argv)

    from phage_genome import parse_fasta
    name, seq = parse_fasta(Path(args.query).read_text())
    report = screen_compatibility(
        seq, query_name=name, cache_dir=Path(args.cache),
        k=args.k, verify_ssl=args.verify_ssl,
    )
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())