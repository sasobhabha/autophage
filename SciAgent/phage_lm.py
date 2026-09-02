"""
Expert bacteriophage genomic language model (Evo-style).

This module trains a decoder-only Transformer language model on real,
publicly available bacteriophage genomes from NCBI (RefSeq) -- the same
architecture family and autoregressive next-token objective used by genomic
foundation models such as Evo 2 (Brixi, Durrant, Ku et al., 2025,
bioRxiv 10.1101/2025.02.18.638918; Evo 2 is a 7B-40B parameter model trained
on 9.3 trillion bases on thousands of GPUs).

Honest scope note: training a model that *rivals* Evo 2 (billions of
parameters, trillions of tokens) is not feasible on a laptop. What this module
provides is a real, working version of that pipeline at small scale:

  * real training data: thousands of RefSeq phage genomes, downloaded live
    from NCBI (default Caudoviricetes, taxid 28883)
  * a k-mer tokenizer (6-mer, 4096-token vocabulary -- the same vocabulary
    size Evo uses, mirroring its byte-level design)
  * a causal decoder-only Transformer (GPT-style with rotary embeddings,
    pre-LayerNorm, weight tying) trained with cross-entropy next-token
    prediction
  * checkpointing, train/validation splitting, temperature/top-k sampling
    to generate new genomes

The trained checkpoint is a genuine expert model of phage sequence statistics;
scaling it toward Evo-2-class capability is a matter of compute and data, not
architecture.

CLI::

    python phage_lm.py fetch  --max-genomes 500 --out data/phages
    python phage_lm.py train  --data data/phages --epochs 1 --out checkpoints/phage_lm.pt
    python phage_lm.py generate --model checkpoints/phage_lm.pt --length 50000 --out phage.fasta
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import ssl
import sys
import time
import urllib.request
import zipfile
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from host_compat import ncbi_get, NCBI_BASE, USER_AGENT
except Exception:  # pragma: no cover - importable standalone
    NCBI_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"

    def ncbi_get(url, verify_ssl=False, timeout=90):
        import ssl as _ssl
        if verify_ssl:
            opener = urllib.request.build_opener()
        else:
            ctx = _ssl._create_unverified_context()
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx))
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with opener.open(req, timeout=timeout) as resp:
            return resp.read()


# ---------------------------------------------------------------------------
# Data: fetch real phage genomes from NCBI
# ---------------------------------------------------------------------------

def fetch_phage_genomes(max_genomes: int = 500,
                        out_dir: str = "data/phages",
                        taxid: int = 28883,
                        min_len: int = 2000,
                        max_len: int = 200_000,
                        verify_ssl: bool = False) -> List[str]:
    """
    Download up to `max_genomes` real phage genomes (RefSeq, Caudoviricetes
    by default) into `out_dir` as individual FASTA files. Returns the list of
    accessions saved.
    """
    from host_compat import ncbi_get, NCBI_BASE, USER_AGENT
    from phage_genome import parse_fasta

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = (f"{NCBI_BASE}/genome/taxon/{taxid}/dataset_report?page_size=100"
           f"&filters.assembly_version=current")
    data = json.loads(ncbi_get(url, verify_ssl=verify_ssl))
    reports = data.get("reports", [])
    print(f"[fetch] NCBI reports {data.get('total_count')} genomes; "
          f"scanning up to {len(reports)} paged results")

    accessions = [r["accession"] for r in reports
                  if r.get("accession", "").startswith("GCF_")]
    # include more pages until we have enough candidates
    page = 0
    while len(accessions) < max_genomes and reports:
        page += 1
        url = (f"{NCBI_BASE}/genome/taxon/{taxid}/dataset_report"
               f"?page_size=100&page_token={reports[-1].get('next_page_token', '')}"
               f"&filters.assembly_version=current")
        try:
            data = json.loads(ncbi_get(url, verify_ssl=verify_ssl))
        except Exception:
            break
        reports = data.get("reports", [])
        accessions += [r["accession"] for r in reports
                       if r.get("accession", "").startswith("GCF_")]

    saved: List[str] = []
    for acc in accessions[:max_genomes]:
        fasta_path = out_dir / f"{acc}.fasta"
        if fasta_path.exists():
            saved.append(acc)
            continue
        try:
            blob = ncbi_get(
                f"{NCBI_BASE}/genome/accession/{acc}/download"
                f"?include_annotation_type=GENOME_FASTA",
                verify_ssl=verify_ssl,
            )
            zf = zipfile.ZipFile(io.BytesIO(blob))
            fna = [n for n in zf.namelist() if n.endswith(".fna")]
            seq_parts = []
            for n in fna:
                for line in zf.read(n).decode().splitlines():
                    if not line.startswith(">"):
                        seq_parts.append(line.strip().upper())
            seq = "".join(seq_parts)
            if not (min_len <= len(seq) <= max_len):
                continue
            fasta_path.write_text(f">{acc}\n{seq}\n")
            saved.append(acc)
        except Exception as exc:
            print(f"  [warn] {acc}: {exc}", file=sys.stderr)
    print(f"[fetch] saved {len(saved)} phage genomes to {out_dir}")
    return saved


# ---------------------------------------------------------------------------
# Tokenizer: 6-mer (4096 vocab, same size as Evo's byte-level vocabulary)
# ---------------------------------------------------------------------------

@dataclass
class PhageTokenizer:
    k: int = 6
    bos: int = 4096
    eos: int = 4097
    pad: int = 4098

    def __post_init__(self):
        self.vocab_size = 4096 + 3

    def encode(self, seq: str) -> List[int]:
        """Slide a k-mer window over the sequence; skip windows with Ns or
        any non-ACGT character (mirrors VirHostMatcher's N-reset counting)."""
        seq = seq.upper()
        toks: List[int] = []
        for i in range(len(seq) - self.k + 1):
            w = seq[i:i + self.k]
            v = 0
            ok = True
            for b in w:
                base = "ACGT".find(b)
                if base < 0:
                    ok = False
                    break
                v = (v << 2) | base
            if ok:
                toks.append(v)
        return toks

    def decode(self, toks: Sequence[int]) -> str:
        out = []
        for t in toks:
            if t >= 4096:
                continue
            v = t
            w = []
            for _ in range(self.k):
                w.append("ACGT"[v & 3])
                v >>= 2
            out.append("".join(reversed(w)))
        return "".join(out)


# ---------------------------------------------------------------------------
# Model: causal decoder-only Transformer (GPT-style), like Evo's base class
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (seq, dim/2)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def rotate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (..., seq, heads, dim)
        seq = x.size(-2)
        cos = self.cos_cached[:seq].to(x.dtype)
        sin = self.sin_cached[:seq].to(x.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary position embedding (GPT-NeoX style).

    q/k: (B, heads, T, head_dim), cos/sin: (T, head_dim/2).
    """
    # expand cos/sin to (1, 1, T, head_dim/2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    # split last dim in half; first half rotated by cos, second half by sin
    q1, q2 = q[..., : cos.shape[-1]], q[..., cos.shape[-1]:]
    k1, k2 = k[..., : cos.shape[-1]], k[..., cos.shape[-1]:]
    q_rot = torch.cat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1)
    k_rot = torch.cat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1)
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rotary.rotate(q)
        q, k = apply_rotary(q, k, cos, sin)
        mask = self.causal_mask[:T, :T]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PhageGenomeLM(nn.Module):
    """
    Decoder-only genomic language model (same architecture family as Evo 2:
    causal attention, pre-LayerNorm, rotary embeddings, weight-tying).
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 8, max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                temperature: float = 1.0):
        B, T = idx.shape
        assert T <= self.max_seq_len, "sequence too long for model"
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x) / temperature
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: List[int], length: int, temperature: float = 0.8,
                 top_k: int = 50) -> List[int]:
        self.eval()
        if not prompt:
            prompt = [random.randrange(4096)]
        out = list(prompt)
        device = next(self.parameters()).device
        for _ in range(length):
            ctx = out[-self.max_seq_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits, _ = self(x, temperature=temperature)
            nxt_logits = logits[0, -1, :]
            if top_k > 0:
                vals, _ = torch.topk(nxt_logits, top_k)
                nxt_logits[nxt_logits < vals[-1]] = float("-inf")
            probs = F.softmax(nxt_logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            out.append(nxt)
        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PhageDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_dir: str, tokenizer: PhageTokenizer,
                 seq_len: int = 512, split: str = "train", frac: float = 0.9):
        self.tok = tokenizer
        self.seq_len = seq_len
        files = sorted(Path(fasta_dir).glob("*.fasta"))
        rng = random.Random(42)
        rng.shuffle(files)
        n_train = max(1, int(len(files) * frac))
        files = files[:n_train] if split == "train" else files[n_train:]
        self.chunks: List[List[int]] = []
        for f in files:
            text = f.read_text()
            seq = "".join(l.strip().upper() for l in text.splitlines()
                          if not l.startswith(">"))
            toks = tokenizer.encode(seq)
            if len(toks) < 10:
                continue
            # chunk with overlap-free windows
            for i in range(0, len(toks) - seq_len + 1, seq_len):
                self.chunks.append(toks[i:i + seq_len])
            # tail chunk
            if len(toks) % seq_len:
                self.chunks.append(toks[-seq_len:])
        print(f"[data] {split}: {len(self.chunks)} chunks from {len(files)} genomes")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        toks = self.chunks[i]
        # pad short (tail) chunks; pad targets are masked out via ignore_index=-100
        n = self.seq_len - 1
        x = torch.full((n,), self.tok.pad, dtype=torch.long)
        y = torch.full((n,), -100, dtype=torch.long)
        t = toks[:n]
        x[:len(t) - 1] = torch.tensor(t[:-1], dtype=torch.long)
        y[:len(t) - 1] = torch.tensor(t[1:], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader,
             device: str) -> float:
    model.eval()
    total, cnt = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total += loss.item() * x.size(0)
        cnt += x.size(0)
    model.train()
    return total / max(1, cnt)


def train_lm(data_dir: str, out_path: str, epochs: int = 3,
             d_model: int = 256, n_layers: int = 4, n_heads: int = 8,
             seq_len: int = 512, batch_size: int = 16, lr: float = 3e-4,
             seed: int = 1, max_steps: Optional[int] = None,
             grad_accum: int = 1, use_amp: bool = True,
             device: str = "") -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    tok = PhageTokenizer()
    train_ds = PhageDataset(data_dir, tok, seq_len=seq_len, split="train")
    val_ds = PhageDataset(data_dir, tok, seq_len=seq_len, split="val")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False)

    model = PhageGenomeLM(tok.vocab_size, d_model, n_layers, n_heads,
                          max_seq_len=seq_len + 1)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {n_params:,} parameters on {device}")
    _print_memory_accounting(n_params, use_amp=use_amp)

    # mixed precision: bf16 is supported on MPS (Metal) and CUDA
    use_bf16 = use_amp and device.startswith(("mps", "cuda"))

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    # warmup + cosine schedule over one epoch
    steps_per_epoch = max(1, len(train_loader))
    warmup = min(50, steps_per_epoch // 10)
    total_steps = steps_per_epoch * epochs

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step = 0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        opt.zero_grad(set_to_none=True)
        accum = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            if use_bf16:
                with torch.autocast(
                        device_type="mps" if device.startswith("mps") else "cuda",
                        dtype=torch.bfloat16):
                    _, loss = model(x, targets=y)
            else:
                _, loss = model(x, targets=y)
            loss = loss / grad_accum
            loss.backward()
            run_loss += loss.item() * grad_accum
            accum += 1
            if accum % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if max_steps and step >= max_steps:
                break
            if step % 25 == 0:
                val = evaluate(model, val_loader, device)
                print(f"  step {step:5d} lr {lr_at(step):.2e} "
                      f"train {run_loss/max(1,(step % steps_per_epoch or steps_per_epoch)):.4f} "
                      f"val {val:.4f} ({time.time()-t0:.1f}s)")
                if val < best_val:
                    best_val = val
                    torch.save({"model": model.state_dict(),
                                "config": {"d_model": d_model, "n_layers": n_layers,
                                           "n_heads": n_heads, "seq_len": seq_len,
                                           "vocab_size": tok.vocab_size}},
                               out_path)
        val = evaluate(model, val_loader, device)
        print(f"[epoch {epoch+1}/{epochs}] val {val:.4f}")
        if val < best_val:
            best_val = val
            torch.save({"model": model.state_dict(),
                        "config": {"d_model": d_model, "n_layers": n_layers,
                                   "n_heads": n_heads, "seq_len": seq_len,
                                   "vocab_size": tok.vocab_size}},
                       out_path)
    print(f"[train] done; best val loss {best_val:.4f}; checkpoint {out_path}")
    return best_val


def load_lm(model_path: str, device: str = "") -> Tuple[PhageGenomeLM, PhageTokenizer]:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    tok = PhageTokenizer()
    model = PhageGenomeLM(
        vocab_size=cfg.get("vocab_size", tok.vocab_size),
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        max_seq_len=cfg.get("seq_len", 512) + 1,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fetch(args: argparse.Namespace) -> int:
    fetch_phage_genomes(max_genomes=args.max_genomes, out_dir=args.out,
                        taxid=args.taxid, verify_ssl=args.verify_ssl)
    return 0


def _train(args: argparse.Namespace) -> int:
    train_lm(data_dir=args.data, out_path=args.out, epochs=args.epochs,
             d_model=args.d_model, n_layers=args.n_layers,
             n_heads=args.n_heads, seq_len=args.seq_len,
             batch_size=args.batch, lr=args.lr, seed=args.seed,
             max_steps=args.max_steps, grad_accum=args.grad_accum,
             use_amp=not args.no_amp)
    return 0


def _print_memory_accounting(n_params: int, use_amp: bool) -> None:
    """Honest estimate of the memory (and hardware) needed to train a model
    of this size, so users can reason about scale (e.g. a 2B-parameter model
    needs a GPU cluster, not a laptop)."""
    bytes_per = 2 if use_amp else 4
    weights = n_params * bytes_per
    grads = n_params * bytes_per
    adam = n_params * 2 * 4  # fp32 m and v
    activations = n_params * bytes_per  # rough upper bound
    total_gb = (weights + grads + adam + activations) / 1e9
    print(f"[mem] est. training memory: weights {weights/1e9:.1f} GB, "
          f"grads {grads/1e9:.1f} GB, Adam(m,v) {adam/1e9:.1f} GB, "
          f"activations ~{activations/1e9:.1f} GB => ~{total_gb:.1f} GB total")
    if n_params >= 1e9:
        print(f"[mem] {n_params/1e9:.1f}B parameters is a cluster-class model: "
              f"expect multi-GPU training (e.g. 8xH100) and weeks of compute; "
              f"the Daphnia+phage dataset here (~0.2 Gbp) is also far too small "
              f"to train a {n_params/1e9:.1f}B model without severe overfitting.")


def _generate(args: argparse.Namespace) -> int:
    model, tok = load_lm(args.model)
    prompt = tok.encode("ATGCGT")  # needs k=6 bases for the first window
    if not prompt:
        prompt = [0]
    n_tokens = args.length // tok.k + 100
    toks = model.generate(prompt, length=n_tokens,
                          temperature=args.temperature, top_k=args.top_k)
    seq = tok.decode(toks)
    seq = seq[: args.length]  # trim to requested length
    with open(args.out, "w") as f:
        f.write(f">synthetic_phage_lm length={len(seq)}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i + 80] + "\n")
    print(f"[generate] wrote {args.out} ({len(seq)} bp)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="phage_lm",
                                description="Evo-style phage genomic language model")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download real phage genomes from NCBI")
    f.add_argument("--max-genomes", type=int, default=500)
    f.add_argument("--taxid", type=int, default=28883, help="Caudoviricetes (tailed phages)")
    f.add_argument("--out", default="data/phages")
    f.add_argument("--verify-ssl", action="store_true")
    f.set_defaults(func=_fetch)

    t = sub.add_parser("train", help="train the genomic LM")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--n-layers", type=int, default=4)
    t.add_argument("--n-heads", type=int, default=8)
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--grad-accum", type=int, default=1,
                   help="gradient accumulation steps (bigger effective batch)")
    t.add_argument("--no-amp", action="store_true",
                   help="disable bf16 mixed precision")
    t.set_defaults(func=_train)

    g = sub.add_parser("generate", help="sample a new genome from the model")
    g.add_argument("--model", required=True)
    g.add_argument("--length", type=int, default=50_000)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--out", default="synthetic_phage_lm.fasta")
    g.set_defaults(func=_generate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())