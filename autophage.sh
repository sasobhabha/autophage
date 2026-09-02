#!/usr/bin/env bash
# Autophage — read any genetic dataset, output a phage.
#   ./autophage.sh                    interactive session
#   ./autophage.sh make --input <X>   one-shot (FASTA/FASTQ/GBK/ZIP/dir/accession/DNA)
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi
exec "$PY" Autophage/cli.py "$@"