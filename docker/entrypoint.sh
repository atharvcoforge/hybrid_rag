#!/bin/sh
set -e
INDEX_DIR="${INDEX_DIR:-index}"
CORPUS="${CORPUS:-documents}"
# Idempotent: unchanged files are skipped, a new file is indexed.
python -m rag ingest "$CORPUS" --index "$INDEX_DIR"
exec python -m rag.server
