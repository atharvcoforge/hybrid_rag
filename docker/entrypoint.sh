#!/bin/sh
set -e
INDEX_DIR="${INDEX_DIR:-index}"
CORPUS="${CORPUS:-documents}"
if [ ! -f "$INDEX_DIR/side.sqlite" ]; then
  python -m rag ingest "$CORPUS" --index "$INDEX_DIR"
fi
exec python -m rag.server
