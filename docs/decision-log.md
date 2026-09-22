# Decision log

Short record of choices that closed an argument. Not a changelog.

| Date | Decision | Why | Cost if wrong |
| --- | --- | --- | --- |
| 2026-09 | SQLite-only store, drop Chroma | Corpus is thousands of chunks; exact cosine is simpler and scored | Remount ANN later if the corpus grows 100× |
| 2026-09 | Strictly local inference | Policy text must not leave the machine | Jev / hosted rerankers stay stubs |
| 2026-09 | Contextual retrieval behind a flag | Ships only if it wins on the golden set | One ablation row forever if it loses |
| 2026-09 | Docling as rung 3 only | Fast path is right most of the time | Borderless failures stay flagged prose |
| 2026-09 | Four citation gates, cheapest first | Fabricated figures are the failure mode that matters | Slight answerable abstention |
| 2026-09 | Keep the honesty line in the verdict | Four PDFs is not production; saying so keeps every other number believable | — |
