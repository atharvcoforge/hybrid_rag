# Verification

Ship-gate measurement on 2026-09-23. Every exit code below is taken from a log in `docs/verification/logs/`. Chat output is not a source.

## Environment

From `315-offline-gates.log` and the interpreters used for the later commands:

| Tool | Version |
| --- | --- |
| Python | 3.11.16 (`.venv`, uv-managed) |
| uv | 0.12.5 |
| mypy | 2.3.1 |
| ruff | 0.16.8 |
| pytest | 9.1.1 |
| mutmut | 3.8.0 |
| Docker server | 29.7.2 (`317-docker-build.log` completed against this daemon) |
| Node | `~/.local/node/bin/node` (not on the default `PATH`; used for logs 318–320) |

`pytest` has `pythonpath = ["."]` in `pyproject.toml`. Without it, `uv`’s Python omits the working directory and collection dies with `ModuleNotFoundError: No module named 'tests'` (`259-ci-repro.log`, `PYTEST_EXIT:2`).

## Command table

| Command | Exit | Log |
| --- | --- | --- |
| `uv lock --check` | 0 | `325-uv-lock-check.log` (`LOCK_CHECK_EXIT:0`). The original resolve is `256-uv-lock.log` (`Resolved 183 packages in 1.30s`; that file has no exit-code line) |
| `uv sync --frozen --extra dev` | 0 | `324-uv-sync-frozen.log` (`SYNC_EXIT:0`). `323-uv-sync-frozen.log` is exit 2 from a sandbox `Operation not permitted` while removing `.venv/bin/rag`, not from the lock. `257-uv-sync-dev.log` is the earlier install that dropped chromadb; that file has no exit-code line |
| `pip-audit` | 0 | `258-pip-audit.log` (`No known vulnerabilities found`); repeated in `315-offline-gates.log` (`AUDIT_EXIT:0`) |
| Pre-fix CI pytest (`pythonpath` absent) | 2 | `259-ci-repro.log` (`4 errors during collection`) |
| `ruff check .` | 0 | `316-offline-recheck.log` (`RUFF_EXIT:0`) |
| `mypy --strict src/` | 0 | `316-offline-recheck.log` (`Success: no issues found in 26 source files`, `MYPY_EXIT:0`). Earlier strict run: `312-mypy-strict.log` |
| `pytest -q -p no:randomly` | 0 | `316` (`306 passed`, `NO_RANDOM_EXIT:0`) |
| `pytest -q` (pytest-randomly on) | 0 | `316` (`306 passed`, `RANDOM_EXIT:0`) |
| `pytest -q -n auto -p no:randomly` | 0 | `316` (`306 passed`, `XDIST_EXIT:0`) |
| `pytest --cov=rag --cov-branch --cov-fail-under=100 -q -p no:randomly` | 0 | `316` (`Required test coverage of 100% reached. Total coverage: 100.00%`, `COV_EXIT:0`) |
| `mutmut run` after deleting `mutants/` | 0 | `321-mutmut.log` (`MUTMUT_EXIT:0`). Score is under 0.90; see below |
| `mutmut results` | 0 | `322-mutmut-results.log` |
| `npm ci` in `web/` | 0 | `318-web-npm-ci.log` (167 packages added; wrapper exit 0) |
| `npm run test:coverage` | 0 | `319-web-coverage.log` (9 passed; statements 67.07, branches 75, functions 85.71, lines 67.07) |
| `npm run build` | 0 | `320-web-build.log` (`built in 839ms`) |
| `docker build -t reading-room:ci .` | 0 | `317-docker-build.log` (`DOCKER_EXIT:0`, image `reading-room:ci`) |
| `rag eval` | 0 | `245-rag-eval.log` (`suite gates: PASS`). Not re-run in this closeout |

`315-offline-gates.log` is an earlier attempt of the same offline set. It recorded `RUFF_EXIT:1` (three style findings, later fixed) and `RANDOM_EXIT:1` / `XDIST_EXIT:1` (three tests that were edited again before log 316). Log 316 is the measurement that stands.

## Lockfile

`uv.lock` is the hash lock. Docker and CI install with `uv sync --frozen` (uv 0.12.5). `uv sync --frozen` checks the lock hashes. A hashed `requirements.txt` exported on darwin arm64 would not install on ubuntu CI or the linux image, so that file was not added.

`pip-audit` on the dev install found no known vulnerabilities (`258`, `315`). No ignore was added. Optional extras `ocr` and `layout` are in the lock and are not installed by `uv sync --frozen --extra dev` or by the image (`uv sync --frozen --no-dev`).

## mypy

`[tool.mypy]` has `strict = true`. `mypy --strict src/` exit 0, 26 files (`316`).

`ignore_missing_imports` is set only for these modules and their submodules: `onnxruntime`, `rapidocr_onnxruntime`, `yaml`, `lxml`, `docling`, `sentence_transformers`, `torch`. No `# type: ignore` was added.

`# pragma: no cover` appears once, on `if __name__ == "__main__":` in `src/rag/__main__.py`. No `TYPE_CHECKING` pragma. No skip or xfail was added.

## Branch coverage

Log 316, 306 passed:

`TOTAL 4233 statements, 0 miss, 1616 branches, 0 partial. Total coverage: 100.00%.`

`web/vite.config.js` thresholds stay at 67 / 75 / 85 / 67. Log 319 measured `stream.js` at 67.07 / 75 / 85.71 / 67.07, which meets those floors. `stream.js` was not raised to 100%.

## Mutation score

Scope is `src/rag/chunk.py`, `src/rag/retrieve.py`, `src/rag/gates.py`.

Log 313 reused cached verdicts (`cached results were kept` after non-Python files changed). That log is not the score.

Log 321 is a run started after `mutants/` was deleted:

`2914/2914  killed 1819, timeout 19, survived 1076` (the progress line’s 🙁 count). `MUTMUT_EXIT:0`. `33.54 mutations/second`.

`mutmut results` (`322`) lists 1076 `survived` and 19 `timeout`.

Score used the same ratio as the previous 0.563 figure (killed / (killed + survived)):

`1819 / (1819 + 1076) = 1819 / 2895 = 0.628325`

The 0.90 floor is not met. The gate stays red. CI runs `mutmut run` and will fail on this score. The floor was not lowered.

Every survivor from `322-mutmut-results.log`:

- `rag.gates.x__answer_quantities__mutmut_13: survived`
- `rag.gates.x__asked_quantities__mutmut_4: survived`
- `rag.gates.x_attach_citations__mutmut_1: survived`
- `rag.gates.x_attach_citations__mutmut_5: survived`
- `rag.gates.x_attach_citations__mutmut_25: survived`
- `rag.gates.x_attach_citations__mutmut_29: survived`
- `rag.gates.x_attach_citations__mutmut_31: survived`
- `rag.gates.x_attach_citations__mutmut_42: survived`
- `rag.gates.x_attach_citations__mutmut_45: survived`
- `rag.gates.x_attach_citations__mutmut_48: survived`
- `rag.gates.x_attach_citations__mutmut_49: survived`
- `rag.gates.x_attach_citations__mutmut_55: survived`
- `rag.gates.x_attach_citations__mutmut_58: survived`
- `rag.gates.x_attach_citations__mutmut_61: survived`
- `rag.gates.x_attach_citations__mutmut_62: survived`
- `rag.gates.x_attach_citations__mutmut_63: survived`
- `rag.gates.x_attach_citations__mutmut_64: survived`
- `rag.gates.x_attach_citations__mutmut_65: survived`
- `rag.gates.x_attach_citations__mutmut_66: survived`
- `rag.gates.x_attach_citations__mutmut_68: survived`
- `rag.gates.x_attach_citations__mutmut_71: survived`
- `rag.gates.x_attach_citations__mutmut_72: survived`
- `rag.gates.x_attach_citations__mutmut_73: survived`
- `rag.gates.x_attach_citations__mutmut_74: survived`
- `rag.gates.x_attach_citations__mutmut_75: survived`
- `rag.gates.x_attach_citations__mutmut_77: survived`
- `rag.gates.x_attach_citations__mutmut_80: survived`
- `rag.gates.x_attach_citations__mutmut_81: survived`
- `rag.gates.x_attach_citations__mutmut_82: survived`
- `rag.gates.x_attach_citations__mutmut_83: survived`
- `rag.gates.x_attach_citations__mutmut_84: survived`
- `rag.gates.x_attach_citations__mutmut_85: survived`
- `rag.gates.x_attach_citations__mutmut_86: survived`
- `rag.gates.x_attach_citations__mutmut_97: survived`
- `rag.gates.x_attach_citations__mutmut_99: survived`
- `rag.gates.x_attach_citations__mutmut_102: survived`
- `rag.gates.x_attach_citations__mutmut_104: survived`
- `rag.gates.x_attach_citations__mutmut_106: survived`
- `rag.gates.x_attach_citations__mutmut_108: survived`
- `rag.gates.x_attach_citations__mutmut_112: survived`
- `rag.gates.x_attach_citations__mutmut_117: survived`
- `rag.gates.x_attach_citations__mutmut_119: survived`
- `rag.gates.x_attach_citations__mutmut_130: survived`
- `rag.gates.x_attach_citations__mutmut_132: survived`
- `rag.gates.x_attach_citations__mutmut_135: survived`
- `rag.gates.x_attach_citations__mutmut_145: survived`
- `rag.gates.x_attach_citations__mutmut_146: survived`
- `rag.gates.x_attach_citations__mutmut_147: survived`
- `rag.gates.x_attach_citations__mutmut_148: survived`
- `rag.gates.x_attach_citations__mutmut_149: survived`
- `rag.gates.x_attach_citations__mutmut_150: survived`
- `rag.gates.x_attach_citations__mutmut_151: survived`
- `rag.gates.x_attach_citations__mutmut_153: survived`
- `rag.gates.x_attach_citations__mutmut_154: survived`
- `rag.gates.x_attach_citations__mutmut_155: survived`
- `rag.gates.x_attach_citations__mutmut_156: survived`
- `rag.gates.x_attach_citations__mutmut_165: survived`
- `rag.gates.x_attach_citations__mutmut_166: survived`
- `rag.gates.x_attach_citations__mutmut_168: survived`
- `rag.gates.x_attach_citations__mutmut_182: survived`
- `rag.gates.x_attach_citations__mutmut_183: survived`
- `rag.gates.x_attach_citations__mutmut_186: survived`
- `rag.gates.x_attach_citations__mutmut_188: survived`
- `rag.gates.x_attach_citations__mutmut_189: survived`
- `rag.gates.x_attach_citations__mutmut_190: survived`
- `rag.gates.x_attach_citations__mutmut_200: survived`
- `rag.gates.x_attach_citations__mutmut_202: survived`
- `rag.gates.x_attach_citations__mutmut_203: survived`
- `rag.gates.x_attach_citations__mutmut_206: survived`
- `rag.gates.x_attach_citations__mutmut_212: survived`
- `rag.gates.x_attach_citations__mutmut_213: survived`
- `rag.gates.x__content_tokens__mutmut_8: survived`
- `rag.gates.x__phrase_in__mutmut_10: survived`
- `rag.gates.x__phrase_in__mutmut_13: survived`
- `rag.gates.x__append_cites__mutmut_3: survived`
- `rag.gates.x__append_cites__mutmut_7: survived`
- `rag.gates.x_check_form__mutmut_3: survived`
- `rag.gates.x_check_form__mutmut_5: survived`
- `rag.gates.x_check_form__mutmut_6: survived`
- `rag.gates.x_check_form__mutmut_7: survived`
- `rag.gates.x_check_form__mutmut_9: survived`
- `rag.gates.x_check_form__mutmut_10: survived`
- `rag.gates.x_check_form__mutmut_12: survived`
- `rag.gates.x_check_form__mutmut_13: survived`
- `rag.gates.x_check_form__mutmut_14: survived`
- `rag.gates.x_check_form__mutmut_15: survived`
- `rag.gates.x_check_form__mutmut_19: survived`
- `rag.gates.x_check_form__mutmut_20: survived`
- `rag.gates.x_check_form__mutmut_21: survived`
- `rag.gates.x_check_form__mutmut_23: survived`
- `rag.gates.x_check_form__mutmut_24: survived`
- `rag.gates.x_check_form__mutmut_26: survived`
- `rag.gates.x_check_form__mutmut_27: survived`
- `rag.gates.x_check_form__mutmut_28: survived`
- `rag.gates.x_check_form__mutmut_29: survived`
- `rag.gates.x_check_form__mutmut_50: survived`
- `rag.gates.x_check_form__mutmut_52: survived`
- `rag.gates.x_check_form__mutmut_55: survived`
- `rag.gates.x_check_form__mutmut_59: survived`
- `rag.gates.x_check_form__mutmut_60: survived`
- `rag.gates.x_check_form__mutmut_66: survived`
- `rag.gates.x_check_form__mutmut_67: survived`
- `rag.gates.x_check_form__mutmut_68: survived`
- `rag.gates.x_check_form__mutmut_70: survived`
- `rag.gates.x_check_form__mutmut_71: survived`
- `rag.gates.x_check_form__mutmut_73: survived`
- `rag.gates.x_check_form__mutmut_74: survived`
- `rag.gates.x_check_form__mutmut_75: survived`
- `rag.gates.x_check_form__mutmut_76: survived`
- `rag.gates.x_check_form__mutmut_78: survived`
- `rag.gates.x_check_form__mutmut_80: survived`
- `rag.gates.x_check_form__mutmut_82: survived`
- `rag.gates.x__norm_quantity__mutmut_6: survived`
- `rag.gates.x__norm_quantity__mutmut_7: survived`
- `rag.gates.x__norm_quantity__mutmut_17: survived`
- `rag.gates.x__norm_quantity__mutmut_18: survived`
- `rag.gates.x__norm_quantity__mutmut_19: survived`
- `rag.gates.x__norm_quantity__mutmut_20: survived`
- `rag.gates.x__norm_quantity__mutmut_21: survived`
- `rag.gates.x__norm_quantity__mutmut_22: survived`
- `rag.gates.x__norm_quantity__mutmut_31: survived`
- `rag.gates.x__norm_quantity__mutmut_49: survived`
- `rag.gates.x__passage_blob__mutmut_8: survived`
- `rag.gates.x__passage_blob__mutmut_11: survived`
- `rag.gates.x__passage_blob__mutmut_14: survived`
- `rag.gates.x__passage_blob__mutmut_17: survived`
- `rag.gates.x__passage_blob__mutmut_18: survived`
- `rag.gates.x__passage_blob__mutmut_22: survived`
- `rag.gates.x_check_grounding__mutmut_1: survived`
- `rag.gates.x_check_grounding__mutmut_8: survived`
- `rag.gates.x_check_grounding__mutmut_19: survived`
- `rag.gates.x_check_grounding__mutmut_21: survived`
- `rag.gates.x_check_grounding__mutmut_27: survived`
- `rag.gates.x_check_grounding__mutmut_29: survived`
- `rag.gates.x_check_grounding__mutmut_32: survived`
- `rag.gates.x_check_grounding__mutmut_34: survived`
- `rag.gates.x_check_grounding__mutmut_38: survived`
- `rag.gates.x_check_grounding__mutmut_39: survived`
- `rag.gates.x_check_grounding__mutmut_40: survived`
- `rag.gates.x_check_grounding__mutmut_42: survived`
- `rag.gates.x_check_grounding__mutmut_43: survived`
- `rag.gates.x_check_grounding__mutmut_44: survived`
- `rag.gates.x_check_grounding__mutmut_46: survived`
- `rag.gates.x_check_grounding__mutmut_47: survived`
- `rag.gates.x_check_grounding__mutmut_48: survived`
- `rag.gates.x_check_grounding__mutmut_50: survived`
- `rag.gates.x_check_grounding__mutmut_51: survived`
- `rag.gates.x_check_grounding__mutmut_52: survived`
- `rag.gates.x_check_grounding__mutmut_53: survived`
- `rag.gates.x_check_conflict__mutmut_1: survived`
- `rag.gates.x_check_conflict__mutmut_3: survived`
- `rag.gates.x_check_conflict__mutmut_10: survived`
- `rag.gates.x_check_conflict__mutmut_11: survived`
- `rag.gates.x_check_conflict__mutmut_12: survived`
- `rag.gates.x_check_conflict__mutmut_13: survived`
- `rag.gates.x_check_conflict__mutmut_14: survived`
- `rag.gates.x_check_conflict__mutmut_15: survived`
- `rag.gates.x_check_conflict__mutmut_16: survived`
- `rag.gates.x_check_conflict__mutmut_17: survived`
- `rag.gates.x_check_conflict__mutmut_18: survived`
- `rag.gates.x_check_conflict__mutmut_19: survived`
- `rag.gates.x_check_conflict__mutmut_20: survived`
- `rag.gates.x_check_conflict__mutmut_21: survived`
- `rag.gates.x_check_conflict__mutmut_22: survived`
- `rag.gates.x_check_conflict__mutmut_25: survived`
- `rag.gates.x_check_conflict__mutmut_26: survived`
- `rag.gates.x_check_conflict__mutmut_27: survived`
- `rag.gates.x_check_conflict__mutmut_30: survived`
- `rag.gates.x_check_conflict__mutmut_31: survived`
- `rag.gates.x_check_conflict__mutmut_32: survived`
- `rag.gates.x_check_conflict__mutmut_33: survived`
- `rag.gates.x_check_conflict__mutmut_34: survived`
- `rag.gates.x_check_conflict__mutmut_35: survived`
- `rag.gates.x_check_conflict__mutmut_36: survived`
- `rag.gates.x_check_conflict__mutmut_37: survived`
- `rag.gates.x_check_conflict__mutmut_38: survived`
- `rag.gates.x_check_conflict__mutmut_39: survived`
- `rag.gates.x_check_conflict__mutmut_40: survived`
- `rag.gates.x_check_conflict__mutmut_41: survived`
- `rag.gates.x_check_conflict__mutmut_42: survived`
- `rag.gates.x_check_conflict__mutmut_43: survived`
- `rag.gates.x_check_conflict__mutmut_44: survived`
- `rag.gates.x_check_conflict__mutmut_45: survived`
- `rag.gates.x_check_conflict__mutmut_46: survived`
- `rag.gates.x_check_all__mutmut_1: survived`
- `rag.gates.x_check_all__mutmut_9: survived`
- `rag.gates.x_check_all__mutmut_10: survived`
- `rag.gates.x_check_all__mutmut_11: survived`
- `rag.gates.x_check_all__mutmut_15: survived`
- `rag.gates.x_check_all__mutmut_18: survived`
- `rag.gates.x_check_all__mutmut_19: survived`
- `rag.gates.x_check_all__mutmut_20: survived`
- `rag.gates.x_check_all__mutmut_21: survived`
- `rag.gates.x_check_all__mutmut_22: survived`
- `rag.gates.x_check_all__mutmut_23: survived`
- `rag.gates.x_check_all__mutmut_24: survived`
- `rag.gates.x_check_all__mutmut_25: survived`
- `rag.gates.x_check_all__mutmut_27: survived`
- `rag.gates.x_check_all__mutmut_28: survived`
- `rag.gates.x_check_all__mutmut_29: survived`
- `rag.gates.x_check_all__mutmut_30: survived`
- `rag.gates.x_check_all__mutmut_31: survived`
- `rag.gates.x_check_all__mutmut_33: survived`
- `rag.gates.x_check_all__mutmut_34: survived`
- `rag.gates.x_check_all__mutmut_35: survived`
- `rag.gates.x_check_all__mutmut_36: survived`
- `rag.gates.x_check_all__mutmut_37: survived`
- `rag.gates.x_check_all__mutmut_38: survived`
- `rag.gates.x_check_all__mutmut_39: survived`
- `rag.gates.x_check_all__mutmut_40: survived`
- `rag.gates.x_check_all__mutmut_41: survived`
- `rag.gates.x_check_all__mutmut_42: survived`
- `rag.gates.x_check_all__mutmut_50: survived`
- `rag.gates.x_check_all__mutmut_51: survived`
- `rag.gates.x_check_all__mutmut_52: survived`
- `rag.gates.x_check_all__mutmut_54: survived`
- `rag.gates.x_check_all__mutmut_55: survived`
- `rag.gates.x_check_all__mutmut_56: survived`
- `rag.gates.x_check_all__mutmut_57: survived`
- `rag.gates.x_check_all__mutmut_58: survived`
- `rag.gates.x_check_all__mutmut_59: survived`
- `rag.gates.x_check_all__mutmut_60: survived`
- `rag.gates.x_check_all__mutmut_70: survived`
- `rag.gates.x_check_all__mutmut_71: survived`
- `rag.gates.x_check_all__mutmut_72: survived`
- `rag.gates.x_check_all__mutmut_74: survived`
- `rag.gates.x_check_all__mutmut_75: survived`
- `rag.gates.x_check_all__mutmut_76: survived`
- `rag.gates.x_check_all__mutmut_77: survived`
- `rag.gates.x_check_all__mutmut_78: survived`
- `rag.gates.x_check_all__mutmut_79: survived`
- `rag.gates.x_check_all__mutmut_80: survived`
- `rag.gates.x_check_all__mutmut_81: survived`
- `rag.gates.x_check_all__mutmut_82: survived`
- `rag.gates.x_check_all__mutmut_83: survived`
- `rag.gates.x_check_all__mutmut_84: survived`
- `rag.gates.x_check_all__mutmut_85: survived`
- `rag.gates.x_check_all__mutmut_86: survived`
- `rag.gates.x_check_all__mutmut_88: survived`
- `rag.gates.x_check_all__mutmut_89: survived`
- `rag.gates.x_check_all__mutmut_90: survived`
- `rag.gates.x_check_all__mutmut_91: survived`
- `rag.gates.x_check_all__mutmut_94: survived`
- `rag.gates.x_check_all__mutmut_95: survived`
- `rag.gates.x_check_all__mutmut_96: survived`
- `rag.gates.x_check_all__mutmut_97: survived`
- `rag.gates.x_check_all__mutmut_98: survived`
- `rag.gates.x_check_all__mutmut_101: survived`
- `rag.gates.x_check_all__mutmut_102: survived`
- `rag.gates.x_check_all__mutmut_103: survived`
- `rag.gates.x_check_all__mutmut_104: survived`
- `rag.gates.x_check_all__mutmut_105: survived`
- `rag.gates.x_check_all__mutmut_106: survived`
- `rag.gates.x_check_all__mutmut_108: survived`
- `rag.gates.x_check_all__mutmut_109: survived`
- `rag.gates.x_check_all__mutmut_110: survived`
- `rag.gates.x_check_all__mutmut_111: survived`
- `rag.gates.x_check_all__mutmut_112: survived`
- `rag.gates.x_check_all__mutmut_115: survived`
- `rag.gates.x_check_all__mutmut_116: survived`
- `rag.gates.x_check_all__mutmut_117: survived`
- `rag.gates.x_check_all__mutmut_119: survived`
- `rag.gates.x_check_all__mutmut_120: survived`
- `rag.gates.x_check_all__mutmut_123: survived`
- `rag.gates.x_check_all__mutmut_124: survived`
- `rag.gates.x_check_all__mutmut_126: survived`
- `rag.gates.x_check_all__mutmut_127: survived`
- `rag.gates.x_check_all__mutmut_128: survived`
- `rag.gates.x_check_all__mutmut_129: survived`
- `rag.gates.x_check_all__mutmut_133: survived`
- `rag.gates.x_check_all__mutmut_134: survived`
- `rag.gates.x_check_all__mutmut_135: survived`
- `rag.gates.x_check_all__mutmut_136: survived`
- `rag.gates.x_check_all__mutmut_137: survived`
- `rag.gates.x_check_all__mutmut_138: survived`
- `rag.gates.x_check_all__mutmut_140: survived`
- `rag.gates.x_check_all__mutmut_146: survived`
- `rag.gates.x_check_all__mutmut_147: survived`
- `rag.gates.x_check_all__mutmut_148: survived`
- `rag.gates.x_check_all__mutmut_149: survived`
- `rag.gates.x_check_all__mutmut_150: survived`
- `rag.gates.x_check_all__mutmut_152: survived`
- `rag.gates.x_check_all__mutmut_153: survived`
- `rag.gates.x_check_all__mutmut_154: survived`
- `rag.gates.x_check_all__mutmut_155: survived`
- `rag.gates.x_check_all__mutmut_156: survived`
- `rag.gates.x_check_all__mutmut_157: survived`
- `rag.gates.x_check_all__mutmut_160: survived`
- `rag.gates.x_check_all__mutmut_161: survived`
- `rag.gates.x_check_all__mutmut_162: survived`
- `rag.gates.x_check_all__mutmut_163: survived`
- `rag.gates.x_check_all__mutmut_164: survived`
- `rag.gates.x_check_all__mutmut_165: survived`
- `rag.gates.x_check_all__mutmut_166: survived`
- `rag.gates.x_check_all__mutmut_170: survived`
- `rag.gates.x_check_all__mutmut_171: survived`
- `rag.gates.x_check_all__mutmut_172: survived`
- `rag.gates.x_check_all__mutmut_173: survived`
- `rag.gates.x_check_all__mutmut_175: survived`
- `rag.gates.x_check_all__mutmut_178: survived`
- `rag.gates.x_check_all__mutmut_179: survived`
- `rag.gates.x_check_all__mutmut_180: survived`
- `rag.gates.x_check_all__mutmut_181: survived`
- `rag.gates.x_check_all__mutmut_182: survived`
- `rag.gates.x_check_all__mutmut_183: survived`
- `rag.gates.x_check_all__mutmut_184: survived`
- `rag.gates.x_check_all__mutmut_185: survived`
- `rag.gates.x_check_all__mutmut_186: survived`
- `rag.gates.x_check_all__mutmut_187: survived`
- `rag.gates.x_check_all__mutmut_188: survived`
- `rag.gates.x_check_all__mutmut_189: survived`
- `rag.gates.x_check_all__mutmut_190: survived`
- `rag.gates.x_check_all__mutmut_191: survived`
- `rag.gates.x_check_all__mutmut_192: survived`
- `rag.gates.x_check_all__mutmut_193: survived`
- `rag.gates.x_check_all__mutmut_194: survived`
- `rag.gates.x_check_all__mutmut_195: survived`
- `rag.gates.x_check_all__mutmut_196: survived`
- `rag.chunk.x_chunk_document__mutmut_40: survived`
- `rag.chunk.x__sections__mutmut_7: survived`
- `rag.chunk.x__sections__mutmut_8: survived`
- `rag.chunk.x__sections__mutmut_9: survived`
- `rag.chunk.x__sections__mutmut_12: survived`
- `rag.chunk.x__sections__mutmut_13: survived`
- `rag.chunk.x__sections__mutmut_24: survived`
- `rag.chunk.x__sections__mutmut_25: survived`
- `rag.chunk.x__sections__mutmut_26: survived`
- `rag.chunk.x__sections__mutmut_27: survived`
- `rag.chunk.x__sections__mutmut_35: survived`
- `rag.chunk.x__sections__mutmut_36: survived`
- `rag.chunk.x__sections__mutmut_37: survived`
- `rag.chunk.x__sections__mutmut_38: survived`
- `rag.chunk.x__sections__mutmut_39: survived`
- `rag.chunk.x__sections__mutmut_40: survived`
- `rag.chunk.x__sections__mutmut_44: survived`
- `rag.chunk.x__sections__mutmut_45: survived`
- `rag.chunk.x__sections__mutmut_46: survived`
- `rag.chunk.x__sections__mutmut_47: survived`
- `rag.chunk.x__sections__mutmut_48: survived`
- `rag.chunk.x__sections__mutmut_49: survived`
- `rag.chunk.x__sections__mutmut_50: survived`
- `rag.chunk.x__sections__mutmut_51: survived`
- `rag.chunk.x__sections__mutmut_52: survived`
- `rag.chunk.x__sections__mutmut_53: survived`
- `rag.chunk.x__sections__mutmut_54: survived`
- `rag.chunk.x__sections__mutmut_55: survived`
- `rag.chunk.x__sections__mutmut_56: survived`
- `rag.chunk.x__pack_section__mutmut_5: survived`
- `rag.chunk.x__pack_section__mutmut_8: survived`
- `rag.chunk.x__pack_section__mutmut_9: survived`
- `rag.chunk.x__pack_section__mutmut_11: survived`
- `rag.chunk.x__pack_section__mutmut_12: survived`
- `rag.chunk.x__pack_section__mutmut_13: survived`
- `rag.chunk.x__pack_section__mutmut_14: survived`
- `rag.chunk.x__pack_section__mutmut_16: survived`
- `rag.chunk.x__pack_section__mutmut_17: survived`
- `rag.chunk.x__pack_section__mutmut_18: survived`
- `rag.chunk.x__pack_section__mutmut_19: survived`
- `rag.chunk.x__pack_section__mutmut_20: survived`
- `rag.chunk.x__pack_section__mutmut_21: survived`
- `rag.chunk.x__pack_section__mutmut_28: survived`
- `rag.chunk.x__pack_section__mutmut_29: survived`
- `rag.chunk.x__pack_section__mutmut_31: survived`
- `rag.chunk.x__pack_section__mutmut_40: survived`
- `rag.chunk.x__pack_section__mutmut_49: survived`
- `rag.chunk.x__pack_section__mutmut_50: survived`
- `rag.chunk.x__pack_section__mutmut_55: survived`
- `rag.chunk.x__pack_section__mutmut_57: survived`
- `rag.chunk.x__pack_section__mutmut_58: survived`
- `rag.chunk.x__pack_section__mutmut_67: survived`
- `rag.chunk.x__pack_section__mutmut_68: survived`
- `rag.chunk.x__pack_section__mutmut_72: survived`
- `rag.chunk.x__pack_section__mutmut_73: survived`
- `rag.chunk.x__pack_section__mutmut_74: survived`
- `rag.chunk.x__pack_section__mutmut_78: survived`
- `rag.chunk.x__pack_section__mutmut_88: survived`
- `rag.chunk.x__pack_section__mutmut_92: survived`
- `rag.chunk.x__pack_section__mutmut_99: survived`
- `rag.chunk.x__pack_section__mutmut_100: survived`
- `rag.chunk.x__pack_section__mutmut_102: survived`
- `rag.chunk.x__pack_section__mutmut_111: survived`
- `rag.chunk.x__pack_section__mutmut_114: survived`
- `rag.chunk.x__pack_section__mutmut_144: survived`
- `rag.chunk.x__pack_section__mutmut_145: survived`
- `rag.chunk.x__pack_section__mutmut_146: survived`
- `rag.chunk.x__pack_section__mutmut_150: survived`
- `rag.chunk.x__pack_section__mutmut_160: survived`
- `rag.chunk.x__parents_from_windows__mutmut_1: survived`
- `rag.chunk.x__parents_from_windows__mutmut_9: survived`
- `rag.chunk.x__parents_from_windows__mutmut_14: survived`
- `rag.chunk.x__parents_from_windows__mutmut_15: survived`
- `rag.chunk.x__parents_from_windows__mutmut_16: survived`
- `rag.chunk.x__parents_from_windows__mutmut_17: survived`
- `rag.chunk.x__parents_from_windows__mutmut_19: survived`
- `rag.chunk.x__parents_from_windows__mutmut_28: survived`
- `rag.chunk.x__parents_from_windows__mutmut_29: survived`
- `rag.chunk.x__parents_from_windows__mutmut_30: survived`
- `rag.chunk.x__parents_from_windows__mutmut_31: survived`
- `rag.chunk.x__parents_from_windows__mutmut_34: survived`
- `rag.chunk.x__children_for__mutmut_15: survived`
- `rag.chunk.x__children_for__mutmut_16: survived`
- `rag.chunk.x__children_for__mutmut_28: survived`
- `rag.chunk.x__children_for__mutmut_30: survived`
- `rag.chunk.x__children_for__mutmut_31: survived`
- `rag.chunk.x__children_for__mutmut_34: survived`
- `rag.chunk.x__children_for__mutmut_35: survived`
- `rag.chunk.x__children_for__mutmut_45: survived`
- `rag.chunk.x__children_for__mutmut_46: survived`
- `rag.chunk.x__children_for__mutmut_51: survived`
- `rag.chunk.x__children_for__mutmut_54: survived`
- `rag.chunk.x__children_for__mutmut_63: survived`
- `rag.chunk.x__children_for__mutmut_64: survived`
- `rag.chunk.x__children_for__mutmut_65: survived`
- `rag.chunk.x__children_for__mutmut_66: survived`
- `rag.chunk.x__children_for__mutmut_67: survived`
- `rag.chunk.x__children_for__mutmut_68: survived`
- `rag.chunk.x__children_for__mutmut_70: survived`
- `rag.chunk.x__children_for__mutmut_72: survived`
- `rag.chunk.x__children_for__mutmut_82: survived`
- `rag.chunk.x__children_for__mutmut_91: survived`
- `rag.chunk.x__children_for__mutmut_96: survived`
- `rag.chunk.x__children_for__mutmut_100: survived`
- `rag.chunk.x__children_for__mutmut_101: survived`
- `rag.chunk.x__table_summary_child__mutmut_6: survived`
- `rag.chunk.x__table_summary_child__mutmut_7: survived`
- `rag.chunk.x__table_summary_child__mutmut_8: survived`
- `rag.chunk.x__table_summary_child__mutmut_14: survived`
- `rag.chunk.x__table_summary_child__mutmut_15: survived`
- `rag.chunk.x__table_summary_child__mutmut_17: survived`
- `rag.chunk.x__table_summary_child__mutmut_20: survived`
- `rag.chunk.x__table_summary_child__mutmut_21: survived`
- `rag.chunk.x__table_summary_child__mutmut_23: survived`
- `rag.chunk.x__table_summary_child__mutmut_26: survived`
- `rag.chunk.x__table_summary_child__mutmut_38: survived`
- `rag.chunk.x__looks_like_unit__mutmut_3: survived`
- `rag.chunk.x__looks_like_unit__mutmut_4: survived`
- `rag.chunk.x__looks_like_unit__mutmut_5: survived`
- `rag.chunk.x__looks_like_unit__mutmut_6: survived`
- `rag.chunk.x__looks_like_unit__mutmut_7: survived`
- `rag.chunk.x__looks_like_unit__mutmut_8: survived`
- `rag.chunk.x__looks_like_unit__mutmut_9: survived`
- `rag.chunk.x__looks_like_unit__mutmut_10: survived`
- `rag.chunk.x__looks_like_unit__mutmut_11: survived`
- `rag.chunk.x__looks_like_unit__mutmut_12: survived`
- `rag.chunk.x__label_row__mutmut_14: survived`
- `rag.chunk.x__label_row__mutmut_15: survived`
- `rag.chunk.x__label_row__mutmut_17: survived`
- `rag.chunk.x__child__mutmut_17: survived`
- `rag.chunk.x__child__mutmut_18: survived`
- `rag.chunk.x__child__mutmut_20: survived`
- `rag.chunk.x__child__mutmut_35: survived`
- `rag.chunk.x__child__mutmut_39: survived`
- `rag.chunk.x__child__mutmut_40: survived`
- `rag.chunk.x__child__mutmut_42: survived`
- `rag.chunk.x__child__mutmut_43: survived`
- `rag.chunk.x__table_pieces__mutmut_13: survived`
- `rag.chunk.x__table_pieces__mutmut_24: survived`
- `rag.chunk.x__table_pieces__mutmut_36: survived`
- `rag.chunk.x__table_pieces__mutmut_49: survived`
- `rag.chunk.x__table_pieces__mutmut_50: survived`
- `rag.chunk.x__table_pieces__mutmut_52: survived`
- `rag.chunk.x__table_pieces__mutmut_55: survived`
- `rag.chunk.x__table_pieces__mutmut_59: survived`
- `rag.chunk.x__merge_tiny__mutmut_16: survived`
- `rag.chunk.x__merge_tiny__mutmut_17: survived`
- `rag.chunk.x__merge_tiny__mutmut_18: survived`
- `rag.chunk.x__merge_tiny__mutmut_19: survived`
- `rag.chunk.x__merge_tiny__mutmut_22: survived`
- `rag.chunk.x__merge_tiny__mutmut_23: survived`
- `rag.chunk.x__merge_tiny__mutmut_25: survived`
- `rag.chunk.x__merge_tiny__mutmut_33: survived`
- `rag.chunk.x__merge_tiny__mutmut_40: survived`
- `rag.chunk.x__merge_tiny__mutmut_41: survived`
- `rag.chunk.x__merge_tiny__mutmut_42: survived`
- `rag.chunk.x__merge_tiny__mutmut_44: survived`
- `rag.chunk.x__merge_tiny__mutmut_45: survived`
- `rag.chunk.x__merge_tiny__mutmut_46: survived`
- `rag.chunk.x__merge_tiny__mutmut_47: survived`
- `rag.chunk.x__merge_tiny__mutmut_48: survived`
- `rag.chunk.x__merge_tiny__mutmut_49: survived`
- `rag.chunk.x__merge_tiny__mutmut_50: survived`
- `rag.chunk.x__merge_tiny__mutmut_51: survived`
- `rag.chunk.x__merge_tiny__mutmut_52: survived`
- `rag.chunk.x__merge_tiny__mutmut_53: survived`
- `rag.chunk.x__merge_tiny__mutmut_54: survived`
- `rag.chunk.x__merge_tiny__mutmut_55: survived`
- `rag.chunk.x__merge_tiny__mutmut_70: survived`
- `rag.chunk.x__merge_tiny__mutmut_71: survived`
- `rag.chunk.x__merge_tiny__mutmut_75: survived`
- `rag.chunk.x__merge_tiny__mutmut_76: survived`
- `rag.chunk.x__merge_tiny__mutmut_77: survived`
- `rag.chunk.x__merge_tiny__mutmut_78: survived`
- `rag.chunk.x__merge_tiny__mutmut_80: survived`
- `rag.chunk.x__apply_context__mutmut_5: survived`
- `rag.chunk.x__apply_context__mutmut_8: survived`
- `rag.chunk.x__apply_context__mutmut_9: survived`
- `rag.chunk.x__apply_context__mutmut_12: survived`
- `rag.chunk.x__apply_context__mutmut_13: survived`
- `rag.chunk.x__apply_context__mutmut_16: survived`
- `rag.chunk.x__apply_context__mutmut_21: survived`
- `rag.chunk.x__apply_context__mutmut_25: survived`
- `rag.chunk.x__apply_context__mutmut_26: survived`
- `rag.chunk.x__apply_context__mutmut_28: survived`
- `rag.chunk.x__make_parent__mutmut_1: survived`
- `rag.chunk.x__make_parent__mutmut_12: survived`
- `rag.chunk.x__make_parent__mutmut_14: survived`
- `rag.chunk.x__make_parent__mutmut_26: survived`
- `rag.chunk.x__make_parent__mutmut_31: survived`
- `rag.chunk.x__make_parent__mutmut_32: survived`
- `rag.chunk.x__make_parent__mutmut_34: survived`
- `rag.chunk.x__make_parent__mutmut_35: survived`
- `rag.chunk.x__make_parent__mutmut_36: survived`
- `rag.chunk.x__make_parent__mutmut_38: survived`
- `rag.chunk.x__make_parent__mutmut_39: survived`
- `rag.chunk.x__make_parent__mutmut_41: survived`
- `rag.chunk.x__make_parent__mutmut_42: survived`
- `rag.chunk.x__make_parent__mutmut_43: survived`
- `rag.chunk.x__make_parent__mutmut_45: survived`
- `rag.chunk.x__make_parent__mutmut_47: survived`
- `rag.chunk.x__make_parent__mutmut_48: survived`
- `rag.chunk.x__make_parent__mutmut_49: survived`
- `rag.chunk.x__make_parent__mutmut_50: survived`
- `rag.chunk.x__section_kind__mutmut_3: survived`
- `rag.chunk.x__section_kind__mutmut_4: survived`
- `rag.chunk.x__section_kind__mutmut_8: survived`
- `rag.chunk.x__section_kind__mutmut_9: survived`
- `rag.chunk.x__section_kind__mutmut_13: survived`
- `rag.chunk.x__section_kind__mutmut_14: survived`
- `rag.chunk.x__piece_spans__mutmut_4: survived`
- `rag.chunk.x__piece_spans__mutmut_9: survived`
- `rag.chunk.x__piece_spans__mutmut_18: survived`
- `rag.chunk.x__split_long__mutmut_5: survived`
- `rag.chunk.x__split_long__mutmut_9: survived`
- `rag.chunk.x__split_long__mutmut_10: survived`
- `rag.chunk.x__paragraph_spans__mutmut_4: survived`
- `rag.chunk.x__paragraph_spans__mutmut_10: survived`
- `rag.chunk.x__paragraph_spans__mutmut_11: survived`
- `rag.chunk.x__paragraph_spans__mutmut_12: survived`
- `rag.chunk.x__paragraph_spans__mutmut_13: survived`
- `rag.chunk.x__paragraph_spans__mutmut_19: survived`
- `rag.chunk.x__sentence_spans__mutmut_7: survived`
- `rag.chunk.x__sentence_spans__mutmut_9: survived`
- `rag.chunk.x__sentence_spans__mutmut_12: survived`
- `rag.chunk.x__sentence_spans__mutmut_15: survived`
- `rag.chunk.x__sentence_spans__mutmut_16: survived`
- `rag.chunk.x__word_spans__mutmut_15: survived`
- `rag.chunk.x__word_spans__mutmut_17: survived`
- `rag.chunk.x__word_spans__mutmut_18: survived`
- `rag.chunk.x__word_spans__mutmut_19: survived`
- `rag.chunk.x__word_spans__mutmut_23: survived`
- `rag.chunk.x__word_spans__mutmut_34: survived`
- `rag.chunk.x__word_spans__mutmut_35: survived`
- `rag.chunk.x__line_spans__mutmut_12: survived`
- `rag.chunk.x__line_spans__mutmut_13: survived`
- `rag.chunk.x__line_spans__mutmut_14: survived`
- `rag.chunk.x__line_spans__mutmut_17: survived`
- `rag.chunk.x__line_spans__mutmut_23: survived`
- `rag.chunk.x__line_spans__mutmut_24: survived`
- `rag.chunk.x__line_spans__mutmut_25: survived`
- `rag.chunk.x__windows__mutmut_3: survived`
- `rag.chunk.x__windows__mutmut_6: survived`
- `rag.chunk.x__windows__mutmut_10: survived`
- `rag.chunk.x__windows__mutmut_19: survived`
- `rag.chunk.x__windows__mutmut_22: survived`
- `rag.chunk.x__windows__mutmut_30: survived`
- `rag.chunk.x__windows__mutmut_31: survived`
- `rag.chunk.x__windows__mutmut_37: survived`
- `rag.chunk.x__windows__mutmut_38: survived`
- `rag.chunk.x__windows__mutmut_39: survived`
- `rag.chunk.x__windows__mutmut_40: survived`
- `rag.chunk.x__windows__mutmut_41: survived`
- `rag.chunk.x__windows__mutmut_53: survived`
- `rag.chunk.x__retreat__mutmut_2: survived`
- `rag.chunk.x__retreat__mutmut_4: survived`
- `rag.chunk.x__retreat__mutmut_5: survived`
- `rag.chunk.x__retreat__mutmut_9: survived`
- `rag.chunk.x__retreat__mutmut_10: survived`
- `rag.chunk.x__retreat__mutmut_12: survived`
- `rag.chunk.x__retreat__mutmut_14: survived`
- `rag.chunk.x__retreat__mutmut_15: survived`
- `rag.chunk.x__retreat__mutmut_16: survived`
- `rag.chunk.x__retreat__mutmut_18: survived`
- `rag.chunk.x__retreat__mutmut_19: survived`
- `rag.chunk.x__retreat__mutmut_20: survived`
- `rag.chunk.x__retreat__mutmut_21: survived`
- `rag.chunk.x__retreat__mutmut_22: survived`
- `rag.chunk.x__retreat__mutmut_24: survived`
- `rag.chunk.x__retreat__mutmut_26: survived`
- `rag.chunk.x__retreat__mutmut_27: survived`
- `rag.chunk.x__retreat__mutmut_28: survived`
- `rag.chunk.x__dedupe__mutmut_4: survived`
- `rag.chunk.x__content_id__mutmut_4: survived`
- `rag.chunk.x__content_id__mutmut_5: survived`
- `rag.retrieve.x_fuse__mutmut_8: survived`
- `rag.retrieve.x_fuse__mutmut_13: survived`
- `rag.retrieve.x_fuse__mutmut_23: survived`
- `rag.retrieve.x_fuse__mutmut_25: survived`
- `rag.retrieve.x_agreed_parent__mutmut_1: survived`
- `rag.retrieve.x_agreed_parent__mutmut_8: survived`
- `rag.retrieve.x_agreed_parent__mutmut_16: survived`
- `rag.retrieve.x_retrieve__mutmut_7: survived`
- `rag.retrieve.x_retrieve__mutmut_16: survived`
- `rag.retrieve.x_retrieve__mutmut_26: survived`
- `rag.retrieve.x_retrieve__mutmut_27: survived`
- `rag.retrieve.x_retrieve__mutmut_29: survived`
- `rag.retrieve.x_retrieve__mutmut_30: survived`
- `rag.retrieve.x_retrieve__mutmut_31: survived`
- `rag.retrieve.x_retrieve__mutmut_41: survived`
- `rag.retrieve.x_retrieve__mutmut_42: survived`
- `rag.retrieve.x_retrieve__mutmut_43: survived`
- `rag.retrieve.x_retrieve__mutmut_44: survived`
- `rag.retrieve.x_retrieve__mutmut_45: survived`
- `rag.retrieve.x_retrieve__mutmut_46: survived`
- `rag.retrieve.x_retrieve__mutmut_49: survived`
- `rag.retrieve.x_retrieve__mutmut_50: survived`
- `rag.retrieve.x_retrieve__mutmut_51: survived`
- `rag.retrieve.x_retrieve__mutmut_54: survived`
- `rag.retrieve.x_retrieve__mutmut_55: survived`
- `rag.retrieve.x_retrieve__mutmut_56: survived`
- `rag.retrieve.x_retrieve__mutmut_57: survived`
- `rag.retrieve.x_retrieve__mutmut_58: survived`
- `rag.retrieve.x__gate__mutmut_1: survived`
- `rag.retrieve.x__gate__mutmut_7: survived`
- `rag.retrieve.x__gate__mutmut_8: survived`
- `rag.retrieve.x__gate__mutmut_9: survived`
- `rag.retrieve.x__gate__mutmut_10: survived`
- `rag.retrieve.x__gate__mutmut_11: survived`
- `rag.retrieve.x__gate__mutmut_14: survived`
- `rag.retrieve.x__gate__mutmut_22: survived`
- `rag.retrieve.x__gate__mutmut_23: survived`
- `rag.retrieve.x__gate__mutmut_51: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_6: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_9: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_14: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_18: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_20: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_23: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_24: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_25: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_26: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_36: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_46: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_49: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_52: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_55: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_57: survived`
- `rag.retrieve.x__retain_superseded_siblings__mutmut_58: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_8: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_25: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_26: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_27: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_28: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_29: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_30: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_31: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_34: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_35: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_37: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_38: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_39: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_40: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_41: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_42: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_43: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_44: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_45: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_46: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_47: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_48: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_49: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_50: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_51: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_52: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_65: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_66: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_67: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_68: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_69: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_70: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_71: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_72: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_73: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_74: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_75: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_76: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_77: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_78: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_79: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_80: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_81: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_82: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_87: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_88: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_89: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_90: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_91: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_92: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_95: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_96: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_97: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_98: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_101: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_102: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_103: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_104: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_107: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_108: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_109: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_110: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_113: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_114: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_115: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_116: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_117: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_118: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_119: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_120: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_121: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_122: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_123: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_124: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_125: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_126: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_127: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_128: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_129: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_130: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_131: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_132: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_133: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_134: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_135: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_136: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_137: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_138: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_139: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_140: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_141: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_142: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_143: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_144: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_145: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_146: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_147: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_148: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_149: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_150: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_152: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_159: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_161: survived`
- `rag.retrieve.x__append_fact_siblings__mutmut_162: survived`
- `rag.retrieve.x__retrieve__mutmut_4: survived`
- `rag.retrieve.x__retrieve__mutmut_5: survived`
- `rag.retrieve.x__retrieve__mutmut_6: survived`
- `rag.retrieve.x__retrieve__mutmut_7: survived`
- `rag.retrieve.x__retrieve__mutmut_9: survived`
- `rag.retrieve.x__retrieve__mutmut_10: survived`
- `rag.retrieve.x__retrieve__mutmut_11: survived`
- `rag.retrieve.x__retrieve__mutmut_12: survived`
- `rag.retrieve.x__retrieve__mutmut_14: survived`
- `rag.retrieve.x__retrieve__mutmut_17: survived`
- `rag.retrieve.x__retrieve__mutmut_18: survived`
- `rag.retrieve.x__retrieve__mutmut_19: survived`
- `rag.retrieve.x__retrieve__mutmut_22: survived`
- `rag.retrieve.x__retrieve__mutmut_23: survived`
- `rag.retrieve.x__retrieve__mutmut_24: survived`
- `rag.retrieve.x__retrieve__mutmut_28: survived`
- `rag.retrieve.x__retrieve__mutmut_31: survived`
- `rag.retrieve.x__retrieve__mutmut_32: survived`
- `rag.retrieve.x__retrieve__mutmut_33: survived`
- `rag.retrieve.x__retrieve__mutmut_34: survived`
- `rag.retrieve.x__retrieve__mutmut_36: survived`
- `rag.retrieve.x__retrieve__mutmut_37: survived`
- `rag.retrieve.x__retrieve__mutmut_38: survived`
- `rag.retrieve.x__retrieve__mutmut_39: survived`
- `rag.retrieve.x__retrieve__mutmut_40: survived`
- `rag.retrieve.x__retrieve__mutmut_44: survived`
- `rag.retrieve.x__retrieve__mutmut_47: survived`
- `rag.retrieve.x__retrieve__mutmut_48: survived`
- `rag.retrieve.x__retrieve__mutmut_49: survived`
- `rag.retrieve.x__retrieve__mutmut_50: survived`
- `rag.retrieve.x__retrieve__mutmut_70: survived`
- `rag.retrieve.x__retrieve__mutmut_72: survived`
- `rag.retrieve.x__retrieve__mutmut_75: survived`
- `rag.retrieve.x__retrieve__mutmut_84: survived`
- `rag.retrieve.x__retrieve__mutmut_85: survived`
- `rag.retrieve.x__retrieve__mutmut_86: survived`
- `rag.retrieve.x__retrieve__mutmut_87: survived`
- `rag.retrieve.x__retrieve__mutmut_88: survived`
- `rag.retrieve.x__retrieve__mutmut_89: survived`
- `rag.retrieve.x__retrieve__mutmut_96: survived`
- `rag.retrieve.x__retrieve__mutmut_97: survived`
- `rag.retrieve.x__retrieve__mutmut_100: survived`
- `rag.retrieve.x__retrieve__mutmut_103: survived`
- `rag.retrieve.x__retrieve__mutmut_104: survived`
- `rag.retrieve.x__retrieve__mutmut_109: survived`
- `rag.retrieve.x__retrieve__mutmut_113: survived`
- `rag.retrieve.x__retrieve__mutmut_119: survived`
- `rag.retrieve.x__retrieve__mutmut_120: survived`
- `rag.retrieve.x__retrieve__mutmut_126: survived`
- `rag.retrieve.x__retrieve__mutmut_127: survived`
- `rag.retrieve.x__retrieve__mutmut_128: survived`
- `rag.retrieve.x__retrieve__mutmut_143: survived`
- `rag.retrieve.x__retrieve__mutmut_147: survived`
- `rag.retrieve.x__retrieve__mutmut_153: survived`
- `rag.retrieve.x__retrieve__mutmut_162: survived`
- `rag.retrieve.x__retrieve__mutmut_164: survived`
- `rag.retrieve.x__retrieve__mutmut_165: survived`
- `rag.retrieve.x__retrieve__mutmut_172: survived`
- `rag.retrieve.x__retrieve__mutmut_176: survived`
- `rag.retrieve.x__retrieve__mutmut_182: survived`
- `rag.retrieve.x__retrieve__mutmut_183: survived`
- `rag.retrieve.x__retrieve__mutmut_189: survived`
- `rag.retrieve.x__retrieve__mutmut_190: survived`
- `rag.retrieve.x__retrieve__mutmut_191: survived`
- `rag.retrieve.x__retrieve__mutmut_207: survived`
- `rag.retrieve.x__retrieve__mutmut_208: survived`
- `rag.retrieve.x__retrieve__mutmut_214: survived`
- `rag.retrieve.x__retrieve__mutmut_215: survived`
- `rag.retrieve.x__retrieve__mutmut_217: survived`
- `rag.retrieve.x__retrieve__mutmut_223: survived`
- `rag.retrieve.x__retrieve__mutmut_225: survived`
- `rag.retrieve.x__retrieve__mutmut_226: survived`
- `rag.retrieve.x__retrieve__mutmut_228: survived`
- `rag.retrieve.x__retrieve__mutmut_229: survived`
- `rag.retrieve.x__retrieve__mutmut_230: survived`
- `rag.retrieve.x__retrieve__mutmut_237: survived`
- `rag.retrieve.x__retrieve__mutmut_245: survived`
- `rag.retrieve.x__retrieve__mutmut_249: survived`
- `rag.retrieve.x__retrieve__mutmut_251: survived`
- `rag.retrieve.x__retrieve__mutmut_254: survived`
- `rag.retrieve.x__retrieve__mutmut_255: survived`
- `rag.retrieve.x__retrieve__mutmut_256: survived`
- `rag.retrieve.x__retrieve__mutmut_257: survived`
- `rag.retrieve.x__retrieve__mutmut_258: survived`
- `rag.retrieve.x__retrieve__mutmut_259: survived`
- `rag.retrieve.x__retrieve__mutmut_260: survived`
- `rag.retrieve.x__retrieve__mutmut_267: survived`
- `rag.retrieve.x__retrieve__mutmut_274: survived`
- `rag.retrieve.x__retrieve__mutmut_279: survived`
- `rag.retrieve.x__retrieve__mutmut_280: survived`
- `rag.retrieve.x__retrieve__mutmut_281: survived`
- `rag.retrieve.x__retrieve__mutmut_286: survived`
- `rag.retrieve.x__retrieve__mutmut_290: survived`
- `rag.retrieve.x__retrieve__mutmut_295: survived`
- `rag.retrieve.x__retrieve__mutmut_296: survived`
- `rag.retrieve.x__retrieve__mutmut_297: survived`
- `rag.retrieve.x__retrieve__mutmut_302: survived`
- `rag.retrieve.x__retrieve__mutmut_303: survived`
- `rag.retrieve.x__retrieve__mutmut_305: survived`
- `rag.retrieve.x__retrieve__mutmut_313: survived`
- `rag.retrieve.x__retrieve__mutmut_317: survived`
- `rag.retrieve.x__retrieve__mutmut_322: survived`
- `rag.retrieve.x__retrieve__mutmut_323: survived`
- `rag.retrieve.x__retrieve__mutmut_324: survived`
- `rag.retrieve.x__retrieve__mutmut_329: survived`
- `rag.retrieve.x__retrieve__mutmut_330: survived`
- `rag.retrieve.x__retrieve__mutmut_332: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_1: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_2: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_3: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_4: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_5: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_6: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_7: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_8: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_9: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_10: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_13: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_15: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_18: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_19: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_20: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_21: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_22: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_25: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_27: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_30: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_31: survived`
- `rag.retrieve.x__stored_tau_key__mutmut_32: survived`
- `rag.retrieve.x__parent_scores_from_chunks__mutmut_11: survived`
- `rag.retrieve.x__parent_scores_from_chunks__mutmut_18: survived`
- `rag.retrieve.x__parent_scores_from_chunks__mutmut_23: survived`
- `rag.retrieve.x__parent_scores_from_chunks__mutmut_27: survived`
- `rag.retrieve.x__parent_scores_from_chunks__mutmut_29: survived`
- `rag.retrieve.x__parent_scores_from_chunks__mutmut_31: survived`
- `rag.retrieve.x__order_current_first__mutmut_1: survived`
- `rag.retrieve.x__order_current_first__mutmut_2: survived`
- `rag.retrieve.x__order_current_first__mutmut_3: survived`
- `rag.retrieve.x__order_current_first__mutmut_10: survived`
- `rag.retrieve.x__order_current_first__mutmut_11: survived`
- `rag.retrieve.x__order_current_first__mutmut_14: survived`
- `rag.retrieve.x__order_current_first__mutmut_15: survived`
- `rag.retrieve.x__order_current_first__mutmut_16: survived`
- `rag.retrieve.x__order_current_first__mutmut_17: survived`
- `rag.retrieve.x__order_current_first__mutmut_18: survived`
- `rag.retrieve.x__order_current_first__mutmut_19: survived`
- `rag.retrieve.x__order_current_first__mutmut_20: survived`
- `rag.retrieve.x__order_current_first__mutmut_27: survived`
- `rag.retrieve.x__order_current_first__mutmut_29: survived`
- `rag.retrieve.x__order_current_first__mutmut_30: survived`
- `rag.retrieve.x__order_current_first__mutmut_36: survived`
- `rag.retrieve.x__order_current_first__mutmut_37: survived`
- `rag.retrieve.x__order_current_first__mutmut_38: survived`
- `rag.retrieve.x__order_current_first__mutmut_39: survived`
- `rag.retrieve.x__order_current_first__mutmut_40: survived`
- `rag.retrieve.x__order_current_first__mutmut_41: survived`
- `rag.retrieve.x__order_current_first__mutmut_42: survived`
- `rag.retrieve.x__order_current_first__mutmut_43: survived`
- `rag.retrieve.x__order_current_first__mutmut_44: survived`
- `rag.retrieve.x__order_current_first__mutmut_45: survived`
- `rag.retrieve.x__order_current_first__mutmut_46: survived`
- `rag.retrieve.x__order_current_first__mutmut_47: survived`
- `rag.retrieve.x__order_current_first__mutmut_48: survived`
- `rag.retrieve.x__order_current_first__mutmut_49: survived`
- `rag.retrieve.x__order_current_first__mutmut_50: survived`
- `rag.retrieve.x__order_current_first__mutmut_51: survived`
- `rag.retrieve.x__order_current_first__mutmut_57: survived`
- `rag.retrieve.x__order_current_first__mutmut_58: survived`
- `rag.retrieve.x__order_current_first__mutmut_59: survived`
- `rag.retrieve.x__order_current_first__mutmut_61: survived`
- `rag.retrieve.x__order_current_first__mutmut_63: survived`
- `rag.retrieve.x__order_current_first__mutmut_64: survived`
- `rag.retrieve.x__order_current_first__mutmut_67: survived`
- `rag.retrieve.x__order_current_first__mutmut_68: survived`
- `rag.retrieve.x__order_current_first__mutmut_69: survived`
- `rag.retrieve.x__order_current_first__mutmut_70: survived`
- `rag.retrieve.x__order_current_first__mutmut_71: survived`
- `rag.retrieve.x__order_current_first__mutmut_72: survived`
- `rag.retrieve.x__order_current_first__mutmut_73: survived`
- `rag.retrieve.x__order_current_first__mutmut_74: survived`
- `rag.retrieve.x__order_current_first__mutmut_76: survived`
- `rag.retrieve.x__order_current_first__mutmut_77: survived`
- `rag.retrieve.x__order_current_first__mutmut_78: survived`
- `rag.retrieve.x__order_current_first__mutmut_79: survived`
- `rag.retrieve.x__order_current_first__mutmut_85: survived`
- `rag.retrieve.x__order_current_first__mutmut_86: survived`
- `rag.retrieve.x__order_current_first__mutmut_87: survived`
- `rag.retrieve.x__order_current_first__mutmut_89: survived`
- `rag.retrieve.x__order_current_first__mutmut_90: survived`
- `rag.retrieve.x__order_current_first__mutmut_91: survived`
- `rag.retrieve.x__order_current_first__mutmut_92: survived`
- `rag.retrieve.x__order_current_first__mutmut_93: survived`
- `rag.retrieve.x__order_current_first__mutmut_94: survived`
- `rag.retrieve.x__order_current_first__mutmut_96: survived`
- `rag.retrieve.x__order_current_first__mutmut_97: survived`
- `rag.retrieve.x__order_current_first__mutmut_98: survived`
- `rag.retrieve.x__order_current_first__mutmut_101: survived`
- `rag.retrieve.x__order_current_first__mutmut_102: survived`
- `rag.retrieve.x__order_current_first__mutmut_103: survived`
- `rag.retrieve.x__order_current_first__mutmut_104: survived`
- `rag.retrieve.x__order_current_first__mutmut_105: survived`
- `rag.retrieve.x__order_current_first__mutmut_114: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_1: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_2: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_3: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_4: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_6: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_7: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_8: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_9: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_10: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_11: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_12: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_13: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_20: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_21: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_22: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_23: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_24: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_25: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_26: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_27: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_28: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_29: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_30: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_31: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_32: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_33: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_34: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_35: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_36: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_37: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_38: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_39: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_42: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_43: survived`
- `rag.retrieve.x__prefer_named_year__mutmut_52: survived`
- `rag.retrieve.x__parent_order__mutmut_4: survived`
- `rag.retrieve.x__parent_order__mutmut_11: survived`
- `rag.retrieve.x__parent_order__mutmut_12: survived`
- `rag.retrieve.x__from_parent_ids__mutmut_1: survived`
- `rag.retrieve.x__from_parent_ids__mutmut_8: survived`
- `rag.retrieve.x__from_parent_ids__mutmut_16: survived`
- `rag.retrieve.x__from_parent_ids__mutmut_23: survived`
- `rag.retrieve.x__from_parent_ids__mutmut_25: survived`
- `rag.retrieve.x__from_parent_ids__mutmut_30: survived`
- `rag.retrieve.x__from_parent_ids__mutmut_32: survived`
- `rag.retrieve.x__from_parent_ids__mutmut_39: survived`
- `rag.retrieve.x__from_parent_ids__mutmut_40: survived`
- `rag.retrieve.x__best_child__mutmut_5: survived`
- `rag.retrieve.x__best_child__mutmut_6: survived`
- `rag.retrieve.x__best_child__mutmut_8: survived`
- `rag.retrieve.x__best_child__mutmut_11: survived`
- `rag.retrieve.x__best_child__mutmut_14: survived`
- `rag.retrieve.x__best_child__mutmut_16: survived`
- `rag.retrieve.x__emit__mutmut_1: survived`
- `rag.retrieve.x__emit__mutmut_2: survived`
- `rag.retrieve.x__emit__mutmut_13: survived`
- `rag.retrieve.x__emit__mutmut_14: survived`
- `rag.retrieve.x__emit__mutmut_15: survived`
- `rag.retrieve.x__emit__mutmut_16: survived`
- `rag.retrieve.x__emit__mutmut_17: survived`
- `rag.retrieve.x__emit__mutmut_18: survived`
- `rag.retrieve.x__emit__mutmut_19: survived`
- `rag.retrieve.x__emit__mutmut_23: survived`
- `rag.retrieve.x__emit__mutmut_27: survived`
- `rag.retrieve.x__emit__mutmut_29: survived`
- `rag.retrieve.x__emit__mutmut_32: survived`
- `rag.retrieve.x__emit__mutmut_33: survived`
- `rag.retrieve.x__emit__mutmut_34: survived`
- `rag.retrieve.x__emit__mutmut_36: survived`
- `rag.retrieve.x__emit__mutmut_37: survived`
- `rag.retrieve.x__emit__mutmut_38: survived`
- `rag.retrieve.x__emit__mutmut_39: survived`
- `rag.retrieve.x__emit__mutmut_40: survived`
- `rag.retrieve.x__emit__mutmut_53: survived`
- `rag.retrieve.x__emit__mutmut_54: survived`
- `rag.retrieve.x__emit__mutmut_56: survived`
- `rag.retrieve.x__emit__mutmut_57: survived`
- `rag.retrieve.x__emit__mutmut_58: survived`
- `rag.retrieve.x__emit__mutmut_59: survived`
- `rag.retrieve.x__emit__mutmut_60: survived`
- `rag.retrieve.x__emit__mutmut_63: survived`
- `rag.retrieve.x__emit__mutmut_64: survived`
- `rag.retrieve.x__emit__mutmut_65: survived`
- `rag.retrieve.x__emit__mutmut_66: survived`
- `rag.retrieve.x__emit__mutmut_67: survived`
- `rag.retrieve.x__emit__mutmut_72: survived`
- `rag.retrieve.x__emit__mutmut_77: survived`
- `rag.retrieve.x__emit__mutmut_83: survived`
- `rag.retrieve.x__emit__mutmut_85: survived`
- `rag.retrieve.x__emit__mutmut_86: survived`
- `rag.retrieve.x__emit__mutmut_87: survived`
- `rag.retrieve.x__emit__mutmut_88: survived`
- `rag.retrieve.x__emit__mutmut_89: survived`
- `rag.retrieve.x__emit__mutmut_95: survived`
- `rag.retrieve.x__emit__mutmut_97: survived`
- `rag.retrieve.x__emit__mutmut_98: survived`
- `rag.retrieve.x__emit__mutmut_99: survived`
- `rag.retrieve.x__emit__mutmut_100: survived`
- `rag.retrieve.x__emit__mutmut_101: survived`
- `rag.retrieve.x__emit__mutmut_103: survived`
- `rag.retrieve.x__emit__mutmut_104: survived`
- `rag.retrieve.x__emit__mutmut_105: survived`
- `rag.retrieve.x__emit__mutmut_106: survived`
- `rag.retrieve.x__emit__mutmut_107: survived`
- `rag.retrieve.x__emit__mutmut_108: survived`
- `rag.retrieve.x__emit__mutmut_109: survived`
- `rag.retrieve.x__emit__mutmut_110: survived`
- `rag.retrieve.x__emit__mutmut_111: survived`
- `rag.retrieve.x__emit__mutmut_113: survived`
- `rag.retrieve.x__emit__mutmut_114: survived`
- `rag.retrieve.x__emit__mutmut_118: survived`
- `rag.retrieve.x__emit__mutmut_119: survived`
- `rag.retrieve.x__emit__mutmut_120: survived`
- `rag.retrieve.x__emit__mutmut_121: survived`
- `rag.retrieve.x__emit__mutmut_122: survived`
- `rag.retrieve.x__emit__mutmut_123: survived`
- `rag.retrieve.x__emit__mutmut_124: survived`
- `rag.retrieve.x__emit__mutmut_125: survived`
- `rag.retrieve.x__emit__mutmut_126: survived`
- `rag.retrieve.x__emit__mutmut_127: survived`
- `rag.retrieve.x__emit__mutmut_128: survived`
- `rag.retrieve.x__emit__mutmut_129: survived`
- `rag.retrieve.x__emit__mutmut_132: survived`

Timeouts (19), not counted in the killed+survived denominator:

- `rag.chunk.x__paragraph_spans__mutmut_7: timeout`
- `rag.chunk.x__paragraph_spans__mutmut_9: timeout`
- `rag.chunk.x__paragraph_spans__mutmut_18: timeout`
- `rag.chunk.x__word_spans__mutmut_3: timeout`
- `rag.chunk.x__word_spans__mutmut_6: timeout`
- `rag.chunk.x__word_spans__mutmut_9: timeout`
- `rag.chunk.x__word_spans__mutmut_11: timeout`
- `rag.chunk.x__word_spans__mutmut_16: timeout`
- `rag.chunk.x__word_spans__mutmut_21: timeout`
- `rag.chunk.x__word_spans__mutmut_24: timeout`
- `rag.chunk.x__word_spans__mutmut_28: timeout`
- `rag.chunk.x__word_spans__mutmut_36: timeout`
- `rag.chunk.x__word_spans__mutmut_37: timeout`
- `rag.chunk.x__line_spans__mutmut_7: timeout`
- `rag.chunk.x__line_spans__mutmut_9: timeout`
- `rag.chunk.x__line_spans__mutmut_20: timeout`
- `rag.chunk.x__windows__mutmut_45: timeout`
- `rag.chunk.x__windows__mutmut_66: timeout`
- `rag.chunk.x__windows__mutmut_68: timeout`

## Eval suite

`245-rag-eval.log` ends with `suite gates: PASS`, `live=rrf`, `tau=0.1174`. The rrf `all` line in that log is `recall@5=0.99 mrr=0.82 abstain=0.00 p50=27ms p95=31ms p99=34ms n=152`. This replaces the earlier report text that said `suite gates: FAIL`. The suite was not re-run for this closeout. `evals/suite.yaml` floors were not changed.

## Carbon-neutrality answer

Captured before this closeout, not re-run.

`246-live-carbon-and-unanswerable.log`:

```
Coforge commits to becoming Carbon Neutral in its operations by 2040 [1]. Note: a superseded version (Environmental_Sustainability_Policy_2025.pdf) states 2050 [5].
```

`252-streamed-carbon-rebuilt.log` done payload has the same answer text, `verification.state` `conflict`, `verification.reason` `version_conflict`, current value `2040`, superseded value `2050`.

## First-token latency

`250-ttft-warmed.log`: `n=10 p50=1173 p95=1819 max=1819`. The suite budget is 1200 ms. 1819 ms is over that budget. The budget was not lowered. This measurement was not repeated in this closeout.

## CI

`.github/workflows/ci.yml` installs with `uv sync --frozen --extra dev` and then runs corpus pins, `ruff check .`, `mypy --strict src/`, the pragma scan, pytest without random order, pytest with random order, `pytest -n auto`, branch coverage at 100, `mutmut run`, `pip-audit`, plus web `npm ci` / `test:coverage` / `build`, plus `docker build`. `rag eval` is not in the workflow.

The workflow that was on disk before the rewrite failed collection (`259`, exit 2) because `tests` was not importable. Log 316 is the same pytest commands after `pythonpath = ["."]`.

## Still broken

1. Mutation score is 0.628325, under 0.90. 1076 survivors are listed above. `321`, `322`.
2. Warmed first-token p95 is 1819 ms, over the 1200 ms budget. `250`.

## Not re-run

- `rag eval` (the recorded pass is `245-rag-eval.log`).
- The warmed ttft sample (`250-ttft-warmed.log`).
- Live API queries, including the carbon answer (`246`, `252`).
- Browser screenshots.
- Security probes, concurrent eval, and the earlier bring-up parts.
- `npm audit` findings printed by `npm ci` (1 moderate, 2 critical) were not patched. `pip-audit` on the Python env is the audit the plan required, and that exit is 0.

Passed: lockfile install, pip-audit, ruff, mypy --strict, pytest in three orders, branch coverage 100%, web coverage at the existing floors, web build, docker build, eval suite on the recorded log.
Failed: mutmut 0.628325 against 0.90; warmed ttft p95 1819 ms against 1200 ms.
Could not run: nothing in the closeout command list was skipped; the items under “Not re-run” were left on their existing logs on purpose.
