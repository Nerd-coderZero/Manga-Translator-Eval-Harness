# Manga Translator Eval Harness

An evaluation harness for [Manga Translator](https://huggingface.co/spaces/NerdCoderZero/manga-translator),
a deployed manga-translation pipeline (Japanese/Chinese to English). This
harness calls the live, deployed pipeline the same way a real user does —
through `gradio_client` — runs it through four quality checks grounded in
the pipeline's own documented failure history, and drives a small agentic
retry loop (built on LangGraph) that decides per page whether to retry,
fall back, flag for human review, or pass. Every run produces a plain-text
log readable without re-running any code.

This is not a synthetic test suite. Every check and every fixture in
`tests/` traces back to a real, documented bug or to real data pulled from
the live pipeline — not invented test cases.

## What it found

Across three real batches (18 Japanese pages, 22 Chinese pages, and a
second independent 24-page Chinese batch) against the live pipeline:

- Confirmed a known font-sizing bug (translated text hard-wrapping into
  single-letter stacks inside undersized boxes) reproduces on the live
  Space, and found a real hit on the same failure pattern on the Chinese
  path.
- Found and locked in a regression test for a real gap in the pipeline's
  own garbage-OCR filter: short misread tokens (3-5 characters) evade it
  entirely and get passed through translation unchanged. Reproduced on
  two independent batches.
- Found a previously undocumented failure mode: the translation backend
  sometimes returns conversational text — a clarifying question, or an
  explicit content-policy refusal — instead of a translation or an empty
  string, when given source text it can't or won't translate. Both cases
  are now regression-tested.

Full reasoning, rejected alternatives, and a dated findings log are in
[`docs/DESIGN.md`](docs/DESIGN.md).

## Running it

```
cd harness
pip install -r ../requirements.txt   # gradio_client, langgraph, numpy, Pillow
python run.py <image_path> <ja|zh>              # single image, up to 2 attempts
python batch_run.py <directory> <ja|zh>         # every image in a directory, 1 attempt each
```

Both write a human-readable log per image to `logs/`; `batch_run.py` also
writes an aggregate summary and a raw per-region CSV (every check's
computed values, fired or not).

Test images are not included in this repo (real manga pages are
copyrighted) — source your own and point either script at them.

## Layout

- `harness/` — the four checks, the LangGraph agent loop, the live-pipeline
  client, and the two entry points (`run.py`, `batch_run.py`).
- `fixtures/` — real data used by the regression tests, with provenance
  noted for anything that isn't a pure historical replay.
- `tests/` — regression tests, every case traced to a real source.
- `logs/` — real run output from the batches described above.
- `docs/DESIGN.md` — build decisions, rejected alternatives, and findings.
