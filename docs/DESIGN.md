# Design decisions

Why Project 2 (the evaluation harness) is built the way it is. Written in
the same spirit as Project 1's own `docs/DECISIONS.md`: each entry states
the alternative that was rejected and what it would have cost, and nothing
here claims more certainty than the evidence supports.

---

## What this is

Project 2 is an evaluation harness for Project 1 (the manga translator),
built as a separate project under `Projects/Project 2/`, sibling to
`Projects/Manga Translator/`. It calls the real, deployed
`NerdCoderZero/manga-translator` HF Space through `gradio_client`, exactly
the way a real user reaches it, runs four quality checks against the
returned regions, and drives a small agentic retry loop that decides
per page whether to retry, fall back, flag for human review, or pass.

Every check and every fixture is grounded in Project 1's own documented
failure surface (`LIMITATIONS.md`, `BUGS.md`, `DECISIONS.md`) or in real
data pulled from the live Space, not in invented test cases. The point is
not just to build checks, but to produce a run log that is itself
readable evidence of what happened and why, without needing to re-run
anything.

---

## Why LangGraph over a hand-rolled state machine

The actual control flow here is small: `preflight -> translate -> score ->
decide`, with one conditional edge back from `decide` to `translate` for
retries, and two more terminal exits (a failed preflight, and `decide`
reaching pass/fallback/flag). A plain `while` loop with an enum would
express this in fewer lines than `graph.py` currently takes, and would not
need `langgraph` as a dependency at all.

LangGraph was chosen anyway, because it is the framework that keeps
showing up by name in real job postings for this kind of agentic-systems
work, and a harness built to demonstrate that skill is worth more if it
demonstrates the tool the market is actually asking about. This is
disclosed plainly rather than oversold: this graph does not exercise
LangGraph's persistence, memory, or multi-agent features, and a custom
state machine would have worked functionally just as well. The choice is
about legibility to someone reading this project later, not about a
capability gap a custom loop would have hit.

---

## Why "retry" re-invokes the same call instead of adjusting parameters

The original framing for a retry was "retry with adjusted parameters" --
e.g. a lower `min_acceptable_font_size` or a different placement strategy.
That is not available here: `client.view_api()` against the real Space
confirms its `/translate` endpoint takes only `files` and `source_lang`.
Calling Project 1's pipeline directly with different kwargs would require
paddleocr/manga-ocr/torch installed locally, which has already failed once
in a sandboxed environment like this one over disk space (recorded in the
2026-09-09 career log).

So "retry" means what is actually reachable: re-invoke the same API call
and rely on `translation.py`'s real non-determinism
(`temperature=1.0` in `translate_text_nemotron`) to possibly produce a
translation that fits better. This is verified as real, not assumed --
the same page translated twice in one session came back with different
phrasing ("Butt!!" vs "Your butt!"). Detection/box geometry, by contrast,
is only approximately deterministic across calls (16 vs 17 regions were
observed on two calls against the same real image), so a retry cannot be
relied on to fix a geometry-driven failure.

The rejected alternative -- building a local, parameterizable pipeline
call just so retries could vary something real -- was set aside because it
duplicates Project 1's own dependency-management problems inside Project
2, for a harness whose whole premise is testing the deployed artifact as
a real caller would reach it, not a locally reconstructed one.

---

## Why `decide_node` compares fired checks across attempts by bounds

Because retrying can only possibly help a translation-length-driven
failure, not a geometry-driven one, and there is no way to know which kind
a given failure is just by looking at one attempt. `decide_node` tells
them apart empirically: if the same region (matched by bounds, which do
not change between calls) fails the same check on two consecutive
attempts despite a fresh, non-deterministic translation, that persistence
is treated as evidence the failure is geometry-driven and needs a
pipeline-level fix, not another retry -- so the decision moves to
`fallback` immediately rather than burning the rest of the retry budget.

The rejected alternative was always retrying up to `max_attempts`
regardless of whether the same failure kept recurring. That wastes retry
budget and live API calls on cases retrying cannot fix, and produces a
noisier log (repeated identical failures) instead of a decisive one.

---

## Why preflight runs once, before any real API call

Check 2 (the BUG-1 duplicate-box regression) is a property of the shipped
detection code, not of the specific page being translated -- it produces
the same answer regardless of input image. Running it once per harness
invocation, against a fixture built from BUG-1's own recorded real
coordinates, and short-circuiting straight to `flag` on failure without
ever calling the live Space, avoids spending a live API call (20-60+
seconds, sometimes much longer on a cold ZeroGPU start) on a page when the
detection code itself has already regressed.

---

## Why checks call Project 1's real functions instead of reimplementing the logic

`check_text_fit` calls `pipeline_core.fit_text_in_box`/`wrap_text_to_fit`
directly -- the same functions `placement.py` itself calls -- rather than
reimplementing a text-fit estimate. `check_duplicate_box_regression` calls
the real `_drop_redundant_oversized_boxes` from `ocr.py`. This means a
check can never silently drift from how the pipeline actually measures
text or filters boxes; if Project 1's own logic changes, the check's
behavior changes with it automatically instead of needing a parallel
update.

The cost is dependency management: `ocr.py` imports `paddleocr` at module
load time, and installing the real package has already failed once in a
sandbox like this one over disk space. `pipeline_access.py` solves this by
installing a minimal stand-in `paddleocr` module into `sys.modules` before
importing the real `ocr.py`, satisfying the import without installing the
real package -- safe specifically because the one function this harness
needs from `ocr.py`, `_drop_redundant_oversized_boxes`, is a pure function
over already-detected coordinates and never touches paddleocr itself. This
does not modify `ocr.py` in any way. The rejected alternative -- writing an
independent reimplementation of the fit/duplicate-box logic -- was set
aside because a reimplementation can pass its own tests while the real
pipeline has already diverged from it, which is a worse failure mode than
the dependency-management workaround costs.

---

## Why `detection_plausibility` is a proxy check, not a confidence check

PaddleOCR's `rec_scores` are computed inside `ocr.py` but `pipeline.py`
never attaches them to `merged_boxes`, so `runner.normalise_result()` never
returns them to any real caller, including this harness. A genuine
confidence-based check would need a small additive change threading
`rec_scores` through `pipeline_core.merge_text_boxes` -> `pipeline.py` ->
`runner.py`, and for the Japanese path `ocr.py` would need its hardcoded
`"confidence": 1.0` changed to `null`. That work was deliberately deferred
rather than done alongside a redeploy for its own sake -- the user's own
call, made explicitly to avoid redeploying the live Space for a change
that is not otherwise needed yet. Until that lands, this check falls back
to a bounds/text-density proxy (`chars_per_100px_height`), disclosed in
its own `scope_note` field as a weak proxy, not a substitute for real
detection accuracy -- which this project has never had ground truth for
in the first place (no page has ever been hand-labeled with correct
boxes to compare against).

The proxy's threshold (`dense_chars_per_100px_height=25.0`) was not picked
a priori -- an earlier value of 12.0 was invented without data and, once
14 real fired values across 4 real pages were inspected, turned out to
flag completely ordinary dense dialogue as often as it flagged anything
real. It was recalibrated against that real evidence: every value below
20 was ordinary dialogue on inspection, one real 30.0 outlier (a 3-character
translation crammed into a 10px-tall box) looked genuine, and 25.0 sits
between the two. This is disclosed as a small-sample recalibration
(n=14, fired cases only -- values for regions that did not fire were never
persisted by the original per-run logger, a real gap `batch_run.py`'s raw
CSV output now fixes going forward), not a statistically rigorous one.

---

## Why the per-run log only details fired checks, and why `batch_run.py` writes a separate raw CSV

`logger.py`'s `write_run_log` was built to answer "was this pass
comfortable or a near miss" for a human reading one run cold, so it prints
full detail only for checks that fired -- a passing check's computed
values are not interesting to a reader trying to understand why a page
was flagged. The cost, discovered only once real recalibration was
needed, is that this makes it impossible to reconstruct the full
distribution of a check's values (fired and not) from historical logs --
exactly the gap that made the `detection_plausibility` recalibration
harder than it should have been, since only the 14 fired values were ever
recoverable, not the full population they were drawn from.

Rather than rewriting `logger.py` and losing the readability its
fired-only format gives a human, `batch_run.py` writes a second,
machine-oriented raw CSV (`batch_raw_<timestamp>_<lang>.csv`) recording
every check's computed values for every region, fired or not. The
per-run `.md` logs stay optimized for a human reading one run; the CSV is
what any future recalibration should read from.

---

## Why `batch_run.py` defaults to `--max-attempts 1`

Bulk runs already take a long time per image against a live, sometimes
cold-starting ZeroGPU Space (observed 2-3 minutes per Chinese-language
image across a 22-24 image batch). A real consequence of this default,
worth stating plainly since it is not obvious from the decision counts
alone: `decide_node` only returns `retry` when `attempt < max_attempts`,
so with `max_attempts=1` no batch-run image can ever produce a `retry`
outcome -- every image's result is only ever a preflight-flag, a pipeline-
crash flag, an immediate `pass`, an immediate `fallback` (something fired
below high severity, budget already exhausted at attempt 1), or an
immediate `flag` (something fired at high severity, budget already
exhausted at attempt 1). This is the correct lens for reading every batch
summary in this project: the decision counts describe single-attempt
outcomes, not the full retry/fallback/flag machinery `run.py` exercises
on one image at a time with `max_attempts=2`.

---

## Why crash resilience is separate code from the four checks

The original BUG-5 was a crash (`AttributeError: 'PaddleOCR' object has no
attribute 'text_detector'`), not a bad-but-parseable result -- none of the
four checks can see a crash, because a crash means there is no result to
check in the first place. `translate_node` wraps the live API call in its
own `try`/`except`, storing the error in `attempt_history` instead of
letting the exception take down the whole harness process; `decide_node`
checks for that error first, before even looking at check results, and
flags immediately rather than retrying blindly, on the reasoning that a
crash is much more likely to be a real regression than translation-length
noise. This has since fired for real, independently of any synthetic
test: two images in the 2026-09-09 22-image Chinese batch
(`004.webp`, `006.webp`) hit a real `ConnectTimeout` and a real
`ReadTimeout` against the live Space, both caught cleanly and logged with
full reasoning instead of crashing the batch run.

---

## Why real batches were run against real, varied manga pages

Project 2's four checks were unit-tested from the start against real data
and BUG-1's own recorded verification numbers (`tests/test_checks.py`),
but that only tests the harness's own check code, not whether Project 1's
pipeline actually behaves the way the checks assume in the wild. Two real
batches were run against the live Space: 18 Japanese pages and 22 Chinese
pages (2026-09-09), across multiple different manga series, followed by a
second, independent 24-image Chinese batch (2026-09-12) to check whether
the first Chinese batch's result would reproduce on an unrelated sample.

---

## Current findings (as of 2026-09-12)

Grounded in the batches above, not invented:

- LIM-1 (the font-sizing bug on small/vertical regions) reproduces on the
  live Space with real Japanese data -- confirmed both in the 2026-09-09
  career log (`Manager!` rendering as stacked `M/an/ag/er!`) and again in
  this harness's own `12.webp` case (`"Huh"` -> `H/u/h` in a 12x13px box).

- A real hit on the originally hypothesized Chinese-path equivalent of
  LIM-1 (oversized dialogue vs. undersized text) was found on `09.webp`:
  a 23x13px region translated to `"THWACK"`, which cannot fit even at
  minimum font size and hard-wraps into `TH/WAC/K`.

- A previously undocumented gap in Project 1's own `pipeline_core.is_garbage_text`:
  it only screens repeated-character or digit-heavy strings, and only once
  a string is 6-8+ characters long. Short OCR-misread tokens (`'CSB'`,
  `'d00'`, single letters/digits) evade it entirely and pass through
  translation unchanged, which this harness's `translation_sanity` check
  correctly catches as a suspicious identical-passthrough. This reproduced
  on two independent Chinese batches (13 fires across 9 images in the
  first, 35 fires across 11 images in the second) -- confirmed real and
  reproducible, not a single-batch artifact. Locked in as a regression
  fixture (`test_short_garbage_ocr_evades_is_garbage_text_but_translation_sanity_catches_it`).

- A new failure mode, not in LIM-1/BUG-1 through BUG-5 anywhere: NIM
  sometimes returns conversational meta-commentary instead of a
  translation or an empty string, when given source text with no real
  standalone meaning (e.g. a single particle). Confirmed on `08.webp`
  (Japanese, `と` -> a 139-character explanation) and reproduced with
  different exact wording on two regions in the second Chinese batch
  (`18.webp`, `27.webp`) -- consistent with `translation.py`'s real
  non-determinism (`temperature=1.0`) rather than a fixed template. Both
  `text_fit` and `translation_sanity` catch this, from two different
  angles, on the same region. Locked in as a regression fixture
  (`test_nim_meta_commentary_leak_on_untranslatable_particle`).

- A distinct sub-case of the above, found in the second Chinese batch on
  `29.webp`: NIM declined to translate genuine, coherent, explicit
  dialogue with an explicit content-policy refusal message, rather than a
  "nothing to translate" meta-comment. This is a real behavior of the
  underlying NIM model, external to both Project 1's and Project 2's code
  -- neither project adds or controls this filtering; the harness only
  observes and flags it after the fact via the same length-ratio
  heuristic. Locked in as a regression fixture
  (`test_nim_content_policy_refusal_on_explicit_dialogue`).

- The two live-pipeline network failures described above (`004.webp`,
  `006.webp`, 2026-09-09), confirming the BUG-5-style crash resilience
  works against a real, not synthetic, failure.

---

## Still open

- Whether the elevated Chinese-path flag rate (50% on the first 22-image
  batch, 58% on the second 24-image batch, vs. 11% on 18 Japanese images)
  is a language-specific detection-quality difference, or driven partly or
  wholly by image/scan quality and genre, is not cleanly settled. The
  second Chinese batch changed both the manga source's genre and likely
  scan quality at the same time as holding the language constant, so it
  does not isolate language as the only variable. The specific
  garbage-OCR-passthrough mechanism is confirmed reproducible across both
  Chinese batches; the raw flag-rate comparison across all three batches
  is not a controlled comparison.

- Real recognition confidence (`rec_scores`) is not threaded through
  `pipeline_core.merge_text_boxes` -> `pipeline.py` -> `runner.py`, and
  `ocr.py`'s Japanese-path `"confidence": 1.0` has not been changed to
  `null`. `detection_plausibility` is forward-compatible with this (it
  already reads `region.get("confidence")`) but currently always sees
  `None`. Deliberately deferred to avoid a Space redeploy for its own
  sake; planned to be batched with a future redeploy instead.

- The bulk-upload path in Project 1's deployed UI (`app.py`'s
  `batch_store`) has never been exercised by this harness at all -- every
  call this harness makes goes through the same single-image
  `client.predict(files=[...], source_lang=...)` path, one image per call.
  A separately reported real observation (many images/boxes left
  untouched when 40 images were uploaded at once through the live UI,
  with content-filtering and token/rate limits both suspected) is
  therefore only partially explained by this project's findings: the
  content-policy-refusal fixture above confirms content filtering is a
  real, independent mechanism, but neither theory has been tested against
  the actual bulk code path, since that is out of this harness's scope as
  built.

- No fixture yet exercises the Chinese path's BUG-5 fix end to end
  against the live Space with a crash specifically induced (as opposed to
  the two real, naturally-occurring network timeouts already observed).
