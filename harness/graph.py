"""
The agent loop: preflight -> translate -> score -> decide, with a retry
edge back to translate and fallback/flag/pass as terminal outcomes.

A real constraint shapes this more than anything else, so it is stated
here rather than left implicit: "retry with adjusted parameters" was the
original framing, but the live Space's actual API (files + source_lang
only -- confirmed via client.view_api()) exposes no pipeline parameters
to adjust. Calling the pipeline directly with different kwargs (e.g.
placement.py's min_acceptable_font_size) would need paddleocr/manga-ocr/
torch installed locally, which has already failed once in a sandbox like
this one over disk space (2026-09-09 log). So "retry" here means what is
actually available: re-invoke the same pipeline call and let
translation.py's real non-determinism (temperature=1.0 in
translate_text_nemotron) possibly produce a translation that fits better.
This is verified as real, not assumed -- the same page translated twice
in this session already came back with different phrasing ("Butt!!" vs
"Your butt!").

That mechanism can only ever help checks whose failure is driven by
translation length/shape (check 3, and check-1 failures where a shorter
translation would fit). It cannot help a check-1 failure where the box
itself is too small for any legible text (e.g. the real 16x10px case) --
detection is deterministic across calls, so the box never changes. The
decide step tells these apart empirically rather than guessing up front:
if the SAME region (matched by bounds, which do not change between calls)
fails the SAME check on two consecutive attempts, retrying is not
helping, and the decision moves to fallback instead of burning the
remaining retry budget.

Preflight (check 2, the BUG-1 regression) runs once, before any real API
call, because it is a property of the shipped code, not of the page being
translated -- it produces the same answer regardless of which image is
passed in. A preflight failure means the detection code itself has
regressed since BUG-1 was fixed, which is a flag-worthy event on its own
and has nothing to do with the specific page, so the page is never even
submitted to the live Space in that case.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

import checks
from translate_client import call_live_space


class HarnessState(TypedDict, total=False):
    input_image: str
    source_lang: str
    space_name: str
    attempt: int
    max_attempts: int
    regions: list
    check_results: list
    attempt_history: list
    decision: str
    reasoning: str
    preflight: dict


def preflight_node(state):
    import json
    import os

    fixture_path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "bug1_regression_case.json")
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    result = checks.check_duplicate_box_regression(fixture)
    state = dict(state)
    state["preflight"] = result
    return state


def translate_node(state):
    # BUG-5 regression coverage for the *crash* failure mode: the original
    # bug was `AttributeError: 'PaddleOCR' object has no attribute
    # 'text_detector'` -- the live pipeline call raising outright, not a
    # bad-but-parseable result. Check 2 covers the duplicate-box regression
    # BUG-1 left behind; this try/except covers "the pipeline call itself
    # dies," which check 2 does not touch. Without this, a crash here would
    # take the whole harness process down with it instead of being logged
    # as a flag-worthy event.
    state = dict(state)
    attempt = state.get("attempt", 0) + 1
    state["attempt"] = attempt

    try:
        run = call_live_space(
            state["input_image"], state["source_lang"],
            state.get("space_name", "NerdCoderZero/manga-translator"),
        )
        state["regions"] = run["regions"]
        state.setdefault("attempt_history", []).append({
            "attempt": attempt,
            "status": run["status"],
            "regions": run["regions"],
            "error": None,
        })
    except Exception as e:
        state["regions"] = []
        state.setdefault("attempt_history", []).append({
            "attempt": attempt,
            "status": "PIPELINE_CALL_FAILED",
            "regions": [],
            "error": f"{type(e).__name__}: {e}",
        })
    return state


def score_node(state):
    state = dict(state)
    last_attempt = state["attempt_history"][-1]
    if last_attempt.get("error"):
        # pipeline call itself failed -- nothing to score, checks need a
        # real result to run against.
        state["check_results"] = []
        last_attempt["check_results"] = []
        return state

    per_region = []
    for region in state["regions"]:
        fit = checks.check_text_fit(region)
        sanity = checks.check_translation_sanity(region)
        plaus = checks.check_detection_plausibility(region)
        per_region.append({
            "bounds": region["bounds"],
            "source_text": region["source_text"],
            "translation": region["translation"],
            "checks": [fit, sanity, plaus],
        })
    state["check_results"] = per_region
    state["attempt_history"][-1]["check_results"] = per_region
    return state


def _region_key(bounds):
    return tuple(bounds)


def _fired_map(check_results):
    """{(bounds, check_name): severity} for every fired check this attempt."""
    out = {}
    for region in check_results:
        key_base = _region_key(region["bounds"])
        for c in region["checks"]:
            if c["fired"]:
                out[(key_base, c["name"])] = c["severity"]
    return out


def decide_node(state):
    state = dict(state)

    last_attempt = state["attempt_history"][-1] if state.get("attempt_history") else {}
    if last_attempt.get("error"):
        state["decision"] = "flag"
        state["reasoning"] = (
            f"The live pipeline call itself failed on attempt {state['attempt']}: "
            f"{last_attempt['error']}. This is BUG-5's failure mode (a crash, not a bad-but-"
            f"parseable result) -- flagging immediately rather than retrying blindly, since a "
            f"crash is much more likely to be a real regression than translation-length noise."
        )
        return state

    if state.get("preflight", {}).get("fired"):
        state["decision"] = "flag"
        state["reasoning"] = (
            "Preflight failed: " + state["preflight"]["detail"] +
            " This is a property of the shipped detection code, not of this page -- "
            "the page was never submitted to the live Space."
        )
        return state

    current_fired = _fired_map(state["check_results"])

    if not current_fired:
        state["decision"] = "pass"
        state["reasoning"] = f"All {len(state['check_results'])} regions passed all checks on attempt {state['attempt']}."
        return state

    high_severity = {k: v for k, v in current_fired.items() if v == "high"}
    attempt = state["attempt"]
    max_attempts = state.get("max_attempts", 2)

    # compare to the previous attempt's fired set, if there was one, to
    # tell "retrying might help" apart from "this box will never fit."
    persisted = {}
    if attempt > 1:
        prev_results = state["attempt_history"][-2]["check_results"]
        prev_fired = _fired_map(prev_results)
        persisted = {k: v for k, v in current_fired.items() if k in prev_fired}

    if persisted:
        state["decision"] = "fallback"
        state["reasoning"] = (
            f"{len(persisted)} region/check pair(s) failed identically on attempts "
            f"{attempt - 1} and {attempt} despite a fresh (non-deterministic) translation call: "
            f"{sorted(persisted.keys())}. Box geometry is deterministic across calls, so a check "
            f"that survives a different translation is very unlikely to be a translation-length "
            f"fluke -- it needs a pipeline-level fix (e.g. placement.py's font-size floor or "
            f"box-expansion cap), not another retry. Falling back rather than burning the "
            f"remaining retry budget."
        )
        return state

    if attempt < max_attempts:
        state["decision"] = "retry"
        state["reasoning"] = (
            f"{len(current_fired)} region/check pair(s) fired on attempt {attempt} "
            f"({sorted(current_fired.keys())}), none of them repeats from a prior attempt yet, "
            f"and {max_attempts - attempt} retr{'y' if max_attempts - attempt == 1 else 'ies'} remain. "
            f"Retrying and betting on translation.py's real non-determinism to produce a "
            f"better-fitting translation."
        )
        return state

    state["decision"] = "fallback" if not high_severity else "flag"
    state["reasoning"] = (
        f"Retry budget ({max_attempts}) exhausted on attempt {attempt} with "
        f"{len(current_fired)} region/check pair(s) still firing "
        f"({'high severity present' if high_severity else 'no high severity'}). "
        f"{'Flagging for human review given high-severity unresolved issues.' if high_severity else 'Falling back.'}"
    )
    return state


def _route_after_preflight(state):
    return "flag" if state.get("preflight", {}).get("fired") else "continue"


def _route_after_decide(state):
    return "retry" if state["decision"] == "retry" else "end"


def build_graph():
    graph = StateGraph(HarnessState)
    graph.add_node("preflight", preflight_node)
    graph.add_node("translate", translate_node)
    graph.add_node("score", score_node)
    graph.add_node("decide", decide_node)

    graph.set_entry_point("preflight")
    graph.add_conditional_edges("preflight", _route_after_preflight, {"flag": END, "continue": "translate"})
    graph.add_edge("translate", "score")
    graph.add_edge("score", "decide")
    graph.add_conditional_edges("decide", _route_after_decide, {"retry": "translate", "end": END})

    return graph.compile()
