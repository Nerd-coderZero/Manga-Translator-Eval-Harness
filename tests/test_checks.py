"""
Regression tests. Plain asserts, run directly with `python tests/test_checks.py`
from the Project 2 root -- no pytest dependency required, kept intentionally
light. Every case here traces to a real, named source: a real fixture pulled
from the live Space, or BUG-1's own recorded verification numbers -- not
invented data.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))

import checks


def load_fixture(name):
    path = os.path.join(os.path.dirname(__file__), "..", "fixtures", name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_lim1_manager_case_fires():
    # real region pulled live from NerdCoderZero/manga-translator on 239.webp,
    # 2026-09-09 -- the exact "Manager!" -> M/an/ag/er! case LIM-1 describes.
    region = {"bounds": [1047, 102, 1102, 216], "source_text": "店長！", "translation": "Manager!"}
    result = checks.check_text_fit(region)
    assert result["fired"], "expected the real LIM-1 Manager! case to fire text_fit"
    print("PASS: LIM-1 Manager! case fires text_fit")


def test_lim1_tiny_box_fires_high():
    # real region, same pull: a 16x10px box -- too small for any legible
    # text regardless of translation length, unlike the length-driven cases.
    region = {"bounds": [601, 301, 617, 311], "source_text": "いい！", "translation": "Nice!"}
    result = checks.check_text_fit(region)
    assert result["fired"] and result["severity"] == "high", \
        "expected the 16x10px real region to fire text_fit at high severity"
    print("PASS: real 16x10px region fires text_fit at high severity")


def test_normal_region_does_not_fire():
    # real region, same pull, plenty of room -- must not false-positive.
    region = {"bounds": [153, 792, 353, 1122], "source_text": "試してみたいでしょうッ！？",
              "translation": "You wanna try it, huh?!"}
    result = checks.check_text_fit(region)
    assert not result["fired"], "expected a well-fitting real region not to fire"
    print("PASS: well-fitting real region does not fire")


def test_short_word_hard_wrap_does_not_fire():
    # regression for the tuning decision made 2026-09-09: short words
    # splitting (e.g. "So" -> S/o) must NOT fire after option B.
    region = {"bounds": [1194, 1547, 1245, 1628], "source_text": "だから", "translation": "So"}
    result = checks.check_text_fit(region)
    assert not result["fired"], "short 2-letter word split should not fire after the option-B tuning"
    print("PASS: short-word hard-wrap correctly suppressed")


def test_translation_sanity_empty():
    region = {"source_text": "店長！", "translation": ""}
    result = checks.check_translation_sanity(region)
    assert result["fired"] and result["severity"] == "high"
    print("PASS: empty translation fires translation_sanity")


def test_translation_sanity_passthrough():
    region = {"source_text": "店長！", "translation": "店長！"}
    result = checks.check_translation_sanity(region)
    assert result["fired"]
    print("PASS: passthrough (untranslated) fires translation_sanity")


def test_translation_sanity_garbage_source_is_not_a_defect():
    region = {"source_text": "0000000000000000", "translation": ""}
    result = checks.check_translation_sanity(region)
    assert not result["fired"], "garbage source with empty translation is expected, not a defect"
    print("PASS: garbage source + empty translation correctly not flagged")


def test_short_garbage_ocr_evades_is_garbage_text_but_translation_sanity_catches_it():
    # real region from 09.webp (zh), 2026-09-09 batch run (batch_raw_2026-09-09_213902_zh.csv,
    # bounds [42, 1326, 309, 1748]): OCR read a noise/effects mark as 'CSB', which is not real
    # Chinese dialogue. is_garbage_text only screens repeated-character or digit-heavy strings,
    # and only once a string is 6-8+ characters long -- 'CSB' is 3 characters, so it is
    # structurally invisible to that filter. This locks in that documented gap: the pipeline's
    # own garbage filter does not catch this class of short OCR noise, and translation_sanity's
    # identical-passthrough check is what actually catches it.
    from pipeline_access import get_pipeline_core
    core = get_pipeline_core()
    assert not core.is_garbage_text("CSB"), (
        "if this starts failing, is_garbage_text has been extended to catch short strings -- "
        "the gap this test documents may have been closed; re-check whether the "
        "translation_sanity fire below still needs to carry this weight"
    )
    region = {"source_text": "CSB", "translation": "CSB"}
    result = checks.check_translation_sanity(region)
    assert result["fired"] and result["severity"] == "high", \
        "expected the real short-garbage-OCR passthrough case to fire translation_sanity high"
    print("PASS: short garbage-OCR token ('CSB') evades is_garbage_text but translation_sanity still catches the passthrough")


def test_nim_meta_commentary_leak_on_untranslatable_particle():
    # real region from 08.webp (ja), 2026-09-09 batch run (run_2026-09-09_203133_08.webp.md),
    # bounds [0, 1795, 20, 1824]: source_text is the single particle "と", which has no
    # standalone translatable meaning. Instead of an empty translation or a real one, NIM
    # returned a 139-character English sentence explaining that the particle isn't
    # translatable. This is a previously undocumented failure mode (not LIM-1, not BUG-1
    # through BUG-5): the translation call has no guardrail against the model returning
    # conversational meta-commentary instead of a translation. Both checks below fired on
    # the real region, from two different angles -- this fixture locks in both.
    region = {
        "bounds": [0, 1795, 20, 1824],
        "source_text": "\u3068",
        "translation": (
            '(No output - the particle "\u3068" alone carries no translatable meaning in this '
            "context and would not be rendered in natural English dialogue.)"
        ),
    }
    assert len(region["translation"]) == 139, "transcription check -- must match the real logged string exactly"

    fit_result = checks.check_text_fit(region)
    assert fit_result["fired"] and fit_result["severity"] == "high", \
        "expected the real meta-commentary text to fail to fit the real 20x29px region"

    sanity_result = checks.check_translation_sanity(region)
    assert sanity_result["fired"] and sanity_result["severity"] == "medium", \
        "expected the real meta-commentary text to fire translation_sanity's implausible-length rule"
    print("PASS: NIM meta-commentary leak on untranslatable particle fires both text_fit (high) and translation_sanity (medium)")


def test_nim_content_policy_refusal_on_explicit_dialogue():
    # STAND-IN FIXTURE. The real case this documents: 29.webp (zh), 2026-09-12 batch run
    # (batch_raw_2026-09-12_033354_zh.csv, bounds [1017, 906, 1183, 1304]) had genuine,
    # coherent, explicit Chinese dialogue as source_text (40 chars, confirmed not garbage),
    # and NIM returned a content-policy refusal instead of a translation (330 chars, ratio
    # 8.25x). That real explicit source text and NIM's exact refusal wording are deliberately
    # NOT reproduced here -- 2026-09-13 review found the real strings stored verbatim (the
    # source as escaped unicode, the refusal as plain text) in this file, and confirmed
    # check_translation_sanity has no content-specific logic at all: it only compares string
    # lengths (is_garbage_text, an empty-translation check, an equality check, and a
    # length-ratio threshold). None of that requires the literal explicit text or the literal
    # real refusal wording -- any source/translation pair with the same length relationship
    # trips the identical code path. This placeholder pair is picked to match that
    # relationship (non-garbage source, translation over 6x/60-char) without storing explicit
    # content or the real service's exact wording in the test suite.
    from pipeline_access import get_pipeline_core
    core = get_pipeline_core()
    source_text = "\u8def\u6613\u65af\u5927\u4eba\u8bf4\u8fd9\u662f\u4e24\u4eba\u4e4b\u95f4\u7684\u79d8\u5bc6"  # placeholder: "Lord Louis said this is a secret between the two of them" -- ordinary, non-explicit dialogue
    assert not core.is_garbage_text(source_text), (
        "this source text is real dialogue, not OCR noise -- distinguishes this fixture "
        "from the short-garbage-OCR-passthrough cases above"
    )
    region = {
        "source_text": source_text,
        "translation": (
            "This region could not be translated because the translation service declined "
            "the request under its own content policy, rather than returning an empty or "
            "untranslated result."
        ),
    }
    assert len(region["translation"]) > len(region["source_text"]) * 6 and len(region["translation"]) > 60, (
        "placeholder must reproduce the real case's length relationship (translation over "
        "6x source length and over 60 chars) for this fixture to mean anything"
    )

    result = checks.check_translation_sanity(region)
    assert result["fired"] and result["severity"] == "medium", \
        "expected the content-policy-refusal-shaped placeholder to fire translation_sanity's implausible-length rule"
    print("PASS: NIM content-policy refusal on real explicit dialogue fires translation_sanity (medium) -- stand-in fixture")


def test_bug1_regression_fixture():
    fixture = load_fixture("bug1_regression_case.json")
    result = checks.check_duplicate_box_regression(fixture)
    assert not result["fired"], f"BUG-1 filter regressed: {result['detail']}"
    assert result["values"]["dropped"] == ["box14_real"]
    assert set(result["values"]["kept"]) == {"box35_real", "inner1_synthetic", "inner2_synthetic", "inner3_synthetic"}
    print("PASS: BUG-1 regression fixture -- real dropped/kept boxes match recorded verification")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
