"""
The four quality checks. Every check takes a plain dict `region` with at
least: bounds [x1,y1,x2,y2], source_text, translation -- exactly the shape
runner.normalise_result() already returns via main.py's /api/translate and
app.py's Gradio path. No check here requires anything the real API does
not already expose; where a check needs something the API does not expose
(recognition confidence), that is called out explicitly rather than
silently worked around.

Each check returns a dict: {"name", "fired", "severity", "detail", "values"}.
"fired" means the check flagged a problem. "values" carries the raw
numbers, not just the verdict, because the run log is meant to be read by
a human deciding whether a "pass" was a comfortable one or a near miss.
"""

from pipeline_access import get_pipeline_core


# ---------------------------------------------------------------------------
# Check 1: does the translation actually fit inside the ORIGINAL detected
# region at a legible size? This targets LIM-1 (Japanese vertical regions
# sized from the wrong, full-region avg_original_height) and the documented
# Chinese-path equivalent, at the level of the actual practical outcome
# rather than by inspecting internal fields the API never returns anyway.
#
# It reuses pipeline_core.fit_text_in_box/wrap_text_to_fit directly -- the
# real function placement.py itself calls -- so this check can never
# silently drift from how the pipeline actually measures text. It is run
# with a purely geometric ceiling (min(box_width, box_height)), not
# placement.py's target_font_size, because that value is exactly the field
# LIM-1 shows cannot be trusted; this check asks "could this box ever have
# held this text legibly," independent of whatever font size the pipeline
# actually chose.
# ---------------------------------------------------------------------------

def check_text_fit(region, project1_backend_path=None, min_acceptable_font_size=10):
    from PIL import Image, ImageDraw

    core = get_pipeline_core(project1_backend_path)
    font_path = core.find_available_font()

    x1, y1, x2, y2 = region["bounds"]
    box_width, box_height = x2 - x1, y2 - y1
    translated = region.get("translation", "") or ""

    result = {
        "name": "text_fit",
        "fired": False,
        "severity": "none",
        "detail": "",
        "values": {"box_width": box_width, "box_height": box_height},
    }

    if not translated:
        result["detail"] = "no translation to place; not this check's concern (see translation_sanity)"
        return result

    ceiling_font_size = max(min(box_width, box_height), min_acceptable_font_size)

    canvas = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(canvas)

    font, lines, line_heights, line_spacing = core.fit_text_in_box(
        draw, translated, box_width, box_height, font_path,
        ceiling_font_size, min_acceptable_font_size,
    )

    max_line_width = max(
        (draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in lines),
        default=0,
    )
    total_height = sum(line_heights) + line_spacing * (len(lines) - 1) if lines else 0

    fits = (
        font.size >= min_acceptable_font_size
        and max_line_width <= box_width
        and total_height <= box_height
    )

    # mid-word hard-wrap without a hyphen: the "Stil / l" artifact class.
    # wrap_text_to_fit falls back to character-by-character wrapping only
    # when the source string has no spaces for it to break on. Tuned
    # against real output on 2026-09-09: firing on ANY multi-line split
    # (including "So" -> ['S','o'], "..." -> ['..','.']) was too noisy --
    # those are cosmetically harmless, not "Stil/l"-grade. Scoped to real
    # words (>=4 letters) breaking into 3+ pieces, which is what the
    # actual documented artifact looked like.
    stripped = translated.strip()
    is_single_token = " " not in stripped
    is_real_word = len(stripped) >= 4
    hard_wrapped = (
        is_single_token
        and is_real_word
        and len(lines) >= 3
        and not any(l.endswith("-") for l in lines[:-1])
    )

    result["values"].update({
        "chosen_font_size": font.size,
        "ceiling_font_size": ceiling_font_size,
        "line_count": len(lines),
        "lines": lines,
        "max_line_width": max_line_width,
        "total_text_height": total_height,
        "width_overflow_px": max(0, max_line_width - box_width),
        "height_overflow_px": max(0, total_height - box_height),
        "box_aspect_h_over_w": round(box_height / box_width, 2) if box_width else None,
        "hard_wrapped_single_word": hard_wrapped,
    })

    if not fits:
        result["fired"] = True
        result["severity"] = "high" if font.size <= min_acceptable_font_size else "medium"
        result["detail"] = (
            f"'{translated}' does not fit legibly in the original {box_width}x{box_height}px region "
            f"even at font size {font.size} (min allowed {min_acceptable_font_size}): "
            f"needs {max_line_width}x{total_height}px. Placement.py would have to expand this box "
            f"beyond its detected bounds or accept an illegible size -- this is LIM-1's signature."
        )
    elif hard_wrapped:
        result["fired"] = True
        result["severity"] = "low"
        result["detail"] = (
            f"'{translated}' fits by area but got split across {len(lines)} lines with no spaces "
            f"and no hyphen ({lines}) -- the mid-word hard-wrap pattern from the 2026-09-09 "
            f"'Stil/l' artifact, not the oversized-font pattern LIM-1 describes."
        )
    else:
        result["detail"] = f"fits at font size {font.size} within the original bounds."

    return result


# ---------------------------------------------------------------------------
# Check 3: translation length/shape sanity. Catches empty output, silent
# untranslated passthrough, and implausible length ratios. The "implausible"
# threshold below is not invented for this harness -- it is the exact
# max_chars_per_source_char=6 rule placement.py already enforces server-side
# (erase_and_paste_text skips a box outright above this, reason
# "implausible_length"), so a fire here predicts what the real pipeline's
# own skipped_boxes would already show for the same box.
# ---------------------------------------------------------------------------

def check_translation_sanity(region, project1_backend_path=None,
                              max_chars_per_source_char=6, min_chars_absolute=60):
    core = get_pipeline_core(project1_backend_path)
    source = region.get("source_text", "") or ""
    translated = region.get("translation", "") or ""

    result = {
        "name": "translation_sanity",
        "fired": False,
        "severity": "none",
        "detail": "",
        "values": {"source_text": source, "translation": translated,
                   "source_len": len(source), "translation_len": len(translated)},
    }

    if core.is_garbage_text(source):
        result["detail"] = "source_text is classified as garbage by the pipeline's own is_garbage_text; an empty translation here is expected, not a defect."
        return result

    if not translated.strip():
        result["fired"] = True
        result["severity"] = "high"
        result["detail"] = "translation is empty for non-garbage source text -- likely the NIM translation call exhausted its retries (LIM-3-adjacent) and fell back to '' silently."
        return result

    if translated.strip() == source.strip():
        result["fired"] = True
        result["severity"] = "high"
        result["detail"] = "translation is identical to source_text -- looks like an untranslated passthrough rather than a real translation."
        return result

    source_len = max(len(source), 1)
    implausibly_long = len(translated) > source_len * max_chars_per_source_char and len(translated) > min_chars_absolute
    result["values"]["ratio"] = round(len(translated) / source_len, 2)

    if implausibly_long:
        result["fired"] = True
        result["severity"] = "medium"
        result["detail"] = (
            f"translation is {len(translated)} chars for a {source_len}-char source, "
            f"over placement.py's own {max_chars_per_source_char}x/{min_chars_absolute}-char "
            f"implausible-length rule -- this is the exact condition erase_and_paste_text "
            f"already uses to skip a box as 'implausible_length'."
        )
    else:
        result["detail"] = f"length ratio {result['values']['ratio']}x looks plausible."

    return result


# ---------------------------------------------------------------------------
# Check 2: BUG-1 regression tripwire. Calls the REAL
# _drop_redundant_oversized_boxes from ocr.py (via pipeline_access's
# paddleocr stand-in) against a fixture built from BUG-1's own recorded
# real coordinates plus a minimal synthetic scaffold -- see
# fixtures/bug1_regression_case.json for exactly what is real vs
# scaffolded and why. This does not test today's detection; it tests
# whether the shipped filter still behaves the way BUG-1 verified it did.
# ---------------------------------------------------------------------------

def check_duplicate_box_regression(fixture, project1_backend_path=None):
    from pipeline_access import get_ocr_module_with_stub_paddleocr

    ocr = get_ocr_module_with_stub_paddleocr(project1_backend_path)

    id_by_bounds = {tuple(b["bounds"]): b["id"] for b in fixture["boxes"]}
    # _drop_redundant_oversized_boxes expects polygon coords (a list of
    # (x, y) points), since it calls _box_bounds(coords) which does
    # min/max over p[0]/p[1]. a rectangle is the four corners.
    dt_boxes = []
    for b in fixture["boxes"]:
        x1, y1, x2, y2 = b["bounds"]
        dt_boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    kept = ocr._drop_redundant_oversized_boxes(dt_boxes)
    kept_ids = set()
    for coords in kept:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        kept_ids.add(id_by_bounds.get(bounds, f"UNKNOWN{bounds}"))

    all_ids = set(id_by_bounds.values())
    dropped_ids = all_ids - kept_ids

    expected_dropped = set(fixture["expected_dropped"])
    expected_kept = set(fixture["expected_kept"])

    matches_expected = dropped_ids == expected_dropped and kept_ids == expected_kept

    result = {
        "name": "duplicate_box_regression",
        "fired": not matches_expected,
        "severity": "none" if matches_expected else "high",
        "detail": "",
        "values": {
            "dropped": sorted(dropped_ids),
            "kept": sorted(kept_ids),
            "expected_dropped": sorted(expected_dropped),
            "expected_kept": sorted(expected_kept),
        },
    }
    if matches_expected:
        result["detail"] = "the shipped BUG-1 filter still drops the real oversized-duplicate box and keeps the real legitimate outlier, matching BUG-1's own recorded verification."
    else:
        result["detail"] = (
            f"regression: expected to drop {sorted(expected_dropped)} and keep {sorted(expected_kept)}, "
            f"actually dropped {sorted(dropped_ids)} and kept {sorted(kept_ids)}. "
            f"Someone changed _drop_redundant_oversized_boxes's behavior since BUG-1 was fixed."
        )
    return result


# ---------------------------------------------------------------------------
# Check 4: detection plausibility (v1 scope).
#
# What this is NOT: a confidence-based check. PaddleOCR's rec_scores are
# computed inside ocr.py/detect_and_read_text but pipeline.py never
# attaches them to merged_boxes, so runner.normalise_result() never
# returns them, so the harness (calling through the real API/Gradio path,
# same as any real caller) has no confidence numbers to check at all.
# Making a real confidence-based check would need a small additive change
# to pipeline.py/runner.py to thread rec_scores through -- flagged
# separately for a decision, not made silently here.
#
# What this check CAN do with only bounds + source_text, which the API
# already gives us: flag boxes whose shape/text-density combination looks
# like a plausible detection failure rather than a real dialogue region --
# e.g. a box with an aspect ratio and text length that don't correspond to
# how manga text is actually laid out. This is a weak proxy, disclosed as
# such, not a substitute for ground-truth detection accuracy (which this
# project has never had -- no page has ever been hand-labeled with correct
# boxes to compare against).
# ---------------------------------------------------------------------------

def check_detection_plausibility(region, min_box_dimension_px=4, dense_chars_per_100px_height=25.0,
                                  low_confidence_thresh=0.5):
    # dense_chars_per_100px_height was originally 12.0 (3.0 * 4), picked
    # without real data. Recalibrated 2026-09-09 against the first real
    # multi-page evidence: 14 real fired-check values across 4 real pages
    # (two different mangas) ranged 14.9-30.0, median 16.55, and every
    # value below 20 turned out, on inspection of the actual source/
    # translation text, to be completely ordinary dense dialogue -- not a
    # detection problem. Only one real 30.0 case (the 16x10px "Nice!" box,
    # 3 chars in a 10px-tall region) looked like a genuine outlier. 25.0
    # sits above the ordinary-dialogue cluster and below that outlier.
    # Small-sample recalibration (n=14, fired cases only -- values for
    # non-fired regions were never persisted, a real logger gap), not a
    # statistically rigorous one.
    x1, y1, x2, y2 = region["bounds"]
    box_width, box_height = x2 - x1, y2 - y1
    source_len = len(region.get("source_text", "") or "")
    confidence = region.get("confidence")  # only present once task #11 lands; None until then

    result = {
        "name": "detection_plausibility",
        "fired": False,
        "severity": "none",
        "detail": "",
        "values": {"box_width": box_width, "box_height": box_height, "source_len": source_len,
                   "confidence": confidence},
        "scope_note": (
            "real confidence signal used when present (currently zh-only, once task #11 lands); "
            "falls back to a bounds/text proxy otherwise -- not a ground-truth detection "
            "accuracy check either way. See this function's docstring for why."
        ),
    }

    if confidence is not None and confidence < low_confidence_thresh:
        result["fired"] = True
        result["severity"] = "medium"
        result["detail"] = (
            f"real recognition confidence {confidence:.2f} is below {low_confidence_thresh} -- "
            f"treat any passing checks on this region with less certainty, per the caveat-logging "
            f"approach discussed for the OCR-accuracy concern."
        )
        return result

    if box_width < min_box_dimension_px or box_height < min_box_dimension_px:
        result["fired"] = True
        result["severity"] = "medium"
        result["detail"] = f"region is {box_width}x{box_height}px, suspiciously small for a real detection -- possible noise box."
        return result

    if source_len > 0:
        chars_per_100px_height = source_len / max(box_height, 1) * 100
        result["values"]["chars_per_100px_height"] = round(chars_per_100px_height, 2)
        if chars_per_100px_height > dense_chars_per_100px_height:
            result["fired"] = True
            result["severity"] = "low"
            result["detail"] = (
                f"{source_len} source characters packed into a {box_height}px-tall region "
                f"({chars_per_100px_height:.1f} chars/100px) -- denser than typical manga text, "
                f"worth a look at whether detection under-segmented multiple lines into one box."
            )
            return result

    result["detail"] = "box shape and text density look ordinary; this is a weak proxy and a pass here says nothing about geometric correctness."
    return result
