"""
Run the harness across every image in a directory, one language at a time
(the API/Gradio path takes one source_lang per call, and filenames don't
reliably encode it, so this stays explicit rather than guessing).

Usage:
    python batch_run.py <directory> <ja|zh> [--max-attempts N]

Writes one per-image log (same as run.py) plus one aggregate summary in
../logs/. Also writes a raw data file recording EVERY check's computed
values for EVERY region, fired or not -- run.py's normal log only records
detail for checks that fired, which is enough to read a single run but not
enough to recalibrate a threshold afterward (this is exactly the gap that
made the 2026-09-09 detection_plausibility recalibration harder than it
needed to be: fired-only values were available, not-fired values were not).
"""

import argparse
import csv
import glob
import os
import sys
from collections import Counter

from graph import build_graph
from logger import write_run_log

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")


def main():
    parser = argparse.ArgumentParser(description="Batch-run the harness over a directory of images")
    parser.add_argument("directory")
    parser.add_argument("source_lang", choices=["ja", "zh"])
    parser.add_argument("--max-attempts", type=int, default=1,
                         help="default 1 -- bulk runs already take a while per image; raise this "
                              "only if you also want the retry path exercised on every image")
    parser.add_argument("--space", default="NerdCoderZero/manga-translator")
    args = parser.parse_args()

    images = sorted(
        p for p in glob.glob(os.path.join(args.directory, "*"))
        if p.lower().endswith(IMAGE_EXTS)
    )
    if not images:
        print(f"No images found in {args.directory} (looked for {IMAGE_EXTS})")
        sys.exit(1)

    print(f"Found {len(images)} image(s): {[os.path.basename(i) for i in images]}")

    app = build_graph()
    decisions = Counter()
    check_fires = Counter()  # (check_name, severity) -> count
    raw_rows = []  # every region x check, fired or not -- for future recalibration

    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    for image_path in images:
        name = os.path.basename(image_path)
        print(f"\n--- {name} ---")
        try:
            final_state = app.invoke({
                "input_image": image_path,
                "source_lang": args.source_lang,
                "space_name": args.space,
                "max_attempts": args.max_attempts,
                "attempt": 0,
            })
        except Exception as e:
            print(f"  batch-level failure (outside the harness's own crash handling): {e}")
            decisions["BATCH_ERROR"] += 1
            continue

        log_path = write_run_log(final_state)
        decisions[final_state["decision"]] += 1
        print(f"  decision: {final_state['decision']}  |  log: {log_path}")

        for h in final_state.get("attempt_history", []):
            for region in h.get("check_results", []):
                for c in region["checks"]:
                    if c["fired"]:
                        check_fires[(c["name"], c["severity"])] += 1
                    raw_rows.append({
                        "image": name,
                        "attempt": h["attempt"],
                        "bounds": region["bounds"],
                        "check": c["name"],
                        "fired": c["fired"],
                        "severity": c["severity"],
                        "values": c.get("values", {}),
                    })

    ts = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H%M%S")
    summary_path = os.path.join(log_dir, f"batch_summary_{ts}_{args.source_lang}.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Batch run: {len(images)} image(s), source_lang={args.source_lang}\n\n")
        f.write(f"Images: {[os.path.basename(i) for i in images]}\n\n")
        f.write("## Decisions\n\n")
        for decision, count in decisions.most_common():
            f.write(f"- {decision}: {count}\n")
        f.write("\n## Check fires (name, severity) -> count\n\n")
        for (name, severity), count in check_fires.most_common():
            f.write(f"- {name} ({severity}): {count}\n")

    raw_path = os.path.join(log_dir, f"batch_raw_{ts}_{args.source_lang}.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "attempt", "bounds", "check", "fired", "severity", "values"])
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)

    print(f"\n=== Batch summary: {summary_path}")
    print(f"=== Raw per-region data (fired AND not-fired): {raw_path}")
    print(f"Decisions: {dict(decisions)}")


if __name__ == "__main__":
    main()
