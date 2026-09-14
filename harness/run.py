"""
CLI entrypoint. Usage:

    python run.py <image_path> <ja|zh> [--max-attempts N] [--space NAME]

Runs the full preflight -> translate -> score -> decide loop against the
real, live pipeline (the deployed HF Space by default) and writes a
human-readable log to ../logs/.
"""

import argparse
import sys

from graph import build_graph
from logger import write_run_log


def main():
    parser = argparse.ArgumentParser(description="Manga translator evaluation harness")
    parser.add_argument("image_path")
    parser.add_argument("source_lang", choices=["ja", "zh"])
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--space", default="NerdCoderZero/manga-translator")
    args = parser.parse_args()

    app = build_graph()
    final_state = app.invoke({
        "input_image": args.image_path,
        "source_lang": args.source_lang,
        "space_name": args.space,
        "max_attempts": args.max_attempts,
        "attempt": 0,
    })

    log_path = write_run_log(final_state)

    print(f"Decision: {final_state['decision']}")
    print(f"Reasoning: {final_state['reasoning']}")
    print(f"Attempts: {final_state['attempt']}")
    print(f"Log written to: {log_path}")

    sys.exit(0 if final_state["decision"] in ("pass",) else 1)


if __name__ == "__main__":
    main()
