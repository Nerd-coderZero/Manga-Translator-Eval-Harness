"""
Writes one human-readable log per harness run. Plain text/markdown, meant
to be read without re-running any code -- per the original requirement,
this log is itself the evaluation-harness evidence, so it has to stand on
its own for someone skimming it cold.
"""

import datetime
import os


def write_run_log(final_state, log_dir=None):
    log_dir = log_dir or os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    input_name = os.path.basename(final_state.get("input_image", "unknown"))
    path = os.path.join(log_dir, f"run_{ts}_{input_name}.md")

    lines = []
    lines.append(f"# Harness run: {input_name} ({final_state.get('source_lang', '?')})")
    lines.append(f"\nStarted: {ts}")
    lines.append(f"\nFinal decision: **{final_state.get('decision', 'unknown').upper()}**")
    lines.append(f"\nReasoning: {final_state.get('reasoning', '')}")

    preflight = final_state.get("preflight")
    if preflight:
        lines.append("\n## Preflight (BUG-1 duplicate-box regression)")
        lines.append(f"\nFired: {preflight['fired']}")
        lines.append(f"\n{preflight['detail']}")
        if preflight["fired"]:
            lines.append(f"\nValues: {preflight['values']}")

    for h in final_state.get("attempt_history", []):
        lines.append(f"\n## Attempt {h['attempt']}")
        lines.append(f"\nPipeline status: {h.get('status', '?')}")
        lines.append(f"\n{len(h.get('regions', []))} regions returned.\n")

        any_fired = False
        for region in h.get("check_results", []):
            fired_here = [c for c in region["checks"] if c["fired"]]
            if not fired_here:
                continue
            any_fired = True
            lines.append(
                f"\n- Region {region['bounds']}: `{region['source_text']}` -> `{region['translation']}`"
            )
            for c in fired_here:
                lines.append(f"  - **{c['name']}** ({c['severity']}): {c['detail']}")
                if c.get("values"):
                    lines.append(f"    - values: {c['values']}")

        if not any_fired:
            lines.append("\nNo checks fired on this attempt.")

    lines.append(f"\n---\nTotal attempts: {final_state.get('attempt', 0)}")

    content = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
