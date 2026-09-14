"""
Access into Project 1's real backend code, reused rather than duplicated.

Two different problems are solved here, because two different parts of
Project 1 have two different dependency profiles:

- pipeline_core.py imports only numpy and PIL. It is always safe to import
  directly.
- ocr.py imports paddleocr at module load time (`from paddleocr import
  PaddleOCR`), even though the specific function this harness needs from
  it, `_drop_redundant_oversized_boxes`, never touches paddleocr itself --
  it is a pure function over already-detected box coordinates. Installing
  the real paddleocr/paddlepaddle just to reach a pure function is the same
  class of problem BUG-4 documents (a transitive import dragging in a
  dependency the code never actually needed), and paddleocr's own install
  has already failed once in a sandboxed environment very like this one
  (see the 2026-09-09 career log: manga-ocr/torch install failures on disk
  space). So a minimal stand-in module is installed into sys.modules
  before ocr.py is imported, satisfying the import statement without
  installing the real package. This does not touch or modify ocr.py.
"""

import importlib
import sys
import types


def _find_project1_backend(start_path):
    # anchored on this file's own location, not the caller's cwd, so it
    # works the same whether invoked from Project 2's root, from harness/,
    # or from anywhere else -- Project 2 and "Manga Translator" are sibling
    # folders under the same Projects/ directory.
    import os

    this_dir = os.path.dirname(os.path.abspath(__file__))  # .../Project 2/harness
    candidates = [
        os.path.join(this_dir, "..", "..", "Manga Translator", "backend"),  # sibling of Project 2
        os.path.join(this_dir, "..", "Manga Translator", "backend"),
        os.path.join(start_path, "..", "Manga Translator", "backend"),
        os.path.join(start_path, "Manga Translator", "backend"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "pipeline_core.py")):
            return c
    raise FileNotFoundError(
        "Could not locate Project 1's backend/ directory (looked for pipeline_core.py "
        f"under: {[os.path.abspath(c) for c in candidates]}). Pass the path explicitly."
    )


def get_pipeline_core(project1_backend_path=None):
    """Import the real pipeline_core module. Safe: numpy + PIL only."""
    import os

    path = project1_backend_path or _find_project1_backend(os.getcwd())
    if path not in sys.path:
        sys.path.insert(0, path)
    if "pipeline_core" in sys.modules:
        importlib.reload(sys.modules["pipeline_core"])
    import pipeline_core
    return pipeline_core


def get_ocr_module_with_stub_paddleocr(project1_backend_path=None):
    """
    Import the real ocr.py, with a stand-in paddleocr module installed
    into sys.modules first so `from paddleocr import PaddleOCR` succeeds
    without the real package. The stand-in's PaddleOCR is never
    instantiated by anything this harness calls (_drop_redundant_oversized_boxes,
    _containment_ratio, _box_bounds, _box_area are all pure functions), so a
    dummy class body is sufficient.
    """
    import os

    path = project1_backend_path or _find_project1_backend(os.getcwd())
    if path not in sys.path:
        sys.path.insert(0, path)

    if "paddleocr" not in sys.modules:
        stub = types.ModuleType("paddleocr")

        class _StubPaddleOCR:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "This is the harness's stand-in for paddleocr, installed only so "
                    "ocr.py's top-level import succeeds. It is not a real OCR engine "
                    "and must never be instantiated -- if you see this error, the "
                    "harness is calling something in ocr.py that needs real detection, "
                    "which is out of scope for the pure-function checks this stand-in "
                    "supports."
                )

        stub.PaddleOCR = _StubPaddleOCR
        sys.modules["paddleocr"] = stub

    if "ocr" in sys.modules:
        importlib.reload(sys.modules["ocr"])
    import ocr
    return ocr
