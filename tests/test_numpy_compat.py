"""Regression guard for numpy 2.x compatibility.

The project runs on numpy 2.x (pulled in by torch 2.11). In numpy 2.0 the
scalar-constructor aliases ``np.int`` / ``np.float`` / ``np.bool`` /
``np.object`` / ``np.str`` / ``np.complex`` were *removed* — any reuse
crashes at runtime with ``AttributeError`` (this bit ``main_test_ircnn_-
denoiser.py`` and ``models/model_plain4.py`` before being fixed).

This test scans the source tree and fails if any such deprecated call
reappears, so the break can never silently creep back.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The removed scalar ctors are e.g. np.int / np.float / np.bool / np.object /
# np.str / np.complex (np.long / np.unicode are the py2 leftovers, also gone).
DEPRECATED = re.compile(r"\bnp\.(int|float|bool|object|str|complex|long|unicode)\(")

# Directories that are not project source (tests guards source, not itself).
EXCLUDE_DIRS = {".venv", ".git", ".workbuddy", "node_modules", "__pycache__", "tests"}


def _source_files():
    for p in ROOT.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        yield p


def test_no_deprecated_numpy_scalar_ctors():
    hits = []
    for p in _source_files():
        text = p.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if DEPRECATED.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not hits, (
        "Deprecated numpy 2.x scalar constructors found "
        "(np.int/np.float/np.bool/np.object/np.str/np.complex). "
        "Replace with the builtin int()/float()/bool() or np.int64 etc.:\n"
        + "\n".join(hits)
    )
