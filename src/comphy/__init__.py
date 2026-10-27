"""Shared helpers for the Computational Physics exercises.

The notebooks in ``notebooks/`` import from this package so that code used in
several exercises (plotting, training loops, potentials, graph utilities,
physics-informed networks) lives in one place.
"""

from pathlib import Path

__version__ = "1.0.0"

#: Repository root (``src/comphy`` -> ``src`` -> root).
ROOT_DIR = Path(__file__).resolve().parents[2]
#: Folder with the datasets handed out in the course.
DATA_DIR = ROOT_DIR / "data"
