"""
seqlgn — Sequential / recurrent Logic Gate Networks.
====================================================

Research infrastructure for malcolm's recurrent-LGN program (built on the ``difflogic``
fork). Shared by Paper #1 (logic-native gating) and Paper #2 (latch primitives, parked).

Public API:
    from mlgn.seqlgn.cells import LogicRecurrentCell, LogicMLP, MECHANISMS
    from mlgn.seqlgn.models import SequenceClassifier
    from mlgn.seqlgn.data import get_task, AVAILABLE_TASKS

See README.md and docs/ for the design, benchmarks, experiment protocol, and difflogic
API notes.
"""

# Make `difflogic` importable on CPU-only machines (no-op if the CUDA extension exists).
# Must run BEFORE any `import difflogic`.
from ._cpu_compat import ensure_cpu_importable as _ensure_cpu_importable
_HAS_CUDA_EXT = _ensure_cpu_importable()

from .cells import LogicRecurrentCell, LogicMLP, MECHANISMS
from .models import SequenceClassifier
from .data import get_task, AVAILABLE_TASKS

__all__ = [
    "LogicRecurrentCell",
    "LogicMLP",
    "MECHANISMS",
    "SequenceClassifier",
    "get_task",
    "AVAILABLE_TASKS",
]
