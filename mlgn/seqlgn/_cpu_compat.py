"""
_cpu_compat.py — let `difflogic` import on a CPU-only machine.
==============================================================

The upstream ``difflogic`` package does ``import difflogic_cuda`` at the top of
``difflogic.py`` and ``packbitstensor.py``, so the whole library is unimportable without
the compiled CUDA extension — even though there is a pure-Python CPU path
(``LogicLayer.forward_python``) that never touches CUDA.

``ensure_cpu_importable()`` injects a harmless stub ``difflogic_cuda`` module into
``sys.modules`` (only if the real one is missing) so the import succeeds. The stub's
functions raise a clear error if a CUDA-only code path is actually invoked, so you can
develop and smoke-test on CPU with ``device='cpu'`` / ``implementation='python'`` while
PackBitsTensor and the fused kernels remain (correctly) unavailable.

This is dev convenience only. Real experiments are slow on the Python CPU path — run them
on a GPU machine / cluster. Import this BEFORE importing ``difflogic`` (the seqlgn package
``__init__`` does so automatically).
"""

from __future__ import annotations

import sys
import types


def ensure_cpu_importable() -> bool:
    """Return True if the real CUDA extension is present, False if a stub was injected."""
    if "difflogic_cuda" in sys.modules:
        return not getattr(sys.modules["difflogic_cuda"], "_IS_CPU_STUB", False)

    try:
        import difflogic_cuda  # noqa: F401  (real extension available)
        return True
    except Exception:
        pass

    stub = types.ModuleType("difflogic_cuda")
    stub.__doc__ = "CPU stub injected by mlgn.seqlgn._cpu_compat (real CUDA extension absent)."
    stub._IS_CPU_STUB = True

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "difflogic_cuda (the CUDA extension) is not installed — this is a CPU-only "
            "environment. Only device='cpu' / implementation='python' works here; "
            "PackBitsTensor and the fused CUDA kernels are unavailable. Run real "
            "experiments on a GPU machine."
        )

    for name in ("forward", "backward_x", "backward_w", "eval", "tensor_packbits_cuda"):
        setattr(stub, name, _unavailable)

    sys.modules["difflogic_cuda"] = stub
    return False
