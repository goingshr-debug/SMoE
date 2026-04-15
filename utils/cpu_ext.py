"""
cpu_ext.py — JIT loader for the cpu_experts_ext C++ extension.

Compiles once on first call via torch.utils.cpp_extension.load(), then caches
the module for all subsequent calls.  The compiled .so is stored in a
build/ directory next to csrc/.
"""

import os
import torch
from torch.utils.cpp_extension import load

_ext = None

def get_cpu_ext():
    """
    Returns the compiled cpu_experts_ext module, building it on first call.

    The module exposes:
        silu_mlp_batch_forward(tokens_list, gate_ws, up_ws, down_ws)
            → List[Tensor]  (see csrc/cpu_experts_ext.cpp for full docstring)
    """
    global _ext
    if _ext is not None:
        return _ext

    src_dir  = os.path.join(os.path.dirname(__file__), '..', 'csrc')
    src_file = os.path.join(src_dir, 'cpu_experts_ext.cpp')
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'cpu_experts_ext')
    os.makedirs(build_dir, exist_ok=True)

    _ext = load(
        name='cpu_experts_ext',
        sources=[src_file],
        build_directory=build_dir,
        extra_cflags=[
            '-O3',
            '-march=native',   # enables AVX2/AVX-512 on the build machine
            '-ffast-math',     # allow reassociation for dot-product vectorisation
            '-fopenmp',
        ],
        extra_ldflags=['-fopenmp'],
        verbose=False,
    )
    return _ext
