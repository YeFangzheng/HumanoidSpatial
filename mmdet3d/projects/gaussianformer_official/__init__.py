"""Glue code to train the upstream GaussianFormer repo (OpenOcc / mmseg BEVSegmentor).

Import this module (or call :func:`ensure_registered`) *before* building OpenOcc datasets
so that custom datasets register into the official ``OPENOCC_DATASET`` registry.
"""

from __future__ import annotations

import os
import sys

__all__ = [
    'ensure_registered',
    'run_official_gaussianformer_training',
    'run_official_gaussianformer_raymetric_test',
]


def ensure_registered(gf_root: str | None = None) -> str:
    """Put official GaussianFormer on ``sys.path`` and register XHumanoid OpenOcc extensions.

    Args:
        gf_root: Path to the cloned GaussianFormer repository. If None, uses env
            ``GAUSSIANFORMER_ROOT`` or the default path used in this workspace.

    Returns:
        Resolved absolute path to the official repository root.
    """
    _default_gf = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'GaussianFormer'),
    )
    root = gf_root or os.environ.get('GAUSSIANFORMER_ROOT', _default_gf)
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f'GAUSSIANFORMER_ROOT is not a directory: {root}. '
            'Set env GAUSSIANFORMER_ROOT to your clone of GaussianFormer.',
        )
    if root not in sys.path:
        sys.path.insert(0, root)

    # Register OpenOcc dataset + transforms (imports official ``dataset`` registry).
    from mmdet3d.projects.gaussianformer_official import xhumanoid_openocc  # noqa: F401

    return root


# 延迟导入以避免循环导入问题
def run_official_gaussianformer_training(*args, **kwargs):
    """Wrapper to avoid circular import."""
    from .train import run_official_gaussianformer_training as _func
    return _func(*args, **kwargs)


def run_official_gaussianformer_raymetric_test(*args, **kwargs):
    """Wrapper to avoid circular import."""
    from .raymetric_eval import run_official_gaussianformer_raymetric_test as _func
    return _func(*args, **kwargs)
