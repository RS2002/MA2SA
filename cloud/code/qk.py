"""Final square-L2 query/key head used by both training phases.

Extracted from this project's UPMSP adaptation, ``upmsp_core/networks.py``.
Only the square-L2 path used by the released Cloud method is included.
Keys are squared and L2-normalized; queries retain their magnitude.
"""
from __future__ import annotations

import torch


def normalized_square(logits):
    """Unit-norm squared keys with an explicit undefined-direction failure.

    Scaling each nonzero vector by its maximum absolute entry before squaring
    is algebraically equivalent to logits.square()/||logits.square()||_2.
    It avoids squaring extreme finite logits directly. No epsilon or fallback
    changes the direction: an all-zero vector is an experimental failure.
    Near zero, the true derivative can still be large despite finite output.
    """
    if not torch.isfinite(logits).all():
        raise FloatingPointError('square_l2 key logits must be finite')
    scale = logits.abs().amax(dim=-1, keepdim=True)
    if (scale == 0).any():
        raise FloatingPointError('square_l2 is undefined for an all-zero key vector')
    squared = (logits / scale).square()
    keys = squared / squared.norm(p=2, dim=-1, keepdim=True)
    if not torch.isfinite(keys).all():
        raise FloatingPointError('square_l2 normalized keys must be finite')
    return keys



def qk_scores(queries, key_logits, mode='square_l2'):
    """Compute raw QK scores, normalizing only squared keys."""
    if mode != 'square_l2':
        raise ValueError('This release supports only the final square_l2 QK mode')
    if not torch.isfinite(queries).all():
        raise FloatingPointError('square_l2 queries must be finite')
    keys = normalized_square(key_logits)
    scores = queries @ keys.transpose(0, 1)
    if not torch.isfinite(scores).all():
        raise FloatingPointError('square_l2 scores must be finite')
    return scores
