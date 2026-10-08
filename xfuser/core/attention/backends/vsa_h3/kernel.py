"""VSA-H3 on any of its kernels.

All select the same key tiles and differ only in how they read them, so
everything around the call is shared. Nothing is permuted here: the padded tile
buffers FlexAttention and AITER need are built inside the callee, which keeps
the gate and the compression branch out of tile order entirely.
"""

import logging

from xfuser.core.attention.backends.sdpa.kernel import sdpa
from xfuser.core.attention.requirements import resolve
from xfuser.core.attention.spec import AttnCall
from xfuser.logger import init_logger, log_once

from . import aiter_kernel
from . import attention as h3

logger = init_logger(__name__)

# Calls without VSA-H3 metadata run dense. Unlike AITER_VSA, this backend does
# not require AITER -- it runs on CUDA too -- so aiter/kernel.py, which imports
# aiter at module level, cannot be imported outright here.
_DENSE = resolve("xfuser.core.attention.backends.aiter.kernel:aiter_attention") or sdpa


def _vsa_h3(query, key, value, call: AttnCall, *, kernel: str, recipe=None):
    """USP has already gathered the sequence; the compression gate rides the
    same exchange. Without that metadata -- the MiniMax-H3 token refiner --
    this runs dense, the way VSA does without thw.

    ``kernel`` is "flex", "triton" or "aiter", and ``recipe`` names the mha_v4
    row for the last."""
    kwargs = call.attention_kwargs
    metadata = kwargs.get("vsa_h3_metadata")
    gate = kwargs.get("vsa_h3_gate")
    if metadata is None or gate is None:
        return _DENSE(query, key, value, call)

    if kernel == "triton" and not h3.h3_vsa_triton_is_usable(query.device):
        log_once(
            logger,
            ("vsa_h3_triton", str(query.device)),
            f"TRITON_VSA_H3 cannot run its kernel on {query.device}, falling "
            f"back to the FlexAttention path. Select FLEX_VSA_H3 to ask for "
            f"it directly.",
            level=logging.WARNING,
        )
        kernel = "flex"
    if kernel == "aiter" and not aiter_kernel.vsa_h3_aiter_row_available(recipe):
        # Per recipe rather than once overall because the answer is per recipe:
        # a device can serve one of the two and not the other.
        log_once(
            logger,
            ("vsa_h3_aiter", recipe),
            f"VSA-H3 has no AITER {recipe} 64x64 block-sparse MHA v4 row on "
            f"this device, falling back to the FlexAttention path. That "
            f"geometry is gfx950 only; gfx942's finest is 256x64.",
            level=logging.WARNING,
        )
        kernel = "flex"

    sequence_length = metadata.total_seq_length
    gathered_length = query.shape[2]
    query, key, value, gate = (t[:, :, :sequence_length] for t in (query, key, value, gate))

    if kernel == "aiter":
        packed = aiter_kernel.aiter_h3_vsa_attention(query, key, value, gate, metadata, recipe=recipe)
    else:
        packed = h3.h3_vsa_attention(query, key, value, gate, metadata, use_triton=kernel == "triton")

    if gathered_length > sequence_length:
        padded = packed.new_zeros(packed.shape[0], packed.shape[1], gathered_length, packed.shape[3])
        padded[:, :, :sequence_length] = packed
        packed = padded

    return packed, None


def flex_vsa_h3(query, key, value, call: AttnCall):
    """Through FlexAttention: portable, and the selection reference."""
    return _vsa_h3(query, key, value, call, kernel="flex")


def triton_vsa_h3(query, key, value, call: AttnCall):
    """Through the hand-written kernel, FlexAttention where it cannot run."""
    return _vsa_h3(query, key, value, call, kernel="triton")


def aiter_vsa_h3(query, key, value, call: AttnCall, *, recipe: str):
    """Through AITER's 64x64 sorted-sparse MHA v4 row for ``recipe``,
    FlexAttention where this device has none.

    The FP8 row stages its LUT in LDS, which caps the key length at its 1919
    entries -- 122,816 tokens -- and FastH3's default 768x1344x124 render is
    2024 tiles, so it lands just past that and the launcher refuses it by name.
    The BF16 row walks the LUT from memory instead and has no such ceiling.
    """
    return _vsa_h3(query, key, value, call, kernel="aiter", recipe=recipe)
