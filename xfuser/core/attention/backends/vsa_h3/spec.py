"""FastH3 VSA-H3: 64-token tiles, on any of three kernels.

All select the same key tiles and differ only in how they read them, so they
share everything around the call. The Triton kernel reads packed rows through
the tile map; FlexAttention and AITER's 64x64 sorted-sparse MHA v4 rows need
the padded tile buffers built for them.
"""

from xfuser.core.attention.constraints import NO_DROPOUT, NON_CAUSAL, NO_VARLEN
from xfuser.core.attention.requirements import NEVER, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Sparsity, Spec

_H3 = "xfuser.core.attention.backends.vsa_h3.attention"

# None produces an LSE, so none can join a ring.
_CALLS = NON_CAUSAL & NO_DROPOUT & NO_VARLEN

# The AITER rows, and which mha_v4 recipe each dispatches. The 64x64
# sorted-sparse geometry VSA-H3's tile needs exists in these two precisions and
# no others.
VSA_H3_AITER_RECIPE_BY_BACKEND = {
    AttentionBackendType.AITER_BF16_VSA_H3: "bf16",
    AttentionBackendType.AITER_FP8_VSA_H3: "fp8",
}


def _prime_aiter_row(recipe):
    """Answer the row query now, outside any compiled region, so the attention
    call's lookup is a constant under the transformer's compile rather than a
    trace of aiter's manifest read."""

    def prime():
        from xfuser.core.attention.backends.vsa_h3.aiter_kernel import vsa_h3_aiter_row_available

        vsa_h3_aiter_row_available(recipe)

    return prime


def _aiter_spec(backend, recipe):
    # Only the build is required, not the device. Whether this GPU has the
    # 64x64 row for the recipe is a per-device answer that the call warns about
    # and falls back to FlexAttention for, as the Triton row does; a build with
    # no mha_v4_packed at all cannot run either of them.
    return Spec(
        backend,
        impl=Impl("kernel:aiter_vsa_h3", {"recipe": recipe}),
        ring=NEVER,
        sparsity=Sparsity.H3,
        low_precision=recipe != "bf16",
        accepts=_CALLS,
        requires=SYMBOL(f"{_H3}:h3_vsa_attention") & SYMBOL("aiter.ops.mha_v4:mha_v4_packed"),
        initializers=(_prime_aiter_row(recipe),),
    )


SPECS = [
    Spec(
        AttentionBackendType.FLEX_VSA_H3,
        impl=Impl("kernel:flex_vsa_h3"),
        ring=NEVER,
        sparsity=Sparsity.H3,
        accepts=_CALLS,
        requires=SYMBOL(f"{_H3}:h3_vsa_attention"),
    ),
    # Triton is not in `requires`: where the kernel cannot run this falls back
    # to FlexAttention rather than refusing, so the backend stays selectable
    # and says so once at the first call.
    Spec(
        AttentionBackendType.TRITON_VSA_H3,
        impl=Impl("kernel:triton_vsa_h3"),
        ring=NEVER,
        sparsity=Sparsity.H3,
        accepts=_CALLS,
        requires=SYMBOL(f"{_H3}:h3_vsa_attention"),
    ),
] + [_aiter_spec(backend, recipe) for backend, recipe in VSA_H3_AITER_RECIPE_BY_BACKEND.items()]
