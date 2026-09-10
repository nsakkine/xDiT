"""Absorb one known-broken import so test outcomes do not depend on collection order.

`xfuser/__init__.py` pulls in the pipelines, which import `parallelize_decoder` from `distvae.vae`.
The DistVAE installed here (0.0.0b5) has no `vae` submodule and no `parallelize_decoder` anywhere,
so that import raises. This is a dependency skew in the environment, unrelated to any test.

It matters only because of how Python unwinds it. By the time `xfuser/__init__.py` fails, the
submodules underneath it are already fully imported and left in `sys.modules`, but `xfuser` itself
is removed. So whichever test imports first absorbs the ImportError and every later one succeeds
against the cached submodules -- which makes a single test fail for reasons that have nothing to do
with it, and makes which test that is depend on collection order.

Doing the import once here, before collection, makes that deterministic. Only the DistVAE skew is
swallowed; anything else propagates, so a real import regression in xfuser still fails loudly.
"""

import warnings


def _absorb_distvae_skew():
    try:
        import xfuser  # noqa: F401
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "distvae":
            raise
        warnings.warn(
            f"xfuser package import failed on an unrelated optional dependency ({exc}); "
            "its submodules remain importable and the tests run against those.",
            stacklevel=2,
        )


_absorb_distvae_skew()
