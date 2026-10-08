import os
import warnings


def _absorb_distvae_skew():
    """Absorb one known-broken import so test outcomes do not depend on collection order.

    `xfuser/__init__.py` pulls in the pipelines, which import `parallelize_decoder` from
    `distvae.vae`. A DistVAE without a `vae` submodule (0.0.0b5 has none) makes that import raise.
    This is a dependency skew in the environment, unrelated to any test.

    It matters only because of how Python unwinds it. By the time `xfuser/__init__.py` fails, the
    submodules underneath it are already fully imported and left in `sys.modules`, but `xfuser`
    itself is removed. So whichever test imports first absorbs the ImportError and every later one
    succeeds against the cached submodules -- which makes a single test fail for reasons that have
    nothing to do with it, and makes which test that is depend on collection order.

    Doing the import once, before collection, makes that deterministic. Only the DistVAE skew is
    swallowed; anything else propagates, so a real import regression in xfuser still fails loudly.
    """
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


def pytest_configure():
    """Keep a GPU fault from writing a core dump into the working tree.

    The ROCm HSA runtime writes ``gpucore.<pid>.gpu`` on a GPU exception unless
    this is set before the first device call. pytest_configure runs before
    collection imports those tests, and the xfuser import below can reach the
    device through AITER, so it comes after.
    """
    os.environ.setdefault("HSA_DISABLE_COREDUMP_ON_EXCEPTION", "1")
    _absorb_distvae_skew()
