"""Which models may select a Sol-Attn backend, and what they must not silently acquire with it.

Config validation only -- no kernel, no device. Whether the machine actually carries the manifest
row for a recipe is a separate check that runtime_state makes against the device.
"""

import pytest

from xfuser.config import xFuserArgs
from xfuser.model_executor.models.runner_models.flux import (
    xFuserFlux2Klein9BModel,
    xFuserFlux2Model,
)
from xfuser.model_executor.models.runner_models.hunyuan import (
    xFuserHunyuanvideo15Model,
    xFuserHunyuanvideoModel,
)

SOL_BACKENDS = ["aiter_fp8_sol", "aiter_i8fp8_sol"]

# The model class, a name it accepts, and any extra args its own validation demands.
MODELS = [
    (xFuserHunyuanvideoModel, "HunyuanVideo", {}),
    (xFuserHunyuanvideo15Model, "HunyuanVideo-1.5", {"task": "t2v"}),
    (xFuserFlux2Model, "FLUX.2-dev", {}),
    (xFuserFlux2Klein9BModel, "FLUX.2-klein-9B", {}),
]


def _build(cls, model, **kwargs):
    return cls(xFuserArgs(model=model, **kwargs))


@pytest.mark.parametrize("cls,model,extra", MODELS)
@pytest.mark.parametrize("backend", SOL_BACKENDS)
def test_sol_backends_are_accepted(cls, model, extra, backend):
    _build(cls, model, attention_backend=backend, **extra)


@pytest.mark.parametrize("cls,model,extra", MODELS)
@pytest.mark.parametrize("backend", ["aiter_sparge", "aiter_fp8_sparge"])
def test_sol_does_not_drag_in_sparge(cls, model, extra, backend):
    """Opting into Sol-Attn must not also opt a model into Sparge.

    The two used to share one capability, so enabling Sol on a model with no backend allowlist
    would have handed it every Sparge row as well, none of them checked for it.
    """
    with pytest.raises(ValueError, match="does not support Sparge"):
        _build(cls, model, attention_backend=backend, **extra)


@pytest.mark.parametrize("cls,model,extra", MODELS)
def test_sol_refuses_ring_parallelism_at_config_time(cls, model, extra):
    """Sol-Attn cannot be a ring rank, and finding that out mid-denoise is too late.

    Every model here advertises ring_degree, so the combination is reachable by configuration
    rather than by mistake.
    """
    with pytest.raises(ValueError, match="does not support ring parallelism"):
        _build(
            cls, model, attention_backend="aiter_fp8_sol", ring_degree=2, **extra
        )


@pytest.mark.parametrize("cls,model,extra", MODELS)
def test_ulysses_stays_available(cls, model, extra):
    """Ulysses is the sequence parallelism Sol-Attn does support, so it must survive the ring check."""
    _build(cls, model, attention_backend="aiter_fp8_sol", ulysses_degree=2, **extra)


def test_a_model_without_the_capability_still_refuses_sol():
    """The gate is opt-in, so Flux1 sitting next to Flux2 must keep saying no."""
    from xfuser.model_executor.models.runner_models.flux import xFuserFluxModel

    with pytest.raises(ValueError, match="does not support Sol-Attn"):
        xFuserFluxModel(xFuserArgs(model="FLUX.1-dev", attention_backend="aiter_fp8_sol"))
