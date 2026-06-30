import pytest
import torch
import torch.nn as nn

from xfuser.core.distributed.runtime_state import Fp8CommsState


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = nn.Module()


class _FakeTransformer(nn.Module):
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock() for _ in range(num_layers)])
        for i, block in enumerate(self.blocks):
            block.attn1.register_buffer("fp8_q_scale", torch.ones(1, dtype=torch.float32))
            block.attn1.register_buffer("fp8_k_scale", torch.ones(1, dtype=torch.float32))
            block.attn1.register_buffer("fp8_v_scale", torch.ones(1, dtype=torch.float32))
            block.attn1.register_buffer(
                "fp8_comms_layer_idx", torch.tensor([i], dtype=torch.long)
            )


def test_per_layer_running_max_and_scatter():
    fp8 = Fp8CommsState()
    model = _FakeTransformer(num_layers=2)
    fp8.register_model(model, num_layers=2)

    layer_idx = torch.tensor([0], dtype=torch.long)
    q = torch.tensor([[[[2.0, -1.0]]]])
    k = torch.tensor([[[[4.0]]]])
    v = torch.tensor([[[[0.5]]]])
    fp8.update_running_max(model, layer_idx, q, k, v)

    layer_idx_1 = torch.tensor([1], dtype=torch.long)
    fp8.update_running_max(model, layer_idx_1, q * 3, k * 2, v * 4)

    model_state = fp8.get_model_state(model)
    assert model_state.q_running_max[0].item() == 2.0
    assert model_state.q_running_max[1].item() == 6.0
    assert model_state.k_running_max[1].item() == 8.0
    assert model_state.v_running_max[1].item() == 2.0

    scales_q = model_state.q_running_max / 100.0
    scales_k = model_state.k_running_max / 100.0
    scales_v = model_state.v_running_max / 100.0
    fp8._scatter_scales_to_model(model, scales_q, scales_k, scales_v)

    assert model.blocks[0].attn1.fp8_q_scale.item() == pytest.approx(0.02)
    assert model.blocks[1].attn1.fp8_k_scale.item() == pytest.approx(0.08)


def test_fixed_scale_broadcast():
    fp8 = Fp8CommsState(fixed_scale=0.5)
    model = _FakeTransformer(num_layers=2)
    fp8.register_model(model, num_layers=2)

    assert fp8.get_model_state(model).synced is True
    assert model.blocks[0].attn1.fp8_q_scale.item() == 0.5
    assert model.blocks[1].attn1.fp8_v_scale.item() == 0.5


def test_unexercised_model_has_zero_running_max():
    fp8 = Fp8CommsState()
    model = _FakeTransformer(num_layers=1)
    fp8.register_model(model, num_layers=1)

    model_state = fp8.get_model_state(model)
    assert model_state.synced is False
    assert model_state.q_running_max.max() == 0
