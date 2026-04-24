"""
Shape, finiteness, and gradient tests for the three pipeline components:
  - LogMelFrontend
  - SSTFRFrontend
  - ResNet18Head

and for the full end-to-end SSTFR -> ResNet-18 model.
"""

from __future__ import annotations

import pytest
import torch

from sstfr.models.classifier import ResNet18Head
from sstfr.models.logmel_frontend import LogMelFrontend
from sstfr.models.sstfr_frontend import SSTFRFrontend
from sstfr.models.ssm_layer import SSTFRConfig


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)


# --------------------------------------------------------------------------
# Log-Mel
# --------------------------------------------------------------------------

def test_logmel_output_shape():
    fe = LogMelFrontend(sample_rate=16000, n_mels=128, hop_length=160)
    x = torch.randn(4, 16000)  # 1 s
    y = fe(x)
    print(f"\n  LogMel output: {tuple(y.shape)}")
    assert y.dim() == 3
    assert y.shape[0] == 4
    assert y.shape[1] == 128
    assert torch.isfinite(y).all()


# --------------------------------------------------------------------------
# SSTFR front-end
# --------------------------------------------------------------------------

def test_sstfr_frontend_output_shape_fast_path():
    """Default path: no complex H, feature map shape is still correct."""
    cfg = SSTFRConfig(num_channels=64, sample_rate=16000)
    fe = SSTFRFrontend(cfg, hop_length=160)
    fe.train()  # training mode so caching logic runs
    x = torch.randn(2, 16000)
    y = fe(x)
    print(f"\n  SSTFR fast-path output: {tuple(y.shape)}")
    assert y.dim() == 3
    assert y.shape == (2, 64, 100)  # 16000 / 160 = 100
    assert torch.isfinite(y).all()
    # Fast path: H is not cached
    assert fe.last_hidden_states is None, (
        "Fast path should not cache complex H"
    )


def test_sstfr_frontend_output_shape_complex_path():
    """Explicit complex path: H is cached as complex tensor."""
    cfg = SSTFRConfig(num_channels=64, sample_rate=16000)
    fe = SSTFRFrontend(cfg, hop_length=160)
    fe.train()
    fe.set_need_complex_H(True)
    x = torch.randn(2, 16000)
    y = fe(x)
    print(f"\n  SSTFR complex-path output: {tuple(y.shape)}")
    assert y.shape == (2, 64, 100)
    assert torch.isfinite(y).all()
    # Complex path: H is cached with correct shape and dtype
    assert fe.last_hidden_states is not None
    assert torch.is_complex(fe.last_hidden_states)
    assert fe.last_hidden_states.shape == (2, 16000, 64)


def test_sstfr_and_logmel_output_same_shape():
    """Matched interface: both front-ends produce (B, 128, T) for the same input."""
    B, L = 2, 16000
    x = torch.randn(B, L)
    logmel = LogMelFrontend(sample_rate=16000, n_mels=128, hop_length=160)
    cfg = SSTFRConfig(num_channels=128, sample_rate=16000)
    sstfr = SSTFRFrontend(cfg, hop_length=160)

    y_logmel = logmel(x)
    y_sstfr = sstfr(x)
    print(f"\n  LogMel: {tuple(y_logmel.shape)}  SSTFR: {tuple(y_sstfr.shape)}")
    # T may differ by 1 due to mel's centered vs our non-centered pooling.
    assert y_logmel.shape[:2] == y_sstfr.shape[:2]  # B, D match exactly
    assert abs(y_logmel.shape[2] - y_sstfr.shape[2]) <= 2  # T close


# --------------------------------------------------------------------------
# ResNet-18 head
# --------------------------------------------------------------------------

def test_resnet18_forward_shape():
    head = ResNet18Head(num_classes=50, pretrained=False)
    features = torch.randn(4, 128, 100)  # typical spectrogram shape
    logits = head(features)
    print(f"\n  ResNet-18 logits: {tuple(logits.shape)}")
    assert logits.shape == (4, 50)
    assert torch.isfinite(logits).all()


def test_resnet18_param_count():
    """~11.19M params (matches paper's claim)."""
    head = ResNet18Head(num_classes=50, pretrained=False)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"\n  ResNet-18 total params: {n_params:,}")
    assert 11_000_000 < n_params < 12_000_000


# --------------------------------------------------------------------------
# End-to-end: SSTFR -> ResNet-18
# --------------------------------------------------------------------------

def test_full_model_forward_backward():
    cfg = SSTFRConfig(num_channels=128, sample_rate=16000)
    frontend = SSTFRFrontend(cfg, hop_length=160)
    head = ResNet18Head(num_classes=50, pretrained=False)

    x = torch.randn(2, 16000, requires_grad=False)
    features = frontend(x)
    logits = head(features)
    loss = logits.sum()
    loss.backward()

    # Check gradients flow into SSTFR parameters
    assert frontend.ssm.omega.grad is not None
    assert frontend.ssm.alpha_raw.grad is not None
    assert torch.isfinite(frontend.ssm.omega.grad).all()
    assert torch.isfinite(frontend.ssm.alpha_raw.grad).all()
    print(f"\n  End-to-end forward+backward OK, omega.grad norm: "
          f"{frontend.ssm.omega.grad.norm():.4e}")


def test_matched_total_param_count_frontends():
    """ResNet-18 head dominates; front-ends contribute < 1% difference."""
    cfg = SSTFRConfig(num_classes := 50, sample_rate=16000)
    head = ResNet18Head(num_classes=50)
    sstfr = SSTFRFrontend(SSTFRConfig(num_channels=128, sample_rate=16000))
    logmel = LogMelFrontend(n_mels=128)

    n_head = sum(p.numel() for p in head.parameters())
    n_sstfr_fe = sum(p.numel() for p in sstfr.parameters() if p.requires_grad)
    n_logmel_fe = sum(p.numel() for p in logmel.parameters() if p.requires_grad)

    print(f"\n  ResNet-18:         {n_head:,} params")
    print(f"  SSTFR front-end:   {n_sstfr_fe:,} params")
    print(f"  Log-Mel frontend:  {n_logmel_fe:,} params")

    total_sstfr = n_head + n_sstfr_fe
    total_logmel = n_head + n_logmel_fe
    rel_diff = abs(total_sstfr - total_logmel) / total_logmel
    print(f"  Total SSTFR:       {total_sstfr:,}")
    print(f"  Total Log-Mel:     {total_logmel:,}")
    print(f"  Relative diff:     {rel_diff:.4%}")
    assert rel_diff < 0.01, f"Parameter counts differ by {rel_diff:.4%}, should be <1%"
