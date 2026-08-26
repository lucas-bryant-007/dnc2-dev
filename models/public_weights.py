"""Pinned public ResNet-50 encoders used by the paper experiments.

The VICReg architecture follows the official implementation at revision
``4e12602fd495af83efd1631fbe82523e6db092e0``.  Its only material difference
from torchvision's ResNet-50 forward is explicit one-pixel zero padding before
a 7x7 convolution configured with padding two.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.resnet import Bottleneck, ResNet


OFFICIAL_VICREG_REVISION = "4e12602fd495af83efd1631fbe82523e6db092e0"


@dataclass(frozen=True)
class PublicWeights:
    name: str
    url: str
    sha256: str
    size_bytes: int
    filename: str


PUBLIC_WEIGHTS = {
    "vicreg-imagenet": PublicWeights(
        name="vicreg-imagenet",
        url="https://dl.fbaipublicfiles.com/vicreg/resnet50.pth",
        sha256="c843e7652491ff2f712734619fd74d85e5e92ac902615665ad3fd50dc6ada591",
        size_bytes=94_345_885,
        filename="resnet50.pth",
    ),
    "supervised-imagenet": PublicWeights(
        name="supervised-imagenet",
        url=ResNet50_Weights.IMAGENET1K_V1.url,
        sha256="0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
        size_bytes=102_530_333,
        filename="resnet50-0676ba61.pth",
    ),
}


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_public_weights_file(name: str, path: str | Path) -> Path:
    spec = PUBLIC_WEIGHTS[name]
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {name} weights: {resolved}")
    if resolved.stat().st_size != spec.size_bytes:
        raise RuntimeError(
            f"{name} size mismatch: expected {spec.size_bytes}, "
            f"observed {resolved.stat().st_size}"
        )
    observed_hash = sha256_file(resolved)
    if observed_hash != spec.sha256:
        raise RuntimeError(
            f"{name} SHA-256 mismatch: expected {spec.sha256}, "
            f"observed {observed_hash}"
        )
    return resolved


class OfficialVICRegResNet50(ResNet):
    """Network-equivalent local form of the official VICReg ResNet-50."""

    def __init__(self) -> None:
        super().__init__(Bottleneck, [3, 4, 6, 3])
        self.padding = nn.ConstantPad2d(1, 0.0)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=2, bias=False
        )
        self.fc = nn.Identity()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def load_vicreg_imagenet_resnet50(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> OfficialVICRegResNet50:
    resolved = verify_public_weights_file("vicreg-imagenet", path)
    state_dict = torch.load(resolved, map_location="cpu", weights_only=True)
    model = OfficialVICRegResNet50()
    model.load_state_dict(state_dict, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def load_supervised_imagenet_resnet50(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    resolved = verify_public_weights_file("supervised-imagenet", path)
    state_dict = torch.load(resolved, map_location="cpu", weights_only=True)
    model = resnet50(weights=None)
    model.load_state_dict(state_dict, strict=True)
    model.fc = nn.Identity()
    model.eval().requires_grad_(False)
    return model.to(device)
