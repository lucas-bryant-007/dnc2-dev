from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from data_utils.celebA_datamodule import CelebADataModule
from data_utils.dsprites_datamodule import DSpritesDataModule
from data_utils.mini_imagenet_datamodule import MiniImageNetDataModule
from data_utils.mpi3d_datamodule import Mpi3dDataModule
from data_utils.shapes3d_datamodule import Shapes3DDataModule


@pytest.mark.parametrize(
    "module_class",
    [
        CelebADataModule,
        MiniImageNetDataModule,
        DSpritesDataModule,
        Shapes3DDataModule,
        Mpi3dDataModule,
    ],
)
def test_rank_zero_probe_loader_can_disable_distributed_sampler(
    module_class, monkeypatch
):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    module = object.__new__(module_class)
    module.cfg = SimpleNamespace(batch_size=2)
    dataset = TensorDataset(torch.arange(6))
    loader = module._make_loader(
        dataset,
        train=False,
        collate_fn=None,
        shuffle=False,
        num_workers=0,
        distributed=False,
    )
    assert not isinstance(loader.sampler, torch.utils.data.DistributedSampler)
    assert len(loader) == 3
