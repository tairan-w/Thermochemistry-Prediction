import logging

from torch.utils.data import DataLoader

from chemprop.data.collate import collate_batch, collate_multicomponent
from chemprop.data.datasets import MoleculeDataset, MulticomponentDataset, ReactionDataset
from chemprop.data.samplers import ClassBalanceSampler, SeededSampler

logger = logging.getLogger(__name__)


def build_dataloader(
    dataset: MoleculeDataset | ReactionDataset | MulticomponentDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    class_balance: bool = False,
    seed: int | None = None,
    shuffle: bool = True,
    **kwargs,
):


    if class_balance:
        sampler = ClassBalanceSampler(dataset.Y, seed, shuffle)
    elif shuffle and seed is not None:
        sampler = SeededSampler(len(dataset), seed)
    else:
        sampler = None

    if isinstance(dataset, MulticomponentDataset):
        collate_fn = collate_multicomponent
    else:
        collate_fn = collate_batch

    if len(dataset) % batch_size == 1:
        logger.warning(
            f"Dropping last batch of size 1 to avoid issues with batch normalization \
(dataset size = {len(dataset)}, batch_size = {batch_size})"
        )
        drop_last = True
    else:
        drop_last = False

    return DataLoader(
        dataset,
        batch_size,
        sampler is None and shuffle,
        sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        **kwargs,
    )
