# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect

import numpy as np
from torch.utils.data.dataloader import default_collate

from . import FairseqDataset


class ConcatDataset(FairseqDataset):
    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            curr_len = int(ratio * len(e))
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sample_ratios=1, pure_batch=False):
        super(ConcatDataset, self).__init__()

        # Added by Eachan:
        self.pure_batch = pure_batch

        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
        self.sample_ratios = sample_ratios
        self.cumulative_sizes = self.cumsum(self.datasets, sample_ratios)
        self.real_sizes = [len(d) for d in self.datasets]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx][sample_idx]

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if hasattr(self.datasets[0], "collater"):
            return self.datasets[0].collater(samples, **extra_args)
        else:
            return default_collate(samples, **extra_args)

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx].size(sample_idx)

    def num_tokens(self, index: int):
        return np.max(self.size(index))

    def attr(self, attr: str, index: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return getattr(self.datasets[dataset_idx], attr, None)

    @property
    def sizes(self):
        """
        self.datasets: a list of language_pair_dataset
        self.sample_ratios: a list of 1

        return a list (ndarray), size of each sample
        """
        _dataset_sizes = []
        for ds, sr in zip(self.datasets, self.sample_ratios):
            if isinstance(ds.sizes, np.ndarray):  # True
                _dataset_sizes.append(np.tile(ds.sizes, sr))
            else:
                # Only support underlying dataset with single size array.
                assert isinstance(ds.sizes, list)
                _dataset_sizes.append(np.tile(ds.sizes[0], sr))
        return np.concatenate(_dataset_sizes)

    @property
    def supports_prefetch(self):
        return all(d.supports_prefetch for d in self.datasets)

    def ordered_indices_old(self):
        """
        self.sizes: a list, each element is a sample size(src_size, tgt_size)
        Returns indices sorted by length. So less padding is needed.
        """

        if isinstance(self.sizes, np.ndarray) and len(self.sizes.shape) > 1:  # true
            # special handling for concatenating lang_pair_datasets
            indices = np.arange(len(self))
            sizes = self.sizes
            tgt_sizes = (
                sizes[:, 1] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else None
            )
            src_sizes = (
                sizes[:, 0] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else sizes
            )
            # sort by target length, then source length
            if tgt_sizes is not None:
                indices = indices[np.argsort(tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(src_sizes[indices], kind="mergesort")]
        else:
            return np.argsort(self.sizes)

    def ordered_indices(self):
        """
        Re-implemented by Eachan:
            I want this function return a list, whose length is the number of self.datasets
            And each element of this list is sizes of samples in corresponding dataset.

        self.sizes: a list, each element is a sample size(src_size, tgt_size)
        Returns indices sorted by length.
        So less padding is needed.
        """

        # If using mix batch, call the old version ordered_indices:
        # print(f"DEBUG | pure-batch in ConcatDataset: {self.pure_batch}")
        if not self.pure_batch:
            return self.ordered_indices_old()

        if isinstance(self.sizes, np.ndarray) and len(self.sizes.shape) > 1:  # true
            # special handling for concatenating lang_pair_datasets
            indices = np.arange(len(self))
            indices_each_dataset = []
            for i, cum_size in enumerate(self.cumulative_sizes):
                indices_each_dataset.append(
                    np.arange(
                        0 if i == 0 else self.cumulative_sizes[i - 1], cum_size
                    )
                )
            sizes = self.sizes
            tgt_sizes = (
                sizes[:, 1] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else None
            )
            src_sizes = (
                sizes[:, 0] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else sizes
            )
            # sort by target length, then source length
            for i, indices in enumerate(indices_each_dataset):
                if tgt_sizes is not None:
                    indices_each_dataset[i] = indices[np.argsort(tgt_sizes[indices], kind="mergesort")]
                indices_each_dataset[i] = indices[np.argsort(src_sizes[indices], kind="mergesort")]
            return indices_each_dataset
        else:
            return np.argsort(self.sizes)

    def prefetch(self, indices):
        frm = 0
        for to, ds in zip(self.cumulative_sizes, self.datasets):
            real_size = len(ds)
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch([(i - frm) % real_size for i in indices if frm <= i < to])
            frm = to

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(d.can_reuse_epoch_itr_across_epochs for d in self.datasets)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
