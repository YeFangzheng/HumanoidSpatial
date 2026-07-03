# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import copy
import logging
import warnings
from typing import List, Sequence, Tuple, Set, Union

import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from mmengine.dataset import BaseDataset, force_full_init

from mmengine.logging import print_log
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as ``torch.utils.data.dataset.ConcatDataset`` and support lazy_init.

    Note:
        ``ConcatDataset`` should not inherit from ``BaseDataset`` since
        ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
        sub-dataset which conflicts with original dataset. If you want to use
        a sub-dataset of ``ConcatDataset``, you should set ``indices``
        arguments for wrapped dataset which inherit from ``BaseDataset``.

    Args:
        datasets (Sequence[BaseDataset] or Sequence[dict]): A list of datasets
            which will be concatenated.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
        ignore_keys (List[str] or str): Ignore the keys that can be
            unequal in `dataset.metainfo`. Defaults to None.
            `New in version 0.3.0.`
    """

    def __init__(self,
                 datasets: Sequence[Union[BaseDataset, dict]],
                 lazy_init: bool = False,
                 ignore_keys: Union[str, List[str], None] = None):
        self.datasets: List[BaseDataset] = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    'elements in datasets sequence should be config or '
                    f'`BaseDataset` instance, but got {type(dataset)}')
        max_flag = -1
        flag = []
        for dataset in self.datasets:
            dataset_flag = dataset.flag + max_flag + 1
            max_flag = np.max(dataset_flag)
            flag.append(dataset_flag)
        self.flag = np.hstack(flag)

        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError('ignore_keys should be a list or str, '
                            f'but got {type(ignore_keys)}')

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # Only use metainfo of first dataset.
        self._metainfo = self.datasets[0].metainfo
        for i, dataset in enumerate(self.datasets, 1):
            for key in meta_keys:
                if key in self.ignore_keys:
                    continue
                if key not in dataset.metainfo:
                    raise ValueError(
                        f'{key} does not in the meta information of '
                        f'the {i}-th dataset')
                first_type = type(self._metainfo[key])
                cur_type = type(dataset.metainfo[key])
                if first_type is not cur_type:  # type: ignore
                    raise TypeError(
                        f'The type {cur_type} of {key} in the {i}-th dataset '
                        'should be the same with the first dataset '
                        f'{first_type}')
                if (isinstance(self._metainfo[key], np.ndarray)
                        and not np.array_equal(self._metainfo[key],
                                               dataset.metainfo[key])
                        or (not isinstance(self._metainfo[key], np.ndarray)
                            and self._metainfo[key] != dataset.metainfo[key])):
                    raise ValueError(
                        f'The meta information of the {i}-th dataset does not '
                        'match meta information of the first dataset')

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the first dataset in ``self.datasets``.

        Returns:
            dict: Meta information of first dataset.
        """
        # Prevent `self._metainfo` from being modified by outside.
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return
        for d in self.datasets:
            d.full_init()
        # Get the cumulative sizes of `self.datasets`. For example, the length
        # of `self.datasets` is [2, 3, 4], the cumulative sizes is [2, 5, 9]
        super().__init__(self.datasets)
        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> Tuple[int, int]:
        """Convert global idx to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            Tuple[int, int]: The index of ``self.datasets`` and the local
            index of data.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    f'absolute value of index({idx}) should not exceed dataset'
                    f'length({len(self)}).')
            idx = len(self) + idx
        # Get `dataset_idx` to tell idx belongs to which dataset.
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        # Get the inner index of single dataset.
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return dataset_idx, sample_idx

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_data_info(sample_idx)

    @force_full_init
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to '
                'accelerate the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx][sample_idx]

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``ConcatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`ConcatDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `ConcatDataset`.')

    def get_subset(self, indices: Union[List[int], int]) -> 'BaseDataset':
        """Not supported in ``ConcatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`ConcatDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `ConcatDataset`.')
    

@DATASETS.register_module()
class CBGSDataset:
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`BaseDataset` or dict): The dataset to be class sampled.
        lazy_init (bool): Whether to load annotation during instantiation.
            Defaults to False.
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 lazy_init: bool = False) -> None:
        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        self._metainfo = self.dataset.metainfo

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self) -> None:
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        # Get sample_indices
        self.sample_indices = self._get_sample_indices(self.dataset)

        self._fully_initialized = True

    def _get_sample_indices(self, dataset: BaseDataset) -> List[int]:
        """Load sample indices according to ann_file.

        Args:
            dataset (:obj:`BaseDataset`): The dataset.

        Returns:
            List[dict]: List of indices after class sampling.
        """
        classes = self.metainfo['classes']
        cat2id = {name: i for i, name in enumerate(classes)}
        class_sample_idxs = {cat_id: [] for cat_id in cat2id.values()}
        for idx in range(len(dataset)):
            sample_cat_ids = dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                if cat_id != -1:
                    # Filter categories that do not need to be cared.
                    # -1 indicates dontcare in MMDet3D.
                    class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx (int): Global index of ``CBGSDataset``.

        Returns:
            int: Local index of data.
        """
        return self.sample_indices[idx]

    @force_full_init
    def get_cat_ids(self, idx: int) -> Set[int]:
        """Get category ids of class balanced dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            Set[int]: All categories in the sample of specified index.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_cat_ids(sample_idx)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``CBGSDataset``.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    def __getitem__(self, idx: int) -> dict:
        """Get item from infos according to the given index.

        Args:
            idx (int): The index of self.sample_indices.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if not self._fully_initialized:
            warnings.warn('Please call `full_init` method manually to '
                          'accelerate the speed.')
            self.full_init()

        ori_index = self._get_ori_dataset_idx(idx)
        return self.dataset[ori_index]

    @force_full_init
    def __len__(self) -> int:
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``CBGSDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`CBGSDataset` does not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `CBGSDataset`.')

    def get_subset(self, indices: Union[List[int], int]) -> BaseDataset:
        """Not supported in ``CBGSDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`CBGSDataset` does not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `CBGSDataset`.')
