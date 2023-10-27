import itertools
import logging
import math
from collections import defaultdict
from typing import Optional
import torch
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
import utilities.distributed as dist
import numpy as np
from tqdm import tqdm


class BalancedTrainingSampler(WeightedRandomSampler):
    """
    Blanced TrainingSampler, sample frenquency function of classfrequency.
    """

    def __init__(self, weights, num_samples, replacement=True):
        """
        Args:
            weights (sequence)   : a sequence of weights, not necessary summing up to one
            num_samples (int): number of samples to draw
            replacement (bool): if ``True``, samples are drawn with replacement.
                                If not, they are drawn without replacement, which means that when a
                                sample index is drawn for a row, it cannot be drawn again for that row.
        """
        WeightedRandomSampler.__init__(self, weights, num_samples, replacement)

    @staticmethod
    def computeweights(dataset, thresh=[0.0004, 0.0008, 0.0012], weights=[1, 2, 4, 10]):
        """
        Compute  per-image weight based on category frequency.
        The weight factor for an image is a function of the frequency of the rarest
        category labeled in that image. 
        Args:
            dataset: ImageDataset.
            thresh: list of thresholds
            weights: list of weights
        Returns:
            numpy array:
                the i-th element is the repeat factor for the dataset image at index i.
        """
        weight_factors = []
        print('computing data weights')
        for index in tqdm(range(len(dataset))):
            curve = dataset.get_ego_curvature(index)
            if curve < thresh[0]:
                weight_factor = weights[0]
            elif curve < thresh[1]:
                weight_factor = weights[1]
            elif curve < thresh[2]:
                weight_factor = weights[2]
            else:
                weight_factor = weights[-1]

            weight_factors.append(weight_factor)

        return np.asarray(weight_factors, dtype=np.float32)


class DistributedBalancedTrainingSampler(Sampler):

    def __init__(self, weights, seed, shuffle=True):
        """
        Args:
            weights (sequence)   : a sequence of weights, not necessary summing up to one
            num_samples (int): number of samples to draw
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers.
        """
        self._shuffle = shuffle
        self._seed = int(seed)
        self.epoch = 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self.num_samples = int(len(weights) / self._world_size)
        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        self._int_part = torch.trunc(weights)
        self._frac_part = weights - self._int_part

    @staticmethod
    def computeweights(dataset, thresh=[0.0004, 0.0008, 0.0012], weights=[1, 2, 4, 10]):
        """
        Compute  per-image weight based on category frequency.
        The weight factor for an image is a function of the frequency of the rarest
        category labeled in that image. 
        Args:
            dataset: ImageDataset.
            thresh: list of thresholds
            weights: list of weights
        Returns:
            numpy array:
                the i-th element is the repeat factor for the dataset image at index i.
        """
        weight_factors = []
        print('computing data weights')
        for index in tqdm(range(len(dataset))):
            curve = dataset.get_ego_curvature(index)
            if curve < thresh[0]:
                weight_factor = weights[0]
            elif curve < thresh[1]:
                weight_factor = weights[1]
            elif curve < thresh[2]:
                weight_factor = weights[2]
            else:
                weight_factor = weights[-1]

            weight_factors.append(weight_factor)

        return torch.tensor(weight_factors, dtype=torch.float32)

    def __len__(self) -> int:
        return self.num_samples

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.
        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.
        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._indices(), start, start + 2 * self.num_samples, self._world_size)

    def _indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed + self.epoch)

        indices = self._get_epoch_indices(g)
        if self._shuffle:
            randperm = torch.randperm(len(indices), generator=g)
            yield from indices[randperm].tolist()
        else:
            yield from indices.tolist()

    def set_epoch(self, epoch: int) -> None:

        self.epoch = epoch
