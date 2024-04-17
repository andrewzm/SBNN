"""
Raw data sampler
"""

import torch
import numpy as np
from copy import deepcopy
from scipy.signal import detrend
from scipy.io import netcdf

class raw_data_generator(torch.nn.Module):
    def __init__(self, data_file):
        """
        Raw data generator.

        :param data_file: file containing numpy array of dimensions N x 64 x 64, where N is the number of samples
        """
        super().__init__()

        self.data = np.load(data_file)
        self.unsampled_indices = list(range(0, self.data.shape[0]))        
    
    def sample_functions(self, n_samples, replace=False, flatten_order='C'):
        """
        Produce samples from the SST data set, with or without replacement.

        :param n_samples: int, number of sampled functions
        :param flatten_order: str, specify whether to flatten samples row-wise ('C') or col-wise ('F')
        :param replace: bool, specify is sampling is done with replacement or not (default: without replacement)
        :return: torch.Tensor, size (n_inputs, n_samples), with samples in columns
        """
        if self.data is None:
            raise Exception('Data was not loaded properly')
        if self.unsampled_indices is None:
            self.unsampled_indices = list(range(0, self.data.shape[0]))

        # If sampling without replacement, and there are not enough remaining samples, raise exception
        if not replace:
            n_samples_left = len(self.unsampled_indices)
            if n_samples_left < n_samples:
                print('WARNING: Not enough samples remaining ({}) to sample without replacement (need {}), '
                      'reusing previous samples.'.format(n_samples_left, n_samples))
                self.unsampled_indices = list(range(0, self.data.shape[0]))

        # Randomly sample and resize panels of SST data
        np.random.shuffle(self.unsampled_indices)  # randomly shuffle along first mode
        sample_indices = self.unsampled_indices[:n_samples]
        samples = self.data[sample_indices, :, :]
        samples_ = np.empty((64**2, n_samples))
        for ss in range(n_samples):
            samples_[:, ss] = samples[ss, :, :].flatten(order=flatten_order)  # default order is C (row-wise)

        # If sampling without replacement, remove the indices just extracted, so that they cannot be resampled
        if not replace:
            self.unsampled_indices = self.unsampled_indices[n_samples:]

        return torch.from_numpy(samples_)

