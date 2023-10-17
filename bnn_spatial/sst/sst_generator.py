"""
Generate SST data
"""

import torch
import numpy as np
from copy import deepcopy
from scipy.signal import detrend


class SST(torch.nn.Module):
    def __init__(self, sst_data, latitude_flip=False, panels=None):
        """
        SST data generator.

        :param sst_data: np.ndarray, contains SST data (in degrees celsius), dimensions [time, lat, long]
        :param latitude_flip: bool, specify whether latitudinal index is flipped when extracting SST values
        :param panels: list [int], (optional) specify which panels to extract samples from (if None, use all panels)
        """
        super().__init__()

        self.latitude_flip = latitude_flip
        self.time_limit = sst_data.shape[0]
        self.data = None
        self.samples = None
        self.panels = panels  # numbering starts at 1 for first (bottom left) panel, increases left to right and upwards

        if latitude_flip:
            self.sst_data = np.flip(np.squeeze(sst_data), axis=1)
        else:
            self.sst_data = np.squeeze(sst_data)

    def generate_data(self, normalise=True, remove_trend=True):
        """
        Generate the SST data panels to sample from.

        :param normalise: bool, specify if data panels are normalised or not
        :param remove_trend: bool, specify if data panels are de-trended (north-south and east-west trends removed)
        """
        if remove_trend and not normalise:
            return Exception('Must normalise if removing trend.')

        if self.panels is None:
            n_panels = 30
        else:
            n_panels = len(self.panels)
        self.data = np.empty((self.time_limit * n_panels, 64, 64))  # stores data panels
        self.data_loc = np.empty((self.time_limit * n_panels, 2))  # stores (row, col) location of each data panel

        for tt in range(self.time_limit):
            cc = 0  # number of panels used at time tt
            for ss in range(30):

                # Only extract desired panels (if self.panels is specified)
                if self.panels is not None:
                    if ss + 1 not in self.panels:
                        continue
                    else:
                        cc += 1

                # Count rows bottom-to-top, and columns left-to-right

                # Row 1 (starting from bottom)
                if ss <= 7:
                    sst_panel = self.sst_data[tt, 0:64, ss*64:(ss+1)*64]
                    row = 1
                    col = ss + 1
                # Row 2
                elif ss <= 13:
                    vv = ss - 8
                    sst_panel = self.sst_data[tt, 64:2*64, 64+vv*64:64+(vv+1)*64]
                    row = 2
                    col = (vv + 1) + 1
                # Row 3
                elif ss <= 18:
                    vv = ss - 14
                    sst_panel = self.sst_data[tt, 2*64:3*64, 4*64+vv*64:4*64+(vv+1)*64]
                    row = 3
                    col = (vv + 1) + 4
                # Row 4
                elif ss <= 23:
                    vv = ss - 19
                    sst_panel = self.sst_data[tt, 3*64:4*64, 4*64+vv*64:4*64+(vv+1)*64]
                    row = 4
                    col = (vv + 1) + 4
                # Row 5
                else:
                    vv = ss - 24
                    sst_panel = self.sst_data[tt, 4*64:5*64, 3*64+vv*64:3*64+(vv+1)*64]
                    row = 5
                    col = (vv + 1) + 3

                # Standardise the sample panels (if specified)
                if normalise:
                    trend = np.mean(sst_panel)
                    sd = np.std(sst_panel)
                    if remove_trend:
                        lon_trend = detrend(sst_panel, axis=1, type='linear', overwrite_data=False)
                        lat_trend = detrend(sst_panel, axis=0, type='linear', overwrite_data=False)
                        trend += lon_trend + lat_trend
                    sst_panel = (sst_panel - trend) / (sd + 1e-10)

                # Save the sample panel
                if self.panels is None:
                    cc = ss + 1
                if self.latitude_flip:
                    self.data[n_panels * tt + cc - 1, :, :] = np.flip(sst_panel, axis=0)
                else:
                    self.data[n_panels * tt + cc - 1, :, :] = sst_panel
                self.data_loc[n_panels * tt + cc - 1, :] = row, col  # store (row, column) of panel

    def sample_functions(self, n_samples, replace=False, flatten_order='C'):
        """
        Produce samples from the SST data set, with or without replacement.

        :param n_samples: int, number of sampled functions
        :param flatten_order: str, specify whether to flatten samples row-wise ('C') or col-wise ('F')
        :param replace: bool, specify is sampling is done with replacement or not (default: without replacement)
        :return: torch.Tensor, size (n_inputs, n_samples), with samples in columns
        """
        if self.data is None:
            raise Exception('Generate the SST samples first by calling SST.generate_samples()')
        if self.samples is None:
            self.samples = deepcopy(self.data)

        # If sampling without replacement, and there are not enough remaining samples, raise exception
        if not replace:
            n_samples_left = self.samples.shape[0]
            if n_samples_left < n_samples:
                print('WARNING: Not enough samples remaining ({}) to sample without replacement (need {}), '
                      'reusing previous samples.'.format(n_samples_left, n_samples))
                self.samples = deepcopy(self.data)

        # Randomly sample and resize panels of SST data
        np.random.shuffle(self.samples)  # randomly shuffle along first mode
        samples = self.samples[:n_samples, :, :]
        samples_ = np.empty((64**2, n_samples))
        for ss in range(n_samples):
            samples_[:, ss] = samples[ss, :, :].flatten(order=flatten_order)  # default order is C (row-wise)

        # If sampling without replacement, remove the panels just extracted, so that they cannot be resampled
        if not replace:
            self.samples = self.samples[n_samples:, :, :]

        return torch.from_numpy(samples_)

