from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from scipy.stats import pearsonr
import xarray as xr
import utils

def get_responses(day: int) -> Tuple[List[xr.DataArray]]:
    """
    Get time-averaged responses to all elements, divided into
    stimulus and gray period groups.
    """
    stim_arrays, gray_arrays = [], []
    for sequence in ('ABCD', 'ABBD', 'ACBD'):
        data = utils.load_traces(day, sequence)
        arrays = utils.split_by_event(data)
        stim_arrays += arrays[:4]
        gray_arrays.append(arrays[4])

    for i, arr in enumerate(stim_arrays):
        stim_arrays[i] = arr.isel(time=slice(2, None)).mean('time')

    for i, arr in enumerate(gray_arrays):
        gray_arrays[i] = arr.isel(time=slice(2, None)).mean('time')

    return stim_arrays, gray_arrays


def compute_correlations(arrays: List[xr.DataArray]) -> np.ndarray:

    n_trials = 500
    all_corrs = []
    for arr in arrays:
        mat = np.zeros([n_trials, n_trials])
        for i in range(n_trials - 1):
            for j in range(i, n_trials):
                score = pearsonr(arr[i], arr[j]).statistic
                mat[i, j] = score
        # corrs: 499-length array, where corrs[i] is the mean correlation
        # between responses `i` trials apart.
        corrs = np.zeros(n_trials - 1)
        for i in range(n_trials - 1):
            corrs[i] = np.diagonal(mat, i).mean()
        all_corrs.append(corrs)
    all_corrs = np.stack(all_corrs)
    return all_corrs


if __name__ == '__main__':

    stim_0_arrays, gray_0_arrays = get_responses(0)
    stim_5_arrays, gray_5_arrays = get_responses(5)

    stim_0 = compute_correlations(stim_0_arrays)
    gray_0 = compute_correlations(gray_0_arrays)
    stim_5 = compute_correlations(stim_5_arrays)
    gray_5 = compute_correlations(gray_5_arrays)

    with h5py.File('data/drift_correlations.h5', 'w') as f:
        group = f.create_group('day_0')
        group.create_dataset('stim', data=stim_0)
        group.create_dataset('gray', data=gray_0)
        group = f.create_group('day_5')
        group.create_dataset('stim', data=stim_5)
        group.create_dataset('gray', data=gray_5)

