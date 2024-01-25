from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr

__all__ = [
    'annotate_onsets',
    'get_IDs',
    'load_traces',
    'normalize_confusion_matrices',
    'normalize_confusion_matrix',
    'remove_ticks',
    'split_by_event',
]

"""
Data Loading
-------------------------------------------------------------------------------
"""

def get_IDs(day: int, letter: str) -> np.ndarray:
    """Get indices of ROIs that are selective for a particular stimulus.

    Parameters
    ----------
    day: int
      Must be 0 or 5.
    letter: str
      A stimulus name (e.g., 'B').

    Returns
    -------
    indices: np.ndarray
      A 1-dimensional (int) array containing indices of ROIs selective for a given stimulus.
    """

    path = Path(__file__).parent / 'data/stimulus_selectivity.xlsx'
    df = pd.read_excel(path, sheet_name=f'day_{day}', index_col=0).fillna("")
    indices = []
    for r in df.iterrows():
        stims = r[1].stimulus.split(",")
        # Handle letter in sequence non-specific list (e.g., 'C' in ['A', 'C']).
        if letter in stims:
            indices.append(r[0])
            continue
        # Handle letter in sequence-specific response (e.g., 'C' in ['AB(C)D']).
        for elt in stims:
            if f'({letter})' in elt:
                indices.append(r[0])

    return np.array(indices, dtype=int)


def load_traces(day: int, sequence: str, drop_gray: bool = False) -> xr.DataArray:
    """Load deconvolved fluorescence data from 'traces.h5'.

    Parameters
    ----------

    day: int
      Must be 0 or 5.
    sequence: str
      Must be 'ABCD', 'ABBD', or 'ACBD'.
    drop_gray: bool, (optional, default=`False`).
      Whether to drop the data for the gray period following sequence presentations.

    Returns
    -------
    arr: xr.DataArray
      Array with dimensions ('trial', 'time', 'roi').

    """
    # Load the data.
    path = Path(__file__).parent / 'data/traces.h5'
    with h5py.File(path, 'r') as f:
        data = f[f'day_{day}/{sequence}'][:]
        arr = xr.DataArray(data, dims=('trial', 'time', 'roi'))

    # Add coordinate array indicating which event each timepoint corresponds to.
    events = xr.DataArray(np.zeros(56, dtype='int'), dims=('time',))
    events[0:8] = 1
    events[8:16] = 2
    events[16:24] = 3
    events[24:32] = 4
    events[32:56] = 5
    if sequence == 'ABBD':
        events += 5
    elif sequence == 'ACBD':
        events += 10
    arr.coords['event'] = events

    # Optionally, drop post-sequence gray periods.
    if drop_gray:
        arr = arr.isel(time=slice(0, 32))

    return arr


"""
Plotting utilities
-------------------------------------------------------------------------------
"""

def annotate_onsets(
    ax: "Axes",
    events: ArrayLike,
    symbol: str = r'$\Delta$',
    skip_first: bool = False,
    last: Optional[str] = None,
    vline: bool = True,
    shift: Number = 0,
    **kw,
) -> None:
    """
    Put onset indicators where labels change. Optionally, add a vertical
    line, usually for heatmaps or traces.

    Parameters
    ----------
    ax

    Returns
    -------

    """

    events = np.asarray(events)

    # Add onset indicators.
    xticks = []
    if not skip_first:
        xticks.append(0 + shift)
    for i in range(1, len(events)):
        if events[i] != events[i - 1]:
            xticks.append(i + shift)
    xticklabels = [symbol] * len(xticks)
    if last:
        if last is True:
            xticklabels[-1] = symbol
        else:
            xticklabels[-1] = last

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # add vertical lines
    if vline:
        kw = dict(kw)
        kw['color'] = kw.get('color', kw.get('c', 'gray'))
        kw['linestyle'] = kw.get('linestyle', kw.get('ls', '--'))
        kw['linewidth'] = kw.get('linewidth', kw.get('lw', 1))
        kw['alpha'] = kw.get('alpha', 1)
        for x in xticks:
            ax.axvline(x, **kw)


def remove_ticks(ax: "Axes") -> "Axes":
    """
    Remove all ticks and tick labels from axes. Usually for images.

    Parameters
    ----------
    ax

    Returns
    -------

    """
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    return ax


def argwhere(arr: ArrayLike) -> ArrayLike:
    """
    Wraps numpy.argwhere so that 1d input arrays will result in 1d output.

    Parameters
    ----------
    arr

    Returns
    -------

    """

    arr = np.asarray(arr)
    ndim = arr.ndim

    inds = np.argwhere(arr)
    if ndim == 1:
        if inds.size == 0:
            return np.array([], dtype=np.intp)
        return np.atleast_1d(inds.squeeze())

    return inds

def split_by_event(arr: xr.DataArray) -> List[xr.DataArray]:
    """
    Split a sequence array into individual arrays based on events.

    Parameters
    ----------
    arr

    Returns
    -------

    """
    labels = arr.coords['event'].data.astype(int)
    inds = argwhere(labels[:-1] != labels[1:]) + 1
    inds = np.r_[0, inds, len(labels)]
    chunks = []
    for i in range(len(inds) - 1):
        start, stop = inds[i], inds[i + 1]
        chunks.append(arr.isel(time=slice(start, stop)))
    return chunks


def normalize_confusion_matrix(cm: np.ndarray, axis: int = 1) -> np.ndarray:
    cm = np.asarray(cm, float)
    return cm / cm.sum(axis=axis)[:, np.newaxis]


def normalize_confusion_matrices(mat: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Batch normalization of confusion matrices.
    """
    lst = [normalize_confusion_matrix(mat[i], axis) for i in range(len(mat))]
    return np.stack(lst)
