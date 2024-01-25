import copy
import datetime
import logging
import dataclasses
import types
from numbers import Number
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from typing import (
    Any, Callable,
    List,
    Mapping,
    NamedTuple, Optional,
    Sequence,
    Tuple,
    Union,
)

import dask.array as da
import h5py
import ndindex as nd
import pandas as pd
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from statsmodels.distributions.empirical_distribution import ECDF
import xarray as xr
import yaml

from ca_analysis import *
from ca_analysis.io import *
from ca_analysis.stats import *
from ca_analysis.persistence import *
from ca_analysis.plot import *

from main import *
from processing import *




def annotate_onsets(
    ax: "Axes",
    labels: ArrayLike,
    skip_first: bool = False,
    last: Optional[str] = None,
    vline: bool = True,
    color: Union[ArrayLike, str] = "white",
    ls: str = "--",
    alpha: Number = 1,
) -> None:

    # Add onset indicators.
    xticks = []
    if not skip_first:
        xticks.append(0)
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            xticks.append(i - 0.5)
    xticklabels = [r'$\Delta$'] * len(xticks)
    if last is not None:
        xticklabels[-1] = last
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # add vertical lines
    if vline:
        for x in xticks:
            ax.axvline(x, color=color, ls=ls, alpha=alpha)


def concat(
    arrays: Sequence[xr.DataArray],
    dim: str,
    array_coord: Optional[str] = None,
    **kw,
) -> xr.DataArray:

    out = xr.concat(arrays, dim, **kw)
    if array_coord:
        chunks = []
        for i, arr in enumerate(arrays):
            chunks.append(np.full(arr.sizes[dim], i, dtype=int))
        chunks = np.hstack(chunks)
        out.coords[array_coord] = xr.DataArray(chunks, dims=(dim,))
    return out

def ranked(arr: xr.DataArray, rank_by: Union[str, ArrayLike]) -> xr.DataArray:
    if is_str(rank_by):
        stat = getattr(seq[1], rank_by)("time")
        inds = np.flipud(np.argsort(stat))
        arr = arr.isel(roi=inds)
    elif isinstance(rank_by, xr.DataArray):
        arr = arr.isel(roi=rank_by)
    else:
        raise NotImplementedError
    return arr

def get_seq_data_one(
    s: Session,
    seq: Sequence,
    resample_factor: Optional[int] = None,
    gray_size: Optional[int] = None,
    trial_average: bool = True,
) -> xr.DataArray:

    # collect data
    seq = s.data.split(seq)
    if gray_size is not None:
        seq[0] = seq[0].isel(time=slice(-gray_size, None))
        seq[-1] = seq[-1].isel(time=slice(0, gray_size))
    if trial_average:
        for i, arr in enumerate(seq):
            seq[i] = arr.mean("trial")

    # combine arrays
    arr = concat(seq, "time", array_coord="chunk").transpose("roi", "time", ...)

    # resample
    if resample_factor is not None and resample_factor != 1:
        arr = resample1d(arr, "time", factor=resample_factor, preserve_ints=True)

    return arr


def get_seq_data(
    ses: Union[Session, Sequence[Session]],
    seq: Sequence,
    resample_factor: Optional[int] = None,
    gray_size: Optional[int] = None,
    trial_average: bool = True,
) -> xr.DataArray:

    if isinstance(ses, Session):
        return get_seq_data_one(
            ses,
            seq,
            resample_factor=resample_factor,
            gray_size=gray_size,
            trial_average=trial_average,
        )
    arrays = []
    for s in ses:
        arr = get_seq_data_one(
            s,
            seq,
            resample_factor=resample_factor,
            gray_size=gray_size,
            trial_average=trial_average,
        )
        arrays.append(arr)
    out = xr.concat(arrays, "roi")
    if is_str(seq):
        out.name = seq
    return out


def get_chunk(arr: xr.DataArray, num: int) -> xr.DataArray:
    out = arr.sel(time=arr['chunk'] == num)
    return out


def reorder_chunks(arr: xr.DataArray, order: ArrayLike) -> xr.DataArray:
    lst = [get_chunk(arr, num) for num in order]
    out = concat(lst, "time")
    out.attrs.update(arr.attrs)
    out.name = arr.name
    return out


def split_chunks(arr: xr.DataArray) -> xr.DataArray:
    chunk_vals = []
    for val in arr.coords["chunk"].values:
        if val not in chunk_vals:
            chunk_vals.append(val)
    return [get_chunk(arr, num) for num in chunk_vals]


def get_nth(arr: xr.DataArray, num: int) -> xr.DataArray:
    start = num - 2
    inds = np.arange(100) * (num - 2) + num - 2
    out = arr.isel(trial=inds)
    return out


sessions = get_sessions()
ses = sessions[:]

s = sessions[0]

# ABCD_1 = get_seq_data(ses, "ABCD_1")
ACBD_1 = get_seq_data(ses, "ACBD_1", trial_average=False)
# ABCD_5 = get_seq_data(ses, "ABCD_5")
# ACBD_5 = get_seq_data(ses, "ACBD_5")
ABCD_10 = get_seq_data(ses, "ABCD_10", trial_average=False)
ACBD_10 = get_seq_data(ses, "ACBD_10", trial_average=False)

seq1 = ACBD_1
# seq1 = get_nth(seq1, 10)
seq2 = ACBD_10

seq1 = seq1.mean("trial")
seq2 = seq2.mean("trial")

seq1 = resample1d(seq1, "time", factor=2, preserve_ints=True)
seq2 = resample1d(seq2, "time", factor=2, preserve_ints=True)

# seq2 = reorder_chunks(seq2, [0, 2, 1, 3])
diff = seq2 - seq1
diff.coords["chunk"] = seq1.coords["chunk"]
diff.name = f"{seq2.name} - {seq1.name}"

obj = diff.sel(time=diff['chunk'] == 3)
stat = obj.mean("time")
inds = np.flipud(np.argsort(stat))
seq1 = seq1.isel(roi=inds)
seq2 = seq2.isel(roi=inds)
diff = diff.isel(roi=inds)

fig = plt.figure(figsize=(12, 12))
axes = [fig.add_subplot(3, 1, i) for i in range(1, 4)]

ax = axes[0]
arr = seq1
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
ax.imshow(cdata, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["chunk"], skip_first=False, alpha=0.5)
ax.set_title(seq1.name)

ax = axes[1]
arr = seq2
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
ax.imshow(cdata, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["chunk"], skip_first=False, alpha=0.5)
ax.set_title(seq2.name)

ax = axes[2]
arr = diff
smap = get_smap("coolwarm", data=diff, qlim=(50, 97.5))
cdata = smap(arr)
ax.imshow(cdata, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["chunk"], skip_first=False, alpha=0.5)
ax.set_title(diff.name)

fig.tight_layout()
plt.show()
