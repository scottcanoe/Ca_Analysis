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
from scipy.stats import kstest, zscore

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
            xticks.append(i)
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


def split_chunks(arr: xr.DataArray) -> xr.DataArray:
    chunk_vals = []
    for val in arr.coords["chunk"].values:
        if val not in chunk_vals:
            chunk_vals.append(val)
    return [get_chunk(arr, num) for num in chunk_vals]


def reorder_chunks(arr: xr.DataArray, order: ArrayLike) -> xr.DataArray:
    lst = [get_chunk(arr, num) for num in order]
    out = concat(lst, "time")
    out.attrs.update(arr.attrs)
    out.name = arr.name
    return out




sessions = get_sessions(day=5)

ses = sessions[0:1]
visual_only = True

for s in ses:
    visual_spikes = s.data.attrs.get("visual_spikes")
    if visual_spikes is None:
        visual_ids = np.load(s.fs.getsyspath("visual.npy"))
        spks = s.data.get("spikes")
        spks = spks.isel(roi=visual_ids)
        s.data.attrs["visual_spikes"] = spks
        s.data.attrs["all_spikes"] = s.data.get("spikes")
if visual_only:
    s.data.attrs["spikes"] = s.data.attrs["visual_spikes"]
else:
    s.data.attrs["spikes"] = s.data.attrs["all_spikes"]

gray_size = 10
ABCD = get_seq_data(ses, ["gray", "A", "B", "C", "D", "gray"], gray_size=gray_size)
ABCD.name = "ABCD"
ABBD = get_seq_data(ses, ["gray", "A", "B", "B", "D", "gray"], gray_size=gray_size)
ABBD.name = "ABBD"
ACBD = get_seq_data(ses, ["gray", "A", "C", "B", "D", "gray"], gray_size=gray_size)
ACBD.name = "ACBD"
ABBD_diff = ABBD - ABCD
ABBD_diff.name = "ABBD - ABCD"
ACBD_diff = ACBD - ABCD
ACBD_diff.name = "ACBD - ABCD"


def handle_XY(mat: xr.DataArray):
    y = mat.mean("roi")
    y = resample1d(y, "time", factor=1, preserve_ints=True, method="cubic")
    y = y - y.min().item()
    # y.data = zscore(y.data, axis=y.get_axis_num("time"))
    x = np.arange(y.sizes["time"])
    return x, y


fig, axes = plt.subplots(2, 1, figsize=(8, 8))

ax = axes[0]

arr = ABCD
X, Y = handle_XY(arr)
ax.plot(X, Y, color="black", label=arr.name)

arr = ABBD
X, Y = handle_XY(arr)
ax.plot(X, Y, color="red", label=arr.name)

arr = ACBD
X, Y = handle_XY(arr)
ax.plot(X, Y, color="blue", label=arr.name)

annotate_onsets(ax, Y.coords["chunk"], color='gray', skip_first=True, last="-")
ax.legend(loc='upper left')
title = "mice: " + ", ".join([s.mouse for s in ses])
ax.set_title(title)
ax.set_xlim([X[0], X[-1]])

ax = axes[1]

arr = ABBD_diff
X, Y = handle_XY(arr)
ax.plot(X, Y, color="red", label=arr.name)

arr = ACBD_diff
X, Y = handle_XY(arr)
ax.plot(X, Y, color="blue", label=arr.name)

annotate_onsets(ax, Y.coords["chunk"], color='gray', skip_first=True, last="-")
ax.legend(loc='upper left')
ax.set_xlim([X[0], X[-1]])
plt.show()

plotdir = Path.home() / "plots/seq_learn_2/mean_traces"
fig.savefig(plotdir / f"{title}.pdf")
