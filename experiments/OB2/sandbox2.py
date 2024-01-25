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


def get_seq_data_one(
    s: Session,
    seq: Sequence,
    resample_factor: Optional[int] = None,
    gray_size: Optional[int] = None,
    trial_average: bool = True,
) -> xr.DataArray:

    # collect data
    seq = s.data.split(seq, target="spikes")
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


def get_nth(arr: xr.DataArray, num: int) -> xr.DataArray:
    start = num - 2
    inds = np.arange(100) * (num - 2) + num - 2
    out = arr.isel(trial=inds)
    return out


def make_plot(ses):

    ABCD_1 = get_seq_data(ses, "ABCD_1", trial_average=False)
    ACBD_1 = get_seq_data(ses, "ACBD_1", trial_average=False)
    ABCD_5 = get_seq_data(ses, "ABCD_5", trial_average=False)
    ACBD_5 = get_seq_data(ses, "ACBD_5", trial_average=False)
    ABCD_10 = get_seq_data(ses, "ABCD_10", trial_average=False)
    ACBD_10 = get_seq_data(ses, "ACBD_10", trial_average=False)

    def handle_XY(mat: xr.DataArray):
        y = mat.mean("trial").mean("roi")
        y = resample1d(y, "time", factor=2, preserve_ints=True, method="cubic")
        # y.data = zscore(y.data, axis=y.get_axis_num("time"))
        x = np.arange(y.sizes["time"])
        return x, y

    def norm_Y(Y: xr.DataArray):
        # Y.data = zscore(Y.data, axis=Y.get_axis_num("time"))
        Y.data = Y.data - Y.data[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    ylim = [-3, 3]

    ax = axes[0]
    arr = ABCD_1
    X1, Y1 = handle_XY(arr)
    arr = ACBD_1
    X2, Y2 = handle_XY(arr)
    X, Y = X1, Y2 - Y1
    norm_Y(Y)
    ax.plot(X, Y, color="black", label="ACBD_1 - ABCD_1")
    annotate_onsets(ax, Y.coords["chunk"], color='gray')
    # ax.set_ylim(ylim)
    ax.axhline(0)
    ax.legend()

    ax = axes[1]
    arr = ABCD_5
    arr = get_nth(arr, 5)
    X1, Y1 = handle_XY(arr)
    arr = ACBD_5
    X2, Y2 = handle_XY(arr)
    X, Y = X1, Y2 - Y1
    norm_Y(Y)
    ax.plot(X, Y, color="black", label="ACBD_5 - ABCD_5")
    annotate_onsets(ax, Y.coords["chunk"], color='gray')
    ax.axhline(0)
    # ax.set_ylim(ylim)
    ax.legend()

    ax = axes[2]
    arr = ABCD_10
    arr = get_nth(arr, 10)
    X1, Y1 = handle_XY(arr)
    arr = ACBD_10
    X2, Y2 = handle_XY(arr)
    X, Y = X1, Y2 - Y1
    norm_Y(Y)
    ax.plot(X, Y, color="black", label="ACBD_10 - ABCD_10")
    annotate_onsets(ax, Y.coords["chunk"], color='gray')
    ax.axhline(0)
    # ax.set_ylim(ylim)
    ax.legend()

    ylims = [ax.get_ylim() for ax in axes]
    ymin = min(ylims[i][0] for i in range(len(ylims)))
    ymax = max(ylims[i][1] for i in range(len(ylims)))
    for ax in axes:
        ax.set_ylim([ymin, ymax])
    title = "mice:" + ", ".join([s.mouse for s in ses])
    axes[0].set_title(title)
    plt.show()
    return fig

plotdir = Path.home() / "plots/OB2"
sessions = get_sessions()
# ses = sessions[:]
ses = sessions[0:2]

fig = make_plot([sessions[0]])
plt.show()
fig.savefig(plotdir / "diff_0.pdf")

fig = make_plot([sessions[1]])
plt.show()
fig.savefig(plotdir / "diff_1.pdf")

fig = make_plot([sessions[2]])
plt.show()
fig.savefig(plotdir / "diff_2.pdf")

fig = make_plot(sessions[0:2])
plt.show()
fig.savefig(plotdir / "diff_3.pdf")

fig = make_plot(sessions)
plt.show()
fig.savefig(plotdir / "diff_4.pdf")

# ABCD_1 = get_seq_data(ses, "ABCD_1", trial_average=False)
# ACBD_1 = get_seq_data(ses, "ACBD_1", trial_average=False)
# ABCD_5 = get_seq_data(ses, "ABCD_5", trial_average=False)
# ACBD_5 = get_seq_data(ses, "ACBD_5", trial_average=False)
# ABCD_10 = get_seq_data(ses, "ABCD_10", trial_average=False)
# ACBD_10 = get_seq_data(ses, "ACBD_10", trial_average=False)
#

plotdir = Path.home() / "plots/OB2/mean_traces"
# fig.savefig(plotdir / "4.pdf")
