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

from main import *
from processing import *


def backward_counts(seq: Sequence, normalize: bool = False) -> np.ndarray:
    mat = np.zeros([7, 7], dtype=int)
    for i in range(1, len(seq)):
        pre, post = seq[i - 1], seq[i]
        mat[pre, post] += 1
    mat = mat[2:, 2:]
    counts = np.zeros([5, 2], dtype=int)
    for i in range(5):
        counts[i, 0] = mat[(i - 1) % 5, i]
        counts[i, 1] = mat[(i - 2) % 5, i]
    if normalize:
        counts = counts.astype(float)
        sums = counts.sum(axis=1)
        for i in range(5):
            counts[i] = counts[i] / sums[i]

    return counts


def forward_counts(seq: Sequence, normalize: bool = False) -> NDArray:

    mat = np.zeros([7, 7], dtype=int)
    for i in range(len(seq) - 1):
        pre, post = seq[i], seq[i + 1]
        mat[pre, post] += 1
    mat = mat[2:, 2:]
    counts = np.zeros([5, 2], dtype=int)
    for i in range(5):
        counts[i, 0] = mat[i, (i + 1) % 5]
        counts[i, 1] = mat[i, (i + 2) % 5]
    if normalize:
        counts = counts.astype(float)
        sums = counts.sum(axis=1)
        for i in range(5):
            counts[i] = counts[i] / sums[i]
    return counts


def item_frequency(seq):
    freq = np.zeros(7)
    for i in range(len(freq)):
        freq[i] = np.sum(seq == i)
    freq = freq / len(seq)
    return freq



nas = get_fs("ca-nas").opendir("sessions")
ssd = get_fs("ssd").opendir("sessions")

sessions = get_sessions()
for s in sessions:
    fs_remote = nas.opendir(f"{s.mouse}/{s.date}/{s.exp}")
    s.fs = fs_remote
    # fs_local = s.fs
    # if not s.reg["thor_sync"].exists():
    #     s.reg["thorlabs"].path.mkdir(exist_ok=True)
    #     src = fs_remote.getsyspath("thorlabs/Episode001.h5")
    #     dst = fs_local.getsyspath("thorlabs/Episode001.h5")
    #     shutil.copy2(src, dst)
    process_events(s, force=True)

# s = sessions[1]
# schema = s.events.schema
# attrs = s.attrs
#
# df = s.data.get("event_table")
# df = df[df['block_id'] > 1]
# seq = np.array(df.event_id)
# seq = seq - 2
# seq = seq[seq >= 0]
#
# f = forward_counts(seq, True)
# b = backward_counts(seq, True)


# freqs = item_frequency(seq)
# f = f_counts / f_counts.sum(axis=1).reshape(5, 1)
# b = b_counts / b_counts.sum(axis=1).reshape(5, 1)
#
# sessions = get_sessions()
# for s in sessions:
#     df = s.data.get('event_table')
#     df = df[df['block_id'] > 1]
#     seq = np.array(df.event_id)
#     seq = seq - 2
#     freq = item_frequency(seq)
#     print(freq)
#
# X, Y = b_counts[:, 0], b_counts[:, 1]
# Z = X - Y
# X, Y = b[:, 0], b[:, 1]
# Z = X - Y
#
#
# out = np.roll(c, -2, axis=0)
# m = s.data.split("spikes", ["A", "B"])
# (trials, rois, time)
# shape = (5, 3, 5)
# mat = np.arange(np.prod(shape)).reshape(shape)
# arr = xr.DataArray(mat, dims=('trial', 'roi', 'time'))
# A = arr
#
# shape = (4, 5, 5)
# mat = np.arange(np.prod(shape)).reshape(shape)
# arr = xr.DataArray(mat, dims=('trial', 'roi', 'time'))
# B = arr
#
# a, b = A, B
# a, b = a.mean('time'), b.mean('time')
# aa, bb = a.mean('trial'), b.mean('trial')
#
# dct = {'a': a, 'b': b}
# ds = xr.Dataset(dct)
