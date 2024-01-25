import copy
import datetime
import json

import multiprocessing
# multiprocessing.set_start_method('forkserver')  #?

import os
from pathlib import Path
import shutil

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

import ndindex as nd
import pandas as pd
import matplotlib

from matplotlib.animation import FFMpegWriter
import matplotlib.colors as mpc
from matplotlib.figure import Figure
from ca_analysis.plot import get_smap
import matplotlib.pyplot as plt
import numpy as np

from ca_analysis import *

from ca_analysis.plot import *
from ca_analysis.plot.images import *
from ca_analysis.resampling import *

from seq_learn_3.main import *
from seq_learn_3.utils import *


def get_IDs(day:int, letter:str) -> np.array:
    path = Path(__file__).parent.parent / f'selectivity/cells_day{day}_scott.ods'
    df = pd.read_excel(path, index_col=0).fillna("")

    lst = []

    for r in df.iterrows():
        stims = r[1].stimulus.split(",")
        if letter in stims:
            lst.append(r[0])
            continue
        for elt in stims:
            if f'({letter})' in elt:
                lst.append(r[0])
    return np.array(lst, dtype=int)


def bootstrap_means(arr: xr.DataArray) -> Tuple[xr.DataArray]:
    n_iters = 1000
    n_rois = arr.sizes['roi']
    means = []
    for i in range(n_iters):
        inds = np.random.choice(n_rois, n_rois, replace=True)
        sub = arr.isel(roi=inds)
        means.append(sub.mean('roi'))
    means = np.stack(means)
    low = xr.DataArray(np.percentile(means, 2.5, axis=0), dims=('time',))
    mean = xr.DataArray(np.mean(means, axis=0), dims=('time',))
    high = xr.DataArray(np.percentile(means, 97.5, axis=0), dims=('time',))
    return low, mean, high


sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)

sub_IDs_0 = get_IDs(0, 'B')
sub_IDs_5 = get_IDs(5, 'B')

lpad = None
rpad = 24
drop_gray = True

# sequences = ('ABCD', 'ABBD', 'ACBD')
sequences = ('ABCD', 'ABBD',)
split_kwargs = {
    'lpad': lpad,
    'rpad': rpad,
    'drop_gray': drop_gray,
}
arrays_0 = {}
for seq_name in sequences:
    arr = flex_split(sessions_0, seq_name, **split_kwargs)
    arr = arr.isel(roi=sub_IDs_0)
    data = arr.mean('trial')
    arrays_0[seq_name] = data

arrays_5 = {}
for seq_name in sequences:
    arr = flex_split(sessions_5, seq_name, **split_kwargs)
    arr = arr.isel(roi=sub_IDs_5)
    data = arr.mean('trial')
    arrays_5[seq_name] = data

plt.rcParams['font.size'] = 8

fig, ax = plt.subplots(figsize=(2.35, 1.5))

arr = arrays_0['ABBD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(X, mean, color='black', label='day 0')

arr = arrays_5['ABBD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(X, mean, color='red', label='day 5')

annotate_onsets(ax, arr.coords['event'])
ax.legend(handlelength=1, frameon=False, loc='upper right')
ax.set_xlim([0, 48])
ax.set_title('ABBD: B-cells')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('activity (AU)')
ax.set_xlabel('time')
fig.tight_layout(pad=0.25)
plt.show()
fig.savefig('figures/mean_traces_omission_ABBD.eps')

"""
--------------------------------------------------------------------------------
"""


fig, ax = plt.subplots(figsize=(2.35, 1.5))

arr = arrays_0['ABCD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(X, mean, color='black', label='day 0')

arr = arrays_5['ABCD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(X, mean, color='red', label='day 5')

annotate_onsets(ax, arr.coords['event'])
ax.legend(handlelength=1, frameon=False, loc='upper right')
ax.set_xlim([0, 48])
ax.set_title('ABCD: B-cells')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('activity (AU)')
ax.set_xlabel('time')
fig.tight_layout(pad=0.25)
plt.show()
fig.savefig('figures/mean_traces_omission_ABCD.eps')
