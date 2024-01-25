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

import h5py
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.genmod.families.links import identity, log
import xarray as xr

from ca_analysis import *

from main import *
from utils import *


def normalize(lst):
    return [arr / np.linalg.norm(arr) for arr in lst]


def get_similarity(vecs):
    n = len(vecs)
    mat = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            a, b = vecs[i], vecs[j]
            mat[i, j] = np.dot(a, b)
    return mat


sessions_0 = get_sessions(day=0, fs=0)
sessions_5 = get_sessions(day=5, fs=0)

kind = 'visual'
sequence = 'ABCD'
savedir = Path.home() / 'plots/seq_learn_3/pattern_similarity'
savedir.mkdir(parents=True, exist_ok=True)

apply_roi_filter(sessions_0, 'gratings')
apply_roi_filter(sessions_5, 'gratings')

ABCD_0 = split_by_event(flex_split(sessions_0, sequence))
ABCD_0 = [arr.mean('time').mean('trial') for arr in ABCD_0]
ABCD_0 = normalize(ABCD_0)
mat_0 = get_similarity(ABCD_0)

ABCD_5 = split_by_event(flex_split(sessions_5, sequence))
ABCD_5 = [arr.mean('time').mean('trial') for arr in ABCD_5]
ABCD_5 = normalize(ABCD_5)
mat_5 = get_similarity(ABCD_5)


fig, axes = plt.subplots(1, 2)

ax = axes[0]
im = ax.imshow(mat_0, cmap='gray')
ax.set_title(f'day 0: {sequence}')

ax = axes[1]
im = ax.imshow(mat_5, cmap='gray')
ax.set_title(f'day 5: {sequence}')



plt.show()

path = savedir / f'{sequence}.png'
fig.savefig(path)

# fig, ax = plt.subplots()
# diff = mat_5 - mat_0
# im = ax.imshow(diff, cmap='coolwarm')
# plt.colorbar(im)
# fig.suptitle(f'day 5 - day 0: {sequence}')


