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


plt.rcParams['font.size'] = 8

day = 0
sessions = get_sessions(day=day)

B_ids = get_IDs(0, 'B')

# ABCD_all = flex_split(sessions, 'ABCD', drop_gray=False)
# ABBD_all = flex_split(sessions, 'ABBD', drop_gray=False)
# ACBD_all = flex_split(sessions, 'ACBD', drop_gray=False)

roi = 9

ABCD = ABCD_all.isel(roi=roi)
ABBD = ABBD_all.isel(roi=roi)
ACBD = ACBD_all.isel(roi=roi)

fig, ax = plt.subplots(figsize=(2.35, 1.5))
qlim = [0.40, 0.60]

arr = ABCD
mean = arr.mean('trial')
ax.plot(mean, color='black', label='ABCD')
# low = arr.quantile(qlim[0], 'trial')
# high = arr.quantile(qlim[1], 'trial')
# X = np.arange(len(low))
# ax.fill_between(X, low, high, color='black', alpha=0.5)

arr = ABBD
mean = arr.mean('trial')
ax.plot(mean, color='red', label='ABBD')

arr = ACBD
mean = arr.mean('trial')
ax.plot(mean, color='blue', label='ABBD')

ax.legend(handlelength=1, frameon=False)
annotate_onsets(ax, ABCD.coords['event'], skip_first=False, last='-')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('activity (AU)')
ax.set_xlabel('time')
# ax.set_xlim([0, ABCD.sizes['time'] - 1])
ax.set_xlim([0, 40])
fig.tight_layout(pad=0.5)
plt.show()

#
# plt.rcParams['font.size'] = 8
# fig, axes = plt.subplots(2, 1, figsize=(2.75, 2.5))
#
# ax, arrays = axes[0], arrays_0
# ax.plot(arrays['ABCD'], label='ABCD', color='black')
# ax.plot(arrays['ABBD'], label='ABBD', color='blue')
# ax.plot(arrays['ACBD'], label='ACBD', color='red')
# ax.set_title('day 0')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylabel('activity')
# ax.legend(handlelength=1, frameon=False)
#
# ax, arrays = axes[1], arrays_5
# ax.plot(arrays['ABCD'], label='ABCD', color='black')
# ax.plot(arrays['ABBD'], label='ABBD', color='blue')
# ax.plot(arrays['ACBD'], label='ACBD', color='red')
# ax.set_title('day 5')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylabel('activity')
# # ax.legend(frameon=True, handlelength=0.1, bbox_to_anchor=(1.2, 0.7))
# # ax.legend(frameon=False, framealpha=0.0, loc='upper right')
# # ax.legend()
#
# for ax in axes:
#     annotate_onsets(
#         ax,
#         ABCD.coords['event'],
#         skip_first=False,
#         last='-',
#         color='gray',
#         lw=0.5,
#     )
#     ax.set_xlim([0, arrays_0['ABCD'].sizes['time'] - 1])
#     ax.set_ylim([0.5, 1])
#
# # axes[0].legend(framealpha=1, frameon=False, loc='upper right')
# axes[1].set_xlabel('time')
# axes[1].set_ylabel('activity')
# fig.tight_layout(pad=0.5)
#
# plt.show()
#
# fig.savefig('figures/mean_traces.eps')
