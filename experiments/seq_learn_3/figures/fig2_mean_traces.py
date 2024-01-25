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



sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)


lpad = None
rpad = 24
drop_gray = True
trials = slice(0, None)
visual = 'all'
sequences = ('ABCD', 'ABBD', 'ACBD')

apply_roi_filter(sessions_0 + sessions_5, visual)
split_kwargs = {
    'lpad': lpad,
    'rpad': rpad,
    'drop_gray': drop_gray,
}
arrays_0 = {}
for seq_name in sequences:
    arr = flex_split(sessions_0, seq_name, **split_kwargs)
    m0 = arr.sizes['roi']
    arr = arr.isel(trial=trials)
    data = arr.mean('trial').mean('roi')
    arrays_0[seq_name] = data

arrays_5 = {}
for seq_name in sequences:
    arr = flex_split(sessions_5, seq_name, **split_kwargs)
    m5 = arr.sizes['roi']
    arr = arr.isel(trial=trials)
    data = arr.mean('trial').mean('roi')
    arrays_5[seq_name] = data


plt.rcParams['font.size'] = 8
fig, axes = plt.subplots(2, 1, figsize=(2.75, 2.5))

ax, arrays = axes[0], arrays_0
ax.plot(arrays['ABCD'], label='ABCD', color='black')
ax.plot(arrays['ABBD'], label='ABBD', color='blue')
ax.plot(arrays['ACBD'], label='ACBD', color='red')
ax.set_title('day 0')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('activity')
ax.legend(handlelength=1, frameon=False)

ax, arrays = axes[1], arrays_5
ax.plot(arrays['ABCD'], label='ABCD', color='black')
ax.plot(arrays['ABBD'], label='ABBD', color='blue')
ax.plot(arrays['ACBD'], label='ACBD', color='red')
ax.set_title('day 5')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('activity')
# ax.legend(frameon=True, handlelength=0.1, bbox_to_anchor=(1.2, 0.7))
# ax.legend(frameon=False, framealpha=0.0, loc='upper right')
# ax.legend()

for ax in axes:
    annotate_onsets(
        ax,
        ABCD.coords['event'],
        skip_first=False,
        last='-',
        color='gray',
        lw=0.5,
    )
    ax.set_xlim([0, arrays_0['ABCD'].sizes['time'] - 1])
    ax.set_ylim([0.5, 1])

# axes[0].legend(framealpha=1, frameon=False, loc='upper right')
axes[1].set_xlabel('time')
axes[1].set_ylabel('activity')
fig.tight_layout(pad=0.5)

plt.show()

fig.savefig('figures/mean_traces.eps')

#
#
# fig, axes = plt.subplots(3, 1, figsize=[3.5, 2])
#
# ax, event = axes[0], 'ABCD'
# ax.plot(arrays_0[event], label='naive', color='black')
# ax.plot(arrays_5[event], label='trained', color='red')
# ax.set_title(event)
# ax.legend()
#
# ax, event = axes[1], 'ABBD'
# ax.plot(arrays_0[event], label='naive', color='black')
# ax.plot(arrays_5[event], label='trained', color='red')
# ax.set_title(event)
#
# ax, event = axes[2], 'ACBD'
# ax.plot(arrays_0[event], label='naive', color='black')
# ax.plot(arrays_5[event], label='trained', color='red')
# ax.set_title(event)
#
#
# skip_first = lpad is not None
# last = '-' if rpad is not None else None
# ABCD = arrays_0['ABCD']
# for ax in axes:
#     annotate_onsets(
#         ax,
#         ABCD.coords['event'],
#         skip_first=skip_first,
#         last=last,
#         color='gray',
#         lw=0.5,
#     )
#     ax.set_xlim([0, ABCD.sizes['time'] - 1])
#
# fig.tight_layout(pad=1)
# plt.show()
#
# fname = 'figures/fig2b_traces.png'
# fig.savefig('figures/fig2b_traces.png')
#
#
#
#
