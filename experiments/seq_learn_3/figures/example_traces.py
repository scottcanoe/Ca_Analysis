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

plt.rcParams['font.size'] = 8

# sessions = get_sessions(day=0)
# lpad = None
# rpad = 24
# drop_gray = True
#
# apply_roi_filter(sessions, 'gratings')
# split_kwargs = {
#     'lpad': lpad,
#     'rpad': rpad,
#     'drop_gray': drop_gray,
# }
#
# spks_all = flex_split(sessions, 'ABCD', target='spikes')
# F_all = flex_split(sessions, 'ABCD', target='F')
# events = spks_all.coords['event']
# rois = range(20, 40)
# trial = 15
# for r in rois:
#     spks = spks_all.isel(roi=r).isel(trial=trial)
#     F = F_all.isel(roi=r).isel(trial=trial)
#     fig, axes = plt.subplots(2, 1)
#     ax = axes[0]
#     ax.plot(F)
#     ax = axes[1]
#     ax.plot(spks)
#     fig.suptitle(f'roi={r}, trial={trial}')
#     for ax in axes:
#         annotate_onsets(ax, events)
#     plt.show()
# plt.rcParams['font.size'] = 8
# r = 29
# trial = 15
# spks = spks_all.isel(roi=r).isel(trial=trial)
# F = F_all.isel(roi=r).isel(trial=trial)
# fig, axes = plt.subplots(2, 1, figsize=(2, 1.5))
# ax = axes[0]
# ax.plot(F)
# ax = axes[1]
# ax.plot(spks)
# # fig.suptitle(f'roi={r}, trial={trial}')
# # for ax in axes:
# #     annotate_onsets(ax, events)
# fig.tight_layout(pad=0.2)
#
# for ax in axes:
#     ax.set_xticks([])
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
# plt.show()
#
# fig.savefig('figures/example_traces.png')

roi = 0
time = slice(23500, 24000)
lw = 0.75
sessions = get_sessions(day=0)
s = sessions[0]
F_all = s.F.data
spikes_all = s.spikes.data
F = F_all.isel(roi=roi).isel(time=time)
spikes = spikes_all.isel(roi=roi).isel(time=time)
fig, axes = plt.subplots(2, 1, figsize=(2, 1.5))
ax = axes[0]
ax.plot(F, color='black', lw=lw)
ax = axes[1]
ax.plot(spikes, color='blue', lw=lw)
plt.tight_layout(pad=0.2)
for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
plt.show()
fig.savefig('figures/example_traces_separated.eps')


sessions = get_sessions(day=0)
s = sessions[0]
F_all = s.F.data
spikes_all = s.spikes.data
F = F_all.isel(roi=roi).isel(time=time)
spikes = spikes_all.isel(roi=roi).isel(time=time)
fig, ax = plt.subplots(1, 1, figsize=(2, 1))
ax.plot(F, color='black', label='F', lw=lw)
ax.plot(spikes, color='blue', label='deconv.', lw=lw)
# ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.legend()
plt.tight_layout(pad=0.2)
plt.show()
fig.savefig('figures/example_traces_combined.eps')
