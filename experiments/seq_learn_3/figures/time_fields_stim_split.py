from types import SimpleNamespace
from typing import Sequence

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
from scipy.stats import *

from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *
from main import *
from seq_learn_3.utils import *


"""
Create histograms of max firing rates from onset of a sequence.
"""




def filter_consistent(arr: xr.DataArray, tol: int) -> xr.DataArray:
    even_inds = np.arange(0, arr.sizes['trial'], 2)
    even = arr.isel(trial=even_inds).mean('trial')
    odd = arr.isel(trial=even_inds + 1).mean('trial')
    even_scores = even.argmax('time')
    odd_scores = odd.argmax('time')
    delta = np.abs(even_scores - odd_scores)
    arg_good = argwhere(delta < tol)
    print(len(arg_good) / len(delta))
    out = arr.isel(roi=arg_good)
    return out


def get_counts(sessions, drop_gray, tol, trials=None):
    arr = flex_split(sessions, 'ABCD', drop_gray=drop_gray)
    arr = filter_consistent(arr, tol)
    if trials is not None:
        arr = arr.isel(trial=trials)
    scores = arr.mean('trial').argmax('time')
    bins = np.arange(0, arr.sizes['time'] + 1) - 0.5
    vals, _ = np.histogram(scores, bins=bins)
    counts = vals[0 + offset:8 + offset]
    counts += vals[8 + offset:16 + offset]
    counts += vals[16 + offset:24 + offset]
    counts += vals[24 + offset:32 + offset]

    arr = flex_split(sessions, 'ABBD', drop_gray=drop_gray)
    arr = filter_consistent(arr, tol)
    if trials is not None:
        arr = arr.isel(trial=trials)
    scores = arr.mean('trial').argmax('time')
    vals, _ = np.histogram(scores, bins=bins)
    counts += vals[0 + offset:8 + offset]
    counts += vals[8 + offset:16 + offset]
    # counts += vals[16:24]
    counts += vals[24 + offset:32 + offset]

    arr = flex_split(sessions, 'ACBD', drop_gray=drop_gray)
    arr = filter_consistent(arr, tol)
    if trials is not None:
        arr = arr.isel(trial=trials)
    scores = arr.mean('trial').argmax('time')
    vals, _ = np.histogram(scores, bins=bins)
    counts += vals[0 + offset:8 + offset]
    counts += vals[8 + offset:16 + offset]
    counts += vals[16 + offset:24 + offset]
    counts += vals[24 + offset:32 + offset]

    return counts


def norm_counts(c):
    return c / c.sum()


plt.rcParams['font.size'] = 8
day = 5
roi_filter = 'all'
drop_gray = False
tol = 4
offset = 2
ylim = [0, 0.2]

sessions = get_sessions(day=day)
apply_roi_filter(sessions, roi_filter)

# ---- day 0

counts_front = get_counts(sessions, drop_gray, tol, trials=slice(0, 250))
counts_back = get_counts(sessions, drop_gray, tol, trials=slice(250, None))
counts_all = get_counts(sessions, drop_gray, tol)

# ------------------------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(3, 4))
xticks = np.array([1, 4, 7]) + 0.5
xticklabels = ['100', '200', '300']

ax, counts = axes[0], counts_front
counts = norm_counts(counts)
X = np.arange(len(counts)) + 0.5
ax.bar(X, counts, width=0.9)
# annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
# ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time (msec)')
ax.set_ylabel('frac')
ax.set_title('front')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(ylim)

ax, counts = axes[1], counts_back
counts = norm_counts(counts)
X = np.arange(len(counts)) + 0.5
ax.bar(X, counts, width=0.9)
# annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
# ax.set_xlim([-1, arr.sizes['time']])
# ax.set_xlabel('time (msec)')
ax.set_ylabel('frac')
ax.set_title('back')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(ylim)

ax, counts = axes[2], counts_all
counts = norm_counts(counts)
X = np.arange(len(counts)) + 0.5
ax.bar(X, counts, width=0.9)
# annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
# ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time (msec)')
ax.set_ylabel('frac')
ax.set_title('all')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(ylim)

fig.tight_layout(pad=0.8)
plt.show()
fig.savefig(f'figures/time_fields_stim_split_day_{day}.eps')

