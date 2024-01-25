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



sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)
roi_filter = 'all'
drop_gray = False
tol = 4
offset = 2
# l_offset = 2
# r_offset = 2
apply_roi_filter(sessions_0 + sessions_5, roi_filter)

# ---- day 0
arr = flex_split(sessions_0, 'ABCD', drop_gray=drop_gray)
arr = filter_consistent(arr, tol)
scores = arr.mean('trial').argmax('time')
bins = np.arange(0, arr.sizes['time'] + 1) - 0.5
vals, _ = np.histogram(scores, bins=bins)
counts = vals[0 + offset:8 + offset]
counts += vals[8 + offset:16 + offset]
counts += vals[16 + offset:24 + offset]
counts += vals[24 + offset:32 + offset]

arr = flex_split(sessions_0, 'ABBD', drop_gray=drop_gray)
arr = filter_consistent(arr, tol)
scores = arr.mean('trial').argmax('time')
vals, _ = np.histogram(scores, bins=bins)
counts += vals[0 + offset:8 + offset]
counts += vals[8 + offset:16 + offset]
# counts += vals[16:24]
counts += vals[24 + offset:32 + offset]

arr = flex_split(sessions_0, 'ACBD', drop_gray=drop_gray)
arr = filter_consistent(arr, tol)
scores = arr.mean('trial').argmax('time')
vals, _ = np.histogram(scores, bins=bins)
counts += vals[0 + offset:8 + offset]
counts += vals[8 + offset:16 + offset]
counts += vals[16 + offset:24 + offset]
counts += vals[24 + offset:32 + offset]
counts_0 = counts


# ---- day 5
arr = flex_split(sessions_5, 'ABCD', drop_gray=drop_gray)
arr = filter_consistent(arr, tol)
scores = arr.mean('trial').argmax('time')
bins = np.arange(0, arr.sizes['time'] + 1) - 0.5
vals, _ = np.histogram(scores, bins=bins)
counts = vals[0 + offset:8 + offset]
counts += vals[8 + offset:16 + offset]
counts += vals[16 + offset:24 + offset]
counts += vals[24 + offset:32 + offset]

arr = flex_split(sessions_5, 'ABBD', drop_gray=drop_gray)
arr = filter_consistent(arr, tol)
scores = arr.mean('trial').argmax('time')
vals, _ = np.histogram(scores, bins=bins)
counts += vals[0 + offset:8 + offset]
counts += vals[8 + offset:16 + offset]
# counts += vals[16:24]
counts += vals[24 + offset:32 + offset]

arr = flex_split(sessions_5, 'ACBD', drop_gray=drop_gray)
arr = filter_consistent(arr, tol)
scores = arr.mean('trial').argmax('time')
vals, _ = np.histogram(scores, bins=bins)
counts += vals[0 + offset:8 + offset]
counts += vals[8 + offset:16 + offset]
counts += vals[16 + offset:24 + offset]
counts += vals[24 + offset:32 + offset]
counts_5 = counts


# ------------------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(2.75, 2.5))
xticks = np.array([1, 4, 7]) + 0.5
xticklabels = ['100', '200', '300']

ax, counts = axes[0], counts_0
X = np.arange(len(counts)) + 0.5
ax.bar(X, counts, width=0.9)
# annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
# ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time (msec)')
ax.set_ylabel('count')
ax.set_title('day 0')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax, counts = axes[1], counts_5
X = np.arange(len(counts)) + 0.5
ax.bar(X, counts, width=0.9)
# annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
# ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time (msec)')
ax.set_ylabel('count')
ax.set_title('day 5')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig.tight_layout(pad=0.8)
plt.show()
fig.savefig('figures/time_fields_stim.eps')

counts = counts_0
vals = []
for i in range(8):
    vals = vals + [i] * counts[i]
v0 = np.array(vals)

counts = counts_5
vals = []
for i in range(8):
    vals = vals + [i] * counts[i]
v5 = np.array(vals)
