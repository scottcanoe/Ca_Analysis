from types import SimpleNamespace
from typing import Sequence

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
from scipy.stats import *

from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *
from seq_learn_3.main import *
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


def high_pass(arr, thresh = 2):
    maxs = arr.max('time')
    tf = maxs > thresh
    out = arr.isel(roi=tf)
    return out

plt.rcParams['font.size'] = 8
tol = 4
day = 5
chunksize = 250
ylim = [0, 0.3]

sessions = get_sessions(day=day)
ABCD_ = flex_split(sessions, 'ABCD._')
ABBD_ = flex_split(sessions, 'ABBD._')
ACBD_ = flex_split(sessions, 'ACBD._')
arrays = [ABCD_, ABBD_, ACBD_]
lst = []
for i in range(5):
    slc = slice(i * 100, (i + 1) * 100)
    for j in range(3):
        arr = arrays[j]
        lst.append(arr.isel(trial=slc))
day_0 = xr.concat(lst, 'trial')
day_0 = filter_consistent(day_0, tol)

# plot distance from onset.
fig, axes = plt.subplots(3, 1, figsize=[3, 4])
bins = np.arange(0, day_0.sizes['time'] + 1) - 0.5

ax, arr = axes[0], day_0
arr = arr.isel(trial=slice(0, chunksize))
arr = arr.mean('trial')
scores_0 = arr.argmax('time')
hist = ax.hist(scores_0, bins=bins, width=0.9, density=True)
ax.set_xlim([-1, arr.sizes['time']])
# ax.set_xlabel('time (msec)')
ax.set_ylabel('fraction')
ax.set_title(f'day {day}: front')
# ax.set_xlim([1.4, arr.sizes['time']])
# ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
# ax.set_xticklabels([100, 200, 300, 400, 500, 600, 700])
ax.set_xticks([0, 8, 16, 24])
ax.set_xticklabels([0, 250, 500, 750])
ax.set_ylim(ylim)

ax, arr = axes[1], day_0
arr = arr.isel(trial=slice(500 - chunksize, 500))
arr = arr.mean('trial')
scores_0 = arr.argmax('time')
ax.hist(scores_0, bins=bins, width=0.9, density=True)
ax.set_xlim([-1, arr.sizes['time']])
# ax.set_xlabel('time (msec)')
ax.set_ylabel('fraction')
ax.set_title(f'day {day}: back')
# ax.set_xlim([1.4, arr.sizes['time']])
# ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
# ax.set_xticklabels([100, 200, 300, 400, 500, 600, 700])
ax.set_xticks([0, 8, 16, 24])
ax.set_xticklabels([0, 250, 500, 750])
ax.set_ylim(ylim)

ax, arr = axes[2], day_0
# arr = arr.isel(trial=slice(500 - chunksize, 500))
arr = arr.mean('trial')
scores_0 = arr.argmax('time')
ax.hist(scores_0, bins=bins, width=0.9, density=True)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time (msec)')
ax.set_ylabel('fraction')
ax.set_title(f'day {day}: all')
# ax.set_xlim([1.4, arr.sizes['time']])
# ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
# ax.set_xticklabels([100, 200, 300, 400, 500, 600, 700])
ax.set_xticks([0, 8, 16, 24])
ax.set_xticklabels([0, 250, 500, 750])
ax.set_ylim(ylim)

# fig.suptitle('gray')
fig.tight_layout(pad=1)

plt.show()

fig.savefig(f'figures/time_fields_gray_split_day_{day}.eps')
