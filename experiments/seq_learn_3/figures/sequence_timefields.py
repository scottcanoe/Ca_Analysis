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


"""
Create histograms of max firing rates from onset of a sequence.
"""

plt.rcParams['font.size'] = 8

sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)

visual = 'all'
event = 'ACBD'
drop_gray = False
tol = 4
ylim = [0, 0.06]

apply_roi_filter(sessions_0 + sessions_5, visual)
day_0_all = flex_split(sessions_0, event, drop_gray=drop_gray)
day_5_all = flex_split(sessions_5, event, drop_gray=drop_gray)

day_0_all = filter_consistent(day_0_all, tol)
day_5_all = filter_consistent(day_5_all, tol)

onset_kwargs = {
    'shift': -0.5,
    'last': '',
}
if drop_gray:
    onset_kwargs['last'] = None

# plot distance from onset.
fig, axes = plt.subplots(2, 1, figsize=(3, 2))

day_0 = day_0_all.mean('trial')
day_5 = day_5_all.mean('trial')
bins = np.arange(0, day_0.sizes['time'] + 1) - 0.5

ax, arr = axes[0], day_0
scores_0 = arr.argmax('time')
hist = ax.hist(scores_0, bins=bins, width=0.9, density=True)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([0, arr.sizes['time']])
# ax.set_xlabel('time')
ax.set_ylabel('frac')
ax.set_title('day 0')

ax, arr = axes[1], day_5
scores_0 = arr.argmax('time')
ax.hist(scores_0, bins=bins, width=0.9, density=True)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([0, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('frac')
ax.set_title('day 5')

for ax in axes:
    ax.set_ylim(ylim)
if not drop_gray:
    for ax in axes:
        ax.set_xlim([-0.5, 31.5])

fig.tight_layout(pad=0.5)

if drop_gray:
    fname = f'figures/sequence_timefields/sequence_timefields_{event}_drop.eps'
else:
    fname = f'figures/sequence_timefields/sequence_timefields_{event}.eps'
fig.savefig(fname)

plt.show()

# # plot distance from onset.
# fig, axes = plt.subplots(2, 1)
# trials = slice(0, 50)
# day_0 = day_0_all.isel(trial=trials).mean('trial')
# day_5 = day_5_all.isel(trial=trials).mean('trial')
# bins = np.arange(0, day_0.sizes['time'] + 1) - 0.5
