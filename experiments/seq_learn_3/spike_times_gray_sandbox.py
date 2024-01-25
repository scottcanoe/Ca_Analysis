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


plt.rcParams['font.size'] = 8
tol = 4

"""
Create histograms of max firing rates from onset of a sequence.
"""


sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)


event = 'ACBD._'
day_0 = flex_split(sessions_0, event)
day_5 = flex_split(sessions_5, event)

day_0 = filter_consistent(day_0, tol)
day_5 = filter_consistent(day_5, tol)

day_0 = day_0.mean('trial')
day_5 = day_5.mean('trial')

# plot distance from onset.
fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.5))
bins = np.arange(0, day_0.sizes['time'] + 1) - 0.5

ax, arr = axes[0], day_0
scores_0 = arr.argmax('time')
hist = ax.hist(scores_0, bins=bins, width=0.9)
ax.set_xlim([1.4, arr.sizes['time']])
ax.set_xlabel('time (msec)')
ax.set_ylabel('count')
ax.set_title('day 0')
ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
ax.set_xticklabels([100, 200, 300, 400, 500, 600, 700])
ax.set_ylim([0, 100])

ax, arr = axes[1], day_5
scores_0 = arr.argmax('time')
ax.hist(scores_0, bins=bins, width=0.9)
ax.set_xlim([1.4, arr.sizes['time']])
ax.set_xlabel('time (msec)')
ax.set_ylabel('count')
ax.set_title('day 5')
ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
ax.set_xticklabels([100, 200, 300, 400, 500, 600, 700])
ax.set_ylim([0, 100])

fig.tight_layout(pad=0.8)

ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
for ax in axes:
    ax.set_ylim([0, ymax])
fig.suptitle(event)
plt.show()
# fig.savefig('figures/spike_times_gray.eps')

