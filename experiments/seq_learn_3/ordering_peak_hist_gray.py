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
from utils import *


"""
Create histograms of max firing rates from onset of a sequence.
"""

def high_pass(arr, thresh = 2):
    maxs = arr.max('time')
    tf = maxs > thresh
    out = arr.isel(roi=tf)
    return out

sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)

# trials = slice(0, 100)

arrays = []
for event in ('ABCD._', 'ABBD._', 'ACBD._'):
    obj = flex_split(sessions_0, event)
    arrays.append(obj)
day_0 = xr.concat(arrays, 'trial')

arrays = []
for event in ('ABCD._', 'ABBD._', 'ACBD._'):
    obj = flex_split(sessions_5, event)
    arrays.append(obj)
day_5 = xr.concat(arrays, 'trial')

# day_0 = day_0.isel(trial=slice(0, 100))
# day_5 = day_5.isel(trial=slice(0, 100))

day_0 = day_0.isel(trial=slice(-100, None))
day_5 = day_5.isel(trial=slice(-100, None))

day_0 = day_0.mean('trial')
day_5 = day_5.mean('trial')

# plot distance from onset.
fig, axes = plt.subplots(2, 1)
bins = np.arange(0, day_0.sizes['time'] + 1) - 0.5

ax, arr = axes[0], day_0
scores_0 = arr.argmax('time')
hist = ax.hist(scores_0, bins=bins, width=0.9)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time (msec)')
ax.set_ylabel('count')
ax.set_title('day 0')
ax.set_xticks([0, 8, 16, 24])
ax.set_xticklabels([0, 250, 500, 750])

ax, arr = axes[1], day_5
scores_0 = arr.argmax('time')
ax.hist(scores_0, bins=bins, width=0.9)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time (msec)')
ax.set_ylabel('count')
ax.set_title('day 5')
ax.set_xticks([0, 8, 16, 24])
ax.set_xticklabels([0, 250, 500, 750])
fig.suptitle('gray')
fig.tight_layout(pad=1)

ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
for ax in axes:
    ax.set_ylim([0, ymax])

plt.show()

plotdir = Path.home() / 'plots/seq_learn_3/peak_histograms'
# fig.savefig(plotdir / f'gray.png')
