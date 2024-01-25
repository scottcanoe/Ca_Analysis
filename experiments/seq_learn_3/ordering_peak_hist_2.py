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
event_name = 'ABBD'
lpad = None
drop_gray = False
tol = 4

apply_roi_filter(sessions_0 + sessions_5, roi_filter)
day_0 = flex_split(sessions_0, event_name, lpad=lpad, drop_gray=drop_gray)
day_5 = flex_split(sessions_5, event_name, lpad=lpad, drop_gray=drop_gray)

day_0 = filter_consistent(day_0, tol)
day_5 = filter_consistent(day_5, tol)

# plot distance from onset.
fig, axes = plt.subplots(2, 1)
bins = np.arange(0, day_0.sizes['time'] + 1) - 0.5

onset_kwargs = {
    'skip_first': lpad is not None,
    'last': '-',
    'shift': -0.5,
}
ax, arr = axes[0], day_0
scores = arr.mean('trial').argmax('time')
hist = ax.hist(scores, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title('day 0')

ax, arr = axes[1], day_5
scores = arr.mean('trial').argmax('time')
ax.hist(scores, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title('day 5')
fig.suptitle(event_name)

fig.tight_layout(pad=1)
plt.show()
plotdir = Path.home() / 'plots/seq_learn_3/peak_histograms'
fig.savefig(plotdir / f'{event_name}_{roi_filter}.png')

