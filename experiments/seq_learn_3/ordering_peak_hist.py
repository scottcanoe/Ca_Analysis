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


def get_scores(arr: xr.DataArray):

    # arr = arr.isel(trial=slice(0, 2))

    scores = arr.mean('trial').argmax('time')
    #
    # scores = []
    # for i in range(arr.sizes['trial']):
    #     scores.append(arr.isel(trial=i).argmax('time'))
    # scores = np.concatenate(scores)

    return scores


sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)
roi_filter = 'all'
event_name = 'ACBD'
drop_gray = False
apply_roi_filter(sessions_0 + sessions_5, roi_filter)
day_0 = flex_split(sessions_0, event_name, drop_gray=drop_gray)
day_5 = flex_split(sessions_5, event_name, drop_gray=drop_gray)

# plot distance from onset.
fig, axes = plt.subplots(2, 1)

bins = np.arange(0, day_0.sizes['time'] + 1) - 0.5

ax, arr = axes[0], day_0
scores = get_scores(arr)
hist = ax.hist(scores, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], shift=-0.5)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title('day 0')

ax, arr = axes[1], day_5
scores = get_scores(arr)
ax.hist(scores, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], shift=-0.5)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title('day 5')
fig.suptitle(event_name)

fig.tight_layout(pad=1)
plt.show()
plotdir = Path.home() / 'plots/seq_learn_3/peak_histograms'
# fig.savefig(plotdir / f'{event}_{roi_filter}.png')

