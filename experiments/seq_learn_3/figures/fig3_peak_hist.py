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


sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)

visual = 'all'
event = 'ABCD'
drop_gray = False

apply_roi_filter(sessions_0 + sessions_5, visual)
day_0_all = flex_split(sessions_0, event, drop_gray=drop_gray)
day_5_all = flex_split(sessions_5, event, drop_gray=drop_gray)
onset_kwargs = {
    'shift': -0.5,
    'last': '-',
}
if drop_gray:
    onset_kwargs['last'] = None

# plot distance from onset.
fig, axes = plt.subplots(2, 1)

day_0 = day_0_all.mean('trial')
day_5 = day_5_all.mean('trial')
bins = np.arange(0, day_0.sizes['time'] + 1) - 0.5

ax, arr = axes[0], day_0
scores_0 = arr.argmax('time')
hist = ax.hist(scores_0, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title('day 0')

ax, arr = axes[1], day_5
scores_0 = arr.argmax('time')
ax.hist(scores_0, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title('day 5')

fig.tight_layout(pad=1)
plotdir = Path.home() / "plots/seq_learn_3"
fname = f'fig3a_{event}_peaks.png'
# fig.savefig(plotdir / fname)

plt.show()

# plot distance from onset.
fig, axes = plt.subplots(2, 1)
trials = slice(0, 50)
day_0 = day_0_all.isel(trial=trials).mean('trial')
day_5 = day_5_all.isel(trial=trials).mean('trial')
bins = np.arange(0, day_0.sizes['time'] + 1) - 0.5

ax, arr = axes[0], day_0
scores_0 = arr.argmax('time')
hist = ax.hist(scores_0, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title('day 0')

ax, arr = axes[1], day_5
scores_0 = arr.argmax('time')
ax.hist(scores_0, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title('day 5')

fig.tight_layout(pad=1)
plotdir = Path.home() / "plots/seq_learn_3"
fname = f'fig3a_{event}_trials_{trials.start}-{trials.stop}_peaks.png'
# fig.savefig(plotdir / fname)

plt.show()

# plot distance from onset.
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
data = day_5_all
bins = np.arange(0, day_0.sizes['time'] + 1) - 0.5

ax = axes[0, 0]
trials = slice(0, 100)
arr = data.isel(trial=trials).mean('trial')
scores = arr.argmax('time')
hist = ax.hist(scores, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title(f'trials {trials.start}-{trials.stop}')

ax = axes[0, 1]
trials = slice(100, 200)
arr = data.isel(trial=trials).mean('trial')
scores = arr.argmax('time')
hist = ax.hist(scores, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title(f'trials {trials.start}-{trials.stop}')

ax = axes[1, 0]
trials = slice(200, 300)
arr = data.isel(trial=trials).mean('trial')
scores = arr.argmax('time')
hist = ax.hist(scores, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title(f'trials {trials.start}-{trials.stop}')

ax = axes[1, 1]
trials = slice(300, 400)
arr = data.isel(trial=trials).mean('trial')
scores = arr.argmax('time')
hist = ax.hist(scores, bins=bins, width=0.9)
annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_xlim([-1, arr.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title(f'trials {trials.start}-{trials.stop}')

fig.tight_layout(pad=1)
plt.show()

# plotdir = Path.home() / "plots/seq_learn_3"
# fname = f'fig3a_{event}_trials_{trials.start}-{trials.stop}_peaks.png'
# fig.savefig(plotdir / fname)
