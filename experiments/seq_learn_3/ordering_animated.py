from types import SimpleNamespace
from typing import Sequence

import matplotlib
from matplotlib.animation import FFMpegWriter
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.stats import *

from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *
from main import *


def get_data(
    sessions: SessionGroup,
    event: str,
    visual: bool = None,
) -> xr.DataArray:

    for s in sessions:
        s.spikes.prepare()
        data = s.spikes.data
        if visual is None:
            pass
        elif visual in {False, True}:
            vis_inds = np.load(s.fs.getsyspath('visual.npy'))
            if visual is True:
                data = data.sel(roi=vis_inds)
            else:
                all_inds = np.array(data.coords['roi'])
                non_vis_inds = np.setdiff1d(all_inds, vis_inds)
                data = data.sel(roi=non_vis_inds)
        else:
            raise ValueError('invalid "visual" argument')
        s.spikes.data = data
        del s.spikes.data.coords["roi"]

    # split/combine sequence data
    if event in ('ABCD', 'ABBD', 'ACBD'):
        spec = schema.get(sequence=event)[:-1]
    else:
        spec = schema.get(event=event)
    splits = sessions.split('spikes', spec)
    n_trials = min([arr.sizes['trial'] for arr in splits])
    trial_slice = slice(0, n_trials)
    splits = [arr.isel(trial=trial_slice) for arr in splits]
    data = xr.concat(splits, 'roi')

    return data

"""
Trial-average data, get sorting index for first half of trials
"""



visual = None
event = 'ABCD._'
save = Path.home() / f'plots/seq_learn_3/sorting_anim_{event}.mp4'
chunksize = 25

sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)

day_0 = get_data(sessions_0, visual=visual, event=event)
day_5 = get_data(sessions_5, visual=visual, event=event)

n_trials = min(day_0.sizes['trial'], day_5.sizes['trial'])
day_0 = day_0.isel(trial=slice(0, n_trials))
day_5 = day_5.isel(trial=slice(0, n_trials))
day_0_counts = []
day_5_counts = []
chunk_lims = []
n_windows = n_trials - chunksize
bins = np.arange(-0.5, 34.5, 1)
for i in range(n_windows):
    slc = slice(i, i + chunksize)
    chunk_lims.append((i, i + chunksize))

    peaks = day_0.isel(trial=slc).mean('trial').argmax('time')
    counts, _ = np.histogram(peaks, bins=bins)
    day_0_counts.append(counts)

    peaks = day_5.isel(trial=slc).mean('trial').argmax('time')
    counts, _ = np.histogram(peaks, bins=bins)
    day_5_counts.append(counts)


bins = np.arange(-0.5, 34.5, 1)
heights = np.zeros_like(day_0_counts[0])

fig = Figure(figsize=(6, 6))
ax_1 = fig.add_subplot(2, 1, 1)
ax_2 = fig.add_subplot(2, 1, 2)

ax = ax_1
_, _, bars_0 = ax.hist(heights, bins=bins, width=0.9)
annotate_onsets(ax, day_0.coords['event'], shift=-0.5)
ax.set_xlim([-1, day_0.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title('day 0')
ymax = max([max(arr) for arr in day_0_counts]) + 10
ax.set_ylim([0, ymax])

ax = ax_2
_, _, bars_5 = ax.hist(heights, bins=bins, width=0.9)
annotate_onsets(ax, day_0.coords['event'], shift=-0.5)
ax.set_xlim([-1, day_0.sizes['time']])
ax.set_xlabel('time')
ax.set_ylabel('count')
ax.set_title('day 5')
ymax = max([max(arr) for arr in day_5_counts]) + 10
ax.set_ylim([0, ymax])


fig.tight_layout(pad=1)

writer = FFMpegWriter(fps=30)
with writer.saving(fig, save, 190):
    for i in range(len(day_0_counts)):

        bars = bars_0
        heights = day_0_counts[i]
        for j, bar in enumerate(bars):
            bar.set_height(heights[j])

        bars = bars_5
        heights = day_5_counts[i]
        for j, bar in enumerate(bars):
            bars[j].set_height(heights[j])

        writer.grab_frame()
