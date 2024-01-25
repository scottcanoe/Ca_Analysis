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


def get_data(
    sessions: SessionGroup,
    event: str,
    visual: bool = None,
    trials: Optional[slice] = None,
) -> xr.DataArray:

    for s in sessions:
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
        if "roi" in data.coords:
            del data.coords["roi"]
        s.data = SessionData(s, data=data)

    # split/combine sequence data
    if event in ('ABCD', 'ABBD', 'ACBD'):
        spec = schema.get(sequence=event)[:-1]
    else:
        spec = schema.get(event=event)
    splits = sessions.split('data', spec)
    if trials is None:
        n_trials = min([arr.sizes['trial'] for arr in splits])
        trials = slice(0, n_trials)
    splits = [arr.isel(trial=trials) for arr in splits]
    data = xr.concat(splits, 'roi')

    return data


sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)

sessions = sessions_5

visual = None

events = [
    "ABCD._",
    "ABBD._",
    "ACBD._",
]

arrays = []
for ev in events:
    arrays.append(get_data(sessions, visual=visual, event=ev))
data = xr.concat(arrays, 'trial')
# trial_avg = data.mean('trial')
# maxs = trial_avg.max('time')
# tf = maxs > 1
# data = data.isel(roi=tf)

# split into two groups.

n_trials = data.sizes["trial"] - 1
n_trials = n_trials if n_trials % 2 == 0 else n_trials - 1
trial_inds = np.arange(0, n_trials, 2)
even = data.isel(trial=trial_inds).mean('trial')
odd = data.isel(trial=trial_inds + 1).mean('trial')

sorting_index = np.argsort(even.argmax('time'))
even = even.isel(roi=sorting_index)
odd = odd.isel(roi=sorting_index)

even = even.transpose('roi', ...)
odd = odd.transpose('roi', ...)

#-------------------------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(24, 12))

ax, arr = axes[0], even
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
maxs = arr.argmax('time')
for i, peak in enumerate(maxs):
    peak = peak.item()
    cdata[i, peak] = 1
ax.imshow(cdata, aspect="auto", interpolation='none')
# annotate_onsets(ax, arr.coords["event"], skip_first=False, alpha=1, color='white')
ax.set_title(arr.name)

ax, arr = axes[1], odd
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
maxs = arr.argmax('time')
for i, peak in enumerate(maxs):
    peak = peak.item()
    cdata[i, peak] = 1
ax.imshow(cdata, aspect="auto", interpolation='none')
# annotate_onsets(ax, arr.coords["event"], skip_first=False, alpha=1, color='white')
ax.set_title(arr.name)

fig.tight_layout()

plt.show()
