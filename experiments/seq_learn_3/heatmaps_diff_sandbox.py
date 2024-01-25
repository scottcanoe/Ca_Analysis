from types import SimpleNamespace
from typing import Sequence

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from scipy.stats import *

from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *
from main import *
from utils import *


sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)

roi_filter = 'all'
apply_roi_filter(sessions_0 + sessions_5, roi_filter)

event = 'ABCD'
lpad = None
rpad = 24



#-------------------------------------------------------------------------------

sessions = sessions_0
data = flex_split(sessions, event, drop_gray=False)

# rank rois
n_trials = data.sizes["trial"]
if n_trials % 2 == 1:
    n_trials -= 1
all_trial_inds = np.arange(n_trials, dtype=int)
group_1_trial_inds = np.random.choice(all_trial_inds, n_trials // 2, replace=False)
group_2_trial_inds = np.setdiff1d(all_trial_inds, group_1_trial_inds)

group_1_trial_inds = np.arange(0, 400, 2)
group_2_trial_inds = group_1_trial_inds + 1

group_1 = data.isel(trial=group_1_trial_inds).mean('trial')
group_2 = data.isel(trial=group_2_trial_inds).mean('trial')

inds_1 = np.argsort(group_1.argmax('time'))
inds_2 = np.argsort(group_2.argmax('time'))

group_1 = group_1.isel(roi=inds_1).transpose('roi', ...)
group_2 = group_2.isel(roi=inds_1).transpose('roi', ...)

day_0 = group_2

#-------------------------------------------------------------------------------

sessions = sessions_5
data = flex_split(sessions, event, drop_gray=False)

# rank rois
n_trials = data.sizes["trial"]
if n_trials % 2 == 1:
    n_trials -= 1
all_trial_inds = np.arange(n_trials, dtype=int)
group_1_trial_inds = np.random.choice(all_trial_inds, n_trials // 2, replace=False)
group_2_trial_inds = np.setdiff1d(all_trial_inds, group_1_trial_inds)

group_1_trial_inds = np.arange(0, 400, 2)
group_2_trial_inds = group_1_trial_inds + 1

group_1 = data.isel(trial=group_1_trial_inds).mean('trial')
group_2 = data.isel(trial=group_2_trial_inds).mean('trial')

inds_1 = np.argsort(group_1.argmax('time'))
inds_2 = np.argsort(group_2.argmax('time'))

group_1 = group_1.isel(roi=inds_1).transpose('roi', ...)
group_2 = group_2.isel(roi=inds_1).transpose('roi', ...)

day_5 = group_2

# resize images
day = day_0
image = Image.fromarray(day.data)
image = image.resize((1500, 56))
day_0 = xr.DataArray(np.array(image).T, dims=('roi', 'time'), coords=day.coords)

day = day_5
image = Image.fromarray(day.data)
image = image.resize((1500, 56))
day_5 = xr.DataArray(np.array(image).T, dims=('roi', 'time'), coords=day.coords)

diff = day_0 - day_5

#-------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(36, 12))
onset_kwargs = {
    'skip_first': False,
    'color': 'white',
    'alpha': 1,
    'shift': -0.5,
    'last': '-',
}
ax, arr = axes[0], day_0
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
ax.imshow(cdata, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
ax.set_title('day 0')

ax, arr = axes[1], day_5
vmin, vmax = np.percentile(arr, [2.5, 97.5])
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
# ax.imshow(cdata, aspect="auto", interpolation='none')
im = ax.imshow(arr, 'inferno', vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
ax.set_title('day 5')

ax, arr = axes[2], diff
vmax = float(max(arr.max(), abs(arr.min())))
im = ax.imshow(arr, 'coolwarm', vmin=-vmax, vmax=vmax, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["event"], **onset_kwargs)

divider = make_axes_locatable(ax)
ax_cb = divider.append_axes("right", size="5%", pad=0.1)
fig.add_axes(ax_cb)
cbar = plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
ax_cb.yaxis.set_tick_params(labelright=True)

fig.suptitle(event)
fig.tight_layout(pad=3)
plt.show()

plotdir = Path.home() / "plots/seq_learn_3/heatmaps"
fname = f'{event}_day0_vs_day5.png'
# fig.savefig(plotdir / fname)
