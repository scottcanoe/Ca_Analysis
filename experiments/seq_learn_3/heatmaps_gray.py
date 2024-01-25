from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional, Sequence

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import *

from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *

from seq_learn_3.main import *
from seq_learn_3.utils import *



def get_data(sessions: SessionGroup, event: str) -> xr.DataArray:

    data = flex_split(sessions, event, drop_gray=False).isel(time=slice(34, None))
    arr = data.mean('trial')
    maxs = arr.max('time')
    inds = argwhere(maxs >= 1)
    data = data.isel(roi=inds)

    # split into even/odd groups
    group_1_trial_inds = np.arange(0, 500, 2)
    group_2_trial_inds = group_1_trial_inds + 1

    group_1 = data.isel(trial=group_1_trial_inds).mean('trial')
    group_2 = data.isel(trial=group_2_trial_inds).mean('trial')

    inds = np.argsort(group_1.argmax('time'))
    out = group_2.isel(roi=inds).transpose('roi', ...)

    # out = data.mean('trial')
    # inds = np.argsort(out.argmax('time'))
    # out = out.isel(roi=inds).transpose('roi', ...)

    return out


def add_plot(
    ax,
    arr,
    title: str = "",
    colorbar: bool = False,
) -> "matplotlib.image.AxisImage":

    vmin, vmax = np.percentile(arr, [2.5, 97.5])
    # vmin, vmax = np.percentile(arr, [1, 99])
    im = ax.imshow(arr, 'inferno', vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')
    # annotate_onsets(
    #     ax,
    #     arr.coords["event"],
    #     skip_first=False,
    #     color='gray',
    #     alpha=0.3,
    #     last='-',
    #     shift=-0.5,
    #     lw=0.5,
    # )
    ax.set_title(title)

    if colorbar:
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes("right", size="5%", pad=0.1)
        fig.add_axes(ax_cb)
        cbar = plt.colorbar(im, cax=ax_cb)
        ax_cb.yaxis.tick_right()
        ax_cb.yaxis.set_tick_params(labelright=True)

    return im


#-------------------------------------------------------------------------------


plt.rcParams['font.size'] = 8

sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)
# apply_roi_filter(sessions_0 + sessions_5, 'all')

ABCD_0 = get_data(sessions_0, 'ABCD')
ABBD_0 = get_data(sessions_0, 'ABBD')
ACBD_0 = get_data(sessions_0, 'ACBD')
ABCD_5 = get_data(sessions_5, 'ABCD')
ABBD_5 = get_data(sessions_5, 'ABBD')
ACBD_5 = get_data(sessions_5, 'ACBD')

fig, axes = plt.subplots(2, 3, figsize=(7*2, 4.43*2))

ax, arr = axes[0, 0], ABCD_0
add_plot(ax, arr, 'ABCD')
ax.set_xlabel('time')
ax.set_ylabel('ROI')

ax, arr = axes[0, 1], ABBD_0
add_plot(ax, arr, 'ABBD')

ax, arr = axes[0, 2], ACBD_0
add_plot(ax, arr, 'ACBD', colorbar=True)

ax, arr = axes[1, 0], ABCD_5
add_plot(ax, arr)
ax.set_xlabel('time')
ax.set_ylabel('ROI')

ax, arr = axes[1, 1], ABBD_5
add_plot(ax, arr)

ax, arr = axes[1, 2], ACBD_5
add_plot(ax, arr, colorbar=True)


fig.tight_layout(pad=1)
plt.show()

fig.savefig('figures/summary_heatmaps.eps')
