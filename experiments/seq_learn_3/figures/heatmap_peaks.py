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


#-------------------------------------------------------------------------------


plt.rcParams['font.size'] = 8
colorbar = False
tol = 4

sessions = get_sessions(day=5)
# apply_roi_filter(sessions, 'all')

data = flex_split(sessions, 'ABCD', drop_gray=False)

# gray only?
data = data.isel(time=slice(34, None))

# filter for consistency.
data = filter_consistent(data, tol)

# split into even/odd groups.
group_1_trial_inds = np.arange(0, 500, 2)
group_2_trial_inds = group_1_trial_inds + 1

group_1 = data.isel(trial=group_1_trial_inds).mean('trial')
group_2 = data.isel(trial=group_2_trial_inds).mean('trial')

inds = np.argsort(group_1.argmax('time'))
arr = group_2.isel(roi=inds).transpose('roi', ...)

data = data.isel(roi=inds)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

qlim = (2.5, 97.5)
smap = get_smap('inferno', qlim=qlim, data=arr)
mat = smap(arr)

# put maxs on.
imaxs = arr.argmax('time')
# imaxs = group_2.argmax('time')
for i in range(len(imaxs)):
    ind = int(imaxs[i])
    mat[i, ind] = (1, 1, 1, 1)

im = ax.imshow(mat, aspect='auto', interpolation="none")


# im = ax.imshow(arr, 'inferno', vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')
annotate_onsets(
    ax,
    arr.coords["event"],
    skip_first=False,
    color='gray',
    alpha=0.3,
    last='-',
    shift=-0.5,
    lw=0.5,
)

if colorbar:
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right", size="5%", pad=0.1)
    fig.add_axes(ax_cb)
    cbar = plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)


ax.set_ylabel('roi')
ax.set_xlabel('time')
fig.tight_layout(pad=1)
plt.show()
fig.savefig('figures/timefield_heatmap.eps', dpi=600)
