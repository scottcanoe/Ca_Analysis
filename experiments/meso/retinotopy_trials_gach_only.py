
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FFMpegWriter
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import napari
import numpy as np
import pandas as pd
from scipy.io import savemat

from main import *
from processing import *
from ca_analysis.plot import get_smap


def get_trials(s: Session) -> xr.DataArray:
    if s.fs.exists('trials.h5'):
        with h5py.File(s.fs.getsyspath('trials.h5'), 'r') as f:
            data = f['data'][:]
            return xr.DataArray(data, dims=('trial', 'time', 'y', 'x'))
    trials = s.ach_mov.split(1)
    with h5py.File(s.fs.getsyspath('trials.h5'), 'w') as f:
        f.create_dataset('data', data=trials.data)
    return trials


s = open_session('M110', '2023-04-21', '1', fs=0)
outfile = s.fs.getsyspath(f'analysis/trials.mp4')

cmap = "inferno"
qlim = (2.5, 97.5)
kernel = None
fps = 50
dpi = 220
width = 8
frameon = True
facecolor = None
kernel = (1.5, 2, 2)
# Create plot items.

if 'trials' not in locals():
    trials = get_trials(s)
    trials = trials[:, :, 200:, 270:]
    n_trials = trials.sizes['trial']
    for i in range(n_trials):
        trials[i] = gaussian_filter(trials[i], kernel)
    # smap = get_smap(data=trials, cmap=cmap, qlim=qlim)


ypix, xpix = trials.shape[2:]
aspect = ypix / xpix
figsize = (width, width * aspect)
figsize = (8, 5)
fig = Figure(
    figsize=figsize,
    frameon=frameon,
    facecolor=facecolor,
)
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])
tr = trials.isel(trial=0)
im = ax.imshow(np.zeros_like(tr[0]))
fig.tight_layout(pad=3)

rect = Rectangle(
    (10, 10), 40, 40,
    edgecolor='white',
    facecolor='white',
    lw=4,
)
ax.add_patch(rect)

writer = FFMpegWriter(fps=fps)
n_frames = 110
with writer.saving(fig, str(outfile), dpi):

    for trial_num in range(n_trials):
        print(f'trial_num: {trial_num}')
        data = trials.isel(trial=trial_num)
        smap = get_smap(data=data, cmap=cmap, qlim=qlim)
        fig.suptitle(f'trial: {trial_num}')
        for i in range(data.sizes['time']):
            if i < 10:
                rect.set_facecolor('white')
                rect.set_edgecolor('white')
            else:
                rect.set_facecolor('none')
                rect.set_edgecolor('none')
            cdata = smap(data[i])
            cdata[:, :, -1] = 1
            im.set_data(cdata)
            writer.grab_frame()

