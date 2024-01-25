import intake
from intake.source.discovery import drivers, load_plugins_from_module
import weakref
from time import perf_counter as clock
from typing import Mapping

import dask.array as da
import h5py
import matlab.engine
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
import napari
import numpy as np
import pandas as pd
from scipy.io import loadmat

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



s = open_session('M174', '2023-05-12', '3', fs=0)
outfile = s.fs.getsyspath(f'trial_averages.mp4')

cmap = "inferno"
qlim = (2.5, 97.5)
fps = 15
dpi = 220
width = 8
frameon = True
facecolor = None
kernel = (1.5, 2, 2)

# Create plot items.


obj = SimpleNamespace(name='ach')
obj.data = s.ach_mov.split(1).mean('trial')
# obj.data = get_trials(s).mean('trial')

obj.data.data = gaussian_filter(obj.data, kernel)

obj.smap = get_smap(data=obj.data, cmap=cmap, qlim=qlim)

# Setup figure and axes.
ypix, xpix = obj.data.shape[1:]
aspect = ypix / xpix
figsize = (width, width * aspect)
fig = Figure(
    figsize=figsize,
    frameon=frameon,
    facecolor=facecolor,
)


ax = obj.ax = fig.add_subplot(1, 1, 1)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])
obj.im = ax.imshow(np.zeros_like(obj.data[0]))

obj.ax.set_title('ach')

fig.tight_layout(pad=1)

# Save to file.
n_frames = obj.data.shape[0]
writer = FFMpegWriter(fps=fps)
with writer.saving(fig, str(outfile), dpi):
    for i in range(n_frames):
        frame_num = min(i, obj.data.shape[0] - 1)
        cdata = obj.smap(obj.data[frame_num])
        cdata[:, :, -1] = 1
        obj.im.set_data(cdata)
        writer.grab_frame()



