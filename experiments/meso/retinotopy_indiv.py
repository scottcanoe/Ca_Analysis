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


sessions = get_sessions(schema='retinotopy', fs=0)
s = sessions[0]


direction = "down"

outfile = s.fs.getsyspath(f'trial_averages_{direction}_LH.mp4')

cmap = "inferno"
qlim = (0.5, 99.75)
kernel = None
fps = 10
dpi = 220
width = 4
frameon = True
facecolor = None

ylim = 150, None
xlim = 0, 250

# Create plot items.
ca = SimpleNamespace(name="Ca")
ca.data = s.ca_mov.split(direction).mean('trial')


ach = SimpleNamespace(name="ACh")
ach.data = s.ach_mov.split(direction).mean('trial')



items = [ca, ach]
for obj in items:
    obj.data = obj.data.isel(y=slice(ylim[0], ylim[1]), x=slice(xlim[0], xlim[1]))

# Setup figure and axes.
ypix, xpix = items[0].data.shape[1:]
aspect = ypix / xpix
figsize = (width, 2 * width * aspect)
fig = Figure(
    figsize=figsize,
    frameon=frameon,
    facecolor=facecolor,
)
for i, obj in enumerate(items):
    ax = obj.ax = fig.add_subplot(2, 1, i + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title(obj.name)
    obj.smap = get_smap(data=obj.data, cmap=cmap, qlim=qlim)
    obj.im = ax.imshow(np.zeros_like(obj.data[0]))

fig.tight_layout(pad=1)

# Save to file.
n_frames = max([obj.data.shape[0] for obj in items])
writer = FFMpegWriter(fps=fps)
with writer.saving(fig, str(outfile), dpi):
    for i in range(n_frames):
        for obj in items:
            frame_num = min(i, obj.data.shape[0] - 1)
            cdata = obj.smap(obj.data[frame_num])
            cdata[:, :, -1] = 1
            obj.im.set_data(cdata)
        writer.grab_frame()



