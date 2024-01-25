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

target = "ach_mov"

outfile = s.fs.getsyspath(f'trial_averages_{target}.mp4')

cmap = "inferno"
qlim = (0.5, 99.75)
kernel = None
fps = 30
dpi = 220
width = 8
frameon = True
facecolor = None


# Create plot items.

items = []
for name in ["right", "up", "left", "down"]:
    obj = SimpleNamespace(name=name)
    obj.event = s.events.schema.get(event=obj.name)
    obj.data = getattr(s, target).split(name).mean('trial')
    obj.smap = get_smap(data=obj.data, cmap=cmap, qlim=qlim)
    items.append(obj)

# Setup figure and axes.
ypix, xpix = items[0].data.shape[1:]
aspect = ypix / xpix
figsize = (width, width * aspect)
fig = Figure(
    figsize=figsize,
    frameon=frameon,
    facecolor=facecolor,
)
for i, obj in enumerate(items):
    ax = obj.ax = fig.add_subplot(2, 2, i + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title(obj.name)
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



