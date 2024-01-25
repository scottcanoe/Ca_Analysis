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



s = open_session('M173', '2023-05-12', '3', fs=0)
outfile = s.fs.getsyspath(f'trial_averages.mp4')

cmap = "inferno"
qlim = (2.5, 97.5)
kernel = None
fps = 15
dpi = 220
width = 8
frameon = True
facecolor = None
kernel = (1.5, 2, 2)

# Create plot items.

items = []

obj = SimpleNamespace(name='ca')
obj.data = s.ca_mov.split(1).mean('trial')
obj.smap = get_smap(data=obj.data, cmap=cmap, qlim=qlim)
items.append(obj)
print('got ca')

obj = SimpleNamespace(name='ach')
obj.data = s.ach_mov.split(1).mean('trial')
items.append(obj)
print('got ach')

for obj in items:
    obj.data.data = gaussian_filter(obj.data, kernel)
    obj.smap = get_smap(data=obj.data, cmap=cmap, qlim=qlim)


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
    ax = obj.ax = fig.add_subplot(1, 2, i + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    obj.im = ax.imshow(np.zeros_like(obj.data[0]))

items[0].ax.set_title('ca')
items[1].ax.set_title('ach')

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



