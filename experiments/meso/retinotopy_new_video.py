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


s = open_session('M110', '2023-04-21', '1', fs=0)
with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'r') as f:
    dset = f['ach']
    mov = dset[:]
mov = gaussian_filter(mov, sigma=(1, 1, 1))

outfile = s.fs.getsyspath('new_video.mp4')

cmap = "inferno"
qlim = (0.5, 99.75)
kernel = None
fps = 30
dpi = 220
width = 8
frameon = True
facecolor = None


# Create plot items.


# Setup figure and axes.
nframes, ypix, xpix = mov.shape
aspect = ypix / xpix
figsize = (width, width * aspect)
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
im = ax.imshow(np.zeros_like(mov[0]))
smap = get_smap(data=mov, cmap=cmap, qlim=qlim)
fig.tight_layout(pad=1)

hbar_x = 20
hbar_width = 150
hbar_y = 20
ax.plot([hbar_x, hbar_x + hbar_width], [hbar_y, hbar_y], color='white')
ax.plot([hbar_x, hbar_x], [hbar_y + 5, hbar_y - 5], color='white')
ax.plot([hbar_x + hbar_width, hbar_x + hbar_width], [hbar_y + 5, hbar_y - 5], color='white')

points = ax.scatter([hbar_x + 5], [hbar_y], color='white')
x_positions = np.linspace(hbar_x, hbar_x + hbar_width, nframes)
scat_pos = np.zeros([nframes, 2])
scat_pos[:, 1] = hbar_y
scat_pos[:, 0] = x_positions

ax.text(hbar_x, hbar_y - 10, r'$-120 \degree$', fontsize=12, ha='center', color='white')
ax.text(hbar_x + hbar_width, hbar_y - 10, r'$0 \degree$', fontsize=12, ha='center', color='white')

# Save to file.

writer = FFMpegWriter(fps=fps)
# nframes = 10
with writer.saving(fig, str(outfile), dpi):
    for i in range(nframes):
        cdata = smap(mov[i])
        cdata[:, :, -1] = 1
        im.set_data(cdata)
        points.set_offsets(scat_pos[i])
        writer.grab_frame()



