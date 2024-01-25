
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
import napari
import numpy as np
import pandas as pd
from scipy.io import savemat

from main import *
from processing import *
from ca_analysis.plot import get_smap



s = open_session('M110', '2023-04-21', '1', fs=0)


cmap = "inferno"
qlim = (2.5, 97.5)
kernel = None
fps = 50
dpi = 220
width = 8
frameon = True
facecolor = None

# Create plot items.

items = []

obj = SimpleNamespace(name='ca')
obj.data_all = s.ca_mov.split(1)
items.append(obj)
print('got ca')

obj = SimpleNamespace(name='ach')
obj.data_all = s.ach_mov.split(1)
items.append(obj)
print('got ach')


ypix, xpix = items[0].data.shape[1:]
aspect = ypix / xpix
figsize = (width, width * aspect)
figsize = (8, 5)
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
    tr = obj.data_all.isel(trial=0)
    obj.im = ax.imshow(np.zeros_like(tr[0]))

for obj in items:
    obj.smap = get_smap(data=obj.data_all, cmap=cmap, qlim=qlim)

items[0].ax.set_title('ca')
items[1].ax.set_title('ach')

fig.tight_layout(pad=1)

outfile = s.fs.getsyspath(f'analysis/trials_2.mp4')

writer = FFMpegWriter(fps=fps)
n_frames = 110
with writer.saving(fig, str(outfile), dpi):

    for trial_num in range(100):
        print(f'trial_num: {trial_num}')
        for obj in items:
            obj.data = obj.data_all.isel(trial=trial_num)
            # obj.smap = get_smap(data=obj.data, cmap=cmap, qlim=qlim)

        fig.suptitle(f'trial: {trial_num}')

        for i in range(n_frames):
            for obj in items:
                frame_num = min(i, obj.data.shape[0] - 1)
                cdata = obj.smap(obj.data[frame_num])
                cdata[:, :, -1] = 1
                obj.im.set_data(cdata)
            writer.grab_frame()



