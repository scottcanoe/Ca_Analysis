from numbers import Number
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Optional,
    Tuple,
    Union,
)
import h5py
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter

import numpy as np
import xarray as xr

# from ca_analysis.io.common import URL

xr.set_options(keep_attrs=True)

from ca_analysis import *
from ca_analysis.io.thorlabs import *

from ca_analysis.plot import get_smap

from ca_analysis.processing.utils import (
    init_attrs,
    make_sample,
    motion_correct,
    process_events,
    thorimage_raw_to_h5,
)

from experiments.retinotopy.main import *

s = open_session('36662-2', '2022-09-22', '1')
outfile: PathLike = "retinotopy.mp4"
cmap = "inferno"
qlim = (0.5, 99.75)
kernel = None
fps = 30
dpi = 220
width = 6
frameon = True
facecolor = None


# Load the data.
infile = s.fs.getsyspath("mov.h5")
outfile = Path(s.fs.getsyspath(outfile))
with h5py.File(infile, "r") as f:
    mov = f["data"][:]
    if "mask" in f.keys():
        mask = f["mask"][:]
        mov = add_mask(mov, mask, fill_value=0)

# Create plot items.

schema = s.events.schema
fps_capture = s.attrs['capture']['frame']['rate']
table = s.events["events"]

obj = SimpleNamespace(name="left")
obj.event = schema.get(event=schema.get(event=obj.name))

# - find block to extract
df = table[table["event"] == obj.event.id]
starts = df["start"].values
lengths = round(obj.event.duration * fps_capture)
stops = starts + lengths
n_blocks = len(starts)

# - extract and reduce
shape = (n_blocks, lengths, mov.shape[1], mov.shape[2])
stack = np.zeros(shape, dtype=mov.dtype)
for i, (a, b) in enumerate(zip(starts, stops)):
    try:
        stack[i] = mov[a:b]
    except ValueError:
        pass
obj.data = np.mean(stack, axis=0)
if np.ma.is_masked(mov):
    obj.data = add_mask(obj.data, mov.mask[0], fill_value=mov.fill_value)

# - smooth/filter
if kernel:
    obj.data = gaussian_filter(obj.data, kernel)

# - colormapping
obj.smap = get_smap(data=obj.data, cmap=cmap, qlim=qlim)


# Setup figure and axes.
w = 1.5
ypix, xpix = mov[0].shape
aspect = ypix / xpix
figsize = (4 * w, 3 * aspect * w)

# figsize = (6, 2.6)
fig, axes = plt.subplots(
    3,
    4,
    figsize=figsize,
    frameon=True,
    facecolor=None,
    gridspec_kw={'height_ratios': [1, 1, 1]}
)
for ax in axes.flatten():
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.axis('off')
fig.tight_layout(pad=0)
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
frame_nums = np.arange(0, 127, 8)[4:]
# Save to file.
n_frames = obj.data.shape[0]
for i, ax in enumerate(axes.flatten()):
    frame = obj.data[frame_nums[i]]
    cdata = obj.smap(frame)
    cdata[:, :, -1] = 1
    ax.imshow(cdata)
plt.show()
fig.savefig('figures/retinotopy_supplement.eps', dpi=300)
fig.savefig('figures/retinotopy_supplement.png', dpi=300)


# writer = FFMpegWriter(fps=fps)
# with writer.saving(fig, str(outfile), dpi):
#     for i in range(n_frames):
#         for obj in items:
#             frame_num = min(i, obj.data.shape[0] - 1)
#             cdata = obj.smap(obj.data[frame_num])
#             cdata[:, :, -1] = 1
#             obj.im.set_data(cdata)
#         writer.grab_frame()
#
