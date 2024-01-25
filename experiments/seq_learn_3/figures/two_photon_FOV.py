import os
from pathlib import Path
import shutil

from types import SimpleNamespace
from typing import (
    Any, Callable,
    List,
    Mapping,
    NamedTuple, Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from statsmodels.genmod.families.links import identity, log
import xarray as xr

from ca_analysis import *
from ca_analysis.stats import gaussian_kde
# from ca_analysis.plot.images import *
from seq_learn_3.main import *
from seq_learn_3.utils import *



# s = open_session('61106-1', '2023-01-27', '1', fs=-1)
# with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
#     dset = f['data']
#     mov = dset[5000:8000]
#     np.save('mov.npy', mov)

mov = np.load('mov.npy')
image = mov.mean(axis=0)
# image = mov.max(axis=0)

# image = gaussian_filter(image, sigma=(0.5, 0.5))
vmin, vmax = np.percentile(image, [5, 99.5])
fig, ax = plt.subplots(figsize=(2, 2))
im = ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
# remove_ticks(ax)
ax.set_axis_off()

um_per_pix = 412.22 / 512
size_um = 100
size_pix = size_um / um_per_pix
label = '${} \mu $m'.format(size_um)
loc = 3
frameon = False
pad = 0.25
sep = 2.5
fp = FontProperties()
fp.set_size(8)

bar = AnchoredSizeBar(
    ax.transData,
    size_pix,
    label,
    loc,
    color='white',
    frameon=frameon,
    pad=pad,
    sep=sep,
    fontproperties=fp,
)
ax.add_artist(bar)

fig.tight_layout(pad=0.0)
fig.savefig('figures/FOV.eps', dpi=600)
plt.show()



