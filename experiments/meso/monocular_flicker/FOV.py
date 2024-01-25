import intake
from intake.source.discovery import drivers, load_plugins_from_module
import weakref
from time import perf_counter as clock
from typing import Mapping

import dask.array as da
import h5py
import matlab.engine
import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from experiments.meso.main import *
# from processing import *
from ca_analysis.plot import *



s = open_session('M110', '2023-04-21', '2', fs=0)

with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
    mov = f['ach'][1:1000]

frame = mov.max(axis=0)
# mov = gaussian_filter(frame, (0.5, 0.5))

vmin, vmax = np.percentile(frame, [2.5, 97.5])
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax)
remove_ticks(ax)
plt.show()

fig.savefig('figures/FOV_ach.png', dpi=300)
fig.savefig('figures/FOV_ach.eps', dpi=300)

