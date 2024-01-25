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
from statsmodels.genmod.families.links import identity, log
import xarray as xr

from ca_analysis import *

from main import *
from utils import *



def get_diff(
    day: int,
    event_1: str,
    event_2: str,
    roi_filter: str = 'all',
) -> xr.DataArray:

    sessions = get_sessions(day=day, fs=0)
    apply_roi_filter(sessions, roi_filter)

    a = flex_split(sessions, event_1)
    b = flex_split(sessions, event_2)
    # a = a.isel(time=slice(4, None))
    # b = b.isel(time=slice(4, None))
    a = a.mean('time')
    b = b.mean('time')
    a = a.mean('trial')
    b = b.mean('trial')
    diff = b - a
    print(f'Day {day}')
    print(' - ', event_1, ':', describe(a))
    print(' - ', event_2, ':', describe(b))
    print(f'  - {event_2} - {event_1}:', describe(diff))
    return diff


roi_filter = 'all'
A_0 = get_diff(0, 'ABCD.A', 'ABBD.A', roi_filter)
A_5 = get_diff(5, 'ABCD.A', 'ABBD.A', roi_filter)
B_0 = get_diff(0, 'ABBD.B1', 'ABBD.B2', roi_filter)
B_5 = get_diff(5, 'ABBD.B1', 'ABBD.B2', roi_filter)
C_0 = get_diff(0, 'ABCD.C', 'ACBD.C', roi_filter)
C_5 = get_diff(5, 'ABCD.C', 'ACBD.C', roi_filter)
D_0 = get_diff(0, 'ABCD.D', 'ABBD.D', roi_filter)
D_5 = get_diff(5, 'ABCD.D', 'ABBD.D', roi_filter)

fig, ax = plt.subplots()
ax.violinplot([A_0, A_5, B_0, B_5, C_0, C_5, D_0, D_5], showextrema=False)
ax.set_ylim([-5, 5])
plt.show()

print(f'A: {ks_2samp(A_0, A_5)}')
print(f'B: {ks_2samp(B_0, B_5)}')
print(f'C: {ks_2samp(C_0, C_5)}')
print(f'C: {ks_2samp(D_0, D_5)}')

from ca_analysis.stats import gaussian_kde

fig, ax = plt.subplots()
x, y = gaussian_kde(B_0)
ax.plot(x, y, color='black', label='B 0')
x, y = gaussian_kde(B_5)
ax.plot(x, y, color='red', label='B 5')
ax.legend()
ax.set_xlim([-2, 2])
plt.show()

