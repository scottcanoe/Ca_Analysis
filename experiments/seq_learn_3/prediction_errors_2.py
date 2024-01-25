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
from ca_analysis.stats import gaussian_kde
from seq_learn_3.main import *
from seq_learn_3.utils import *



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


def get_IDs(day:int, letter:str) -> np.array:
    df = pd.read_excel(f'selectivity/cells_day{day}_scott.ods', index_col=0)
    df = df.fillna("")
    lst = []
    r = next(df.iterrows())
    for r in df.iterrows():
        stims = r[1].stimulus.split(",")
        if letter in stims:
            lst.append(r[0])
            continue
        for elt in stims:
            if f'({letter})' in elt:
                lst.append(r[0])
    return np.array(lst, dtype=int)


sessions_0 = get_sessions(day=0, fs=0)
sessions_5 = get_sessions(day=5, fs=0)

ABCD_0 = flex_split(sessions_0, 'ABCD')
ABCD_5 = flex_split(sessions_5, 'ABCD')
ABBD_0 = flex_split(sessions_0, 'ABBD')
ABBD_5 = flex_split(sessions_5, 'ABBD')
ACBD_0 = flex_split(sessions_0, 'ACBD')
ACBD_5 = flex_split(sessions_5, 'ACBD')

B_0 = ABBD_0.isel(time=slice(10, 18)).mean('trial').mean('time')
B_nov_0 = ABBD_0.isel(time=slice(18, 26)).mean('trial').mean('time')
B_5 = ABBD_5.isel(time=slice(10, 18)).mean('trial').mean('time')
B_nov_5 = ABBD_5.isel(time=slice(18, 26)).mean('trial').mean('time')

C_0 = ABCD_0.isel(time=slice(18, 26)).mean('trial').mean('time')
C_nov_0 = ACBD_0.isel(time=slice(10, 18)).mean('trial').mean('time')
C_5 = ABCD_5.isel(time=slice(18, 26)).mean('trial').mean('time')
C_nov_5 = ACBD_5.isel(time=slice(10, 18)).mean('trial').mean('time')


# B_0 = ABBD_0.isel(time=slice(10, 18)).mean('trial').mean('time')
# B_nov_0 = ACBD_0.isel(time=slice(18, 26)).mean('trial').mean('time')
# B_5 = ABBD_5.isel(time=slice(10, 18)).mean('trial').mean('time')
# B_nov_5 = ACBD_5.isel(time=slice(18, 26)).mean('trial').mean('time')
#
# C_0 = ABCD_0.isel(time=slice(18, 26)).mean('trial').mean('time')
# C_nov_0 = ACBD_0.isel(time=slice(10, 18)).mean('trial').mean('time')
# C_5 = ABCD_5.isel(time=slice(18, 26)).mean('trial').mean('time')
# C_nov_5 = ACBD_5.isel(time=slice(10, 18)).mean('trial').mean('time')

B_0_IDs = get_IDs(0, 'B')
B_5_IDs = get_IDs(5, 'B')
C_0_IDs = get_IDs(0, 'C')
C_5_IDs = get_IDs(5, 'C')

B_0 = B_0.isel(roi=B_0_IDs)
B_nov_0 = B_nov_0.isel(roi=B_0_IDs)
B_5 = B_5.isel(roi=B_5_IDs)
B_nov_5 = B_nov_5.isel(roi=B_5_IDs)

C_0 = C_0.isel(roi=C_0_IDs)
C_nov_0 = C_nov_0.isel(roi=C_0_IDs)
C_5 = C_5.isel(roi=C_5_IDs)
C_nov_5 = C_nov_5.isel(roi=C_5_IDs)


B_diff_0 = B_nov_0 / B_0
B_diff_5 = B_nov_5 / B_5

C_diff_0 = C_nov_0 / C_0
C_diff_5 = C_nov_5 / C_5


fig, ax = plt.subplots()
ax.violinplot([B_diff_0, B_diff_5, C_diff_0, C_diff_5], showextrema=False)
# ax.set_ylim([0, 5])

ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels([
    'B_nov_0 / B_0',
    'B_nov_5 / B_5',
    'C_nov_0 / C_0',
    'C_nov_0 / C_0',
], rotation=60)
plt.tight_layout(pad=1)
plt.show()

print(f'B diff: {ks_2samp(B_diff_0, B_diff_5)}')
print(f'C_diff: {ks_2samp(C_diff_0, C_diff_5)}')

