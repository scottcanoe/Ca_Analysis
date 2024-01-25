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




def get_IDs(day:int, letter:str) -> np.array:
    path = Path(__file__).parent.parent / f'selectivity/cells_day{day}_scott.ods'
    df = pd.read_excel(path, index_col=0)
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

# B_0 = ABCD_0.isel(time=slice(10, 18)).mean('trial').mean('time')
# B_nov_0 = ABBD_0.isel(time=slice(18, 26)).mean('trial').mean('time')
# B_5 = ABCD_5.isel(time=slice(10, 18)).mean('trial').mean('time')
# B_nov_5 = ABBD_5.isel(time=slice(18, 26)).mean('trial').mean('time')
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


plt.rcParams['font.size'] = 8

fig, ax = plt.subplots(figsize=(2.75, 1.75))
positions = [1, 1.8, 2.6, 3.4]
# positions = [1, 2, 3, 4]
ax.violinplot(
    [B_diff_0, B_diff_5, C_diff_0, C_diff_5],
    positions=positions,
    showextrema=False,
    showmeans=False,
    quantiles=[[0.25, 0.5, 0.75],[0.25, 0.5, 0.75],[0.25, 0.5, 0.75],[0.25, 0.5, 0.75]],
)
ax.set_ylim([0, 5])
ax.set_ylabel('PE ratio')
ax.set_xticks(positions)
ax.set_xticklabels([
    'day 0',
    'day 5',
    'day 0',
    'day 5',
], rotation=60)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout(pad=0.5)
plt.show()
fig.savefig('figures/prediction_errors_violins_wo_mean.eps')
print(f'B diff: {ks_2samp(B_diff_0, B_diff_5)}')
print(f'C_diff: {ks_2samp(C_diff_0, C_diff_5)}')

