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

#
# sessions_0 = get_sessions(day=0, fs=0)
# sessions_5 = get_sessions(day=5, fs=0)
# ABCD_0 = flex_split(sessions_0, 'ABCD')
# ABCD_5 = flex_split(sessions_5, 'ABCD')
# ABBD_0 = flex_split(sessions_0, 'ABBD')
# ABBD_5 = flex_split(sessions_5, 'ABBD')
# ACBD_0 = flex_split(sessions_0, 'ACBD')
# ACBD_5 = flex_split(sessions_5, 'ACBD')
#
# data = {0: {}, 5: {}}
# first = slice(2, 10)
# second = slice(10, 18)
# third = slice(18, 26)
# fourth = slice(26, 34)
#
# data[0]['ABCD.B'] = ABCD_0.isel(time=second).mean('trial').mean('time')
# data[0]['ABBD.B1'] = ABBD_0.isel(time=second).mean('trial').mean('time')
# data[0]['ABBD.B2'] = ABBD_0.isel(time=third).mean('trial').mean('time')
# data[0]['ACBD.B'] = ACBD_0.isel(time=third).mean('trial').mean('time')
# data[0]['ACBD.C'] = ACBD_0.isel(time=second).mean('trial').mean('time')
#
# data[5]['ABCD.B'] = ABCD_5.isel(time=second).mean('trial').mean('time')
# data[5]['ABBD.B1'] = ABBD_5.isel(time=second).mean('trial').mean('time')
# data[5]['ABBD.B2'] = ABBD_5.isel(time=third).mean('trial').mean('time')
# data[5]['ACBD.B'] = ACBD_5.isel(time=third).mean('trial').mean('time')
# data[5]['ACBD.C'] = ACBD_5.isel(time=second).mean('trial').mean('time')
#
#
# B_0_IDs = get_IDs(0, 'B')
# B_5_IDs = get_IDs(5, 'B')
# C_0_IDs = get_IDs(0, 'C')
# C_5_IDs = get_IDs(5, 'C')

"""
Compare AC[B]D vs A[B]CD
"""

standard = 'ABCD.B'
novel = 'ACBD.B'

standard_event = standard.split('.')[-1][0]
novel_event = novel.split('.')[-1][0]

standard_0 = data[0][standard].isel(roi=get_IDs(0, standard_event))
novel_0 = data[0][novel].isel(roi=get_IDs(0, novel_event))

standard_5 = data[5][standard].isel(roi=get_IDs(5, standard_event))
novel_5 = data[5][novel].isel(roi=get_IDs(5, novel_event))

R_0 = novel_0 / standard_0
R_5 = novel_5 / standard_5

fig, ax = plt.subplots()
ax.violinplot([R_0, R_5], showextrema=False)

ax.set_xticks([1, 2])
ax.set_xticklabels(['day 0', 'day 5'])

test = ks_2samp(R_0, R_5)
title = '{} / {}     p={:.3f}'.format(novel, standard, test.pvalue)
ax.set_title(title)

# ax.set_xticklabels([
#     'B_nov_0 / B_0',
#     'B_nov_5 / B_5',
#     'C_nov_0 / C_0',
#     'C_nov_0 / C_0',
# ], rotation=60)
plt.tight_layout(pad=1)
plt.show()

print(f'R-test: {ks_2samp(R_0, R_5)}')

