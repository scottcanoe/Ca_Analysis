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



sessions_0 = get_sessions(day=0, fs=0)
sessions_5 = get_sessions(day=5, fs=0)

kind = 'visual'

apply_roi_filter(sessions_0, 'gratings')
apply_roi_filter(sessions_5, 'gratings')
sizes_0 = flex_split(sessions_0, 'ABCD.A').sizes['roi']
sizes_5 = flex_split(sessions_5, 'ABCD.A').sizes['roi']
events = schema.events[1:]
# events = [ev for ev in events if '_' not in ev.name]

MU, STD, C = [], [], []
for ev in events:
    data = flex_split(sessions_0, ev).mean('time')
    if sizes_0 > sizes_5:
        print('size 0 > size 5')
        SUB_MU, SUB_STD, SUB_C = [], [], []
        for _ in range(100):
            rand_inds = np.random.choice(sizes_0, sizes_5, replace=False)
            sub_data = data.isel(roi=rand_inds)
            sub_mu = float(sub_data.mean())
            sub_std = float(sub_data.std())
            sub_c = sub_std / sub_mu
            SUB_MU.append(sub_mu)
            SUB_STD.append(sub_std)
            SUB_C.append(sub_c)
        SUB_MU, SUB_STD, SUB_C = np.array(SUB_MU), np.array(SUB_STD), np.array(SUB_C)
        MU.append(SUB_MU.mean())
        STD.append(SUB_STD.mean())
        C.append(SUB_C.mean())
    else:
        pass
        mu = float(data.mean())
        std = float(data.std())
        c = std / mu
        MU.append(mu)
        STD.append(std)
        C.append(c)
MU_0 = np.array(MU)
STD_0 = np.array(STD)
C_0 = np.array(C)

MU, STD, C = [], [], []
for ev in events:
    data = flex_split(sessions_5, ev).mean('time')

    mu = float(data.mean())
    std = float(data.std())
    c = std / mu
    MU.append(mu)
    STD.append(std)
    C.append(c)
MU_5 = np.array(MU)
STD_5 = np.array(STD)
C_5 = np.array(C)

dct = {
    'day 0': [MU_0.mean(), STD_0.mean(), C_0.mean()],
    'day 5': [MU_5.mean(), STD_5.mean(), C_5.mean()],
}
df = pd.DataFrame(dct, index=['mu', 'std', 'C'])
print(df)

# pre = []
# for s in sessions_0:
#     spikes = s.spikes.data
#     pre.append(spikes.mean('time'))
# pre = np.concatenate(pre)
#
# post = []
# for s in sessions_5:
#     spikes = s.spikes.data
#     post.append(spikes.mean('time'))
# post = np.concatenate(post)
#
# fig, ax = plt.subplots()
#
# x, y = gaussian_kde(pre)
# ax.plot(x, y, color='black', label='day 0')
#
# x, y = gaussian_kde(post)
# ax.plot(x, y, color='red', label='day 5')
#
# ax.legend()
# fig.suptitle('gray')
# plt.show()
#
# print(f'pre: {pre.mean()}')
# print(f'post: {post.mean()}')
# print(f'P: {ks_2samp(pre, post)}')







