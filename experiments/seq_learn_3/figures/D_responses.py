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



def bootstrap_means(arr: xr.DataArray) -> Tuple[xr.DataArray]:
    n_iters = 1000
    n_rois = arr.sizes['roi']
    means = []
    for i in range(n_iters):
        inds = np.random.choice(n_rois, n_rois, replace=True)
        sub = arr.isel(roi=inds)
        means.append(sub.mean('roi'))
    means = np.stack(means)
    low = xr.DataArray(np.percentile(means, 2.5, axis=0), dims=('time',))
    mean = xr.DataArray(np.mean(means, axis=0), dims=('time',))
    high = xr.DataArray(np.percentile(means, 97.5, axis=0), dims=('time',))
    return low, mean, high


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

ABCD_0 = flex_split(sessions_0, 'ABCD', drop_gray=False)
ABCD_5 = flex_split(sessions_5, 'ABCD', drop_gray=False)
ABBD_0 = flex_split(sessions_0, 'ABBD', drop_gray=False)
ABBD_5 = flex_split(sessions_5, 'ABBD', drop_gray=False)
ACBD_0 = flex_split(sessions_0, 'ACBD', drop_gray=False)
ACBD_5 = flex_split(sessions_5, 'ACBD', drop_gray=False)

t_slc = slice(26, 34)
# t_slc = slice(None, None)
day_0 = {}
day_0['ABCD'] = ABCD_0.isel(time=t_slc).mean('trial')
day_0['ABBD'] = ABBD_0.isel(time=t_slc).mean('trial')
day_0['ACBD'] = ACBD_0.isel(time=t_slc).mean('trial')

day_5 = {}
day_5['ABCD'] = ABCD_5.isel(time=t_slc).mean('trial')
day_5['ABBD'] = ABBD_5.isel(time=t_slc).mean('trial')
day_5['ACBD'] = ACBD_5.isel(time=t_slc).mean('trial')

# D_0_IDs = get_IDs(0, 'D')
# for key, val in day_0.items():
#     day_0[key] = val.isel(roi=D_0_IDs)
#
# D_5_IDs = get_IDs(5, 'D')
# for key, val in day_5.items():
#     day_5[key] = val.isel(roi=D_5_IDs)

#
# plt.rcParams['font.size'] = 8
#
# fig, axes = plt.subplots(2, 3, figsize=(6, 3))
#
# ax = axes[0, 0]
# arr = day_0['ABCD']
# low, mean, high = bootstrap_means(arr)
# X = np.arange(len(low))
# ax.fill_between(X, low, high, color='black', alpha=0.2)
# ax.plot(X, mean, color='black', label='ABCD')
# ax.set_title('ABCD')
#
# ax = axes[0, 1]
# arr = day_0['ABBD']
# low, mean, high = bootstrap_means(arr)
# X = np.arange(len(low))
# ax.fill_between(X, low, high, color='black', alpha=0.2)
# ax.plot(X, mean, color='black', label='ABBD')
# ax.set_title('ABBD')
#
# ax = axes[0, 2]
# arr = day_0['ACBD']
# low, mean, high = bootstrap_means(arr)
# X = np.arange(len(low))
# ax.fill_between(X, low, high, color='black', alpha=0.2)
# ax.plot(X, mean, color='black', label='ACBD')
# ax.set_title('ACBD')
#
# #-------
#
# ax = axes[1, 0]
# arr = day_5['ABCD']
# low, mean, high = bootstrap_means(arr)
# X = np.arange(len(low))
# ax.fill_between(X, low, high, color='black', alpha=0.2)
# ax.plot(X, mean, color='black', label='ABCD')
#
# ax = axes[1, 1]
# arr = day_5['ABBD']
# low, mean, high = bootstrap_means(arr)
# X = np.arange(len(low))
# ax.fill_between(X, low, high, color='black', alpha=0.2)
# ax.plot(X, mean, color='black', label='ABBD')
#
# ax = axes[1, 2]
# arr = day_5['ACBD']
# low, mean, high = bootstrap_means(arr)
# X = np.arange(len(low))
# ax.fill_between(X, low, high, color='black', alpha=0.2)
# ax.plot(X, mean, color='black', label='ACBD')
#
# ylim = [0, 5]
# for ax in axes.flatten():
#     ax.set_ylim(ylim)
#
# plt.show()

#-------------------------------------------------------------------------------


fig, axes = plt.subplots(2, 2, figsize=(6, 4))

# ABBD
ax = axes[0, 0]
arr = day_0['ABCD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(X, mean, color='black', label='ABCD')

arr = day_0['ABBD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(X, mean, color='red', label='ABBD')
ax.set_title('ABBD')
ax.legend(loc='upper left')

# ACBD
ax = axes[0, 1]
arr = day_0['ABCD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(X, mean, color='black', label='ABCD')

arr = day_0['ACBD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(X, mean, color='red', label='ACBD')
ax.set_title('ACBD')
ax.legend(loc='upper left')

#----------

# ABBD
ax = axes[1, 0]
arr = day_5['ABCD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(X, mean, color='black', label='ABCD')

arr = day_5['ABBD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(X, mean, color='red', label='ABBD')
ax.set_title('ABBD')
ax.legend(loc='upper left')

# ACBD
ax = axes[1, 1]
arr = day_5['ABCD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(X, mean, color='black', label='ABCD')

arr = day_5['ACBD']
low, mean, high = bootstrap_means(arr)
X = np.arange(len(low))
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(X, mean, color='red', label='ACBD')
ax.set_title('ACBD')
ax.legend(loc='upper left')


ylim = [0.5, 1.5]
for ax in axes.flatten():
    ax.set_ylim(ylim)

fig.tight_layout(pad=0.5)
plt.show()
fig.savefig('figures/mean_D.png')

#-------------------------------------------------------------------------------
# PE ratios, KS-tests
# difference between PE ratios between days

abcd = day_0['ABCD'].mean('time')
abbd = day_0['ABBD'].mean('time')
acbd = day_0['ACBD'].mean('time')
ABBD_diff_0 = abbd / abcd
ACBD_diff_0 = acbd / abcd

abcd = day_5['ABCD'].mean('time')
abbd = day_5['ABBD'].mean('time')
acbd = day_5['ACBD'].mean('time')
ABBD_diff_5 = abbd / abcd
ACBD_diff_5 = acbd / abcd

r_abbd = ks_2samp(ABBD_diff_0, ABBD_diff_5)
r_acbd = ks_2samp(ACBD_diff_0, ACBD_diff_5)
print(f'ABBD: {r_abbd.pvalue}')
print(f'ACBD: {r_acbd.pvalue}')


fig, ax = plt.subplots(figsize=(2.75, 1.75))
positions = [1, 1.8, 2.6, 3.4]
# positions = [1, 2, 3, 4]
ax.violinplot(
    [ABBD_diff_0, ABBD_diff_5, ACBD_diff_0, ACBD_diff_5],
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
fig.savefig('figures/prediction_errors_violins_D.eps')
fig.savefig('figures/prediction_errors_violins_D.png')

#-------------------------------------------------------------------------------
# PE ratios, KS-tests
# different from standard within days

abcd_0 = day_0['ABCD'].mean('time')
abbd_0 = day_0['ABBD'].mean('time')
acbd_0 = day_0['ACBD'].mean('time')
r_abbd_0 = ks_2samp(abbd_0, abcd_0)
r_acbd_0 = ks_2samp(acbd_0, abcd_0)
print(f'day 0, ABB(D) vs ABC(D): {r_abbd_0.pvalue}')
print(f'day 0, ACB(D) vs ABC(D): {r_acbd_0.pvalue}')

abcd_5 = day_5['ABCD'].mean('time')
abbd_5 = day_5['ABBD'].mean('time')
acbd_5 = day_5['ACBD'].mean('time')
r_abbd_5 = ks_2samp(abbd_5, abcd_5)
r_acbd_5 = ks_2samp(acbd_5, abcd_5)
print(f'day 5, ABB(D) vs ABC(D): {r_abbd_5.pvalue}')
print(f'day 5, ACB(D) vs ABC(D): {r_acbd_5.pvalue}')


fig, ax = plt.subplots(figsize=(2.75, 1.75))
positions = [1, 1.8, 2.6, 4.0, 4.8, 5.6]
ax.violinplot(
    [abcd_0, abbd_0, acbd_0, abcd_5, abbd_5, acbd_5],
    positions=positions,
    showextrema=False,
    showmeans=False,
    quantiles=[[0.25, 0.5, 0.75] for _ in range(6)]
)
ax.set_ylim([0, 5])
ax.set_ylabel('PE ratio')
ax.set_xticks(positions)
ax.set_xticklabels([
    '0: ABCD',
    '0: ABBD',
    '0: ACBD',
    '5: ABCD',
    '5: ABBD',
    '5: ACBD',
], rotation=60)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout(pad=0.5)
plt.show()

fig.savefig('figures/D_violins.png')

