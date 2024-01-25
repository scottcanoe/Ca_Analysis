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
from statsmodels.distributions.empirical_distribution import ECDF

from ca_analysis import *
from ca_analysis.stats import gaussian_kde
from seq_learn_3.main import *
from seq_learn_3.utils import *


"""
roi_filter = 'gratings'

sessions_0 = get_sessions(day=0, fs=0)
sessions_5 = get_sessions(day=5, fs=0)

apply_roi_filter(sessions_0, roi_filter)
apply_roi_filter(sessions_5, roi_filter)

sequence = 'ACBD'

ABCD_0 = flex_split(sessions_0, sequence)
ABCD_5 = flex_split(sessions_5, sequence)
# ABBD_0 = flex_split(sessions_0, 'ABBD')
# ABBD_5 = flex_split(sessions_5, 'ABBD')
# ACBD_0 = flex_split(sessions_0, 'ACBD')
# ACBD_5 = flex_split(sessions_5, 'ACBD')


A_0 = ABCD_0.isel(time=slice(2, 8)).mean('trial').mean('time')
B_0 = ABCD_0.isel(time=slice(10, 16)).mean('trial').mean('time')
C_0 = ABCD_0.isel(time=slice(18, 24)).mean('trial').mean('time')
D_0 = ABCD_0.isel(time=slice(26, 32)).mean('trial').mean('time')

A_5 = ABCD_5.isel(time=slice(2, 8)).mean('trial').mean('time')
B_5 = ABCD_5.isel(time=slice(10, 16)).mean('trial').mean('time')
C_5 = ABCD_5.isel(time=slice(18, 24)).mean('trial').mean('time')
D_5 = ABCD_5.isel(time=slice(26, 32)).mean('trial').mean('time')

lst_0 = [A_0, B_0, C_0, D_0]
lst_5 = [A_5, B_5, C_5, D_5]

corrs = []
lst = lst_0
for i in range(len(lst) - 1):
    for j in range(i + 1, len(lst)):
        c = pearsonr(lst[i], lst[j]).statistic
        # c = np.linalg.norm(lst[i] - lst[j])
        corrs.append(c)
corrs_0 = np.array([corrs])

corrs = []
lst = lst_5
for i in range(len(lst) - 1):
    for j in range(i + 1, len(lst)):
        c = pearsonr(lst[i], lst[j]).statistic
        # c = np.linalg.norm(lst[i] - lst[j])
        corrs.append(c)
corrs_5 = np.array([corrs])

mu_0 = corrs_0.mean()
mu_5 = corrs_5.mean()
print(f'day 0: {mu_0}, day 5: {mu_5}')
"""

for roi_filter in ('all', 'visual', 'gratings'):
# for roi_filter in ('all',):

    sessions_0 = get_sessions(day=0, fs=0)
    sessions_5 = get_sessions(day=5, fs=0)

    apply_roi_filter(sessions_0, roi_filter)
    apply_roi_filter(sessions_5, roi_filter)

    for sequence in ('ABCD', 'ABBD', 'ACBD'):
    # for sequence in ('ABCD',):
        ABCD_0 = flex_split(sessions_0, sequence)
        ABCD_5 = flex_split(sessions_5, sequence)


        A_0 = ABCD_0.isel(time=slice(2, 8)).mean('time')
        B_0 = ABCD_0.isel(time=slice(10, 16)).mean('time')
        C_0 = ABCD_0.isel(time=slice(18, 24)).mean('time')
        D_0 = ABCD_0.isel(time=slice(26, 32)).mean('time')

        A_5 = ABCD_5.isel(time=slice(2, 8)).mean('time')
        B_5 = ABCD_5.isel(time=slice(10, 16)).mean('time')
        C_5 = ABCD_5.isel(time=slice(18, 24)).mean('time')
        D_5 = ABCD_5.isel(time=slice(26, 32)).mean('time')

        lst_0_all = [A_0, B_0, C_0, D_0]
        lst_5_all = [A_5, B_5, C_5, D_5]

        corrs = []
        for t in range(500):
            lst = [elt.isel(trial=t) for elt in lst_0_all]
            for i in range(len(lst) - 1):
                for j in range(i + 1, len(lst)):
                    c = pearsonr(lst[i], lst[j]).statistic
                    # c = np.linalg.norm(lst[i] - lst[j])
                    corrs.append(c)
        corrs_0 = np.array([corrs]).flatten()

        corrs = []
        for t in range(500):
            lst = [elt.isel(trial=t) for elt in lst_5_all]
            for i in range(len(lst) - 1):
                for j in range(i + 1, len(lst)):
                    c = pearsonr(lst[i], lst[j]).statistic
                    # c = np.linalg.norm(lst[i] - lst[j])
                    corrs.append(c)
        corrs_5 = np.array([corrs]).flatten()


        mu_0 = corrs_0.mean()
        mu_5 = corrs_5.mean()
        print(f'sequence: {sequence}, {roi_filter} cells')
        print(f'mean: day 0: {mu_0}, day 5: {mu_5}')
        print(f'medians: day 0: {np.median(corrs_0)}, day 5: {np.median(corrs_5)}')
        print(f'ks test: {ks_2samp(corrs_0, corrs_5)}')
        print('')

        ## this section for gaussian kde/pdf
        # from ca_analysis.stats import gaussian_kde
        #
        # fig, ax = plt.subplots(figsize=(2, 2))
        # X = np.linspace(-0.1, 0.9, 1000)
        # x, y = gaussian_kde(corrs_0, X=X)
        # ax.plot(x, y, color='black', label='day 0')
        # x, y = gaussian_kde(corrs_5, X=X)
        # ax.plot(x, y, color='red', label='day 5')
        # ax.legend()
        # title = f'{sequence}, {roi_filter} cells'
        # ax.set_title(title)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.set_xlim([-0.1, 0.9])
        # ax.set_ylim([0, 5.5])
        # plt.tight_layout(pad=0.5)
        # plt.show()
        # eps_path = f'figures/correlations/{sequence}_{roi_filter}.eps'
        # fig.savefig(eps_path)

        # this section for cdfs
        from scipy.stats import ecdf
        fig, ax = plt.subplots(figsize=(2, 2))

        ecdf0 = ECDF(corrs_0)
        x0, y0 = ecdf0.x, ecdf0.y
        ax.plot(x0, y0, color='black', label='day 0')

        ecdf5 = ECDF(corrs_5)
        x5, y5 = ecdf5.x, ecdf5.y
        ax.plot(x5, y5, color='red', label='day 5')

        cdf0 = ecdf(corrs_0).cdf
        cdf5 = ecdf(corrs_5).cdf
        X = np.linspace(0, 1, 100)
        y0 = cdf0.evaluate(X)
        y5 = cdf5.evaluate(X)
        diffs = np.abs(y5 - y0)
        imax = np.nanargmax(diffs)
        ax.plot([X[imax], X[imax]], [y0[imax], y5[imax]], color='blue')

        ax.set_xlim([-0.1, 0.8])
        ax.set_ylim([0, 1])
        ax.legend()
        title = f'{sequence}, {roi_filter} cells'
        ax.set_title(title)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.set_xlim([0, 1])

        plt.tight_layout(pad=0.5)
        plt.show()
        eps_path = f'figures/correlations/cdfs/{sequence}_{roi_filter}.eps'
        fig.savefig(eps_path)

