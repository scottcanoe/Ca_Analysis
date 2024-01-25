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


#-----------
def fit(data, roi):

    r = data.isel(roi=roi)

    spiketimes = np.array([], dtype=int)
    for i in range(r.sizes['trial']):
        locs = argwhere(r.isel(trial=i))
        spiketimes = np.r_[locs, spiketimes]

    T = r.sizes['time']
    y = spiketimes
    bins = np.arange(T + 1) - 0.5
    counts, _ = np.histogram(y, bins=bins, density=True)

    X = np.arange(T)
    preds = pd.DataFrame(data={'constant': np.ones_like(X), 'X': X, 'X2': X**2})

    glm = sm.GLM(counts, preds, family=sm.families.Gaussian(log()))
    res = glm.fit()
    params = res.params

    res.mu = -params[1] / (2 * params[2])                  # place field center
    res.sigma = np.sqrt(-1 / (2 * params[2]))             # place field size
    res.alpha = np.exp(params[0] - params[1]**2 / 4 / params[2])  # max firing rate

    # fig, ax = plt.subplots()
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(X, counts)
    out = np.exp(params[0] + params[1] * X + params[2] * X**2)
    ax.plot(X, out)

    mu = -params[1] / (2 * params[2])                  # place field center
    sigma = np.sqrt(-1 / (2 * params[2]))             # place field size

    title = 'roi: {}, mu: {:.2f}, sigma: {:.2f}, llf: {:.2f}'.format(roi, mu, sigma, res.llf)
    ax.set_title(title)
    plotdir = Path.home() / 'plots/seq_learn_3/fits'
    plotdir.mkdir(exist_ok=True)
    path = plotdir / f'{roi}.png'
    fig.savefig(path)
    # plt.show()

    return res


def place_fields():
    sessions_0 = get_sessions(day=0, fs=0)
    s = sessions_0[-1]
    chunks = s.spikes.split('ABCD', lpad=lpad, rpad=rpad)
    arr = xr.concat(chunks, 'time')
    r = arr.isel(roi=3)

    spiketimes = np.array([], dtype=int)
    for i in range(r.sizes['trial']):
        locs = argwhere(r.isel(trial=i))
        spiketimes = np.r_[locs, spiketimes]

    for j in range(10):

        r = arr.isel(roi=j)

        spiketimes = np.array([], dtype=int)
        for i in range(r.sizes['trial']):
            locs = argwhere(r.isel(trial=i))
            spiketimes = np.r_[locs, spiketimes]

        T = r.sizes['time']
        y = spiketimes
        bins = np.arange(T + 1) - 0.5
        counts, _ = np.histogram(y, bins=bins, density=True)

        X = np.arange(T)
        preds = pd.DataFrame(data={'constant': np.ones_like(X), 'X': X, 'X2': X ** 2})

        glm = sm.GLM(counts, preds, family=sm.families.Gaussian(log()))
        res = glm.fit()
        params = res.params
        print(res.summary())

        fig, ax = plt.subplots()
        ax.bar(X, counts)
        out = np.exp(params[0] + params[1] * X + params[2] * X ** 2)
        ax.plot(X, out)

        mu = -params[1] / (2 * params[2])  # place field center
        sigma = np.sqrt(-1 / (2 * params[2]))  # place field size
        alpha = np.exp(params[0] - params[1] ** 2 / 4 / params[2])  # max firing rate

        title = 'roi: {}, mu: {:.2f}, sigma: {:.2f}, llf: {:.2f}'.format(j, mu, sigma, res.llf)
        ax.set_title(title)
        plt.show()

        print('mu: {}\nsigma: {}\nalpha: {}'.format(mu, sigma, alpha))



def load(day: int, sequence: str) -> xr.DataArray:

    h5_path = Path(__file__).parent / 'data/data.h5'
    with h5py.File(h5_path, 'r') as f:
        group = f[f'day{day}']
        dset = group[sequence]
        out = xr.DataArray(dset[:], dims=('trial', 'time', 'roi'))
    return out

day = 0
roi = 0

ABCD_all = load(day, 'ABCD')
ABBD_all = load(day, 'ABBD')
ACBD_all = load(day, 'ACBD')

ABCD = ABCD_all.isel(roi=roi)
ABBD = ABBD_all.isel(roi=roi)
ACBD = ACBD_all.isel(roi=roi)

ABCD_A = ABCD.isel(time=slice(2, 8)).mean('time')
ABCD_B = ABCD.isel(time=slice(10, 16)).mean('time')
ABCD_C = ABCD.isel(time=slice(18, 24)).mean('time')
ABCD_D = ABCD.isel(time=slice(26, 32)).mean('time')
ABCD_gr = ABCD.isel(time=slice(34, None)).mean('time')

