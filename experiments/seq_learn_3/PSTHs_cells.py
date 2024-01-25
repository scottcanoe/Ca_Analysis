import json
from types import SimpleNamespace
from typing import Iterable, Mapping

import h5py
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from types import SimpleNamespace

from ca_analysis.plot import *
from main import *
from selectivity import load_regression_table



def make_cell_PSTHs(s: Session):

    lpad = 4
    rpad = 4
    day = s.attrs['day']
    if day in {0, 5}:
        sequences = ('ABCD', 'ABBD', 'ACBD')
    else:
        sequences = ('ABCD',)

    # set up splits
    splits = {}
    for seq_name in sequences:
        spec = schema.get(sequence=seq_name)[:-1]
        arr = s.spikes.split(spec, lpad=lpad, rpad=rpad, concat='time').mean('trial')
        arr.coords['roi'] = s.spikes.data.coords['roi']
        splits[seq_name] = arr

    roi_ids = s.spikes.data.coords['roi'].data
    for roi in roi_ids:

        fig = Figure(figsize=(6, 3))
        ax = fig.add_subplot(1, 1, 1)

        seq = 'ABCD'
        arr = splits[seq].sel(roi=roi)
        ax.plot(arr, color='black', label=seq)
        if day in {0, 5}:
            seq = 'ABBD'
            arr = splits[seq].sel(roi=roi)
            ax.plot(arr, color='red', label=seq)

            seq = 'ACBD'
            arr = splits[seq].sel(roi=roi)
            ax.plot(arr, color='blue', label=seq)

        annotate_onsets(ax, arr.coords['event'], skip_first=True, last='-')
        title = f"{s.mouse}, day {s.attrs['day']}, roi {roi}"
        ax.set_title(title)
        ax.legend()
        ax.set_xlim([0, len(arr)])

        plotdir = Path(s.fs.getsyspath('analysis')) / 'PSTHs/rois'
        plotdir.mkdir(exist_ok=True, parents=True)
        fname = plotdir / f"roi_{roi}.png"
        fig.savefig(fname)
        fig.clear()
        del fig


if __name__ == "__main__":
    sessions = get_sessions(fs=0)
    # s = sessions[-1]
    for s in sessions:
        logger.info(f'Making PSTHs for {s}')
        make_cell_PSTHs(s)
