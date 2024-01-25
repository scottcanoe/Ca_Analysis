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
    df = pd.read_excel(path, index_col=0).fillna("")

    lst = []

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
day_0_all = flex_split(sessions_0, 'ABCD.A').sizes['roi']
day_5_all = flex_split(sessions_5, 'ABCD.A').sizes['roi']

# sessions_0 = get_sessions(day=0, fs=0)
# sessions_5 = get_sessions(day=5, fs=0)
apply_roi_filter(sessions_0 + sessions_5, 'visual')
day_0_vis = flex_split(sessions_0, 'ABCD.A').sizes['roi']
day_5_vis = flex_split(sessions_5, 'ABCD.A').sizes['roi']

print(f'day 0: {day_0_vis / day_0_all}')
print(f'day 5: {day_5_vis / day_5_all}')

