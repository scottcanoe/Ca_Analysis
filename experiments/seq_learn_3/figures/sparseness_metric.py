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


#
# for roi_filter in ('all', 'visual', 'gratings'):
# # for roi_filter in ('all',):
#
#     sessions_0 = get_sessions(day=0, fs=0)
#     sessions_5 = get_sessions(day=5, fs=0)
#
#     apply_roi_filter(sessions_0, roi_filter)
#     apply_roi_filter(sessions_5, roi_filter)
#
#     for sequence in ('ABCD', 'ABBD', 'ACBD'):
#     # for sequence in ('ABCD',):
#         ABCD_0 = flex_split(sessions_0, sequence)
#         ABCD_5 = flex_split(sessions_5, sequence)
#
#
#         A_0 = ABCD_0.isel(time=slice(2, 8)).mean('time')
#         B_0 = ABCD_0.isel(time=slice(10, 16)).mean('time')
#         C_0 = ABCD_0.isel(time=slice(18, 24)).mean('time')
#         D_0 = ABCD_0.isel(time=slice(26, 32)).mean('time')
#
#         A_5 = ABCD_5.isel(time=slice(2, 8)).mean('time')
#         B_5 = ABCD_5.isel(time=slice(10, 16)).mean('time')
#         C_5 = ABCD_5.isel(time=slice(18, 24)).mean('time')
#         D_5 = ABCD_5.isel(time=slice(26, 32)).mean('time')
#
#         lst_0_all = [A_0, B_0, C_0, D_0]
#         lst_5_all = [A_5, B_5, C_5, D_5]

roi_filter = 'all'

sessions_0 = get_sessions(day=0, fs=0)
sessions_5 = get_sessions(day=5, fs=0)

apply_roi_filter(sessions_0, roi_filter)
apply_roi_filter(sessions_5, roi_filter)

sequence = 'ABCD'
ABCD_0 = flex_split(sessions_0, sequence)
ABCD_5 = flex_split(sessions_5, sequence)


A_0 = ABCD_0.isel(time=slice(2, 8)).mean('time').mean('trial')
B_0 = ABCD_0.isel(time=slice(10, 16)).mean('time').mean('trial')
C_0 = ABCD_0.isel(time=slice(18, 24)).mean('time').mean('trial')
D_0 = ABCD_0.isel(time=slice(26, 32)).mean('time').mean('trial')

A_5 = ABCD_5.isel(time=slice(2, 8)).mean('time').mean('trial')
B_5 = ABCD_5.isel(time=slice(10, 16)).mean('time').mean('trial')
C_5 = ABCD_5.isel(time=slice(18, 24)).mean('time').mean('trial')
D_5 = ABCD_5.isel(time=slice(26, 32)).mean('time').mean('trial')

lst_0 = [A_0, B_0, C_0, D_0]
lst_5 = [A_5, B_5, C_5, D_5]

n = 4

lst = lst_0
num = np.sum([R / n for R in lst])**2
den = np.sum([R**2 / n for R in lst])
S0 = (1 - num / den) / (1 - 1 / n)

lst = lst_5
num = np.sum([R / n for R in lst])**2
den = np.sum([R**2 / n for R in lst])
S5 = (1 - num / den) / (1 - 1 / n)

