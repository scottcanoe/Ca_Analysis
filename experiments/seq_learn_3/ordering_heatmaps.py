from types import SimpleNamespace
from typing import Sequence

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
from scipy.stats import *

from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *
from main import *
from utils import *


def add_heatmap(ax, arr, title: str = "", label_peaks: bool = True):
    smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
    cdata = smap(arr)
    if label_peaks:
        maxs = arr.argmax('time')
        for i, peak in enumerate(maxs):
            peak = peak.item()
            cdata[i, peak] = 1
    im = ax.imshow(cdata, aspect="auto", interpolation='none')
    annotate_onsets(ax, arr.coords["event"], skip_first=False, alpha=1, color='gray')
    ax.set_title(title)
    return im


event = 'ABCD'
day = 5
visual = None
lpad = None
rpad = None

sessions = get_sessions(day=day)
apply_roi_filter(sessions, roi_filter)


data = flex_split(sessions, event, drop_gray=False)
# data = np.clip(data, 0, 5)

"""
Trial-average data, get sorting index for first half of trials
"""

# split into groups
group_1_trial_inds = np.arange(0, 400, 2)
group_2_trial_inds = group_1_trial_inds + 1
group_1 = data.isel(trial=group_1_trial_inds).mean('trial')
group_2 = data.isel(trial=group_2_trial_inds).mean('trial')

# rank rois
inds_1 = np.argsort(group_1.argmax('time'))
group_1 = group_1.isel(roi=inds_1).transpose('roi', ...)
group_2 = group_2.isel(roi=inds_1).transpose('roi', ...)


#-------------------------------------------------------------------------------


fig, axes = plt.subplots(1, 2, figsize=(24, 12))

ax, arr = axes[0], group_1
im = add_heatmap(ax, arr, 'even trials', label_peaks=False)

ax, arr = axes[1], group_2
im = add_heatmap(ax, arr, 'odd trials', label_peaks=False)

fig.tight_layout(pad=3)
plt.show()
