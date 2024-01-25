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


for day in (0, 5):
    sessions_base = get_sessions(day=day)


    roi_filter = 'all'
    apply_roi_filters(sessions_base, roi_filter)
    event = 'ABCD'
    lpad = None
    rpad = None

    for i in range(len(sessions_base)):
        sessions = sessions_base[i:i+1]
        data = flex_split(sessions, event, drop_gray=False)

        """
        Trial-average data, get sorting index for first half of trials
        """

        # rank rois
        n_trials = data.sizes["trial"]
        if n_trials % 2 == 1:
            n_trials -= 1
        all_trial_inds = np.arange(n_trials, dtype=int)
        group_1_trial_inds = np.random.choice(all_trial_inds, n_trials // 2, replace=False)
        group_2_trial_inds = np.setdiff1d(all_trial_inds, group_1_trial_inds)

        group_1_trial_inds = np.arange(0, 400, 2)
        group_2_trial_inds = group_1_trial_inds + 1

        group_1 = data.isel(trial=group_1_trial_inds).mean('trial')
        group_2 = data.isel(trial=group_2_trial_inds).mean('trial')

        inds_1 = np.argsort(group_1.argmax('time'))
        inds_2 = np.argsort(group_2.argmax('time'))

        group_1 = group_1.isel(roi=inds_1).transpose('roi', ...)
        group_2 = group_2.isel(roi=inds_1).transpose('roi', ...)


        #-------------------------------------------------------------------------------


        fig, axes = plt.subplots(1, 2, figsize=(24, 12))

        ax, arr = axes[0], group_1
        smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
        cdata = smap(arr)
        ax.imshow(cdata, aspect="auto", interpolation='none')
        annotate_onsets(ax, arr.coords["event"], skip_first=False, alpha=1, color='white')
        ax.set_title('group 1')

        ax, arr = axes[1], group_2
        smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
        cdata = smap(arr)
        ax.imshow(cdata, aspect="auto", interpolation='none')
        annotate_onsets(ax, arr.coords["event"], skip_first=False, alpha=1, color='white')
        ax.set_title('group 2')

        fig.tight_layout(pad=5)
        sv = Path.home() / f"plots/seq_learn_3/heatmaps_mice/{sessions[0].mouse} day {day}.png"
        fig.savefig(sv)
        plt.show()
