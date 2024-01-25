from types import SimpleNamespace
from typing import Sequence, Tuple

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
from scipy.stats import *

from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *
from main import *



def split_trials(
    sessions: SessionGroup,
    event: str,
    visual: Optional[bool] = None,
    trial_mean: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:

    for s in sessions:
        s.spikes.prepare()
        data = s.spikes.data
        if visual is None:
            pass
        elif visual in {False, True}:
            vis_inds = np.load(s.fs.getsyspath('visual.npy'))
            if visual is True:
                data = data.sel(roi=vis_inds)
            else:
                all_inds = np.array(data.coords['roi'])
                non_vis_inds = np.setdiff1d(all_inds, vis_inds)
                data = data.sel(roi=non_vis_inds)
        else:
            raise ValueError('invalid "visual" argument')
        s.spikes.data = data
        del s.spikes.data.coords["roi"]

    """
    Trial-average data, get sorting index for first half of trials
    """

    # split/combine sequence data
    if event in ('ABCD', 'ABBD', 'ACBD'):
        spec = schema.get(sequence=event)[:-1]
    else:
        spec = schema.get(event=event)

    splits = sessions.split('spikes', spec)
    n_trials = min([arr.sizes['trial'] for arr in splits])
    splits = [arr.isel(trial=slice(0, n_trials)) for arr in splits]
    data = xr.concat(splits, 'roi')

    # split into even/odd trials
    n_trials = data.sizes["trial"] - 1
    n_trials = n_trials - 1 if n_trials % 2 == 1 else n_trials

    group_1_trial_inds = np.arange(0, n_trials, 2)
    group_2_trial_inds = group_1_trial_inds + 1

    group_1 = data.isel(trial=group_1_trial_inds)
    group_2 = data.isel(trial=group_2_trial_inds)

    group_1 = group_1.transpose('roi', ...)
    group_2 = group_2.transpose('roi', ...)

    if trial_mean:
        group_1 = group_1.mean('trial')
        group_2 = group_2.mean('trial')

        # order both by sort order of group 1 for use in heatmaps
        inds_1 = np.argsort(group_1.argmax('time'))
        inds_2 = np.argsort(group_2.argmax('time'))

        group_1 = group_1.isel(roi=inds_1)
        group_2 = group_2.isel(roi=inds_1)

    return group_1, group_2


save = False
event = "ACBD"
visual = None
sessions_0 = get_sessions(day=0)
sessions_5 = get_sessions(day=5)

day_0 = split_trials(sessions_0, event, visual, trial_mean=False)
day_5 = split_trials(sessions_5, event, visual, trial_mean=False)

# trial average
day_0 = [arr.mean('trial') for arr in day_0]
day_5 = [arr.mean('trial') for arr in day_5]

# get sorting indiices
# day_0_inds = [np.argsort(arr.argmax('time')) for arr in day_0]
# day_5_inds = [np.argsort(arr.argmax('time')) for arr in day_5]
day_0_inds = [arr.argmax('time') for arr in day_0]
day_5_inds = [arr.argmax('time') for arr in day_5]
# compute spearman's rho
rho_0 = spearmanr(day_0_inds[0], day_0_inds[1])
rho_5 = spearmanr(day_5_inds[0], day_5_inds[1])
