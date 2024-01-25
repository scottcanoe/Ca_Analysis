# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ca_analysis.plot import *
from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *


def bootstrap_means(arr: xr.DataArray) -> Tuple[xr.DataArray]:
    n_iters = 100
    n_trials = arr.sizes['trial']
    means = []
    for i in range(n_iters):
        inds = np.random.choice(n_trials, n_trials, replace=True)
        sub = arr.isel(trial=inds)
        means.append(sub.mean('trial'))
    means = np.stack(means)
    low = xr.DataArray(np.percentile(means, 2.5, axis=0), dims=('time',))
    mean = xr.DataArray(np.mean(means, axis=0), dims=('time',))
    high = xr.DataArray(np.percentile(means, 97.5, axis=0), dims=('time',))
    return low, mean, high


"""
Processing steps:
 >> s = open_session(...)
 >> create_masks(s) # save as 'masks.h5'
 >> process_masks(s) # creates 'rois.h5'

 Now we can open an ROI with something like (import ROI from experiments...
 >> LH = ROI(s, 'LH', 'ach') # assuming ROI named 'LH'
 >> data = LH.split('left', lpad=10, ...)
 - 
"""
from experiments.meso.roi_processing import ROI

lpad = 10
rpad = 10

s = open_session('M110', '2023-04-21', '2', fs=0)
# create_masks(s)
LH = ROI(s, 'LH', 'ach')
RH = ROI(s, 'RH', 'ach')

LH_left = LH.split('left', lpad=lpad, rpad=rpad, concat='time')
LH_right = LH.split('right', lpad=lpad, rpad=rpad, concat='time')
RH_left = RH.split('left', lpad=lpad, rpad=rpad, concat='time')
RH_right = RH.split('right', lpad=lpad, rpad=rpad, concat='time')

baselines = LH_left.isel(time=slice(0, lpad)).min('time')


events = LH_left.coords['event']
del LH_left.coords['event']
del LH_right.coords['event']
del RH_left.coords['event']
del RH_right.coords['event']

contra = xr.concat([LH_right, RH_left], 'trial')
ipsi = xr.concat([LH_left, RH_right], 'trial')

fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.5))
ax.axhline(color='gray', ls='--')
X = np.arange(contra.sizes['time'])

low, mean, high = bootstrap_means(contra)
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(mean, color='black', label='contra')
low, mean, high = bootstrap_means(ipsi)
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(mean, color='red', label='ipsi')
ax.legend(handlelength=0.5, frameon=False)
annotate_onsets(ax, events, skip_first=True, vline=False)
ax.set_xlim([X[0], X[-1]])
# ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig.tight_layout(pad=0.5)
plt.show()
fig.savefig('figures/mean_traces_contra_vs_ipsi2.png')
fig.savefig('figures/mean_traces_contra_vs_ipsi2.eps')


##------------------------------------------------------------------------------
# difference plot


fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.5))
ax.axhline(color='gray', ls='--')
X = np.arange(contra.sizes['time'])
Y = contra - ipsi
low, mean, high = bootstrap_means(Y)
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(mean, color='black')
# ax.set_title('contra - ipsi')

ax.legend(handlelength=1, frameon=False)
annotate_onsets(ax, events, skip_first=True, vline=False)
ax.set_xlim([X[0], X[-1]])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_yticks([])
fig.tight_layout(pad=0.5)
plt.show()
fig.savefig('figures/mean_traces_contra_vs_ipsi_difference2.png')
fig.savefig('figures/mean_traces_contra_vs_ipsi_difference2.eps')

