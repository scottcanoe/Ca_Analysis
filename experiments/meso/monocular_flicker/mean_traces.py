# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ca_analysis.plot import *
from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *


def bootstrap_means(arr: xr.DataArray) -> Tuple[xr.DataArray]:
    n_iters = 1000
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

fig, axes = plt.subplots(2, 1, figsize=(5, 3))
X = np.arange(LH_left.sizes['time'])

ax = axes[0]
low, mean, high = bootstrap_means(LH_left)
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(mean, color='black', label='left')
low, mean, high = bootstrap_means(LH_right)
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(mean, color='red', label='right')
ax.set_title('left V1')

ax = axes[1]
low, mean, high = bootstrap_means(RH_left)
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(mean, color='black', label='left')
low, mean, high = bootstrap_means(RH_right)
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(mean, color='red', label='right')
ax.set_title('right V1')

for ax in axes:
    ax.legend()
    ax.set_xlim([0, mean.sizes['time']])
    annotate_onsets(ax, LH_left.coords['event'])

fig.tight_layout(pad=1)
plt.show()
fig.savefig('mean_traces.png')

"""
Compare LH values between two conditions (left and right stim) w/ violin.
"""

fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.75))
positions = [1, 1.8]

ax = axes[0]
X = LH_left.isel(time=slice(10, 20)).data.flatten()
Y = LH_right.isel(time=slice(10, 20)).data.flatten()
ax.violinplot(
    [X, Y],
    positions=positions,
    showextrema=False,
    showmeans=False,
    quantiles=[[0.25, 0.5, 0.75] for _ in range(2)]
)
# ax.set_ylim([0, 5])
# ax.set_ylabel('PE ratio')
ax.set_xticks(positions)
ax.set_xticklabels(['left', 'right'], rotation=60)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('left V1')

ax = axes[1]
X = RH_left.isel(time=slice(10, 20)).data.flatten()
Y = RH_right.isel(time=slice(10, 20)).data.flatten()
ax.violinplot(
    [X, Y],
    positions=positions,
    showextrema=False,
    showmeans=False,
    quantiles=[[0.25, 0.5, 0.75] for _ in range(2)]
)
# ax.set_ylim([0, 5])
# ax.set_ylabel('PE ratio')
ax.set_xticks(positions)
ax.set_xticklabels(['left', 'right'], rotation=60)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('right V1')
ax.set_xlim([0, 3])

fig.tight_layout(pad=1)
plt.show()
fig.savefig('violins.png')

# --------------------

fig, axes = plt.subplots(2, 1, figsize=(6, 4))
X = np.arange(LH_left.sizes['time'])

ax = axes[0]
low, mean, high = bootstrap_means(LH_left)
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(mean, color='black', label='left V1')
low, mean, high = bootstrap_means(RH_left)
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(mean, color='red', label='right V1')
ax.set_title('left flicker')

ax = axes[1]
low, mean, high = bootstrap_means(LH_right)
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(mean, color='black', label='left V1')
low, mean, high = bootstrap_means(RH_right)
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(mean, color='red', label='right V1')
ax.set_title('right flicker')

for ax in axes:
    ax.legend(loc='upper right')
    ax.set_xlim([0, mean.sizes['time']])
    annotate_onsets(ax, LH_left.coords['event'])

fig.tight_layout(pad=1)
plt.show()
fig.savefig('mean_traces_2.png')

# ------------------


