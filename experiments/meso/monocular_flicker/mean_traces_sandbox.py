import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal

from ca_analysis.plot import *
from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def bootstrap_mean(arr: xr.DataArray) -> Tuple[xr.DataArray]:
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
rpad = 5

s = open_session('M115', '2023-04-21', '2', fs=0)
# create_masks(s)
LV = ROI(s, 'LV', 'ach')
RV = ROI(s, 'RV', 'ach')

cutoff = 0.1
order = 2
LV.data.data = butter_highpass_filter(LV.data, cutoff, 10.0, order=order)
RV.data.data = butter_highpass_filter(RV.data, cutoff, 10.0, order=order)


LV_left = LV.split('left', lpad=lpad, rpad=rpad, concat='time')
LV_right = LV.split('right', lpad=lpad, rpad=rpad, concat='time')
RV_left = RV.split('left', lpad=lpad, rpad=rpad, concat='time')
RV_right = RV.split('right', lpad=lpad, rpad=rpad, concat='time')

events = LV_left.coords['event']
# for arr in (LH_left, LH_right, RH_left, RH_right):
#     del arr.coords['event']

scale = LV.data.max('time') - LV.data.min('time')
print(f"l scale: {LV.data.min('time').item()}  -  {LV.data.max('time').item()}: {scale}")
# scale = 1
for arr in (LV_left, LV_right):
    fronts = arr.isel(time=slice(0, lpad))
    baselines = fronts.mean('time')
    for i in range(baselines.sizes['trial']):
        row = arr[i]
        arr[i] = (row - baselines[i]) / scale
        # arr[i] = row / scale

scale = RV.data.max('time') - RV.data.min('time')
print(f"r scale: {RV.data.min('time').item()}  -  {RV.data.max('time').item()}: {scale}")
# scale = 1
for arr in (RV_left, RV_right):
    fronts = arr.isel(time=slice(0, lpad))
    baselines = fronts.mean('time')
    for i in range(baselines.sizes['trial']):
        row = arr[i]
        arr[i] = (row - baselines[i]) / scale
        # arr[i] = row / scale


contra = xr.concat([LV_right, RV_left], 'trial')
ipsi = xr.concat([LV_left, RV_right], 'trial')

fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.5))
ax.axhline(color='gray', ls='--')
X = np.arange(contra.sizes['time'])

low, mean, high = bootstrap_mean(contra)
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(mean, color='black', label='contra')
low, mean, high = bootstrap_mean(ipsi)
ax.fill_between(X, low, high, color='red', alpha=0.2)
ax.plot(mean, color='red', label='ipsi')
ax.legend(handlelength=0.5, frameon=False, loc='upper right')
annotate_onsets(ax, events, skip_first=True, vline=False)
ax.set_xlim([X[0], X[-1]])
ax.set_yticks([-0.05, 0, 0.05, 0.1])
ax.set_yticklabels(['-5', '0', '5', '10'])
ax.set_ylabel('% change')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig.tight_layout(pad=0.5)
plt.show()
fig.savefig(f'figures/monoc_mean_traces_contra_vs_ipsi_{s.mouse}.png')
fig.savefig(f'figures/monoc_mean_traces_contra_vs_ipsi_{s.mouse}.eps')
#
#
# ##----------------------------------------------------------------------------
# # difference plot


fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.5))
ax.axhline(color='gray', ls='--')
X = np.arange(contra.sizes['time'])
Y = contra - ipsi
low, mean, high = bootstrap_mean(Y)
ax.fill_between(X, low, high, color='black', alpha=0.2)
ax.plot(mean, color='black')
# ax.set_title('contra - ipsi')

ax.legend(handlelength=1, frameon=False)
annotate_onsets(ax, events, skip_first=True, vline=False)
ax.set_xlim([X[0], X[-1]])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_yticks([-0.05, 0, 0.05, 0.1])
ax.set_yticklabels(['-5', '0', '5', '10'])
ax.set_ylabel('% change')


fig.tight_layout(pad=0.5)
plt.show()
fig.savefig(f'figures/monoc_mean_traces_contra_vs_ipsi_difference_{s.mouse}.png')
fig.savefig(f'figures/monoc_mean_traces_contra_vs_ipsi_difference_{s.mouse}.eps')

