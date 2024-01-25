# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *


plt.rcParams['font.size'] = 8


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


from experiments.meso.roi_processing import ROI

lpad = 10
rpad = 10

s = open_session('M110', '2023-04-21', '2', fs=0)
# create_masks(s)
LH = ROI(s, 'LH', 'ach')
RH = ROI(s, 'RH', 'ach')


#-------------------------------------------------------------------------------
# EPDFs

from ca_analysis.stats import gaussian_kde

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
X = np.linspace(-0.04, 0.04, 1000)
ax.axvline(0, color='gray', ls='--')

LH_left, LH_right = LH.split('left'), LH.split('right')
RH_left, RH_right = RH.split('left'), RH.split('right')
contra = xr.concat([LH_right, RH_left], 'trial')
ipsi = xr.concat([LH_left, RH_right], 'trial')
data = (contra - ipsi).data.flatten()
X, Y = gaussian_kde(data, X=X)
ax.plot(X, Y, color='black', label='stim')

LH_left = LH.split('left', rpad=15)[1].isel(time=slice(5, None))
LH_right = LH.split('right', rpad=15)[1].isel(time=slice(5, None))
RH_left = RH.split('left', rpad=15)[1].isel(time=slice(5, None))
RH_right = RH.split('right', rpad=15)[1].isel(time=slice(5, None))
contra = xr.concat([LH_right, RH_left], 'trial')
ipsi = xr.concat([LH_left, RH_right], 'trial')
data = (contra - ipsi).data.flatten()
X, Y = gaussian_kde(data, X=X)
ax.plot(X, Y, color='red', label='gray')

ax.set_xlim([X[0], X[-1]])
ax.legend()
ax.set_title('contra - ipsi')
plt.show()

fig.savefig('figures/diff_pdf.png')
fig.savefig('figures/diff_pdf.eps')

#-------------------------------------------------------------------------------
# ECDFs

from scipy.stats import ecdf, ks_2samp

fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))
X = np.linspace(-0.04, 0.04, 1000)

# stim
LH_left, LH_right = LH.split('left'), LH.split('right')
RH_left, RH_right = RH.split('left'), RH.split('right')
contra = xr.concat([LH_right, RH_left], 'trial')
ipsi = xr.concat([LH_left, RH_right], 'trial')
data_stim = (contra - ipsi).data.flatten()
res_stim = ecdf(data_stim)
q_stim = res_stim.cdf.quantiles
y_stim = res_stim.cdf.evaluate(q_stim)

# gray
LH_left = LH.split('left', rpad=15)[1].isel(time=slice(5, None))
LH_right = LH.split('right', rpad=15)[1].isel(time=slice(5, None))
RH_left = RH.split('left', rpad=15)[1].isel(time=slice(5, None))
RH_right = RH.split('right', rpad=15)[1].isel(time=slice(5, None))
contra = xr.concat([LH_right, RH_left], 'trial')
ipsi = xr.concat([LH_left, RH_right], 'trial')
data_gray = (contra - ipsi).data.flatten()
res_gray = ecdf(data_gray)
q_gray = res_gray.cdf.quantiles
y_gray = res_gray.cdf.evaluate(q_gray)

## Use this version to have the same X values
# X = np.linspace(-0.04, 0.04, 1000)
# y_gray = res_gray.cdf.evaluate(X)
# y_stim = res_stim.cdf.evaluate(X)
# ax.step(X, y_gray, color='black', label='gray')
# ax.step(X, y_stim, color='red', label='stim')

## Or use this version instead
ax.step(q_gray, y_gray, color='black', label='gray')
ax.step(q_stim, y_stim, color='red', label='stim', lw=1)
ax.set_xlim([-0.022, 0.04])

ax.legend(handlelength=1, frameon=False)
ks = ks_2samp(data_stim, data_gray)
x_ks = ks.statistic_location
y1 = res_stim.cdf.evaluate(x_ks)
y2 = res_gray.cdf.evaluate(x_ks)
ax.plot([x_ks, x_ks], [y1, y2], color='blue', lw=0.5)
# ax.set_title('contra - ipsi')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout(pad=0.5)
plt.show()
ax.set_yticks([0, 0.5, 1.0])
fig.savefig('figures/monoc_diff_cdf.png')
fig.savefig('figures/monoc_diff_cdf.eps')
