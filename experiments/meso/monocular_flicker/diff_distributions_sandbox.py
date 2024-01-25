# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal
from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *


plt.rcParams['font.size'] = 8


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


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

#-------------------------------------------------------------------------------
# EPDFs

from ca_analysis.stats import gaussian_kde

fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))
X = np.linspace(-0.04, 0.04, 1000)
ax.axvline(0, color='gray', ls='--')

LV_left = LV.split('left', lpad=lpad, rpad=rpad, concat='time')
LV_right = LV.split('right', lpad=lpad, rpad=rpad, concat='time')
RV_left = RV.split('left', lpad=lpad, rpad=rpad, concat='time')
RV_right = RV.split('right', lpad=lpad, rpad=rpad, concat='time')

"""
Need to rescale gray period (baseline subtraction) using stim data.
"""

L_scale = LV.data.max('time') - LV.data.min('time')
# L_scale = 1
for arr in (LV_left, LV_right):
    fronts = arr.isel(time=slice(0, lpad))
    baselines = fronts.mean('time')
    for i in range(baselines.sizes['trial']):
        arr[i] = (arr[i] - baselines[i]) / L_scale

R_scale = RV.data.max('time') - RV.data.min('time')
# R_scale = 1
for arr in (RV_left, RV_right):
    fronts = arr.isel(time=slice(0, lpad))
    baselines = fronts.mean('time')
    for i in range(baselines.sizes['trial']):
        arr[i] = (arr[i] - baselines[i]) / R_scale

contra = xr.concat([LV_right, RV_left], 'trial')
ipsi = xr.concat([LV_left, RV_right], 'trial')
diff = contra - ipsi

stim_vals = diff.isel(time=slice(10, 20)).data.flatten()
gray_vals = diff.isel(time=slice(0, 10)).data.flatten()
X = np.linspace(-0.4, 0.4, 1000)

X, Y = gaussian_kde(stim_vals, X=X)
ax.plot(X, Y, color='black', label='stim')

X, Y = gaussian_kde(gray_vals, X=X)
ax.plot(X, Y, color='red', label='gray')

ax.set_xlim([X[0], X[-1]])
ax.set_xticks([-0.25, 0, 0.25])
ax.set_xticklabels(['-25', '0', '25'])
ax.legend(handlelength=1, frameon=False)
ax.set_ylabel('density')
ax.set_xlabel('% change ($\Delta$)')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout(pad=0.5)
plt.show()

# fig.savefig('figures/monoc_diff_pdf.png')
fig.savefig(f'figures/monoc_diff_pdf_{s.mouse}.eps')
#
# #-----------------------------------------------------------------------------
# # ECDFs

from scipy.stats import ecdf, ks_2samp

fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))

# stim
res_stim = ecdf(stim_vals)
q_stim = res_stim.cdf.quantiles
y_stim = res_stim.cdf.evaluate(q_stim)

# gray
res_gray = ecdf(gray_vals)
q_gray = res_gray.cdf.quantiles
y_gray = res_gray.cdf.evaluate(q_gray)

## Use this version to have the same X values
X = np.linspace(-0.40, 0.40, 1000)
y_gray = res_gray.cdf.evaluate(X)
y_stim = res_stim.cdf.evaluate(X)
ax.step(X, y_gray, color='black', label='gray')
ax.step(X, y_stim, color='red', label='stim')

## Or use this version instead
# ax.step(q_gray, y_gray, color='black', label='gray')
# ax.step(q_stim, y_stim, color='red', label='stim', lw=1)
# ax.set_xlim([-0.022, 0.04])

ks = ks_2samp(stim_vals, gray_vals)
x_ks = ks.statistic_location
y1 = res_stim.cdf.evaluate(x_ks)
y2 = res_gray.cdf.evaluate(x_ks)
ax.plot([x_ks, x_ks], [y1, y2], color='blue', lw=0.5)
# ax.set_title('contra - ipsi')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([-0.4, 0.4])
ax.set_yticks([0, 0.5, 1.0])
ax.set_xticks([-0.25, 0, 0.25])
ax.set_xticklabels(['-25', '0', '25'])
ax.set_xlabel('% change ($\Delta$)')
# ax.set_yticklabels(['0', '50', '100'])
ax.set_ylabel('probability')
ax.legend(handlelength=1, frameon=False)

fig.tight_layout(pad=0.5)
plt.show()
# fig.savefig('figures/monoc_diff_cdf.png')
fig.savefig(f'figures/monoc_diff_cdf_{s.mouse}.eps')
