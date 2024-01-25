import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal

from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *
from experiments.meso.roi_processing import ROI


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y



lpad = 10
rpad = 5
cutoff = 0.1
order = 2
xlim = slice(1800, None)
s = open_session('M110', '2023-04-21', '2', fs=0)
# create_masks(s)
LV = ROI(s, 'LV', 'ach')

fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))

Y1 = LV.data.data
Y2 = butter_highpass_filter(LV.data, cutoff, 10.0, order=order)
X = np.arange(len(Y1)) / 10

X = X[xlim]
Y1 = Y1[xlim]
Y2 = Y2[xlim]
X -= X[0]
ax.plot(X, Y1, color='black', label='dF/F', lw=1)
ax.plot(X, Y2, color='red', label='dF/F filt', lw=1)
ax.legend(loc='upper right', framealpha=0)
ax.set_xlabel('time (sec)')
ax.set_ylabel('dF/F')
ax.set_xlim([X[0], X[-1]])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout(pad=1)
plt.show()

fig.savefig('figures/supplement_high_pass.png')
fig.savefig('figures/supplement_high_pass.eps')


fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
scale = Y2.max() - Y2.min()
Y3 = Y2 / scale
ax.plot(X, Y3, color='black', label='dF/F', lw=1)
ax.set_xlabel('time (sec)')
ax.set_ylabel('% change')
ax.set_xlim([X[0], X[-1]])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout(pad=1)
plt.show()
fig.savefig('figures/supplement_high_pass_2.eps')
