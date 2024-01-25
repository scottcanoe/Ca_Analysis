import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal

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

X = np.arange(len(LH.data)) / 10


fig, axes = plt.subplots(2, 1)
axes[0].plot(X, LH.data, color='red', label='LH')
axes[0].plot(X, RH.data, color='black', label='RH')
axes[0].legend()

cutoff = 0.1
order = 2

a = butter_highpass_filter(LH.data, cutoff, 10.0, order=order)
b = butter_highpass_filter(RH.data, cutoff, 10.0, order=order)

axes[1].plot(X, a, color='red', label='LH')
axes[1].plot(X, b, color='black', label='RH')
axes[1].legend()

plt.show()