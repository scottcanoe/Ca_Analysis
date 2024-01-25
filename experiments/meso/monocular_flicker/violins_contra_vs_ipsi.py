# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *


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


s = open_session('M110', '2023-04-21', '2', fs=0)
# create_masks(s)
LH = ROI(s, 'LH', 'ach')
RH = ROI(s, 'RH', 'ach')

LH_left = LH.split('left')
LH_right = LH.split('right')
RH_left = RH.split('left')
RH_right = RH.split('right')

# LH_left = LH.split('left').isel(time=slice(0, 3))
# LH_right = LH.split('right').isel(time=slice(0, 3))
# RH_left = RH.split('left').isel(time=slice(0, 3))
# RH_right = RH.split('right').isel(time=slice(0, 3))

contra = xr.concat([LH_right, RH_left], 'trial')
ipsi = xr.concat([LH_left, RH_right], 'trial')

fig, ax = plt.subplots(1, 1, figsize=(2, 2))
positions = [1, 1.8]
X, Y = contra.data.flatten(), ipsi.data.flatten()
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
ax.set_xticklabels(['contra', 'ipsi'], rotation=60)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig.tight_layout(pad=0.5)
plt.show()
fig.savefig('figures/violins_contra_vs_ipsi.png')
fig.savefig('figures/violins_contra_vs_ipsi.eps')

r = ks_2samp(X, Y)
print(r)