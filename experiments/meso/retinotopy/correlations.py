import matplotlib
from skimage.measure import find_contours

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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

s = open_session('M150', '2023-03-30', '1', fs=0)


# s = open_session('M150', '2023-04-21', '1', fs=0)
# with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
#     SHAPE = f['ach'].shape

# fname = 'correlations.npy'
fname = 'correlations_trial_averaged.npy'

# with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
#     ca = xr.DataArray(f['ca'][:], dims=('time', 'y', 'x'))
#     ach = xr.DataArray(f['ach'][:], dims=('time', 'y', 'x'))
#

with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'r') as f:
    ca = xr.DataArray(f['ca'][:], dims=('time', 'y', 'x'))
    ach = xr.DataArray(f['ach'][:], dims=('time', 'y', 'x'))
ca = gaussian_filter(ca, [1, 2, 2])
ach = gaussian_filter(ach, [1, 2, 2])

mat = np.zeros([ca.sizes['y'], ca.sizes['x']])
P = np.zeros([ca.sizes['y'], ca.sizes['x']])
for y in range(ca.sizes['y']):
    print(f'y: {y}')
    for x in range(ca.sizes['x']):
        ca_ts = ca.data[:, y, x]
        ach_ts = ach.data[:, y, x]
        # val = np.corrcoef(ca_ts, ach_ts)[0, 1]
        stat = pearsonr(ca_ts, ach_ts)
        mat[y, x] = stat.statistic
        P[y, x] = stat.pvalue
np.save(fname, mat)


mat = np.load(fname)

# - non-cortex and LH/RH bitmasks, pixels, contours
bitmask = np.zeros(mat.shape, dtype=bool)
l_bitmask = np.zeros(mat.shape, dtype=bool)
r_bitmask = np.zeros(mat.shape, dtype=bool)
with h5py.File(s.fs.getsyspath('rois.h5'), 'r') as f:
    x, y = f['LH']['x'][:], f['LH']['y'][:]
    bitmask[y, x] = True
    l_bitmask[y, x] = True
    x, y = f['RH']['x'][:], f['RH']['y'][:]
    bitmask[y, x] = True
    r_bitmask[y, x] = True
YX = np.argwhere(~bitmask)
Y, X = YX[:, 0], YX[:, 1]
l_ct = find_contours(l_bitmask)[0].astype(int)
l_Y, l_X = l_ct[:, 1], l_ct[:, 0]
r_ct = find_contours(r_bitmask)[0].astype(int)
r_Y, r_X = r_ct[:, 1], r_ct[:, 0]

mat = np.load('correlations.npy')
mat[Y, X] = np.nan
YX = np.argwhere(P > 0.05)
Y, X = YX[:, 0], YX[:, 1]
mat[Y, X] = np.nan
fig, ax = plt.subplots(figsize=(2.5, 2.5))

vmin, vmax = np.nanpercentile(mat, [2.5, 97.5])
vmin, vmax = 0.5, 0.9
# vmin, vmax = np.nanmin(mat), np.nanmax(mat)
im = ax.imshow(mat, cmap='inferno', vmin=vmin, vmax=vmax)
plt.colorbar(im)
plt.show()

# LH = ROI(s, 'LH', 'ach')
# RH = ROI(s, 'RH', 'ach')
#
# LH_left = LH.split('left', lpad=lpad, rpad=rpad, concat='time')
# LH_right = LH.split('right', lpad=lpad, rpad=rpad, concat='time')
# RH_left = RH.split('left', lpad=lpad, rpad=rpad, concat='time')
# RH_right = RH.split('right', lpad=lpad, rpad=rpad, concat='time')
#
# left = load_splits(s, 'ach', 'left', lpad=False, rpad=False)
# right = load_splits(s, 'ach', 'right', lpad=False, rpad=False)
#
# left = left.mean('trial')
# right = right.mean('trial')
# v = napari.view_image(right)
