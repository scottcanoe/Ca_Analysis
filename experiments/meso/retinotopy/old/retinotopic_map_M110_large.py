import matplotlib as mpl
from skimage.measure import find_contours

mpl.use('QtAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import skimage as ski

from ca_analysis.plot import *
from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *

plt.rcParams['font.size'] = 8

ylim = slice(None, None)
xlim = slice(None, None)
figsize = (1.9, 1.5)
qlim = (2.5, 97.5)
cmap = "inferno"
kernel = (1, 1, 1)


s = open_session('M110', '2023-04-21', '1', fs=0)

# load trial-averaged movie, restricted to bottom right region.
with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'r') as f:
    mov = f['ach'][:]
mov = mov[:, ylim, xlim]

# Omit first part of the movie where nothing happens.
mov = mov[45:]

# smooth movie
mov = gaussian_filter(mov, kernel)

# assign the best matching basis function to each pixel ('mat'), and separately
# store a value indicating how good the match is ('A').
mat = np.zeros_like(mov[0])
A = np.zeros_like(mov[0])
for y in range(mov.shape[1]):
    for x in range(mov.shape[2]):
        ts = mov[:, y, x]
        i_best = np.argmax(ts)
        A[y, x] = ts[i_best]
        mat[y, x] = i_best

np.save('../mat.npy', mat)
np.save('../A.npy', A)


with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
    orig_frame_shape = f['ach'].shape[1:]

mat = np.load('../mat.npy')
mat = gaussian_filter(mat, sigma=(2, 2))
mat = mat / mat.max()

A = np.load('../A.npy')
A = gaussian_filter(A, sigma=(2, 2))
A[A < 0] = 0
p = np.percentile(A, 90)
A = A / p

print('adding cortex mask')
bitmask = np.zeros(orig_frame_shape, dtype=bool)
with h5py.File(s.fs.getsyspath('rois.h5'), 'r') as f:
    x = f['LH']['x'][:]
    y = f['LH']['y'][:]
    bitmask[y, x] = True
    x = f['RH']['x'][:]
    y = f['RH']['y'][:]
    bitmask[y, x] = True
bitmask = bitmask[ylim, xlim]
YX = np.argwhere(~bitmask)
Y, X = YX[:, 0], YX[:, 1]

# manually create the RGBA image because we want to control alpha values
cmap = get_cmap('inferno', data=mat, vlim=[0, 1])
image = cmap(mat)
image[:, :, 3] = A
image[Y, X, :] = 1


image = ski.color.rgba2rgb(image)

# plot image and colorbar
figsize = (2, 1.5)
fig, ax = plt.subplots(figsize=figsize)
im = ax.imshow(image)
divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.inferno, orientation='vertical')
fig.add_axes(ax_cb)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.set_ticklabels(
    [
        r'100$^\circ$',
        r'75$^\circ$',
        r'50$^\circ$',
        r'25$^\circ$',
        r'0$^\circ$',
    ]
)
remove_ticks(ax)

fig.tight_layout(pad=0.15)
fig.savefig('figures/retinotopic_map.png')
fig.savefig('figures/retinotopic_map.eps')
# plt.show()


"""
--------------------------------------------------------------------------------
Retinotopic map with ROIs drawn
"""

# Get contour for left monocular area
with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
    orig_frame_shape = f['ach'].shape[1:]

bitmask = np.zeros(orig_frame_shape, dtype=bool)
with h5py.File(s.fs.getsyspath('rois.h5'), 'r') as f:
    lvm_x = f['LVM']['x'][:]
    lvm_y = f['LVM']['y'][:]
    bitmask[lvm_y, lvm_x] = True
lvm_ct = find_contours(bitmask)[0].astype(int)
lvm_Y, lvm_X = lvm_ct[:, 1], lvm_ct[:, 0]

# Get contour for left and right hemispheres
l_bitmask = np.zeros(orig_frame_shape, dtype=bool)
r_bitmask = np.zeros(orig_frame_shape, dtype=bool)
with h5py.File(s.fs.getsyspath('rois.h5'), 'r') as f:
    x = f['LH']['x'][:]
    y = f['LH']['y'][:]
    l_bitmask[y, x] = True
    x = f['RH']['x'][:]
    y = f['RH']['y'][:]
    r_bitmask[y, x] = True

l_ct = find_contours(l_bitmask)[0].astype(int)
l_Y, l_X = l_ct[:, 1], l_ct[:, 0]

r_ct = find_contours(r_bitmask)[0].astype(int)
r_Y, r_X = r_ct[:, 1], r_ct[:, 0]

# plot image and colorbar
figsize = (2, 1.5)
fig, ax = plt.subplots(figsize=figsize)
im = ax.imshow(image)
divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.inferno, orientation='vertical')
fig.add_axes(ax_cb)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.set_ticklabels(
    [
        r'100$^\circ$',
        r'75$^\circ$',
        r'50$^\circ$',
        r'25$^\circ$',
        r'0$^\circ$',
    ]
)
remove_ticks(ax)

# Create a Rectangle patch
locs = [
    (120 + 200, 105 + 200),
    (140 + 200, 100 + 200),
    (160 + 200, 95 + 200),
]
width, height = 5, 5
for loc in locs:
    rect = patches.Rectangle(loc, width, height, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

ax.plot(lvm_Y, lvm_X, color='blue')
ax.plot(l_Y, l_X, color='blue')
ax.plot(r_Y, r_X, color='blue')

fig.tight_layout(pad=0.15)
fig.savefig('figures/retinotopic_map_with_patches.png')
fig.savefig('figures/retinotopic_map_with_patches.eps')
# plt.show()

"""
--------------------------------------------------------------------------------
Traces
"""

# load trial-averaged movie, restricted to bottom right region.
with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'r') as f:
    mov = f['ach'][:]
mov = mov[:, ylim, xlim]
mov = np.concatenate([mov, mov])

# Omit first part of the movie where nothing happens.
# mov = mov[45:]

# smooth movie
mov = gaussian_filter(mov, kernel)

fig, ax = plt.subplots(figsize=(2, 1))
TS = []
colors = [
    (0.387, 0, 0.361),
    (0.8, 0, 0),
    (0.98, 0.99, 0.48),
]
from experiments.meso.roi_processing import ROI
LVM = ROI(s, 'LVM', 'ach')
splits = LVM.split('right')
sp = splits.mean('trial')
mmin, mmax = np.min(sp).item(), np.max(sp).item()
scale = mmax - mmin


for i, loc in enumerate(locs):
    x, y = loc
    ts = mov[:, slice(y, y + height), slice(x, x + width)] # all that 2d ROIs
    ts = np.array([np.mean(patch) for patch in ts])        # average them
    ts = ts / scale
    TS.append(ts)

for i, ts in enumerate(TS):
    ts = ts[13:123]
    ax.plot(ts, color=colors[i])
ax.set_facecolor((0.6, 0.6, 0.6))
ax.set_xlim([0, len(ts)])
ax.set_xticks([0, 27.5, 55, 82.5, 110])
ax.set_xticklabels([
    r'100$^\circ$',
    r'75$^\circ$',
    r'50$^\circ$',
    r'25$^\circ$',
    r'0$^\circ$',
]
)
# ax.set_yticks([])
fig.tight_layout(pad=0.15)
fig.savefig('figures/retinotopic_map_traces.png')
fig.savefig('figures/retinotopic_map_traces.eps')
# plt.show()
