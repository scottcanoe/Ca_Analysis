import matplotlib as mpl
from skimage.measure import find_contours
mpl.use('QtAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ca_analysis.plot import *
from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *

plt.rcParams['font.size'] = 8

ylim = slice(200, None)
xlim = slice(200, None)
figsize = (1.9, 1.5)
qlim = (2.5, 97.5)
cmap = "inferno"
kernel = (1, 1, 1)


s = open_session('M110', '2023-04-21', '1', fs=0)

# # load trial-averaged movie, restricted to bottom right region.
# with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'r') as f:
#     mov = f['ach'][:]
# # mov = mov[:, ylim, xlim]
# #
# # Omit first part of the movie where nothing happens.
# mov = mov[45:]
#
# # smooth movie
# mov = gaussian_filter(mov, kernel)
#
# # assign the best matching basis function to each pixel ('mat'), and separately
# # store a value indicating how good the match is ('A').
# mat = np.zeros_like(mov[0])
# A = np.zeros_like(mov[0])
# for y in range(mov.shape[1]):
#     # print(f'{100 * y / mov.shape[1]} % done')
#     for x in range(mov.shape[2]):
#         ts = mov[:, y, x]
#         # covs = [np.cov(ts, b)[0, 1] for b in bases]
#         # i_best = np.argmax(covs)
#         i_best = np.argmax(ts)
#         A[y, x] = ts[i_best]
#         mat[y, x] = i_best
#         # A[y, x] = covs[i_best]
# np.save('mat.npy', mat)
# np.save('A.npy', A)

#---
# add cortex mask

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
# create whole cortex bitmask
bitmask = np.zeros(orig_frame_shape, dtype=bool)
with h5py.File(s.fs.getsyspath('rois.h5'), 'r') as f:
    lx = f['LH']['x'][:]
    ly = f['LH']['y'][:]
    bitmask[ly, lx] = True
    rx = f['RH']['x'][:]
    ry = f['RH']['y'][:]
    bitmask[ry, rx] = True
YX = np.argwhere(~bitmask)
Y, X = YX[:, 0], YX[:, 1]
# bitmask = bitmask[ylim, xlim]
YX = np.argwhere(~bitmask)
Y, X = YX[:, 0], YX[:, 1]

# find contours for left and right hemisphere ROIs
# ct = find_contours(bitmask)[0].astype(int)
# ct_Y, ct_X = ct[:, 1], ct[:, 0]
l_bitmask = np.zeros(orig_frame_shape, dtype=bool)
l_bitmask[ly, lx] = True
l_bitmask = ~l_bitmask
l_ct = find_contours(l_bitmask)[0]
l_ct = l_ct.astype(int)
l_Y, l_X = l_ct[:, 1], l_ct[:, 0]
# l_Y -= 10
# l_X -= 45

r_bitmask = np.zeros(orig_frame_shape, dtype=bool)
r_bitmask[ry, rx] = True
r_bitmask = ~r_bitmask
r_ct = find_contours(r_bitmask)[0]
r_ct = r_ct.astype(int)
r_Y, r_X = r_ct[:, 1], r_ct[:, 0]
# r_Y -= 10
# r_X -= 45

# manually create the RGBA image because we want to control alpha values
cmap = get_cmap('inferno', data=mat, vlim=[0, 1])
image = cmap(mat)
image[:, :, 3] = A
# image[Y, X, :] = 1
import skimage as ski

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
ax.plot(l_Y, l_X, color='blue')
ax.plot(r_Y, r_X, color='blue')

fig.tight_layout(pad=0.15)
fig.savefig('figures/retinotopic_map.png')
fig.savefig('figures/retinotopic_map.eps')
# plt.show()


#-----------------

import matplotlib.patches as patches
# Plot timeseries

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
    (120, 105),
    (140, 100),
    (160, 95),
]
width, height = 5, 5
for loc in locs:
    rect = patches.Rectangle(loc, width, height, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

# rect = patches.Rectangle((1, 10), width, height, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)

fig.tight_layout(pad=0.15)
fig.savefig('figures/retinotopic_map_with_patches.png')
fig.savefig('figures/retinotopic_map_with_patches.eps')
# plt.show()

