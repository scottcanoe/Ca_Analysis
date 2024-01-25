import matplotlib as mpl
mpl.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage as ski
from skimage.measure import find_contours

from ca_analysis.plot import *
from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *
from experiments.meso.roi_processing import ROI


# parameters
mouse = 'M153'
channel = 'ach'
degrees = 80  # should be 120, 100, or 80
cmap = "inferno"
kernel = (1, 1, 1)
roi_width, roi_height = 5, 5

#-----

mouse_to_session = {
    'M110': open_session('M110', '2023-04-21', '1', fs=0),
    'M115': open_session('M115', '2023-04-21', '1', fs=0),
    'M150': open_session('M150', '2023-03-30', '1', fs=0),
    'M152': open_session('M152', '2023-03-30', '1', fs=0),
    'M153': open_session('M153', '2023-03-30', '1', fs=0),
}
# x, y pairs
mouse_to_roi_locs = {
    'M110': [
        (315, 310),
        (335, 305),
        (355, 300),
    ],
    'M115': [
        (315, 310),
        (335, 305),
        (355, 300),
    ],
    'M150': [
        (302, 274),
        (317, 262),
        (341, 248),
    ],
    'M152': [
        (310, 277),
        (325, 270),
        (341, 263),
    ],
    'M153': [
        (310, 280),
        (326, 274),
        (341, 268),
    ]
}
degrees_to_ticklabels = {
    120: [
            r'120$^\circ$',
            r'90$^\circ$',
            r'60$^\circ$',
            r'30$^\circ$',
            r'0$^\circ$',
        ],
    100: [
            r'100$^\circ$',
            r'75$^\circ$',
            r'50$^\circ$',
            r'25$^\circ$',
            r'0$^\circ$',
        ],
    80: [
        r'80$^\circ$',
        r'60$^\circ$',
        r'40$^\circ$',
        r'20$^\circ$',
        r'0$^\circ$',
    ],
    60: [
        r'60$^\circ$',
        r'45$^\circ$',
        r'30$^\circ$',
        r'15$^\circ$',
        r'0$^\circ$',
    ],
}
s = mouse_to_session[mouse]
roi_locs = mouse_to_roi_locs[mouse]
ticklabels = degrees_to_ticklabels[degrees]

"""
--------------------------------------------------------------------------------
Prepare data
"""

# Load trial-averaged movie, and apply smoothing.
with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'r') as f:
    mov = f[channel][:]
mov = gaussian_filter(mov, kernel)
frame_shape = mov.shape[1:]

# Omit first 20 degrees of the movie where nothing happens.
to_drop = int(110 * (1 - degrees / 120))
mov = mov[to_drop:]

# Compute retinotopic map
#------------------------

# Assign the best matching basis function to each pixel ('mat'), and separately
# store a value indicating how good the match is ('A').
indices = np.zeros_like(mov[0])
amplitudes = np.zeros_like(mov[0])
for y in range(mov.shape[1]):
    for x in range(mov.shape[2]):
        ts = mov[:, y, x]
        i_best = np.argmax(ts)
        indices[y, x] = i_best
        amplitudes[y, x] = ts[i_best]

np.save('indices.npy', indices)
np.save('amplitudes.npy', amplitudes)

# Get bitmasks and contours for hemispheres and LVM
#--------------------------------------------------

# - non-cortex and LH/RH bitmasks, pixels, contours
bitmask = np.zeros(frame_shape, dtype=bool)
l_bitmask = np.zeros(frame_shape, dtype=bool)
r_bitmask = np.zeros(frame_shape, dtype=bool)
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

# - LVM bitmask, pixels, contour
lvm_bitmask = np.zeros(frame_shape, dtype=bool)
with h5py.File(s.fs.getsyspath('rois.h5'), 'r') as f:
    x = f['LVM']['x'][:]
    y = f['LVM']['y'][:]
    lvm_bitmask[y, x] = True
lvm_ct = find_contours(lvm_bitmask)[0].astype(int)
lvm_Y, lvm_X = lvm_ct[:, 1], lvm_ct[:, 0]

# Get scaling factor from LVM
#----------------------------
LVM = ROI(s, 'LVM', channel)
lvm_arr = LVM.split('right').mean('trial')
lvm_min, lvm_max = np.min(lvm_arr).item(), np.max(lvm_arr).item()
lvm_scale = lvm_max - lvm_min

# Extract manual ROIs and rescale traces
#---------------------------------------
ROI_traces = []
for loc in roi_locs:
    x, y = loc
    ts = mov[:, slice(y, y + roi_height), slice(x, x + roi_width)] # all that 2d ROIs
    ts = np.array([np.mean(patch) for patch in ts])        # average them
    ts = ts / lvm_scale
    ROI_traces.append(ts)

"""
Plot retinotopic map with patches and contours.
"""

indices = np.load('indices.npy')
indices = gaussian_filter(indices, sigma=(1, 1))
indices = indices / mov.shape[0]

amplitudes = np.load('amplitudes.npy')
amplitudes = gaussian_filter(amplitudes, sigma=(1, 1))
amplitudes[amplitudes < 0] = 0
amplitudes_2 = amplitudes.copy()
amplitudes_2[Y, X] = np.nan
p = np.nanpercentile(amplitudes_2, 97.5)
# p = np.percentile(amplitudes, 95)
amplitudes = amplitudes / p

# - Create the RGBA image because we want to control alpha values
smap = get_smap(cmap, data=indices, vlim=[0, 1])
image = smap(indices)
image[:, :, 3] = amplitudes        # apply alpha from amplitudes
image[Y, X, :] = 1                 # make non-cortex areas with white
image = ski.color.rgba2rgb(image)  # do this so no alpha in output

# - draw retinotopic map
plt.rcParams['font.size'] = 8
fig, ax = plt.subplots(figsize=(2, 1.5))
im = ax.imshow(image)

# - draw manual ROIs
for loc in roi_locs:
    rect = patches.Rectangle(loc, roi_width, roi_height, lw=0.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

# - draw cortices and LVM
ax.plot(l_Y, l_X, color='blue')
ax.plot(r_Y, r_X, color='blue')
ax.plot(lvm_Y, lvm_X, color='blue')

# - add colorbar
divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
fig.add_axes(ax_cb)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.set_ticklabels(ticklabels)
remove_ticks(ax)
fig.tight_layout(pad=0.15)
# fig.savefig(f'figures/retinotopic_map_{s.mouse}_{channel}.png')
fig.savefig(f'figures/retinotopic_map_{s.mouse}_{channel}.eps')
# plt.show()


"""
Plot time series
"""
fig, ax = plt.subplots(figsize=(2, 1))
colors = [
    (0.387, 0, 0.361),
    (0.8, 0, 0),
    (0.98, 0.99, 0.48),
]
colors = [
    smap([0.3]),
    smap([0.6]),
    smap([0.9])
]
for i, ts in enumerate(ROI_traces):
    ax.plot(ts, color=colors[i])
ax.set_facecolor((0.6, 0.6, 0.6))
ax.set_xlim([0, len(ts)])
ax.set_xticks(np.linspace(0, mov.shape[0], 5))
ax.set_xticklabels(ticklabels)

fig.tight_layout(pad=0.15)
# fig.savefig(f'figures/retinotopic_map_traces_{s.mouse}_{channel}.png')
fig.savefig(f'figures/retinotopic_map_traces_{s.mouse}_{channel}.eps')
# plt.show()
