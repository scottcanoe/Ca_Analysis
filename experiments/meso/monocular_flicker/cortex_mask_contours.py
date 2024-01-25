# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import find_contours

from ca_analysis.plot import *
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

def load_trials(s: Session) -> Tuple[xr.DataArray]:
    with h5py.File(s.fs.getsyspath('trials.h5'), 'r') as f:
        group = f['ach']
        left = xr.DataArray(group['left'][:], dims=('trial', 'time', 'y', 'x'))
        right = xr.DataArray(group['right'][:], dims=('trial', 'time', 'y', 'x'))
    evs = [s.events.schema.get(event=0)] * left.sizes['time']
    evs[10:20] = [s.events.schema.get(event=1)] * 10
    left.coords['event'] = xr.DataArray(evs, dims=('time',))
    evs = [s.events.schema.get(event=0)] * left.sizes['time']
    evs[10:20] = [s.events.schema.get(event=2)] * 10
    right.coords['event'] = xr.DataArray(evs, dims=('time',))

    return left, right

plt.rcParams['font.size'] = 8

figsize = (2.1, 1.5)
qlim = (1, 99)
cmap = "hot"
kernel = (0.5, 2, 2)
colorbar = True

s = open_session('M110', '2023-04-21', '2', fs=0)
_, right = load_trials(s)


# subtract baseline from images
print('subtracting baseline (right)')
scale = 0.04393532
baselines = []
for i in range(right.sizes['trial']):
    front = right.isel(trial=i, time=slice(0, 10))
    img = front.mean('time')
    baselines.append(img)
for i in range(right.sizes['trial']):
    for j in range(right.sizes['time']):
        right.data[i, j] = (right.data[i, j] - baselines[i]) / scale

right = right.mean('trial')

# right.data = right.da
right.data = gaussian_filter(right.data, kernel)

# add cortex mask
print('adding cortex mask')
bitmask = np.zeros([right.sizes['y'], right.sizes['x']], dtype=bool)
with h5py.File(s.fs.getsyspath('rois.h5'), 'r') as f:
    lx = f['LH']['x'][:]
    ly = f['LH']['y'][:]
    bitmask[ly, lx] = True
    rx = f['RH']['x'][:]
    ry = f['RH']['y'][:]
    bitmask[ry, rx] = True
YX = np.argwhere(~bitmask)
Y, X = YX[:, 0], YX[:, 1]

l_bitmask = np.zeros([right.sizes['y'], right.sizes['x']], dtype=bool)
l_bitmask[ly, lx] = True
l_bitmask = ~l_bitmask
l_ct = find_contours(l_bitmask)[0]
l_ct = l_ct.astype(int)
l_Y, l_X = l_ct[:, 1], l_ct[:, 0]
l_Y -= 10
l_X -= 45

r_bitmask = np.zeros([right.sizes['y'], right.sizes['x']], dtype=bool)
r_bitmask[ry, rx] = True
r_bitmask = ~r_bitmask
r_ct = find_contours(r_bitmask)[0]
r_ct = r_ct.astype(int)
r_Y, r_X = r_ct[:, 1], r_ct[:, 0]
r_Y -= 10
r_X -= 45

YLIM = slice(45, None)
XLIM = slice(10, -10)
fm = right.isel(time=17)
# fm.data[Y, X] = np.nan
fm = fm.isel(y=YLIM, x=XLIM)

plt.rcParams['axes.facecolor'] = 'white'
fig, ax = plt.subplots(figsize=figsize)
vmin, vmax = np.nanpercentile(fm, qlim)
vmin, vmax = -0.05, 0.05
im = ax.imshow(fm, cmap=cmap, vmin=vmin, vmax=vmax)
remove_ticks(ax)

ax.plot(l_Y, l_X, color='blue')
ax.plot(r_Y, r_X, color='blue')

if colorbar:
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    cbar = plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)
    # ax_cb.set_xticks([])
    # ax_cb.set_xticklabels([])
    # cbar.set_ticks([])

fig.tight_layout(pad=0.15)
plt.show()

fig.savefig('figures/contours.png')
fig.savefig('figures/contours.eps')
