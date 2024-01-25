from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

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

# load trial-averaged movie, restricted to bottom right region.
with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'r') as f:
    mov = f['ach'][:]
mov = mov[:, ylim, xlim]
mov = mov[:, 40:, 50:]

# Omit first part of the movie where nothing happens.
mov = mov[45:]

# smooth movie
mov = gaussian_filter(mov, kernel)
im1 = mov[5]
im2 = mov[20]
im3 = mov[45]
im4 = mov[60]

# manually create the RGBA image because we want to control alpha values
# cmap = get_cmap('inferno', data=mov, vlim=[0, 1])

vmin, vmax = np.percentile(mov, [2.5, 97.5])
aspect = mov.shape[1] / mov.shape[2]
width = 2.2
height = width * aspect

# plot image and colorbar
figsize = (width, height)
fig, ax = plt.subplots(figsize=figsize)
grid = ImageGrid(
    fig, 111,  # similar to subplot(111)
    nrows_ncols=(2, 2),  # creates 2x2 grid of axes
    axes_pad=0.0,  # pad between axes in inch.
    share_all=True,
    label_mode="L",
)
grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])
for ax, im in zip(grid, [im1, im2, im3, im4]):
    # Iterating over the grid returns the Axes.
    ax.imshow(im, cmap='inferno', vmin=vmin, vmax=vmax)
    remove_ticks(ax)
    ax.axis('off')
# remove_ticks(ax)
#
fig.tight_layout(pad=0.0)
fig.savefig('figures/retinotopy_tiles.png', dpi=300)
fig.savefig('figures/retinotopy_tiles.eps', dpi=300)
plt.show()

# indiv frames
for i in range(mov.shape[0]):
    frame = mov[i]
    width = 2
    height = width * aspect
    # width = 2
    # height = 1.275
    figsize = (width, height)
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(frame, cmap='inferno', vmin=vmin, vmax=vmax, aspect='equal')
    remove_ticks(ax)
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.0)
    fig.savefig(f'figures/retinotopy_tiles/{i}.eps', dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()
