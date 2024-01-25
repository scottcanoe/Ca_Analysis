from types import SimpleNamespace
from typing import Sequence

import matplotlib as mpl
# mpl.use('TkAgg')
from matplotlib.animation import FFMpegWriter
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.decomposition import PCA

from ca_analysis import *
from seq_learn_3.main import *
from seq_learn_3.utils import *



# sequence = 'ABCD'
day = 5

roi_filter = 'all'
lpad: int = None
rpad: int = None

#-------------------------------------------------------------------------------


sessions = get_sessions(day=day, fs=0)
apply_roi_filter(sessions, roi_filter)


# Extract trial-averaged spiking data for each sequence, and put them
# end-to-end.
arrays = []
for seq_name in ('ABCD', 'ABBD', 'ACBD'):
    # spec = schema.get_sequence(seq_name).events[:-1]
    spec = schema.get_sequence(seq_name).events[:]
    chunks = sessions.split('spikes', spec, lpad=lpad, rpad=rpad)
    for i, ch in enumerate(chunks):
        chunks[i] = ch.mean('trial')
    arr = xr.concat(chunks, 'roi')
    arrays.append(arr)
combined = xr.concat(arrays, 'time')

# Do PCA
pca = PCA()
tformed = xr.DataArray(
    pca.fit_transform(combined),
    dims=('time', 'component'),
    coords=dict(event=combined.coords['event']),
)
tformed = tformed.isel(component=slice(0, 3))

for i in range(tformed.sizes['component']):
    arr = tformed.isel(component=i)
    if abs(arr.min()) > abs(arr.max()):
        tformed[:, i] = -1 * arr
if day == 0:
    tformed[:, 0] = -1 * tformed.isel(component=0)

# Split transformed data into ABCD/ABBD/ACBD chunks
edges = np.r_[0, np.cumsum([arr.sizes['time'] for arr in arrays])]
ABCD_T = tformed.isel(time=slice(edges[0], edges[1]))
ABBD_T = tformed.isel(time=slice(edges[1], edges[2]))
ACBD_T = tformed.isel(time=slice(edges[2], edges[3]))


#-------------------------------------------------------------------------------

for sequence in ('ABCD', 'ABBD', 'ACBD'):

    # fig = Figure(figsize=(6, 6))
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.view_init(48, 68)

    if sequence == 'ABCD':
        arr = ABCD_T
    elif sequence == 'ABBD':
        arr = ABBD_T
    elif sequence == 'ACBD':
        arr = ACBD_T
    else:
        raise ValueError('invalid sequence')

    X = arr[:, 0].data
    Y = arr[:, 1].data
    Z = arr[:, 2].data

    n_timepoints = len(X)
    resample_factor = 8
    X = resample(X, resample_factor * n_timepoints)
    Y = resample(Y, resample_factor * n_timepoints)
    Z = resample(Z, resample_factor * n_timepoints)
    change_points = np.array([0, 8, 16, 24, 32]) * resample_factor

    ax.set_xlim([X.min() - 1, X.max() + 1])
    ax.set_ylim([Y.min() - 1, Y.max() + 1])
    ax.set_zlim([Z.min() - 1, Z.max() + 1])

    n_points = len(X)
    values = np.arange(n_points)
    ax.scatter(X, Y, Z, c=values, marker='o', cmap='inferno', s=5)

    X = X[change_points]
    Y = Y[change_points]
    Z = Z[change_points]
    ax.scatter(X, Y, Z, color='red', marker='+', s=500/2)

    path = f'figures/pca3d/{sequence}_{day}.eps'
    # plt.tight_layout(pad=0.2)
    plt.show()
    fig.savefig(path)


# sequence = 'ABCD'
#
# # fig = Figure(figsize=(6, 6))
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.set_xlabel('PC 1')
# ax.set_ylabel('PC 2')
# ax.set_zlabel('PC 3')
# ax.view_init(48, 68)
#
# if sequence == 'ABCD':
#     arr = ABCD_T
# elif sequence == 'ABBD':
#     arr = ABBD_T
# elif sequence == 'ACBD':
#     arr = ACBD_T
# else:
#     raise ValueError('invalid sequence')
#
# X = arr[:, 0].data
# Y = arr[:, 1].data
# Z = arr[:, 2].data
#
# n_timepoints = len(X)
# resample_factor = 8
# X = resample(X, resample_factor * n_timepoints)
# Y = resample(Y, resample_factor * n_timepoints)
# Z = resample(Z, resample_factor * n_timepoints)
# change_points = np.array([0, 8, 16, 24, 32]) * resample_factor
#
# ax.set_xlim([X.min() - 1, X.max() + 1])
# ax.set_ylim([Y.min() - 1, Y.max() + 1])
# ax.set_zlim([Z.min() - 1, Z.max() + 1])
#
# n_points = len(X)
# values = np.arange(n_points)
# ax.scatter(X, Y, Z, c=values, marker='o', cmap='inferno')
#
# X = X[change_points]
# Y = Y[change_points]
# Z = Z[change_points]
# ax.scatter(X, Y, Z, color='red', marker='+', s=500)
#
# path = f'figures/pca3d/{sequence}_{day}.png'
# # fig.savefig(path)
# plt.show()
