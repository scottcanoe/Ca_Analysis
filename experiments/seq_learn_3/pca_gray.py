from types import SimpleNamespace
from typing import Sequence

import matplotlib
from cycler import cycler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from processing import *
from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *

from utils import *


n_components: int = 4
lpad: int = None
rpad: int = None
show: bool = True
roi_filter = 'all'

# save = Path.home() / 'plots/seq_learn_3/plots/pca_gray.png'

#-------------------------------------------------------------------------------

sessions_0 = get_sessions(day=0, fs=0)
sessions_5 = get_sessions(day=5, fs=0)

apply_roi_filters(sessions_0, roi_filter)
apply_roi_filters(sessions_5, roi_filter)

#-------------------------------------------------------------------------------
gray_events = ['ABCD._', 'ABBD._', 'ACBD._']
gray_events = ['ABCD._']
# Extract trial-averaged spiking data.
arrays = []
for event in gray_events:
    a = flex_split(sessions_0, event)
    a = a.isel(trial=slice(0, 10))
    arrays.append(a)
data_0 = xr.concat(arrays, 'trial').mean('trial')
pca_0 = PCA()
tformed_0 = xr.DataArray(
    pca_0.fit_transform(data_0),
    dims=('time', 'component'),
)

# Extract trial-averaged spiking data.
arrays = []
for event in gray_events:
    arrays.append(flex_split(sessions_5, event))
data_5 = xr.concat(arrays, 'trial').mean('trial')
pca_5 = PCA()
tformed_5 = xr.DataArray(
    pca_5.fit_transform(data_5),
    dims=('time', 'component'),
)

# ------------------------------------------------------------------------------


custom_cycler = (cycler(color=['black', '#ff7f00', 'blue',
                               '#f781bf', '#a65628', '#984ea3',
                               '#999999', '#e41a1c', '#dede00']
                        ) +
                 cycler(lw=[2] * 9))

# Plot each
fig, axes = plt.subplots(3, 1, figsize=(6, 6))

def fill_axes(ax, data, name):
    ax.set_prop_cycle(custom_cycler)
    for i in range(n_components):
        arr = data.isel(component=i)
        # if abs(arr.min()) > abs(arr.max()):
        #     arr = -1 * arr
        ax.plot(arr, label=str(i))
    ax.legend(loc='upper right')
    ax.set_xticks([0, 8, 16, 24])
    ax.set_xticklabels(['0', '250', '500', '750'])
    ax.set_xlim([0, len(data)])
    ax.set_title(name)

fill_axes(axes[0], tformed_0, 'day 0')
fill_axes(axes[1], tformed_5, 'day 5')
fig.tight_layout(pad=1)
if show:
    plt.show()
# if save:
#     fig.savefig(save)

# fig, axes = plt.subplots(2, 1)

X = np.arange(n_components)

ax = axes[2]
Y = pca_0.explained_variance_ratio_[:n_components]
ax.bar(
    X - 0.2,
    Y,
    color='blue',
    label='day 0, {:.3f}'.format(Y.sum()),
    width=0.35,
)
Y = pca_5.explained_variance_ratio_[:n_components]
ax.bar(
    X + 0.2,
    Y,
    color='red',
    label='day 5, {:.3f}'.format(Y.sum()),
    width=0.35,
)
ax.legend()
ax.set_title('explained variance')
ax.set_ylabel('fraction')
ax.set_xlabel('component')
ax.set_xticks(X)
plt.show()
