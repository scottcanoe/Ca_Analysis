from types import SimpleNamespace
from typing import Sequence

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *
from seq_learn_3.main import *
from seq_learn_3.utils import *

plt.rcParams['font.size'] = 8

custom_cycler = (cycler(color=['black', '#ff7f00', 'blue',
                               '#f781bf', '#a65628', '#984ea3',
                               '#999999', '#e41a1c', '#dede00']
                        ) +
                 cycler(lw=[2] * 9))



n_components: int = 20
lpad: int = None
rpad: int = 24
show: bool = True
day = 5
roi_filter = 'all'

sessions = get_sessions(day=day, fs=0)
apply_roi_filter(sessions, roi_filter)


savedir = Path.home() / 'plots/seq_learn_3/PCA'
save = savedir / f"PCA_day_{day}.eps"


# Extract trial-averaged spiking data for each sequence, and put them
# end-to-end.

arrays = []
for seq_name in ('ABCD', 'ABBD', 'ACBD'):
    spec = schema.get_sequence(seq_name).events[:-1]
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

# Split transformed data into ABCD/ABBD/ACBD chunks
edges = np.r_[0, np.cumsum([arr.sizes['time'] for arr in arrays])]
ABCD_T = tformed.isel(time=slice(edges[0], edges[1]))
ABBD_T = tformed.isel(time=slice(edges[1], edges[2]))
ACBD_T = tformed.isel(time=slice(edges[2], edges[3]))

to_flip = []
for i in range(n_components):
    arr = ABCD_T.isel(component=i)
    if abs(arr.min()) > abs(arr.max()):
        to_flip.append(i)

# plot
for sequence in ('ABCD', 'ABBD', 'ACBD'):
    fig, axes = plt.subplots(n_components, 1, figsize=(4.2, 20.5))

    if sequence == 'ABCD':
        data = ABCD_T
    elif sequence == 'ABBD':
        data = ABBD_T
    elif sequence == 'ACBD':
        data = ACBD_T
    else:
        raise ValueError('invalid sequence')


    for i in range(n_components):
        ax = axes[i]
        arr = data.isel(component=i)
        if i in to_flip:
            arr = -1 * arr
        ax.plot(arr, label=str(i + 1))
        annotate_onsets(ax, data.coords['event'], skip_first=False, last="-")
        ax.set_xlim([0, len(data)])
        # ax.set_title(name)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(f'component {i}')


    fig.tight_layout(pad=0.5)
    plt.show()
    fig.savefig(f'figures/pca_scratch/pca_day_{sequence}_{day}.png')
