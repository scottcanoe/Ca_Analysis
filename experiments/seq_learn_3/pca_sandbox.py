from types import SimpleNamespace
from typing import Sequence

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
from processing import *
from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *
from seq_learn_3.main import *
from seq_learn_3.utils import *


custom_cycler = (cycler(color=['black', '#ff7f00', 'blue',
                               '#f781bf', '#a65628', '#984ea3',
                               '#999999', '#e41a1c', '#dede00']
                        ) +
                 cycler(lw=[2] * 9))


def plot_pca_for_session(
    s: Session,
    n_components: int = 7,
    lpad: int = None,
    rpad: int = 24,
    show: bool = True,
    save: Optional[PathLike] = "analysis/pca.png",
) -> matplotlib.figure.Figure:

    # Extract trial-averaged spiking data for each sequence, and put them
    # end-to-end.
    arrays = []
    for seq_name in ('ABCD', 'ABBD', 'ACBD'):
        spec = schema.get_sequence(seq_name).events[:-1]
        arr = s.spikes.split(spec, lpad=lpad, rpad=rpad, concat=True).mean('trial')
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
    # Plot each
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))

    def fill_axes(ax, data, name):
        ax.set_prop_cycle(custom_cycler)
        for i in range(n_components):
            arr = data.isel(component=i)
            if i in to_flip:
                arr = -1 * arr
            ax.plot(arr, label=str(i))
        annotate_onsets(ax, data.coords['event'], skip_first=True, last="-")
        ax.legend(loc='upper right')
        ax.set_xlim([0, len(data)])
        ax.set_title(name)

    fill_axes(axes[0], ABCD_T, 'ABCD')
    fill_axes(axes[1], ABBD_T, 'ABBD')
    fill_axes(axes[2], ACBD_T, 'ACBD')

    ax = axes[3]
    evr = pca.explained_variance_ratio_
    csum = np.cumsum(evr)
    thresh_95 = np.sum(csum <= 0.95)
    thresh_99 = np.sum(csum <= 0.99)

    ax.plot(np.arange(thresh_99), csum[:thresh_99])
    ax.set_title('explained variance, cumulative')
    ax.set_xlabel('component')
    ax.set_ylabel('fraction')
    ax.axhline(0.95)
    ax.axvline(thresh_95)

    title = f"{s.mouse}, day={s.attrs['day']}"
    fig.suptitle(title)

    fig.tight_layout(pad=1)
    if show:
        plt.show()
    if save:
        fig.savefig(s.fs.getsyspath(save))

    return fig


n_components: int = 5
lpad: int = None
rpad: int = 24
show: bool = True
day = 5
roi_filter = 'all'

sessions = get_sessions(day=day, fs=0)
apply_roi_filter(sessions, roi_filter)

for s in sessions:
    plot_pca_for_session(s, n_components)

# sessions = sessions[0:1] + sessions[2:3]
# sessions = sessions[0:3]

savedir = Path.home() / 'plots/seq_learn_3/PCA'
save = savedir / f"PCA_day_{day}.png"


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

# Plt each
fig, axes = plt.subplots(4, 1, figsize=(8, 12))

def fill_axes(ax, data, name):
    ax.set_prop_cycle(custom_cycler)
    for i in range(n_components):
        arr = data.isel(component=i)
        if i in to_flip:
            arr = -1 * arr
        ax.plot(arr, label=str(i + 1))
    annotate_onsets(ax, data.coords['event'], skip_first=True, last="-")
    ax.legend(loc='upper right')
    ax.set_xlim([0, len(data)])
    ax.set_title(name)

fill_axes(axes[0], ABCD_T, 'ABCD')
fill_axes(axes[1], ABBD_T, 'ABBD')
fill_axes(axes[2], ACBD_T, 'ACBD')

ax = axes[3]
evr = pca.explained_variance_ratio_
csum = np.cumsum(evr)
thresh_95 = np.sum(csum <= 0.95)
thresh_99 = np.sum(csum <= 0.99)

ax.plot(np.arange(thresh_99), csum[:thresh_99])
ax.set_title('explained variance, cumulative')
ax.set_xlabel('component')
ax.set_ylabel('fraction')
ax.axhline(0.95)
ax.axvline(thresh_95)
ax.set_ylim([0, 1])

fig.tight_layout(pad=1)

if show:
    plt.show()

if save:
    fig.savefig(save)
