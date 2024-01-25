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


def annotate_onsets(
    ax: "Axes",
    events: ArrayLike,
    symbol: str = r'$\Delta$',
    skip_first: bool = False,
    last: Optional[str] = None,
    vline: bool = True,
    shift: Number = 0,
    **kw,
) -> None:
    """
    Put onset indicators where labels change. Optionally, add a vertical
    line, usually for heatmaps or traces.

    Parameters
    ----------
    ax

    Returns
    -------

    """

    events = np.asarray(events)

    # Add onset indicators.
    xticks = []
    if not skip_first:
        xticks.append(0 + shift)
    for i in range(1, len(events)):
        if events[i] != events[i - 1]:
            xticks.append(i + shift)
    xticklabels = [symbol] * len(xticks)
    if last:
        if last is True:
            xticklabels[-1] = symbol
        else:
            xticklabels[-1] = last

    ax.set_xticks([])
    ax.set_xticklabels([])

    # add vertical lines
    if vline:
        kw = dict(kw)
        kw['color'] = kw.get('color', kw.get('c', 'gray'))
        kw['linestyle'] = kw.get('linestyle', kw.get('ls', '--'))
        kw['linewidth'] = kw.get('linewidth', kw.get('lw', 1))
        kw['alpha'] = kw.get('alpha', 1)
        for x in xticks:
            ax.axvline(x, **kw)


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

# figure out which components to flip
to_flip = []
for i in range(n_components):
    arr = ABCD_T.isel(component=i)
    # arr = tformed.isel(component=i)
    if abs(arr.min()) > abs(arr.max()):
        to_flip.append(i)
        continue

# plot
fig, axes = plt.subplots(n_components, 3, figsize=(4.2, 9))

# plot scale lines
for i in range(n_components):
    ax = axes[i, 0]
    ax.plot([1, 1], [0, 5], color='black', lw=1)

for seq_num, sequence in enumerate(('ABCD', 'ABBD', 'ACBD')):

    if sequence == 'ABCD':
        data = ABCD_T
    elif sequence == 'ABBD':
        data = ABBD_T
    elif sequence == 'ACBD':
        data = ACBD_T
    else:
        raise ValueError('invalid sequence')

    for i in range(n_components):
        ax = axes[i, seq_num]
        arr = data.isel(component=i)
        if i in to_flip:
            arr = -1 * arr
        ax.plot(arr, label=str(i + 1))
        annotate_onsets(ax, data.coords['event'], skip_first=False, last="-")
        ax.set_xlim([0, len(data)])

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # if seq_num == 0:
        #     ax.set_title(f'component {i + 1}')
        if seq_num > 0:
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)

for i in range(n_components):
    row = axes[i]
    ylims = [ax.get_ylim() for ax in row]
    ymin = min([y[0] for y in ylims])
    ymax = max([y[1] for y in ylims])
    for ax in row:
        ax.set_ylim([ymin, ymax])

# ymin = min([axes[i, 0].get_ylim()[0] for i in range(n_components)])
# ymax = max([axes[i, 0].get_ylim()[1] for i in range(n_components)])
# for i in range(n_components):
#     ax = axes[i, 0]
#     ax.plot([1, 1], [0, 5], color='black', lw=1)
#     for j in range(3):
#         axes[i, j].set_ylim([ymin, ymax])

# remove yticks
for i in range(n_components):
    ax = axes[i, 0]
    ax.set_yticks([])


fig.tight_layout(pad=0.5)
plt.show()
fig.savefig(f'figures/pca_scratch/pca_many_day_{day}.png')
fig.savefig(f'figures/pca_scratch/pca_many_day_{day}.eps')
