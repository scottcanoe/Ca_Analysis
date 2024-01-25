from types import SimpleNamespace
from typing import Sequence

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
from processing import *
from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *



n_components: int = 5
lpad: int = 12
rpad: int = 12
show: bool = True
day = 0
visual = True

save = Path.home() / f'evr.png'

#-------------------------------------------------------------------------------

sessions_0 = get_sessions(day=0, fs=0)
sessions_5 = get_sessions(day=5, fs=0)

for s in sessions_0 + sessions_5:
    s.spikes.prepare()
    if visual:
        visual_ids = np.load(s.fs.getsyspath('visual.npy'))
        s.spikes.data = s.spikes.data.sel(roi=visual_ids)


#-------------------------------------------------------------------------------

# Extract trial-averaged spiking data for each sequence, and put them
# end-to-end.
sessions = sessions_0
arrays = []
for seq_name in ('ABCD', 'ABBD', 'ACBD'):
    spec = schema.get_sequence(seq_name).events[:-1]
    chunks = sessions.split('spikes', spec, lpad=lpad, rpad=rpad)
    for i, ch in enumerate(chunks):
        chunks[i] = ch.mean('trial')
    arr = xr.concat(chunks, 'roi')
    arrays.append(arr)
combined = xr.concat(arrays, 'time')
pca_0 = PCA()
tformed_0 = xr.DataArray(
    pca_0.fit_transform(combined),
    dims=('time', 'component'),
    coords=dict(event=combined.coords['event']),
)

#-------------------------------------------------------------------------------

sessions = sessions_5
arrays = []
for seq_name in ('ABCD', 'ABBD', 'ACBD'):
    spec = schema.get_sequence(seq_name).events[:-1]
    chunks = sessions.split('spikes', spec, lpad=lpad, rpad=rpad)
    for i, ch in enumerate(chunks):
        chunks[i] = ch.mean('trial')
    arr = xr.concat(chunks, 'roi')
    arrays.append(arr)
combined = xr.concat(arrays, 'time')
pca_5 = PCA()
tformed_5 = xr.DataArray(
    pca_5.fit_transform(combined),
    dims=('time', 'component'),
    coords=dict(event=combined.coords['event']),
)

# -------------------------------------------------------------------------------


# Plt each
fig, ax = plt.subplots(figsize=(6, 3))

evr = pca_0.explained_variance_ratio_
csum = np.cumsum(evr)
csum = np.r_[0, csum]
ax.plot(csum, label='day 0')

evr = pca_5.explained_variance_ratio_
csum = np.cumsum(evr)
csum = np.r_[0, csum]
ax.plot(csum, label='day 5')

ax.set_title('explained variance, cumulative')
ax.set_xlabel('component')
ax.set_ylabel('fraction')
# ax.axhline(0.95)
# ax.axvline(thresh_95)
ax.set_ylim([0, 1])
ax.set_xlim([0, 30])
fig.tight_layout(pad=1)
ax.legend()

if show:
    plt.show()
if save:
    fig.savefig(save)
