from types import SimpleNamespace
from typing import Sequence

import matplotlib
from matplotlib.animation import FFMpegWriter
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.decomposition import PCA

from processing import *
from ca_analysis import *
from ca_analysis.plot import *



lpad: int = None
rpad: int = None
day = 5
visual = False
save = Path.home() / f'plots/seq_learn_3/PCA_day_{day}_visual_{visual}_anim.mp4'


#-------------------------------------------------------------------------------



sessions = get_sessions(day=day, fs=0)


# Filter ROIs
for s in sessions:
    s.spikes.prepare()
    if visual:
        visual_ids = np.load(s.fs.getsyspath('visual.npy'))
        s.spikes.data = s.spikes.data.sel(roi=visual_ids)

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


lw = 1
color = 'blue'
alpha = 0.5

#
fig = Figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.view_init(20, 60)

line = ax.plot([], [], [], lw=lw, color=color, alpha=alpha)[0]
arr = ABCD_T

X = arr[:, 0].data
Y = arr[:, 1].data
Z = arr[:, 2].data

n_timepoints = len(X)
resample_factor = 4
X = resample(X, resample_factor * n_timepoints)
Y = resample(Y, resample_factor * n_timepoints)
Z = resample(Z, resample_factor * n_timepoints)

ax.set_xlim([X.min() - 1, X.max() + 1])
ax.set_ylim([Y.min() - 1, Y.max() + 1])
ax.set_zlim([Z.min() - 1, Z.max() + 1])

writer = FFMpegWriter(fps=30)
with writer.saving(fig, save, 190):
    for i in range(1, len(X)):
        chunk = arr.isel(time=slice(0, i))
        x = X[:i]
        y = Y[:i]
        z = Z[:i]
        line.set_xdata(x)
        line.set_ydata(y)
        line.set_3d_properties(z)
        writer.grab_frame()
