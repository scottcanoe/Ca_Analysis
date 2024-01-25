import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from ca_analysis.plot import *
from main import *
from processing import *

from utils import *




lpad = None
rpad = 24

day = 0

sequences = ("ABCD", "ABBD", "ACBD")
visual = None

plotdir = Path.home() / "plots/seq_learn_3/heatmaps"
outfile = plotdir / f"day_{day}.png"

# -----------------------------------------------------------------------------#

sessions = get_sessions(day=day)
apply_roi_filter(sessions, visual)


# split/combine sequence data
dsets = []
for seq_name in sequences:
    spec = schema.get(sequence=seq_name)[:-1]
    splits = sessions.split('spikes', spec, lpad=lpad, rpad=rpad)
    splits = [arr.mean('trial') for arr in splits]
    data = xr.concat(splits, 'roi')
    data = data.transpose('roi', ...)
    data.name = seq_name
    dsets.append(data)


# rank rois
ABCD, ABBD, ACBD = dsets

# - option 1: rank by peak firing time
# splits = split_by_event(ABBD)
# pre = splits[3].mean("time")
# post = splits[2].mean("time")
# stat = post - pre
# inds = np.flipud(np.argsort(post - pre))

# option 2: rank by difference between B2 and B1
stat = ABCD.argmax('time')
inds = np.argsort(stat)

for i, arr in enumerate(dsets):
    dsets[i] = arr.isel(roi=inds)

fig = plt.figure(figsize=(24, 10))
axes = [fig.add_subplot(1, 3, i) for i in range(1, 4)]
onset_kwargs = {
    'alpha': 0.5,
    'shift': -0.5,
    'skip_first': lpad is not None,
    'last': '-' if rpad is not None else None,
}
ax, arr = axes[0], dsets[0]
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
ax.imshow(cdata, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
ax.set_title(arr.name)

ax, arr = axes[1], dsets[1]
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
ax.imshow(cdata, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
ax.set_title(arr.name)

ax, arr = axes[2], dsets[2]
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
ax.imshow(cdata, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
ax.set_title(arr.name)

fig.tight_layout(pad=5)
plt.show()

# fig.savefig(outfile)

