from matplotlib.figure import Figure
import numpy as np

from ca_analysis.plot import *
from main import *
from processing import *
from seq_learn_3.utils import *



day = 5
sequences = ("ABCD", "ABBD", "ACBD")
roi_filter = 'all'
drop_gray = False



# -----------------------------------------------------------------------------

plotdir = Path.home() / f"plots/seq_learn_3/heatmaps/trials/day_{day}/{roi_filter}"
plotdir.mkdir(exist_ok=True, parents=True)

sessions = get_sessions(day=day)
apply_roi_filter(sessions, roi_filter)

dsets = {}
ABCD = flex_split(sessions, 'ABCD', drop_gray=drop_gray)
ABBD = flex_split(sessions, 'ABBD', drop_gray=drop_gray)
ACBD = flex_split(sessions, 'ACBD', drop_gray=drop_gray)

arr = ABCD.mean('trial')
stat = arr.argmax('time')
inds = np.argsort(stat)
ABCD = ABCD.isel(roi=inds).transpose('roi', ...)
ABBD = ABBD.isel(roi=inds).transpose('roi', ...)
ACBD = ACBD.isel(roi=inds).transpose('roi', ...)

n_trials = ABCD.sizes['trial']

for trial_num in range(n_trials):

    seq_1 = ABCD.isel(trial=trial_num)
    seq_2 = ABBD.isel(trial=trial_num)
    seq_3 = ACBD.isel(trial=trial_num)

    fig = Figure(figsize=(8, 8))
    axes = [fig.add_subplot(3, 1, i) for i in range(1, 4)]

    ax, arr = axes[0], seq_1
    smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
    cdata = smap(arr)
    ax.imshow(cdata, aspect="auto", interpolation='none')
    annotate_onsets(ax, arr.coords["event"], skip_first=True, alpha=0.5, last="-")
    ax.set_title('ABCD')

    ax, arr = axes[1], seq_2
    smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
    cdata = smap(arr)
    ax.imshow(cdata, aspect="auto", interpolation='none')
    annotate_onsets(ax, arr.coords["event"], skip_first=True, alpha=0.5, last="-")
    ax.set_title('ABBD')

    ax, arr = axes[2], seq_3
    smap = get_smap("inferno", data=arr, qlim=(2.5, 97.5))
    cdata = smap(arr)
    ax.imshow(cdata, aspect="auto", interpolation='none')
    annotate_onsets(ax, arr.coords["event"], skip_first=True, alpha=0.5, last="-")
    ax.set_title('ACBD')

    fig.tight_layout(pad=2)
    fig.suptitle(f'trial {trial_num}')
    outfile = plotdir / f'trial_{trial_num}.png'

    fig.savefig(outfile)
