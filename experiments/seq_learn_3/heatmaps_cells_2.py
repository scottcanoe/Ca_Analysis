from matplotlib.figure import Figure
import numpy as np

from ca_analysis.plot import *
from experiments.seq_learn_3.main import *
from experiments.seq_learn_3.utils import *


day = 0
sessions = get_sessions(day=day, fs=0)
lpad: int = 4
drop_gray = False

# set up splits
ABCD = flex_split(sessions, 'ABCD', lpad=lpad, drop_gray=drop_gray)
ABBD = flex_split(sessions, 'ABBD', lpad=lpad, drop_gray=drop_gray)
ACBD = flex_split(sessions, 'ACBD', lpad=lpad, drop_gray=drop_gray)


for roi in range(ABCD.sizes['roi']):
# for roi in range(ABCD.sizes['roi']):
    print(roi)
    fig = Figure(figsize=(8, 8))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)
    axes = [ax1, ax2, ax3, ax4]

    onset_kwargs = {
        "skip_first": lpad is not None,
        "alpha": 0.5,
        "last": "-",
        "shift": -0.5,
    }

    abcd = ABCD.isel(roi=roi)
    abbd = ABBD.isel(roi=roi)
    acbd = ACBD.isel(roi=roi)
    mat = np.stack([abcd, abbd, acbd])
    smap = get_smap('inferno', data=mat, qlim=(2.5, 97.5))

    ax, arr = axes[0], ABCD.isel(roi=roi)
    # smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
    cdata = smap(arr)
    ax.imshow(cdata, aspect="auto", interpolation='none')
    annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
    ax.set_title('ABCD')
    ax.set_ylabel('trial')

    ax, arr = axes[1], ABBD.isel(roi=roi)
    # smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
    cdata = smap(arr)
    ax.imshow(cdata, aspect="auto", interpolation='none')
    annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
    ax.set_title('ABBD')
    ax.set_ylabel('trial')

    ax, arr = axes[2], ACBD.isel(roi=roi)
    # smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
    cdata = smap(arr)
    ax.imshow(cdata, aspect="auto", interpolation='none')
    annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
    ax.set_title('ACBD')
    ax.set_ylabel('trial')

    ax = axes[3]
    ax.plot(abcd.mean('trial'), color='black', label='ABCD')
    ax.plot(abbd.mean('trial'), color='red', label='ABBD')
    ax.plot(acbd.mean('trial'), color='blue', label='ACBD')
    annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
    ax.set_xlim([0, arr.sizes['time'] - 1])
    ymax = max(ax.get_ylim()[-1], 6)
    ax.set_ylim([0, ymax])
    ax.legend()

    fig.suptitle(str(roi))
    fig.tight_layout()

    plotdir = Path.home() / f'plots/seq_learn_3/heatmaps/cells/day{day}'
    plotdir.mkdir(exist_ok=True, parents=True)
    fname = plotdir / f"{roi}.png"
    fig.savefig(fname)
    fig.clear()
    del fig


