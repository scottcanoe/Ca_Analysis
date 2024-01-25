import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from ca_analysis.plot import *

from seq_learn_3.main import *
from seq_learn_3.utils import *


roi_filter = 'non_visual'
for roi_filter in ['all', 'gratings', 'gray', 'visual', 'non_visual']:
    split_kwargs = {
        'lpad': 4,
        'rpad': None,
        'drop_gray': False,
    }

    sessions_0 = get_sessions(day=0)
    sessions_5 = get_sessions(day=5)
    apply_roi_filter(sessions_0 + sessions_5, roi_filter)

    ABCD_0 = flex_split(sessions_0, 'ABCD', **split_kwargs)
    ABBD_0 = flex_split(sessions_0, 'ABBD', **split_kwargs)
    ACBD_0 = flex_split(sessions_0, 'ACBD', **split_kwargs)

    ABCD_5 = flex_split(sessions_5, 'ABCD', **split_kwargs)
    ABBD_5 = flex_split(sessions_5, 'ABBD', **split_kwargs)
    ACBD_5 = flex_split(sessions_5, 'ACBD', **split_kwargs)

    #-------------------------------------------------------------------------------

    fig, axes = plt.subplots(3, 1, figsize=(7, 6.25))
    onset_kwargs = {
        'alpha': 0.5,
        'skip_first': split_kwargs['lpad'] is not None,
        'last': '-' if split_kwargs['rpad'] is not None or split_kwargs['rpad'] is not None else None,
    }
    # ylim = [0.35, 1.6]

    ax = axes[0]
    ax.plot(ABCD_0.mean('trial').mean('roi'), color='black', label='day 0')
    ax.plot(ABCD_5.mean('trial').mean('roi'), color='red', label='day 5')
    annotate_onsets(ax, ABCD_0.coords["event"], **onset_kwargs)
    ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
    ax.legend(loc='upper left')
    ax.set_title('ABCD')
    # ax.set_ylim(ylim)

    ax = axes[1]
    ax.plot(ABBD_0.mean('trial').mean('roi'), color='black', label='day 0')
    ax.plot(ABBD_5.mean('trial').mean('roi'), color='red', label='day 5')
    annotate_onsets(ax, ABCD_0.coords["event"], **onset_kwargs)
    ax.legend(loc='upper left')
    ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
    ax.set_title('ABBD')
    # ax.set_ylim(ylim)

    ax = axes[2]
    ax.plot(ACBD_0.mean('trial').mean('roi'), color='black', label='day 0')
    ax.plot(ACBD_5.mean('trial').mean('roi'), color='red', label='day 5')
    annotate_onsets(ax, ABCD_0.coords["event"], **onset_kwargs)
    ax.legend(loc='upper left')
    ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
    ax.set_title('ACBD')
    # ax.set_ylim(ylim)


    fig.tight_layout(pad=1.5)
    plt.show()

    plotdir = Path.home() / "plots/seq_learn_3/mean_firing"
    plotdir.mkdir(exist_ok=True)
    outfile = plotdir / f"mean_rates_{roi_filter}_sequences.png"

    fig.savefig(outfile)

    #-------------------------------------------------------------------------------

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.0))
    onset_kwargs = {
        'alpha': 0.5,
        'skip_first': split_kwargs['lpad'] is not None,
        'last': '-' if split_kwargs['rpad'] is not None or split_kwargs['rpad'] is not None else None,
    }

    ax = axes[0]
    ax.plot(ABCD_0.mean('trial').mean('roi'), color='black', label='ABCD')
    ax.plot(ABBD_0.mean('trial').mean('roi'), color='red', label='ABBD')
    ax.plot(ACBD_0.mean('trial').mean('roi'), color='blue', label='ACBD')
    annotate_onsets(ax, ABCD_0.coords["event"], **onset_kwargs)
    ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
    ax.legend(loc='upper left')
    ax.set_title('day 0')
    # ax.set_ylim(ylim)

    ax = axes[1]
    ax.plot(ABCD_5.mean('trial').mean('roi'), color='black', label='ABCD')
    ax.plot(ABBD_5.mean('trial').mean('roi'), color='red', label='ABBD')
    ax.plot(ACBD_5.mean('trial').mean('roi'), color='blue', label='ACBD')
    annotate_onsets(ax, ABCD_5.coords["event"], **onset_kwargs)
    ax.legend(loc='upper left')
    ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
    ax.set_title('day 5')
    # ax.set_ylim(ylim)

    fig.tight_layout(pad=1.5)
    plt.show()

    plotdir = Path.home() / "plots/seq_learn_3/mean_firing"
    plotdir.mkdir(exist_ok=True)
    outfile = plotdir / f"mean_rates_{roi_filter}_days.png"

    fig.savefig(outfile)

