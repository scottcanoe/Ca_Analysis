import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from ca_analysis.plot import *
from main import *
from processing import *


def prepare_data(ses: SessionGroup, visual: Optional[bool]) -> None:
    # apply roi filters
    for s in ses:
        data = s.spikes.data
        if visual is None:
            pass
        elif visual in {False, True}:
            vis_inds = np.load(s.fs.getsyspath('visual.npy'))
            if visual is True:
                data = data.sel(roi=vis_inds)
            else:
                all_inds = np.array(data.coords['roi'])
                non_vis_inds = np.setdiff1d(all_inds, vis_inds)
                data = data.sel(roi=non_vis_inds)
        else:
            raise ValueError('invalid "visual" argument')
        s.data = SessionData(s, data=data)

    dsets = []
    for seq_name in ('ABCD', 'ABBD', 'ACBD'):
        spec = schema.get(sequence=seq_name)[:-1]
        splits = ses.split('data', spec, lpad=lpad, rpad=rpad)
        splits = [arr.mean('trial') for arr in splits]
        data = xr.concat(splits, 'roi')
        data = data.transpose('roi', ...)
        data.name = seq_name
        dsets.append(data)

    return dsets



# -----------------------------------------------------------------------------#
# prepare data
def plot_3():

    sessions_0 = get_sessions(day=0)
    dsets_0 = prepare_data(sessions_0, visual)

    sessions_5 = get_sessions(day=5)
    dsets_5 = prepare_data(sessions_5, visual)

    ABCD_0, ABBD_0, ACBD_0 = dsets_0
    ABCD_5, ABBD_5, ACBD_5 = dsets_5

    fig, axes = plt.subplots(3, 1, figsize=(6, 5.25))

    ax = axes[0]
    ax.plot(ABCD_0.mean('roi'), color='black', label='day 0')
    ax.plot(ABCD_5.mean('roi'), color='red', label='day 5')
    annotate_onsets(ax, ABCD_0.coords["event"], skip_first=True, alpha=0.5, last="-")
    ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
    ax.legend(loc='upper left')
    ax.set_title('ABCD')

    ax = axes[1]
    ax.plot(ABBD_0.mean('roi'), color='black', label='day 0')
    ax.plot(ABBD_5.mean('roi'), color='red', label='day 5')
    annotate_onsets(ax, ABCD_0.coords["event"], skip_first=True, alpha=0.5, last="-")
    ax.legend(loc='upper left')
    ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
    ax.set_title('ABBD')

    ax = axes[2]
    ax.plot(ACBD_0.mean('roi'), color='black', label='day 0')
    ax.plot(ACBD_5.mean('roi'), color='red', label='day 5')
    annotate_onsets(ax, ABCD_0.coords["event"], skip_first=True, alpha=0.5, last="-")
    ax.legend(loc='upper left')
    ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
    ax.set_title('ACBD')


    fig.tight_layout(pad=1.5)
    plt.show()

    plotdir = Path.home() / "plots/seq_learn_3/mean_firing"
    plotdir.mkdir(exist_ok=True)
    outfile = plotdir / f"mean_rates_visual={visual}_3plots.png"

    fig.savefig(outfile)


lpad = 4
rpad = 4

sequences = ("ABCD", "ABBD", "ACBD")
visual = True

plotdir = Path.home() / "plots/seq_learn_3/mean_firing"
plotdir.mkdir(exist_ok=True)
outfile = plotdir / f"mean_rates_visual={visual}.png"


# sessions_0 = get_sessions(day=0)
# dsets_0 = prepare_data(sessions_0, visual)
#
# sessions_5 = get_sessions(day=5)
# dsets_5 = prepare_data(sessions_5, visual)
#
# ABCD_0, ABBD_0, ACBD_0 = dsets_0
# ABCD_5, ABBD_5, ACBD_5 = dsets_5


fig, axes = plt.subplots(3, 1, figsize=(6, 5.25))

ax = axes[0]
ax.plot(ABCD_0.mean('roi'), color='black', label='day 0')
ax.plot(ABCD_5.mean('roi'), color='red', label='day 5')
annotate_onsets(ax, ABCD_0.coords["event"], skip_first=True, alpha=0.5, last="-")
ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
ax.legend(loc='upper left')
ax.set_title('ABCD')

ax = axes[1]
ax.plot(ABBD_0.mean('roi'), color='black', label='day 0')
ax.plot(ABBD_5.mean('roi'), color='red', label='day 5')
annotate_onsets(ax, ABCD_0.coords["event"], skip_first=True, alpha=0.5, last="-")
ax.legend(loc='upper left')
ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
ax.set_title('ABBD')

ax = axes[2]
ax.plot(ACBD_0.mean('roi'), color='black', label='day 0')
ax.plot(ACBD_5.mean('roi'), color='red', label='day 5')
annotate_onsets(ax, ABCD_0.coords["event"], skip_first=True, alpha=0.5, last="-")
ax.legend(loc='upper left')
ax.set_xlim([0, ABCD_0.sizes['time'] - 1])
ax.set_title('ACBD')


fig.tight_layout(pad=1.5)
plt.show()

plotdir = Path.home() / "plots/seq_learn_3/mean_firing"
plotdir.mkdir(exist_ok=True)
outfile = plotdir / f"mean_rates_visual={visual}_3plots.png"

fig.savefig(outfile)
