from typing import Sequence

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ca_analysis.plot import *
from main import *


def plot_mean_traces(
    groups: Sequence[SessionGroup],
    lpad: int = 250,
    rpad: int = 250,
    sequences: Sequence[str] = ('ABCD', 'ABBD', 'ACBD'),
    channel: Union[int, str] = 'mean',
    colors: Optional[Sequence] = None,
    ylim: Optional[Sequence] = None,
    show: bool = True,
) -> "Figure":

    n_plots = len(sequences)
    width = 8
    height_per_axes = 8 / 3
    height = n_plots * height_per_axes

    fig, axes = plt.subplots(len(sequences), 1, figsize=(width, height))
    if n_plots == 1:
        axes = [axes]
    for ax_num in range(len(sequences)):
        ax = axes[ax_num]
        seq_name = sequences[ax_num]
        seq = schema.get(sequence=seq_name)[:-1]
        for i, grp in enumerate(groups):
            arr = grp.split('LFP', seq, lpad=lpad, rpad=rpad, concat=True)
            if is_int(channel):
                arr = arr.isel(channel=channel)
            else:
                if channel == 'mean':
                    arr = arr.mean('channel')
                else:
                    raise ValueError(f'bad channel argument: {channel}')
            arr = arr.mean('trial')
            arr = arr - arr[0].item()
            kw = {'label': grp.name}
            if colors is not None:
                kw['color'] = colors[i]
            ax.plot(arr, **kw)

        ax.set_xlim([0, len(arr)])
        if ylim is not None:
            ax.set_ylim(ylim)

        annotate_onsets(ax, arr.coords['event'], skip_first=True)

        ax.legend(loc='upper left')
        ax.set_title(seq_name)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def rename_files():
    top_dir = Path("/home/scott/ephys/experiments/seq_learn_3/data/71216_Run5")
    plx_dir = top_dir / "plexon"
    h5_dir = top_dir / "h5"

    parent_dir = plx_dir
    for p in parent_dir.glob("*"):
        name = p.name
        name = name.replace(" ", "")
        parts = name.split('_')
        new_name = "_".join(parts[:3])
        new_path = parent_dir / new_name
        new_path = new_path.with_suffix(".plx")
        p.rename(new_path)
        print(new_path)

    parent_dir = h5_dir
    for p in parent_dir.glob("*"):
        name = p.name
        name = name.replace(" ", "")
        parts = name.split('_')
        new_name = "_".join(parts[:3])
        new_path = parent_dir / new_name
        new_path = new_path.with_suffix(".h5")
        p.rename(new_path)
        print(new_path)


# day_0 = get_sessions(group='a', day=0, fs=0)
# day_5 = get_sessions(group='a', day=5, fs=0)
#
# day_0.name = 'day 0'
# day_5.name = 'day 5'
#
# lpad = 50
# rpad = 50
# channel = 'mean'
#
# plotdir = Path.home() / 'plots/seq_learn_3/ephys'
# save = plotdir / "day_5.png"
# colors = ['black', 'red']
# kw = {
#     'lpad': lpad,
#     'rpad': rpad,
#     'channel': channel,
#     'colors': colors,
#     'ylim': [-0.3, 0.3],
# }
# fig = plot_mean_traces([day_0, day_5], **kw)
# plt.show()


# top_dir = Path("/home/scott/ephys/experiments/seq_learn_3/data/71216_Run5")
# top_dir = Path("/home/scott/ephys/experiments/seq_learn_3/data/78139_Run1_Layer23")
# plx_dir = top_dir / "plexon"
# h5_dir = top_dir / "h5"
#
# mice = [
#     '78139-1',
#     '78139-2',
#     '78139-3',
#     '78139-4',
# ]
# dates = (
#     '2023-02-17',
#     '2023-02-20',
#     '2023-02-21',
#     '2023-02-22',
#     '2023-02-23',
#     '2023-02-24',
# )
# for mouse in mice:
#     for day, date in enumerate(dates):
#         import_session(mouse, date, 1, day, plx_dir, h5_dir)
# group_name = 'a'
# sessions_0 = get_sessions(group=group_name, day=0)
# sessions_0.name = "day 0"
# sessions_5 = get_sessions(group=group_name, day=5)
# sessions_5.name = "day 5"


sessions_a = get_sessions(group='a', day=5)
sessions_a.name = "L4"
sessions_c = get_sessions(group='c', day=5)
sessions_c.name = "L2/3"

groups: Sequence[SessionGroup] = [sessions_a, sessions_c]
lpad: Optional[int] = None
rpad: Optional[int] = 250
sequences: Sequence[str] = ('ABCD', 'ABBD', 'ACBD')
channel: Union[int, str] = 'mean'
colors: Optional[Sequence] = None
ylim: Optional[Sequence] = None
show: bool = True


n_plots = len(sequences)
width = 8
height_per_axes = 8 / 3
height = n_plots * height_per_axes

fig, axes = plt.subplots(len(sequences), 1, figsize=(width, height))
axes = [axes] if n_plots == 1 else axes

for ax_num in range(len(sequences)):
    ax = axes[ax_num]
    seq_name = sequences[ax_num]
    spec = schema.get(sequence=seq_name)[:-1]
    for i, grp in enumerate(groups):
        arrays = grp.split('LFP', spec, lpad=lpad, rpad=rpad)
        n_trials = min([arr.sizes['trial'] for arr in arrays])
        for j, arr in enumerate(arrays):
            arrays[j] = arr.isel(trial=slice(0, n_trials))
        arr = xr.concat(arrays, 'trial')
        if is_int(channel):
            arr = arr.isel(channel=channel)
        elif channel == 'mean':
            arr = arr.mean('channel')
        else:
            raise ValueError(f'bad channel argument: {channel}')
        arr = arr.mean('trial')
        arr = arr - arr[0].item()
        kw = {'label': grp.name}
        if colors is not None:
            kw['color'] = colors[i]
        ax.plot(arr, **kw)

    ax.set_xlim([0, len(arr)])
    if ylim is not None:
        ax.set_ylim(ylim)

    skip_first = lpad is not None
    last = '-' if rpad is not None else None
    annotate_onsets(ax, arr.coords['event'], skip_first=skip_first, last=last)

    ax.legend(loc='lower right')
    ax.set_title(seq_name)

fig.tight_layout()
if show:
    plt.show()
