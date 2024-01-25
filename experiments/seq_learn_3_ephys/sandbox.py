from typing import Sequence

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ca_analysis.plot import *
from main import *




def plot_mean_traces(
    s: Session,
    lpad: Optional[int] = None,
    rpad: Optional[int] = 250,
    channel: Union[int, str] = 'mean',
    colors: Optional[Sequence] = ['black', 'red', 'mediumslateblue'],
    ylim: Optional[Sequence] = None,
    show: bool = True,
    save: Optional[PathLike] = "mean_traces.png",
) -> Figure:

    day = s.attrs['day']
    sequences = ('ABCD', 'ABBD', 'ACBD') if day in (0, 5) else ('ABCD',)

    width, height = 8, 8 / 3
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
    else:
        fig = Figure(figsize=(width, height))
        ax = fig.add_subplot(1, 1, 1)

    for i, seq_name in enumerate(sequences):
        spec = schema.get(sequence=seq_name)[:-1]
        arr = s.LFP.split(spec, lpad=lpad, rpad=rpad, concat=True)
        if is_int(channel):
            arr = arr.isel(channel=channel)
        elif channel == 'mean':
            arr = arr.mean('channel')
        else:
            raise ValueError(f'bad channel argument: {channel}')
        arr = arr.mean('trial')
        arr = arr - arr[0].item()
        kw = {'label': seq_name}
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

        title = f'{s.mouse} day {s.attrs["day"]}'
        ax.set_title(title)

    fig.tight_layout()
    if show:
        plt.show()

    if save:
        analysis_dir = Path(s.fs.getsyspath("analysis"))
        analysis_dir.mkdir(exist_ok=True)
        savepath = analysis_dir / "mean_traces.png"
        fig.savefig(savepath)
    return fig


plotdir = Path.home() / "plots/seq_learn_3/ephys/mean_traces"
sessions = get_sessions(fs=0)
for s in sessions:
    try:
        fig = plot_mean_traces(s, show=False)
        path = plotdir / f'{s.mouse}_day{s.attrs["day"]}.png'
        fig.savefig(path)
    except:
        print(f'failed on {s}')
