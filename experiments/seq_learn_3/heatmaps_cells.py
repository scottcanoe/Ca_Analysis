import numpy as np

from ca_analysis.plot import *
from main import *
from processing import *

from matplotlib.figure import Figure




def make_cell_heatmaps(
    s: Session,
    lpad: int = 12,
    rpad: int = 12,
    target: str = "spikes",
) -> None:

    day = s.attrs['day']
    if day in {0, 5}:
        sequences = ("ABCD", "ABBD", "ACBD")
    else:
        sequences = ("ABCD",)

    # filter ROIs
    data = getattr(s, target).data
    s.data = SessionData(s, data=data)

    # set up splits
    dsets_all = {}
    for seq_name in sequences:
        spec = schema.get(sequence=seq_name)[:-1]
        splits = s.data.split(spec, lpad=lpad, rpad=rpad)
        arr = xr.concat(splits, 'time')
        arr.coords['roi'] = s.data.data.coords['roi']
        dsets_all[seq_name] = arr

    # iterate through ROIs, making figures for each
    roi_ids = s.data.data.coords['roi'].data

    for roi in roi_ids:
        dsets = {}
        for seq_name in sequences:
            splits = dsets_all[seq_name]
            data = splits.sel(roi=roi)
            data.name = seq_name
            dsets[seq_name] = data
            roi_id = data.coords['roi'].item()

        if day in {0, 5}:
            fig = Figure(figsize=(8, 8))
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)
            axes = [ax1, ax2, ax3]

        else:
            fig = Figure(figsize=(8, 8 / 3))
            ax1 = fig.add_subplot(1, 1, 1)
            axes = [ax1]

        onset_kwargs = {
            "skip_first": True,
            "alpha": 0.5,
            "last": "-",
            "shift": -0.5,
        }

        ax, arr = axes[0], dsets['ABCD']
        smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
        cdata = smap(arr)
        ax.imshow(cdata, aspect="auto", interpolation='none')
        annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
        ax.set_title(arr.name)
        ax.set_ylabel('trial')

        if day in {0, 5}:
            ax, arr = axes[1], dsets['ABBD']
            smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
            cdata = smap(arr)
            ax.imshow(cdata, aspect="auto", interpolation='none')
            annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
            ax.set_title(arr.name)
            ax.set_ylabel('trial')

            ax, arr = axes[2], dsets['ACBD']
            smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
            cdata = smap(arr)
            ax.imshow(cdata, aspect="auto", interpolation='none')
            annotate_onsets(ax, arr.coords["event"], **onset_kwargs)
            ax.set_title(arr.name)
            ax.set_ylabel('trial')


        title = f'{s.mouse}, day {s.attrs["day"]}, roi {roi_id}'
        fig.suptitle(title)
        fig.tight_layout()

        plotdir = Path(s.fs.getsyspath('analysis')) / 'heatmaps/rois'
        plotdir.mkdir(exist_ok=True, parents=True)
        fname = plotdir / f"roi_{roi_id}.png"
        fig.savefig(fname)
        fig.clear()
        del fig


if __name__ == "__main__":

    sessions = get_sessions(day=0, fs=0)

    for s in sessions:
        logger.info(f'Making heatmaps for {s}')
        make_cell_heatmaps(s)


