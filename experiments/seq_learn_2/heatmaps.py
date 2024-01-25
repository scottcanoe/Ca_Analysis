import numpy as np

from ca_analysis.plot import *

from processing import *



def annotate_onsets(
    ax: "Axes",
    labels: ArrayLike,
    skip_first: bool = False,
    last: Optional[str] = None,
    vline: bool = True,
    color: Union[ArrayLike, str] = "white",
    ls: str = "--",
    alpha: Number = 1,
) -> None:

    # Add onset indicators.
    xticks = []
    if not skip_first:
        xticks.append(0)
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            xticks.append(i - 0.5)
    xticklabels = [r'$\Delta$'] * len(xticks)
    if last is not None:
        xticklabels[-1] = last
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # add vertical lines
    if vline:
        for x in xticks:
            ax.axvline(x, color=color, ls=ls, alpha=alpha)


def concat(
    arrays: Sequence[xr.DataArray],
    dim: str,
    array_coord: Optional[str] = None,
    **kw,
) -> xr.DataArray:

    out = xr.concat(arrays, dim, **kw)
    if array_coord:
        chunks = []
        for i, arr in enumerate(arrays):
            chunks.append(np.full(arr.sizes[dim], i, dtype=int))
        chunks = np.hstack(chunks)
        out.coords[array_coord] = xr.DataArray(chunks, dims=(dim,))
    return out


def get_seq_data_one(
    s: Session,
    seq: Sequence,
    resample_factor: int = 1,
    resample_method: str = "linear",
    gray_size: int = 6,
    trial_average: bool = True,
) -> xr.DataArray:

    # collect data
    seq = s.data.split(seq)
    seq[0] = seq[0].isel(time=slice(-gray_size, None))
    seq[-1] = seq[-1].isel(time=slice(0, gray_size))
    if trial_average:
        for i, arr in enumerate(seq):
            seq[i] = arr.mean("trial")

    # combine arrays
    arr = concat(seq, "time", array_coord="chunk").transpose("roi", "time", ...)

    # resample
    if resample_factor != 1:
        arr = resample1d(
            arr,
            "time",
            factor=resample_factor,
            method=resample_method,
        )

    return arr


def get_seq_data(
    ses: Union[Session, Sequence[Session]],
    seq: Sequence,
    resample_factor: int = 1,
    resample_method: str = "linear",
    gray_size: int = 6,
    trial_average: bool = True,
) -> xr.DataArray:

    if isinstance(ses, Session):
        return get_seq_data_one(
            ses,
            seq,
            resample_factor=resample_factor,
            gray_size=gray_size,
            trial_average=trial_average,
        )
    arrays = []
    for s in ses:
        arr = get_seq_data_one(
            s,
            seq,
            resample_factor=resample_factor,
            resample_method=resample_method,
            gray_size=gray_size,
            trial_average=trial_average,
        )
        arrays.append(arr)
    out = xr.concat(arrays, "roi")
    return out


def get_chunk(arr: xr.DataArray, num: int) -> xr.DataArray:
    out = arr.sel(time=arr['chunk'] == num)
    return out


def split_chunks(arr: xr.DataArray) -> xr.DataArray:
    chunk_vals = []
    for val in arr.coords["chunk"].values:
        if val not in chunk_vals:
            chunk_vals.append(val)
    return [get_chunk(arr, num) for num in chunk_vals]


def reorder_chunks(arr: xr.DataArray, order: ArrayLike) -> xr.DataArray:
    lst = [get_chunk(arr, num) for num in order]
    out = concat(lst, "time")
    out.attrs.update(arr.attrs)
    out.name = arr.name
    return out


day = 0

sessions = get_sessions(day=day)

ses = sessions[:]
seqs = ("ABCD", "ABBD")
visual_only = False
ops = {
    "resample_factor": 1,
    "resample_method": "cubic",
    "gray_size": 6,
    "trial_average": True,
}
plotdir = Path.home() / "plots/seq_learn_2/heatmaps"
outfile = plotdir / f"{seqs[1]}-{seqs[0]}_day_{day}.pdf"

for s in ses:
    visual_spikes = s.data.attrs.get("visual_spikes")
    if visual_spikes is None:
        s.data.attrs["all_spikes"] = s.data.get("spikes")
        vispath = s.fs.getsyspath("visual.npy")
        # if Path(vispath).exists():
        visual_ids = np.load(s.fs.getsyspath("visual.npy"))
        spks = s.data.get("spikes")
        spks = spks.isel(roi=visual_ids)
        s.data.attrs["visual_spikes"] = spks
        s.data.attrs["spikes"] = spks

if visual_only:
    s.data.attrs["spikes"] = s.data.attrs["visual_spikes"]
else:
    s.data.attrs["spikes"] = s.data.attrs["all_spikes"]


dsets = {}
for seq_name in seqs:
    sseq = s.events.schema.get(sequence=seq_name)
    elts = [elt.name for elt in sseq]
    elts = [elts[-1]] + elts
    seq = get_seq_data(ses, elts, **ops)
    seq.name = seq_name
    dsets[seq_name] = seq

# for name, arr in dsets.items():
#     dsets[name] = arr.mean("trial")

seq1 = dsets[seqs[0]]
seq2 = dsets[seqs[1]]
diff = seq2 - seq1
diff.name = f"{seq2.name} - {seq1.name}"
dsets["diff"] = diff

# rank rois
obj = dsets['diff']
pre = get_chunk(obj, 3).mean("time")
post = get_chunk(obj, 2).mean("time")
stat = post - pre
inds = np.flipud(np.argsort(stat))
stat = np.argmax(obj.data, axis=1)
inds = np.argsort(stat)

for name, arr in dsets.items():
    dsets[name] = arr.isel(roi=inds)


fig = plt.figure(figsize=(12, 18))
axes = [fig.add_subplot(3, 1, i) for i in range(1, 4)]

ax = axes[0]
arr = dsets[seqs[0]]
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
ax.imshow(cdata, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["chunk"], skip_first=True, alpha=0.5, last="-")
ax.set_title(arr.name)

ax = axes[1]
arr = dsets[seqs[1]]
smap = get_smap('inferno', data=arr, qlim=(2.5, 97.5))
cdata = smap(arr)
ax.imshow(cdata, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["chunk"], skip_first=True, alpha=0.5, last="-")
ax.set_title(arr.name)

ax = axes[2]
arr = dsets["diff"]
smap = get_smap("coolwarm", data=diff, qlim=(50, 97.5))
cdata = smap(arr)
ax.imshow(cdata, aspect="auto", interpolation='none')
annotate_onsets(ax, arr.coords["chunk"], skip_first=True, alpha=0.5, last="-")
ax.set_title(arr.name)

fig.tight_layout()
plt.show()
fig.savefig(outfile)

fig, ax = plt.subplots()
obj = dsets["ABCD"]
sums = obj.sum("roi").data
ax.plot(sums, color='red', label='ABCD')
obj = dsets["ABBD"]
sums = obj.sum("roi").data
ax.plot(sums, color='black', label='ABBD')
ax.legend()
plt.show()
