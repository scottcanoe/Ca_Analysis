from types import SimpleNamespace

from scipy.stats import *

from processing import *

sessions = get_sessions()

s = sessions[0]

schema = s.events.schema
n_cells = s.data.n_cells
stim_events = [str(ev) for ev in schema.events if not ev.is_gray]

# split master spike array by event
gray = s.data.split(["-"])
A = s.data.split(["A_1"])

spikes = {}
for ev in stim_events:
    spikes[ev] = s.data.split(ev)

stim = xr.concat([spikes[key] for key in stim_events], "trial")
gray = s.data.split("-")
spikes["gray"] = gray

# gray = xr.concat([spikes["-"][:, i*6:(i+1)*6, :] for i in range(4)], "trial")

stim = stim.mean("time")
gray = gray.mean("time")

all_info = []
alpha = 0.05
for i in range(n_cells):
    obj = SimpleNamespace(roi=i)
    stim_array = stim.isel(roi=i)
    gray_array = gray.isel(roi=i)
    stat = ks_2samp(stim_array, gray_array)
    obj.pvalue = stat.pvalue
    obj.significant = stat.pvalue < alpha
    obj.visual = False
    if obj.significant:
        obj.visual = gray_array.mean().item() < stim_array.mean().item()
    all_info.append(obj)

significant = [info for info in all_info if info.significant]
visual = [info for info in significant if info.visual]
n_sig = len(significant)
n_vis = len(visual)

print('significant: {} / {}  ({:.2f}%)'.format(n_sig, n_cells, 100 * (n_sig / n_cells)))
print('visual: {} / {}  ({:.2f}%)'.format(n_vis, n_cells, 100 * (n_vis / n_cells)))

ids = np.array([info.roi for info in visual])
np.save(s.fs.getsyspath("visual.npy"), ids)
print('finished')
