from matplotlib.figure import Figure
import numpy as np

from ca_analysis.plot import *
from experiments.seq_learn_3.main import *
from experiments.seq_learn_3.utils import *


day = 0
sessions = get_sessions(day=day, fs=0)
lpad: int = None
drop_gray = False

# set up splits
ABCD_all = flex_split(sessions, 'ABCD', lpad=lpad, drop_gray=drop_gray)
ABBD_all = flex_split(sessions, 'ABBD', lpad=lpad, drop_gray=drop_gray)
ACBD_all = flex_split(sessions, 'ACBD', lpad=lpad, drop_gray=drop_gray)

roi = 9
ABCD = ABCD_all.isel(roi=roi)
ABBD = ABBD_all.isel(roi=roi)
ACBD = ACBD_all.isel(roi=roi)

onset_kwargs = {
    "skip_first": lpad is not None,
    "alpha": 0.5,
    "last": "-",
    "shift": -0.5,
}

fig, ax = plt.subplots()

arr = ABCD.mean('trial')
ax.plot(arr, color='black', label='ABCD')

arr = ABBD.mean('trial')
ax.plot(arr, color='blue', label='ABBD')

arr = ACBD.mean('trial')
ax.plot(arr, color='red', label='ACBD')

ax.legend()
ax.set_xlabel('time')

annotate_onsets(ax, arr.coords['event'], **onset_kwargs)
ax.set_title(f'roi {roi}')
plt.show()
