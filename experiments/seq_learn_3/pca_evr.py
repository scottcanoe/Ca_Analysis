from types import SimpleNamespace
from typing import Sequence

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
from processing import *
from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *
from seq_learn_3.main import *
from seq_learn_3.utils import *


def get_evr(day: int) -> np.ndarray:

    lpad: int = None
    rpad: int = 24
    roi_filter = 'all'

    sessions = get_sessions(day=day, fs=0)
    apply_roi_filter(sessions, roi_filter)

    # Extract trial-averaged spiking data for each sequence, and put them
    # end-to-end.
    arrays = []
    for seq_name in ('ABCD', 'ABBD', 'ACBD'):
        spec = schema.get_sequence(seq_name).events[:-1]
        chunks = sessions.split('spikes', spec, lpad=lpad, rpad=rpad)
        for i, ch in enumerate(chunks):
            chunks[i] = ch.mean('trial')
        arr = xr.concat(chunks, 'roi')
        arrays.append(arr)
    combined = xr.concat(arrays, 'time')

    # Do PCA
    pca = PCA()
    tformed = xr.DataArray(
        pca.fit_transform(combined),
        dims=('time', 'component'),
        coords=dict(event=combined.coords['event']),
    )

    evr = pca.explained_variance_ratio_

    return evr


evr_0 = get_evr(0)
csum_0 = np.cumsum(evr_0)

evr_5 = get_evr(5)
csum_5 = np.cumsum(evr_5)

fig, ax = plt.subplots()
ax.plot(csum_0, color='black', label='day 0')
ax.plot(csum_5, color='red', label='day 5')
ax.legend()
ax.set_xlim([0, 25])
plt.show()

# thresh_95 = np.sum(csum <= 0.95)
# thresh_99 = np.sum(csum <= 0.99)

