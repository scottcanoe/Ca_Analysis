from types import SimpleNamespace
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
from scipy.stats import *

from sklearn.decomposition import PCA
from ca_analysis import *
from ca_analysis.plot import *
from experiments.seq_learn_3.main import *



def apply_roi_filter(
    sessions: Union[Session, Iterable[Session]],
    name: str,
) -> None:

    import h5py

    if isinstance(sessions, Session):
        sessions = [sessions]

    for s in sessions:
        with h5py.File(s.fs.getsyspath('roi_filters.h5'), 'r') as f:
            inds = f[name][:]
        for attr in ['spikes', 'F', 'Fneu']:
            obj = getattr(s, attr)
            obj.prepare()
            obj.data = obj.data.sel(roi=inds)
            del obj.data.coords["roi"]


def flex_split(
    sessions: SessionGroup,
    event: str,
    lpad: Optional[int] = None,
    rpad: Optional[int] = None,
    drop_gray: bool = True,
    target="spikes",
) -> Tuple[xr.DataArray, xr.DataArray]:


    # split/combine sequence data
    if event in ('ABCD', 'ABBD', 'ACBD'):
        spec = schema.get(sequence=event)
        spec = spec[:-1] if drop_gray else spec
    else:
        spec = schema.get(event=event)
    splits = sessions.split(target, spec, lpad=lpad, rpad=rpad)

    # force number of trials to be equal
    n_trials = min([arr.sizes['trial'] for arr in splits])
    splits = [arr.isel(trial=slice(0, n_trials)) for arr in splits]
    data = xr.concat(splits, 'roi')

    return data


def split_all_gray(sessions, *args, **kw):
    ABCD_ = flex_split(sessions, 'ABCD._', *args, **kw)
    ABBD_ = flex_split(sessions, 'ABBD._', *args, **kw)
    ACBD_ = flex_split(sessions, 'ACBD._', *args, **kw)
    out = xr.concat([ABCD_, ABBD_, ACBD_], 'trial')
    return out


def normalize_confusion_matrix(cmat):
    out = np.zeros_like(cmat, dtype=float)
    for i in range(cmat.shape[0]):
        sum_ = cmat[i].sum()
        if sum_ > 0:
            out[i] = cmat[i] / sum_
    return out


def get_soft_decoder_accuracy(cm: np.ndarray) -> np.ndarray:

    n = cm.shape[0]
    out = np.zeros(n)
    for i in range(n):
        row = cm[i]
        score = 0
        if i - 2 >= 0:
            score += 0.25 * row[i - 1]
        if i - 1 >= 0:
            score += 0.5 * row[i - 1]
        score += row[i]
        if i + 1 <= n - 1:
            score += 0.5 * row[i + 1]
        if i + 2 <= n - 1:
            score += 0.25 * row[i + 1]
        out[i] = score / row.sum()
    return out
