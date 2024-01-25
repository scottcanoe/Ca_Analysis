from time import perf_counter as clock
from experiments.meso.main import *


def make_retinotopy_mean_mov(s: Session) -> None:

    channels = get_channels(s)
    if 'ca' in channels:
        s.ca_mov.load()
        ca = s.ca_mov.split('right').mean('trial')
        s.ca_mov.clear()
    s.ach_mov.load()
    ach = s.ach_mov.split('right').mean('trial')
    s.ach_mov.clear()

    with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'w') as f:
        if 'ca' in channels:
            f.create_dataset('ca/right', data=ca.data)
        f.create_dataset('ach/right', data=ach.data)


sessions = [
    open_session('M110', '2023-04-21', '1', fs=0),
    open_session('M115', '2023-04-21', '1', fs=0),
    open_session('M150', '2023-03-30', '1', fs=0),
    open_session('M152', '2023-03-30', '1', fs=0),
    open_session('M153', '2023-03-30', '1', fs=0),
]

for s in sessions:
    make_retinotopy_mean_mov(s)
