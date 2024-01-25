from time import perf_counter as clock
from experiments.meso.main import *


def make_mean_mov(s: Session) -> None:

    channels = get_channels(s)
    if 'ca' in channels:
        s.ca_mov.load()
        ca_left = s.ca_mov.split('left').mean('trial')
        ca_right = s.ca_mov.split('right').mean('trial')
        s.ca_mov.clear()
    s.ach_mov.load()
    ach_left = s.ach_mov.split('left').mean('trial')
    ach_right = s.ach_mov.split('right').mean('trial')
    s.ach_mov.clear()

    with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'w') as f:
        if 'ca' in channels:
            f.create_dataset('ca/left', data=ca_left.data)
            f.create_dataset('ca/right', data=ca_right.data)
        f.create_dataset('ach/left', data=ach_left.data)
        f.create_dataset('ach/right', data=ach_right.data)


sessions = [
    open_session('M110', '2023-04-21', '2', fs=0),
    open_session('M115', '2023-04-21', '2', fs=0),
]

for s in sessions:
    make_mean_mov(s)
