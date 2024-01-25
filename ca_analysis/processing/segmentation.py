import shutil
from typing import (
    Mapping,
    Optional,
    Union,
    Sequence,
)

from ca_analysis.io import *
from ..common import *
from ..environment import *

__all__ = [
    "run_roi_detection",
]


def run_roi_detection(
    s: "Session",
    mov: PathLike,
    ops: Optional[Mapping] = None,
    cleanup: bool = True,
) -> None:
    import suite2p

    default_ops = suite2p.default_ops()
    if ops:
        ops = default_ops.update(ops)
    else:
        ops = default_ops

    root = s.fs.getsyspath("")
    mov_url = URL(mov)
    dset_name = mov_url.query.get("name", "data")

    ops["h5py"] = str(mov_url.path)
    ops["h5py_key"] = dset_name
    ops["data_path"] = [root]
    ops['save_path0'] = root
    ops["save_folder"] = "suite2p"
    ops['fast_disk'] = get_fs(0).getsyspath("temp")
    try:
        ops['fs'] = s.attrs["capture"]["fps"]
    except:
        ops['fs'] = s.attrs["capture"]["frame"]["rate"]
    ops['delete_bin'] = True

    # Bidirectional phase offset
    ops['do_bidiphase'] = True

    # Registration.
    ops["do_registration"] = False

    # Cell detection.
    ops['max_overlap'] = 0.75
    try:
        pixel_size = s.attrs["capture"]['pixel_size'][0]
    except:
        pixel_size = s.attrs["capture"]["frame"]["shape"][0] / \
                     s.attrs["capture"]["frame"]["size"][0]
    ops["diameter"] = int(20 / pixel_size)

    suite2p.run_s2p(ops=ops, db={})

    if cleanup:
        cleanup_segmentation_results(s)


def cleanup_segmentation_results(
    s: "Session",
    backup: Optional[Union[str, Sequence[str]]] = 'iscell.npy',
) -> None:
    """
    Move suite2p results into 'suite2p' directory.
    
    ``cleanup_results(session)``
    
    """

    # Move results ('F.npy', 'ops.npy', etc.) into main session folder.
    s2pdir = Path(s.fs.getsyspath('suite2p'))
    planedir = s2pdir / 'plane0'
    for f in planedir.glob('*.npy'):
        f.rename(s2pdir / f.name)

    # Remove 'plane0' directory.
    s.fs.removedir('suite2p/plane0')

    # Backup 'iscell.npy' and anything else specified by the 'backup' kwarg.
    if backup is not None:
        if isinstance(backup, str):
            backup = [backup]
        backupdir = s2pdir / "backup"
        backupdir.mkdir(exist_ok=True)
        for fname in backup:
            src = s2pdir / fname
            dest = backupdir / fname
            try:
                shutil.copyfile(src, dest)
            except FileNotFoundError:
                pass
