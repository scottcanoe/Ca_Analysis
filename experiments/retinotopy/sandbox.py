from main import *
from processing import *
from ca_analysis.processing.utils import fix_frame_trigger
from ca_analysis.processing.utils import pull_session, push_session
import matplotlib.pyplot as plt

import fs as pyfs
from fs.errors import *


def do_import(mouse, date, run="1"):
    path = f"sessions/{mouse}/{date}/{run}"
    pyfs.copy.copy_dir_if(
        get_fs(-1),
        path,
        get_fs(0),
        path,
        'newer',
        preserve_time=True,
    )
    s = open_session(mouse, date, run, fs=0, require=True)
    import_session(s)

    pyfs.copy.copy_dir_if(
        get_fs(0),
        path,
        get_fs(-1),
        path,
        'newer',
        preserve_time=True,
    )
    s.fs.rmtree("thorlabs", missing_ok=True)
    s.fs.remove("mov.h5")


do_import("71765-1", "2023-03-09")
