
from .common import *

from .io import *
from intake.source.discovery import drivers, load_plugins_from_module
dct = load_plugins_from_module('ca_analysis.io')
for key, val in dct.items():
    drivers.register_driver(key, val)
del dct, drivers, load_plugins_from_module

import xarray as xr
xr.set_options(keep_attrs=True)

# setup configurations, io, etc.
from .environment import *

# utilities
from .indexing import *
from .ndimage import *
from .resampling import *

# interfaces
from .event import *

from .roi import *
from .session import *

from .utils import *




