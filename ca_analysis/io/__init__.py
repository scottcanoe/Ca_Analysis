# import intake
# from intake.source.discovery import drivers, load_plugins_from_module
# dct = load_plugins_from_module('ca_analysis.io')
# for key, val in dct:
#     drivers.register_driver(key, val)
# intake.make_open_functions()

from .common import *
from ._fs_patch import *
from .h5 import *
from .thorlabs import *
from .utils import *
