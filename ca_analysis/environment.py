import copy
import logging
import os
from collections import ChainMap, UserDict
from pathlib import Path
from typing import (
    Any,
    List,
    Mapping,
    Optional,
    Union,
)

import yaml
from fs import open_fs
from fs.base import FS
from fs.errors import CreateFailed
from fs.multifs import MultiFS

from .common import *

__all__ = [
    "get_config",
    "get_fs",
    "open_tempdir",
]

"""
-------------------------------------------------------------------------------------
- module-level variables
"""

# config
_config = None
_userdir = Path.home() / ".ca_analysis"
_user_config_defaults = {
    "filesystems": {},
}

# logging
_logger = None

# io
_filesystem = None


"""
-------------------------------------------------------------------------------------
- Config
"""


class UserConfig(UserDict):
    """
    An immutable wrapper around dictionaries that implements the complete :py:class:`collections.Mapping`
    interface. It can be used as a drop-in replacement for dictionaries where immutability is desired.
    
    Wraps a chain map with two underlying maps. The lowest priority are the default
    settings, and the highest priority is loaded from '~/.ca_analysis/config.yaml'.
    
    """

    def __init__(self):

        self.data = ChainMap({}, copy.deepcopy(_user_config_defaults))
        try:
            self.load()
        except:
            logger = logging.getLogger("ca_analysis")
            logger.warning("unable to load user config file")

    @property
    def maps(self) -> List[Mapping]:
        return self.data.maps

    def load(self, path: Optional[PathLike] = None) -> None:

        if path:
            path = _userdir / path
        else:
            path = _userdir / "config.yaml"

        if not path.exists() and path.with_suffix(".yaml").exists():
            path = path.with_suffix(".yaml")

        with open(path, "r") as f:
            d = yaml.load(f, Loader=yaml.Loader)
            self.update(d)

    def save(self, path: Optional[PathLike] = None) -> None:

        if path:
            path = _userdir / path
        else:
            path = _userdir / "config.yaml"

        if not path.suffix:
            path = path.with_suffix(".yaml")

        data = dict(self.data.maps[0])
        with open(path, "w") as f:
            yaml.dump(data, f)


def get_config(key: Optional[str] = None) -> Any:
    global _config
    if _config is None:
        _config = UserConfig()
    if key is None:
        return _config
    return _config[key]


"""
-------------------------------------------------------------------------------------
- Logging
"""


_logger = logging.getLogger("ca_analysis")
_logger.addHandler(logging.StreamHandler())
if _userdir.exists():
    filename = (_userdir / "logs") / os.fspath("ca_analysis.log")
    filename.parent.mkdir(exist_ok=True, parents=True)
    _logger.addHandler(logging.FileHandler(filename))
    del filename
formatter = logging.Formatter(
    fmt='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)

for handler in _logger.handlers:
    handler.setFormatter(formatter)


"""
------------------------------------------------------------------------------------------
- io
"""


def get_fs(fs: Optional[Union[int, str]] = None) -> FS:
    global _filesystem

    # create the master io if necessary
    if _filesystem is None:
        _filesystem = MultiFS()

        config = get_config("filesystems")
        lst_1, lst_2 = [], []
        for entry in config:
            entry = entry.copy()
            p = entry.pop("priority", None)
            if p is None:
                lst_2.append(entry)
            else:
                entry["priority"] = p
                lst_1.append(entry)

        lst_1 = sorted(lst_1, key=lambda elt: elt["priority"], reverse=True)
        for elt in lst_1:
            elt.pop("priority")

        lst_2 = list(reversed(lst_2))
        entries = lst_2 + lst_1

        if entries:
            entries[-1]["write"] = True

        for entry in entries:
            name = entry.pop("name")
            root = entry.pop("root")
            try:
                obj = open_fs(root)
            except CreateFailed:
                msg = f"unable to access io '{name}'"
                _logger.warning(msg)
                continue

            _filesystem.add_fs(name, obj, **entry)
        _filesystem.write_fs = _filesystem[0]

    if fs is None:
        return _filesystem

    return _filesystem[fs]


def open_tempdir(fs: Union[int, str] = 0) -> "FS.base.FSBase":
    fs = get_fs(fs)
    if fs.is_dir("temp"):
        return fs.opendir("temp")
    return fs.makedir("temp")

