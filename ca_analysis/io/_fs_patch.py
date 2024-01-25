"""
Patches the `fs` module's classes to extend functionality.
"""
import fnmatch
import os
from pathlib import Path
from typing import (
    Iterable,
    Mapping, Optional,
    Sequence,
    Tuple,
    Union,
)

import fs as pyfs
from fs.base import FS
from fs.copy import copy_dir_if, copy_file_if
from fs.error_tools import unwrap_errors
from fs.errors import *
from fs.info import Info
from fs.multifs import MultiFS
from fs.opener.parse import parse_fs_url as _parse_fs_url
from fs.osfs import OSFS
from fs.path import relpath
from fs.subfs import SubFS
from fs.wrapfs import WrapFS

import natsort

from ..common import *
from .common import *


__all__ = []

"""
--------------------------------------------------------------------------------------
- pathlib compatibility and behavior changes
"""



def _registry_open(
        fs_url: PathLike,
        writeable: bool = True,
        create: bool = False,
        cwd: PathLike = ".",
        default_protocol: str = "osfs",
) -> Tuple[FS, str]:
    """
    Wrapped to support path-like objects.

    Open a io from a FS URL.

    Returns a tuple of a io object and a path. If there is
    no path in the FS URL, the path value will be `None`.

    Arguments:
        fs_url (str): A io URL.
        writeable (bool, optional): `True` if the io must be
            writeable.
        create (bool, optional): `True` if the io should be
            created if it does not exist.
        cwd (str): The current working directory.

    Returns:
        (FS, str): a tuple of ``(<io>, <path from url>)``

    """

    fs_url = os.fspath(fs_url)
    cwd = os.fspath(cwd)
    if "://" not in fs_url:
        # URL may just be a path
        fs_url = "{}://{}".format(default_protocol, fs_url)

    parse_result = _parse_fs_url(fs_url)
    protocol = parse_result.protocol
    open_path = parse_result.path
    opener = pyfs.opener.registry.get_opener(protocol)
    opened_fs = opener.open_fs(fs_url, parse_result, writeable, create, cwd)
    return opened_fs, open_path


pyfs.opener.registry.open = pyfs.opener.open = _registry_open


"""
--------------------------------------------------------------------------------
 Info
--------------------------------------------------------------------------------
Gives Info object a reference to FS that created it. This enables Info objects 
to implement
- 'refresh()': a method to update info.
- `relpath`: a property that returns the info object's path relative to the 
             owning FS.
- `syspath`: a property that returns the corresponding system path.
- `__fspath__()`: method that implements `os.fspath` behavior. 

"""


def refresh(self) -> None:
    if "fs" not in self.namespaces:
        raise ValueError("needs 'fs' namespace")
    new_info = self.fs.getinfo(self.path, namespaces=self.namespaces)
    self.raw = new_info.raw


def __fspath__(self) -> str:
    return os.fspath(self.syspath)



# attributes/properties
Info.fs = property(fget=lambda self: self.raw["fs"]["fs"])
Info.relpath = property(fget=lambda self: self.raw["fs"]["relpath"])
Info.syspath = property(fget=lambda self: self.raw["fs"]["syspath"])

# methods
Info.refresh = refresh
Info.__fspath__ = __fspath__


"""
------------------------------------------------------------------------------------------------
 FS (base class)
------------------------------------------------------------------------------------------------
Adds methods to the base class that provide a pathlib/os-like interface.
  - mkdir()
  - rm()
  - rmdir()
  - rmtree()
  
Also adds a `name` attribute with default value `None`.
 
"""


def cleardir(self, path: PathLike, missing_ok: bool = False) -> None:
    """
    Delete the contents of a directory.

    Parameters
    ----------
    self
    path
    missing_ok

    Returns
    -------

    """
    try:
        subfs = self.opendir(path)
    except ResourceNotFound as exc:
        if missing_ok:
            return
        raise exc

    for name in subfs.listdir(""):
        subfs.delete(name)


def delete(self, path: PathLike, missing_ok: bool = False) -> None:
    if self.isdir(path):
        self.rmtree(path, missing_ok=missing_ok)
    else:
        self.rm(path, missing_ok=missing_ok)


def ls(self, path: PathLike = "") -> None:

    fnames = self.listdir(path)
    fnames = natsort.natsorted(fnames)
    return fnames


def mkdir(
        self,
        path: PathLike,
        *args,
        exist_ok: bool = False,
        parents: bool = False,
        **kw,
) -> FS:
    """
    Pathlib-like interface to directory creation/opening.
    """

    if self.exists(path):
        if exist_ok:
            return self.opendir(path, *args, **kw)
        return self.makedir(path, *args, **kw)
    if parents:
        return self.makedirs(path, *args, **kw)
    return self.makedir(path, *args, **kw)


def rm(self, path: PathLike, missing_ok: bool = False) -> None:
    try:
        self.remove(path)
    except ResourceNotFound as exc:
        if not missing_ok:
            raise exc


def rmdir(self, path: PathLike, missing_ok: bool = True) -> None:
    try:
        self.removedir(path)
    except ResourceNotFound as exc:
        if not missing_ok:
            raise exc


def rmtree(self, path: PathLike, missing_ok: bool = True) -> None:
    try:
        self.removetree(path)
    except ResourceNotFound as exc:
        if not missing_ok:
            raise exc


@property
def syspath(self) -> Path:
    try:
        return self._syspath
    except AttributeError:
        self._syspath = Path(self.getsyspath(""))
        return self._syspath


def __truediv__(self, path: PathLike) -> Path:
    return self.syspath / path


# attributes/properties
FS.name = None

# methods
FS.cleardir = cleardir
FS.delete = delete
FS.ls = ls
FS.mkdir = mkdir
FS.rm = rm
FS.rmdir = rmdir
FS.rmtree = rmtree
FS.syspath = syspath
FS.__truediv__ = __truediv__
WrapFS.__truediv__ = __truediv__


"""
WrapFS methods

"""


def wrapfs_method(method_name: str):
    """
    Decorator that handles delegate legwork before calling delegate fs.
    """
    def wrapped(self, path: PathLike, *args, **kw):

        self.check()
        _fs, _path = self.delegate_path(path)
        with unwrap_errors(path):
            method = getattr(_fs, method_name)
            return method(_path, *args, **kw)

    return wrapped


def as_multifs(self, require: bool = False):
    self.check()
    _fs = self.delegate_fs()
    _path = self._sub_dir
    mfs = MultiFS()
    for fs_name, fs_branch in _fs.iterate_fs():
        if fs_branch.isdir(_path):
            subfs = fs_branch.opendir(_path)
        else:
            if require:
                subfs = fs_branch.makedirs(_path)
            else:
                continue

        info = self.get_fs_info(fs_name)
        mfs.add_fs(info["name"], subfs, priority=info["priority"], write=info["write"])

    return mfs


def get_fs(self, val, require: bool = False):
    """
    For compatibility with MultiFS
    """
    _fs = self.delegate_fs()
    if not _fs.exists(self._sub_dir) and require:
        out = _fs[val].makedirs(self._sub_dir)
    else:
        out = _fs[val].opendir(self._sub_dir)
    return out


def get_fs_info(self, val: Union[int, str]) -> Mapping:
    """
    For compatibility with MultiFS
    """
    self.check()
    _fs = self.delegate_fs()
    info = _fs.get_fs_info(val)
    return info


def getinfo(self, path, namespaces=None) -> Info:

    self.check()
    _fs, _path = self.delegate_path(path)
    with unwrap_errors(path):
        info = _fs.getinfo(_path, namespaces=namespaces)

    if "fs" in info.namespaces:
        info.raw["fs"]["fs"] = self
        info.raw["fs"]["relpath"] = relpath(path)
        info.raw["fs"]["syspath"] = Path(_fs.getsyspath(_path))
    return info


def get_root_fs(self) -> Tuple[FS, str]:
    parent = self.delegate_fs()
    parts = [self._sub_dir]
    while isinstance(parent, SubFS):
        parts.append(parent._sub_dir)
        parent = parent.delegate_fs()
    sub_dir = "".join(reversed(parts))
    return parent, sub_dir


def iterate_fs(self):
    root_fs, sub_dir = self.get_root_fs()
    fs_names = [key for key, _ in root_fs.iterate_fs()]
    for name in fs_names:
        try:
            subfs = self[name]
            yield name, subfs
        except ResourceNotFound:
            continue


@property
def write_fs(self):
    """
    For compatibility with MultiFS
    """

    _fs = self.delegate_fs()
    return _fs.write_fs


def __getitem__(self, key):
    """
    For compatibility with MultiFS
    """
    return self.get_fs(key)


SubFS.as_multifs = as_multifs
SubFS.get_fs = get_fs
SubFS.get_fs_info = get_fs_info
SubFS.get_root_fs = get_root_fs
SubFS.iterate_fs = iterate_fs
SubFS.write_fs = write_fs
SubFS.__getitem__ = __getitem__

WrapFS.cleardir = wrapfs_method("cleardir")
WrapFS.delete = wrapfs_method("delete")
WrapFS.getinfo = getinfo
WrapFS.ls = wrapfs_method("ls")
WrapFS.mkdir = wrapfs_method("mkdir")
WrapFS.rm = wrapfs_method("rm")
WrapFS.rmdir = wrapfs_method("rmdir")
WrapFS.rmtree = wrapfs_method("rmtree")



"""
------------------------------------------------------------------------------------------------
 OSFS
------------------------------------------------------------------------------------------------
  - Modifies `getinfo()` so that `Info` objects have references to the `FS` instance that created it.  
  - Adds a `root` property that returns an absolute pathlib.Path object to the OSFS's directory. 
    
"""


_getinfo_orig = OSFS.getinfo


def getinfo(
        self: OSFS,
        path: PathLike,
        namespaces: Optional[Sequence[str]] = None,
) -> Info:

    _path = self.validatepath(path)
    namespaces = namespaces or ()
    info = _getinfo_orig(self, _path, namespaces=namespaces)
    if "fs" in namespaces:
        info.raw["fs"] = info.raw.get("fs", {})
        info.raw["fs"]["fs"] = self
        info.raw["fs"]["relpath"] = relpath(_path)
        info.raw["fs"]["syspath"] = Path(self.getsyspath(_path))
    return info


# attributes/properties
OSFS.root = OSFS.syspath

# methods
OSFS.getinfo = getinfo


"""
------------------------------------------------------------------------------------------------
 MultiFS
------------------------------------------------------------------------------------------------

"""


def as_multifs(self, require: bool = False) -> MultiFS:
    return self


def get_fs(self, val: Union[str, int], require: bool = False) -> FS:

    if is_str(val):
        return self._filesystems[val].fs
    return list(self.iterate_fs())[val][1]


def get_fs_info(self, val: Union[str, int]) -> Mapping:

    if isinstance(val, FS):
        name = None
        for fs_name, fs in self.iterate_fs():
            if fs == val:
                name = fs_name
                break
        if name is None:
            raise ValueError(f'{val} is not a member of this MultiFS')
        val = name

    if is_int(val):
        index = val
        name = list(self.iterate_fs())[index][0]
    elif is_str(val):
        name = val
        index = [key for key, _ in self.iterate_fs()].index(name)
    else:
        raise ValueError(f'invalid fs argument: {val}')

    info = {
        "index": index,
        "name": name,
        "priority": self._filesystems[name].priority,
        "write": self._filesystems[name].fs == self.write_fs,
    }
    return info


def get_root_fs(self) -> Tuple[FS, str]:
    return self, ""


def set_write_fs(self, val: Optional[Union[int, str, FS]]) -> None:
    if val is None:
        fs = None
    elif isinstance(val, FS):
        assert val in self._filesystems.values()
        fs = val
    else:
        fs = self.get_fs(val)
    self.write_fs = fs


def __getitem__(self, val: Union[str, int]) -> FS:
    return self.get_fs(val)


MultiFS.as_multifs = as_multifs
MultiFS.get_fs = get_fs
MultiFS.get_fs_info = get_fs_info
MultiFS.get_root_fs = get_root_fs
MultiFS.set_write_fs = set_write_fs
MultiFS.__getitem__ = __getitem__

