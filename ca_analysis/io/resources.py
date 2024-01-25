
import os
from collections import UserDict
from pathlib import Path
from typing import (
    Optional,
    Union,
)

from fs.base import FS

from .common import *

__all__ = [
    "Resource",
    "ResourceCatalog",
]




class Resource:

    _parent: "ResourceCatalog"
    _name: str
    _url: URL
    _path: str

    def __init__(
        self,
        parent: "ResourceCatalog",
        name: str,
        url: PathLike,
    ):
        self._parent = parent
        self._name = name
        self._url = URL(url)


    @property
    def parent(self) -> "ResourceCatalog":
        return self._parent

    @property
    def name(self) -> str:
        return self._name

    @property
    def url(self) -> URL:
        return self._url

    @property
    def path(self) -> Path:
        return self._url.path

    def abspath(self) -> Path:
        if os.path.isabs(self.path):
            return Path(self.path)
        return Path(self._parent.fs.getsyspath(self.path))

    resolve = abspath

    def exists(self) -> bool:
        return self.abspath().exists()

    def is_dir(self) -> bool:
        return self.abspath().is_dir()

    def is_file(self) -> bool:
        return self.abspath().is_file()

    def mkdir(self, *args, **kw) -> None:
        self.abspath().mkdir(*args, **kw)

    def __fspath__(self) -> str:
        return str(self.abspath())

    def __str__(self) -> str:
        return str(self.abspath())


class ResourceCatalog(UserDict):

    def __init__(self, fs: Optional["FS"]):
        self.fs = fs
        self.data = {}

    def add(
        self,
        name: str,
        obj: Union[PathLike, Resource],
    ) -> None:
        if isinstance(obj, Resource):
            obj = Resource(self, obj.name, obj.path)
        else:
            obj = Resource(self, name, obj)

        self.data[name] = obj

    def __getattr__(self, key) -> Resource:
        return self[key]

    def __setitem__(self, key: str, val: Union[PathLike, Resource]) -> None:
        self.add(key, val)

