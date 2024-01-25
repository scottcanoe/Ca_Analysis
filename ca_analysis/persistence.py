"""
This module defines the `PersistentMapping` class, which is a dictionary backed by a YAML file.
It is used primarily for `Session.attrs`.

"""
import collections
import copy
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import glom
import yaml

from .common import *

__all__ = [
    "PersistentMapping",
]

_DEFAULT_LOAD_KW = {
}

_DEFAULT_SAVE_KW = {
    "sort_keys": False,
}


class PersistentMapping(collections.abc.Mapping):

    sort_keys: bool = False
    _is_loaded: bool = False
    _is_changed: bool = True

    def __init__(
        self,
        path: PathLike,
        load: bool = True,
        readonly: bool = False,
        save_on_close: bool = False,
        load_kw: Optional[Mapping] = None,
        save_kw: Optional[Mapping] = None,
    ):
        self._data = {}
        self._path = Path(path)
        self._readonly = readonly
        self.save_on_close = save_on_close
        self.load_kw = dict(_DEFAULT_LOAD_KW)
        if load_kw:
            self.load_kw.update(load_kw)
        self.save_kw = dict(_DEFAULT_SAVE_KW)
        if save_kw:
            self.save_kw.update(save_kw)

        if load and self.path.exists():
            self.load()

    @property
    def data(self) -> Mapping:
        return self._data

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_changed(self) -> bool:
        return self._is_changed

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def readonly(self) -> bool:
        return self._readonly

    def clear(self) -> None:
        self._data.clear()
        self._is_changed = True

    def close(self) -> None:
        if self.save_on_close:
            self.save()

    def copy(
        self,
        path: Optional[Any] = None,
        deep: bool = False,
    ) -> "PersistentMapping":

        out = copy.deepcopy(self) if deep else copy.copy(self)
        out._path = Path(path) if path else self.path
        return out

    def load(self, update: bool = False, merge: bool = False, **kw) -> None:

        kwargs = dict(self.load_kw)
        kwargs.update(kw)
        loader = kwargs.pop("Loader", yaml.Loader)
        with open(self.path, "r") as f:
            data = yaml.load(f, Loader=loader, **kwargs)

        if merge:
            self._data = glom.merge([self._data, data])
        elif update:
            self._data.update(data)
        else:
            self._data = data

        self._is_loaded = True
        self._is_changed = True

    def save(self, **kw) -> None:

        if self._readonly:
            raise OSError('read-only target')
        kwargs = dict(self.save_kw)
        kwargs.update(kw)
        dumper = kw.pop("Dumper", yaml.Dumper)
        with open(self.path, "w") as f:
            yaml.dump(self._data, f, Dumper=dumper, **kwargs)

        self._is_changed = False

    def merge(self, other: Mapping) -> None:
        self._ensure_loaded()
        self._data = glom.merge([self._data, other])
        self._is_changed = True

    def update(self, other: Mapping) -> None:
        self._ensure_loaded()
        self._data.update(other)
        self._is_changed = True

    def _ensure_loaded(self) -> None:
        if self._path and os.path.exists(self._path) and not self._is_loaded:
            self.load()

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __setitem__(self, key: Any, val: Any) -> None:
        self._data[key] = val
        self._is_changed = True

    def __delitem__(self, key: Any) -> Any:
        del self._data[key]
        self._is_changed = True

    def __contains__(self, key: Any):
        return key in self._data

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        s = json.dumps(self._data, indent=2, default=str)
        return s

    def __enter__(self) -> "PersistentMapping":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

