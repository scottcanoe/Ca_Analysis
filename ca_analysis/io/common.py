import copy
import glob
import json
import os
import urllib
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    List, Mapping, Optional,
    Union,
)

from ..common import *

__all__ = [
    "find_file",
    "find_files",
    "is_pathlike",
    "is_wildcard",
    "PathLike",
    "URL",
]


def find_file(
    pathname: PathLike,
    *,
    root_dir: Optional[PathLike] = None,
    recursive: bool = False,
    absolute: bool = False,
) -> str:
    """
    Return the unique path specified by `pathname` or raise `FileNotFoundError`.

    Parameters
    ----------
    pathname: path-like
        File/pattern to search for. If it is relative, `root_dir` must be
        specified and be absolute.
    root_dir: path-like, optional
        If not `None`, it  should be an absolute path-like object specifying
        the root directory for searching.
    recursive: bool, optional
        If `True`, the pattern “**” will match any files and zero or more
        directories, subdirectories and symbolic links to directories.
        If the pattern is followed by an `os.sep` or `os.altsep` then files
        will not match.
    absolute: bool, optional
        If `True`, always returns absolute paths. This overrides the default
        behavior in which relative paths are returned if `root_dir` was
        specified.

    Returns
    -------
    path: str
        Unique path matching the given pattern. If `root_dir` was specified,
        paths will be relative to `root_dir` unless `absolute` is `True`.

    Raises
    ------
    `ValueError`:
        Raised if pattern is not absolute after optionally accounting
        for `root_dir`.
    `FileNotFoundError`:
        Raised if 0 or >1 files found.
    """

    matches = find_files(
        pathname,
        root_dir=root_dir,
        recursive=recursive,
        absolute=absolute,
    )
    if len(matches) != 1:
        msg = f"{len(matches)} files found for pathname {pathname} with " \
              f"root_dir {root_dir}, recursive={recursive}"
        raise FileNotFoundError(msg)
    return matches[0]


def find_files(
    pathname: PathLike,
    *,
    root_dir: Optional[PathLike] = None,
    recursive: bool = False,
    absolute: bool = True,
) -> List[Union[str, Path]]:
    """
    Return a list of files that match a given pattern.


    Parameters
    ----------
    pathname: path-like
        File/pattern to search for. If it is relative, `root_dir` must be
        specified and be absolute.
    root_dir: path-like, optional
        If not `None`, it  should be an absolute path-like object specifying
        the root directory for searching.
    recursive: bool, optional
        If `True`, the pattern “**” will match any files and zero or more
        directories, subdirectories and symbolic links to directories.
        If the pattern is followed by an `os.sep` or `os.altsep` then files
        will not match.
    relative: bool, optional
        If `True` and `root_dir` was specified, will return paths to matching
        files relative to `root_dir` as strings.

    Returns
    -------
    paths: list[str]
        Zero or more paths matching the given pattern. If `root_dir` was
        specified, paths will be relative to `root_dir` unless `absolute`
        is `True`.

    Raises
    ------
    `ValueError`:
        Raised if pattern is not absolute after optionally accounting
        for `root_dir`.
    """

    path = Path(os.path.expanduser(pathname))
    root = Path(os.path.expanduser(root_dir)) if root_dir else None
    if root and not root.is_absolute():
        raise ValueError('root dir must be absolute')
    pattern = root / path if root else path
    if not pattern.is_absolute():
        raise ValueError("path must be absolute if root_dir not specified. "
                         "Otherwise root_dir must be absolute"
                         )

    # handle wildcard/glob patterns
    matches = glob.glob(str(pattern), recursive=recursive)

    if not absolute and root and root.is_absolute():
        matches = [os.path.relpath(p, root) for p in matches]
    else:
        matches = [Path(m) for m in matches]
    return matches


def is_pathlike(obj: Any) -> bool:
    try:
        os.fspath(obj)
        return True
    except (TypeError, ValueError):
        return False



WILDCARD_CHARS = frozenset("*?[]")


def is_wildcard(path: PathLike) -> bool:
    """Check if a path contains glob characters
    """
    return not WILDCARD_CHARS.isdisjoint(os.fspath(path))


class URL:
    """
    URL interface. Parses queries into dictionaries using JSON decoding (and
    unparses similarly). Created to make referring to datasets inside HDF5 files
    easier.

    - immutable
    - scheme, path, or query in kw will override anything parsed
    - other keyword args will be added to the query

    """

    __slots__ = (
        "_scheme",
        "_netloc",
        "_path",
        "_query",
        "_fragment",
    )

    _scheme: str
    _netloc: str
    _path: Path
    _query: Mapping
    _fragment: str

    def __init__(self, url: PathLike, **kw):

        if isinstance(url, URL):
            # closure
            scheme = kw.pop("scheme", url.scheme)
            netloc = kw.pop("netloc", url.netloc)
            path = kw.pop("path", url.path)
            query = kw.pop("query", url.query)
            fragment = kw.pop("fragment", url.fragment)
        else:
            # Parse a regular path-like object.
            parts = urllib.parse.urlparse(os.fspath(url))
            scheme = kw.pop("scheme", parts.scheme)
            netloc = kw.pop("netloc", parts.netloc)
            path = kw.pop("path", parts.path)
            query = kw.pop("query", parts.query)
            fragment = kw.pop("fragment", parts.fragment)

        self._scheme = scheme
        self._netloc = netloc
        self._path = str(path) if path else ""

        self._query = {}
        if query is None:
            pass
        elif is_str(query):
            self._query = self._decode_query(query)
        else:
            self._query.update(query)
        self._query.update(kw)

        self._fragment = fragment

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def netloc(self) -> str:
        return self._netloc

    @property
    def path(self) -> str:
        return self._path

    @property
    def query(self) -> Mapping[str, Any]:
        return self._query

    @property
    def fragment(self) -> str:
        return self._fragment

    @fragment.setter
    def fragment(self, val: Any):
        self._fragment = val

    def copy(self) -> "URL":
        return copy.deepcopy(self)

    def unparse(self) -> str:
        scheme = self._scheme if self._scheme else ""
        netloc = self._netloc if self._netloc else ""
        path = str(self._path) if self._path else ""
        query = self._encode_query()
        fragment = self._fragment if self._fragment else ""

        return urllib.parse.urlunparse(
            (scheme, netloc, path, "", query, fragment)
        )

    def _decode(self, val: str) -> Any:
        try:
            return json.loads(val)
        except json.decoder.JSONDecodeError:
            return val

    def _encode(self, val: Any) -> str:
        try:
            return json.dumps(val)
        except json.encoder.JSONEncoder:
            return str(val)

    def _decode_query(self, qs: str) -> dict:
        if qs:
            query = {}
            for item in qs.split("&"):
                key, val = item.split("=", maxsplit=1)
                query[key] = self._decode(val)
            return query
        return {}

    def _encode_query(self) -> str:
        return "&".join(
            [f"{key}={self._encode(val)}" for key, val in self._query.items()]
        )

    def __fspath__(self) -> str:
        return str(self._path)

    def __str__(self) -> str:
        return self.unparse()

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(scheme='{self.scheme}', " + \
            f"netloc='{self.netloc}', " + \
            f"path='{str(self.path)}', " + \
            f"query={self.query}, " + \
            f"fragment='{self.fragment}')"
        return s
