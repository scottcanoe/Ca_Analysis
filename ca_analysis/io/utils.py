from typing import (
    Any, Callable,
    Optional,
    Sequence,
    Union,
)

from fs.base import FS
import fs.copy
from fs.copy import _copy_is_necessary as copy_is_necessary
import yaml

from ..common import is_str, PathLike


__all__ = [
    "copy_is_necessary",
    "pull",
    "push",
    "read_yaml",
    "write_yaml",
]


def pull(
    src_fs: FS,
    dst_fs: FS,
    **kw,
) -> None:

    return push(dst_fs, src_fs, **kw)


def push(
    src_fs: FS,
    dst_fs: FS,
    filter: Optional[Union[str, Sequence[str]]] = None,
    exclude: Optional[Union[str, Sequence[str]]] = None,
    filter_dirs: Optional[Union[str, Sequence[str]]] = None,
    exclude_dirs: Optional[Union[str, Sequence[str]]] = None,
    condition: str = "newer",
    max_depth: Optional[int] = None,
    on_copy: Optional[Callable] = None,
) -> None:

    filter = [filter] if is_str(filter) else filter
    exclude = [exclude] if is_str(exclude) else exclude
    filter_dirs = [filter_dirs] if is_str(filter_dirs) else filter_dirs
    exclude_dirs = [exclude_dirs] if is_str(exclude_dirs) else exclude_dirs

    for step in src_fs.walk(
            filter=filter,
            exclude=exclude,
            filter_dirs=filter_dirs,
            exclude_dirs=exclude_dirs,
            max_depth=max_depth,
    ):
        for src_info in step.files:
            path = f"{step.path}/{src_info.name}"
            do_copy = copy_is_necessary(src_fs, path, dst_fs, path, condition)
            if do_copy:
                fs.copy.copy_file(src_fs, path, dst_fs, path)
                fs.copy.copy_modified_time(src_fs, path, dst_fs, path)
                if on_copy:
                    on_copy(src_fs, path, dst_fs, path)



def read_yaml(path: PathLike, **kw) -> Any:
    with open(path, "r") as f:
        out = yaml.load(f, Loader=yaml.Loader, **kw)
    return out


def write_yaml(path: PathLike, data: Any, sort_keys: bool = False, **kw) -> None:
    kw["sort_keys"] = sort_keys
    with open(path, "w") as f:
        yaml.dump(data, f, Dumper=yaml.Dumper, **kw)
