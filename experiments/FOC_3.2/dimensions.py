import copy
from typing import (
    Any,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ca_analysis.common import *


def validate_dims(obj: Sequence) -> Tuple[str]:
    """
    Checks that `obj` is a sequence of unique strings.

    Parameters
    ----------
    obj

    Returns
    -------
    valid tuple of strings
    """

    dims = np.asarray(obj)
    if not is_str_array(dims, ndim=1):
        msg = "dimension names must be a 1d sequence of unique strings: {}"
        raise ValueError(msg.format(obj))
    return tuple(dims)


def validate_shape(obj: Sequence) -> Tuple[Optional[int]]:
    """
    Checks that `obj` is a sequence of non-negative integers or `None`.

    Parameters
    ----------
    obj

    Returns
    -------

    """
    shape = tuple(obj)
    msg = "shape must be a 1d sequence of non-negative ints or `None`: {}"
    if np.ndim(shape) != 1:
        raise ValueError(msg.format(obj))
    for size in shape:
        if not (size is None or (is_int(size) and size >= 0)):
            raise ValueError(msg.format(obj))
    return shape


def validate_dims_and_shape(
    dims: Sequence,
    shape: Sequence,
) -> Tuple[Tuple[str], Tuple[int]]:
    """
    Checks that dimension names and shape are valid and have the same length.

    Parameters
    ----------
    dims
    shape

    Returns
    -------

    """
    dims = validate_dims(dims)
    shape = validate_shape(shape)
    if len(dims) != len(shape):
        msg = f'number of dimensions does not match shape: {dims}, {shape}'
        raise ValueError(msg)
    return dims, shape


class Dimensions:

    #: tuple of dimension names
    _dims: Tuple[str]

    #: tuple of dimension sizes
    _shape: Tuple[int]

    #: mapping from dimension names to dimension sizes
    _sizes: Mapping[str, Union[Optional[int]]]

    __slots__ = ("_dims", "_shape", "_sizes")

    def __init__(
        self,
        dims: Optional[Union[Sequence[str], "Dimensions"]] = None,
        shape: Optional[Sequence[int]] = None,
    ):
        if isinstance(dims, Dimensions):
            if shape is None:
                dims, shape = dims.dims, dims.shape
            else:
                dims, shape = validate_dims_and_shape(dims.dims, shape)
        else:
            if dims is None:
                if shape is None:
                    dims, shape = tuple(), tuple()
                else:
                    shape = validate_shape(shape)
                    dims = tuple(f"dim_{i}" for i in range(len(shape)))
            elif shape is None:
                shape = (None,) * len(dims)
            dims, shape = validate_dims_and_shape(dims, shape)

        self._dims = dims
        self._shape = shape
        self._sizes = FrozenDict(dict(zip(self._dims, self._shape)))

    @property
    def dims(self) -> Tuple[str]:
        return self._dims

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def sizes(self) -> Mapping[Any, Optional[int]]:
        return self._sizes

    @property
    def ndim(self) -> int:
        return len(self._dims)

    def as_axis(self, val: Union[int, str]) -> int:
        """
        Go from dimension name to axis number, or just return the axis number
        if the argument already was one. Note that it will be normalized (i.e,
        negative indices will be resolved).

        Parameters
        ----------
        val

        Returns
        -------
        axis: int
        """
        return self.dims.index(val) if is_str(val) else range(self.ndim)[val]

    def as_dim(self, val: Union[int, str]) -> str:
        """
        Go from axis number to dimension name, or just return the dimension
        name if the argument was one.

        Parameters
        ----------
        val

        Returns
        -------
        dim: str
        """
        return self.dims[self.as_axis(val)]

    def copy(self) -> "Dimensions":
        return copy.copy(self)

    def drop(self, dims_or_axes: Union[int, str, List[Union[int, str]]]) -> None:
        """
        Return a dimensions object with one or more dimensions removed.
        Accepts strings for dimension names or ints for axes.

        Parameters
        ----------
        dims_or_axes

        Returns
        -------

        """

        dims_to_drop = tuple(self.as_dim(val) for val in dims_or_axes)
        dims = tuple(val for val in self.dims if val not in dims_to_drop)
        shape = tuple(self.sizes[key] for key in dims)
        return Dimensions(dims, shape)

    def drop_nones(self) -> "Dimensions":
        """
        Return a dimensions object without `None` dimensions.

        Returns
        -------

        """
        if None in self.shape:
            dims = tuple(key for key, val in self.sizes.items() if val is not None)
            shape = tuple(self.sizes[key] for key in dims)
            return Dimensions(dims, shape)
        return self.copy()

    def expand_dims(
        self,
        name: str,
        axis: int,
        size: Optional[int] = None,
    ) -> "Dimensions":

        dims = list(self.dims)
        dims.insert(axis, name)
        shape = list(self.shape)
        shape.insert(axis, size)
        return Dimensions(dims, shape)

    def get_nones(self) -> Tuple[str]:
        """
        Return a dimensions object from only dimensions with no size.

        Returns
        -------

        """
        if None in self._shape:
            dims = tuple(key for key, val in self._sizes.items() if val is None)
            shape = tuple(self.sizes[key] for key in dims)
            return Dimensions(dims, shape)
        return self.copy()

    def get_transposition(
        self,
        obj: Union[Sequence] = None,
    ) -> "Transpose":

        axes = np.array([self.as_axis(val) for val in obj])
        if not np.array_equal(np.sort(axes), np.arange(self.ndim)):
            raise ValueError(f'invalid transposition: {obj}')
        dims = tuple(self.dims[ax] for ax in axes)
        return Transpose(dims, axes)

    def index(self, key: str) -> int:
        return self.dims.index(key)

    def squeeze(self, drop_none: bool = False) -> "Dimensions":
        dims, shape = [], []
        for i, size in enumerate(self.shape):
            if size is None:
                if drop_none:
                    continue
            elif size < 2:
                continue
            dims.append(self.dims[i])
            shape.append(size)
        return Dimensions(dims, shape)

    def transpose(self, *args) -> "Dimensions":
        n_args = len(args)
        if n_args == 0:
            dims = list(reversed(self.dims))
            shape = list(reversed(self.shape))
            return Dimensions(dims, shape)
        if n_args == 1:
            obj = args[0]
            if is_str(obj):
                obj = [obj]
        else:
            if args[0] is ...:
                obj = list(self.dims)
                for name in args[1:]:
                    obj.remove(name)
                obj.extend(args[1:])
            elif args[-1] is ...:
                obj = list(self.dims)
                for name in args[-1]:
                    obj.remove(name)
                obj = list(args[1:]) + obj
            else:
                obj = args

        axes = np.array([self.as_axis(val) for val in obj])
        if not np.array_equal(np.sort(axes), np.arange(self.ndim)):
            raise ValueError(f'invalid transposition: {obj}')
        dims = tuple(self.dims[ax] for ax in axes)
        shape = tuple(self.shape[ax] for ax in axes)
        return Dimensions(dims, shape)

    @classmethod
    def from_shape(cls, shape: Sequence[int]) -> "Dimensions":
        lst = [(f"dim_{i}", dimlen) for i, dimlen in enumerate(shape)]
        return Dimensions(lst)

    def __contains__(self, key: Union[int, str]) -> bool:
        return key in self.dims

    def __eq__(self, other: "Dimensions") -> bool:
        return self.dims == other.dims and self.shape == other.shape

    def __getitem__(self, key: Union[int, str]) -> Union[int, str]:
        return self.dims[key] if is_int(key) else self._dims.index(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.dims)

    def __len__(self) -> int:
        return self.ndim

    def __repr__(self) -> str:
        sizes = self.sizes
        lst = [f"{dim}: {sizes[dim]}" for dim in self.dims]
        s = "Dimensions(" + ", ".join(lst) + ")"
        return s




class Transpose:
    """
    Class for performing transpose operations on numpy or xarray arrays.
    """
    def __init__(
        self,
        dims: Optional[Sequence[str]] = None,
        axes: Optional[Sequence[int]] = None,
    ):
        self._dims = None if dims is None else tuple(dims)
        self._axes = None if axes is None else tuple(axes)
        self._inv = None

    @property
    def inverse(self) -> "Transpose":
        if self._inv is None:
            dims, axes = None, None
            if self._dims:
                dims = [None] * len(self._dims)
                for i, ax in enumerate(self._axes):
                    dims[ax] = self._dims[i]
            if self._axes:
                axes = [None] * len(self._axes)
                for i, ax in enumerate(self._axes):
                    axes[ax] = i
            self._inv = Transpose(dims, axes)
            self._inv._inv = self
        return self._inv

    @property
    def inv(self) -> "Transpose":
        return self.inverse

    def __call__(self, obj: ArrayLike):
        if hasattr(obj, "dims"):
            return obj.transpose(*self._dims)
        obj = np.asarray(obj)
        return np.transpose(obj, self._axes)

    def __repr__(self) -> str:
        return f"<Transpose: dims={self._dims}, axes={self._axes}>"
