import abc
import datetime
from numbers import Number
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pint
from numpy.typing import ArrayLike, DTypeLike, NDArray
from tzlocal import get_localzone

__all__ = [

    # - types
    "Array",
    "ArrayLike",
    "DTypeLike",
    "NDArray",
    "Number",
    "Path",
    "PathLike",

    # - scalar typing
    "is_bool",
    "is_int",
    "is_float",
    "is_number",
    "is_real",
    "is_str",

    # - array typing
    "is_array",
    "is_str_array",

    # singletons
    "MISSING",

    # unit handling
    "units",
    "format_bytes",
    "with_units",

    # time
    "as_date",
    "as_datetime",
    "format_timedelta",

    # etc.
    "FrozenDict",
]

"""
--------------------------------------------------------------------------------
- types
"""


@runtime_checkable
class Array(Protocol):

    ndim: int
    shape: Tuple[int]
    size: int

    def __array__(self) -> NDArray: ...

    def __getitem__(self) -> NDArray: ...


"""
--------------------------------------------------------------------------------
- scalar typing
"""


# - scalars

def _maybe_unwrap_generic(obj: Any) -> Any:
    return obj.item() if isinstance(obj, np.generic) else obj


def is_bool(obj: Any) -> bool:
    """Determine if an object is a boolean (including numpy booleans)"""
    return isinstance(_maybe_unwrap_generic(obj), bool)


def is_int(obj: Any) -> bool:
    """Determine if an object is an integer (including numpy integers)"""
    return isinstance(_maybe_unwrap_generic(obj), int)


def is_float(obj: Any) -> bool:
    """Determine if an object is floating point (including numpy floats)"""
    return isinstance(_maybe_unwrap_generic(obj), float)


def is_number(obj: Any) -> bool:
    """Determine if an object is a real-number (including numpy integers and floats)"""
    return isinstance(_maybe_unwrap_generic(obj), Number)


def is_real(obj: Any) -> bool:
    """Determine if an object is a real-number (including numpy integers and floats)"""
    return np.isreal(obj)


def is_str(obj: Any) -> bool:
    """Determine if an object is a real-number (including numpy integers and floats)"""
    return isinstance(_maybe_unwrap_generic(obj), str)


# - arrays


def is_array(
    obj: Any,
    *,
    shape: Optional[Sequence[int]] = None,
    ndim: Optional[int] = None,
    dtype: Optional[DTypeLike] = None,
    kind: Optional[DTypeLike] = None,
) -> bool:
    """
    Check whether an object is array-like and optionally check
    its shape/dimensionality/dtype/kind. If a dtype is given, then the
    array's dtype must match exactly. For more a more permissive check,
    use `kind`.

    Parameters
    ----------
    obj
    shape
    ndim
    dtype
    kind

    Returns
    -------

    """

    if not isinstance(obj, Array):
        return False

    if shape is not None and obj.shape != tuple(shape):
        return False

    if ndim is not None and obj.ndim != ndim:
        return False

    if dtype is not None:
        dtype = np.dtype(dtype)
        if np.dtype(dtype) != obj.dtype:
            return False

    if kind is not None:
        kind = np.dtype(kind).kind
        if obj.dtype.kind != kind:
            return False

    return True


def is_str_array(
    obj: Any,
    *,
    shape: Optional[Sequence[int]] = None,
    ndim: Optional[int] = None,
) -> bool:
    """
    Determine whether an object is a 1-d string array. The strings may
    be native python objects or numpy unicode.
    """

    # basic check
    if not is_array(obj, shape=shape, ndim=ndim):
        return False

    if obj.dtype == str or obj.dtype.kind == "U":
        return True

    # handle object-typed string arrays
    if obj.dtype.kind == "O":
        if len(obj) == 0:
            return False
        if not is_str(obj[0]):
            return False
        try:
            return np.array_equal(obj, obj.astype(str))
        except (TypeError, ValueError):
            return False

    return False


"""
--------------------------------------------------------------------------------
- singletons
"""


class MISSING(abc.ABC):
    pass


"""
--------------------------------------------------------------------------------
- units
"""

units = pint.UnitRegistry()


def format_bytes(size: Union[int, pint.Quantity]) -> str:
    size = with_units(size, 'byte')
    if size.m < 10 ** 3:
        return f"{size.m} bytes"
    elif size.m < 10 ** 6:
        return f"{size.to('kilobyte').m} kilobytes"
    elif size.m < 10 ** 9:
        return f"{size.to('megabyte').m:.2f} MB"
    elif size.m < 10 ** 12:
        return f"{size.to('gigabyte').m:.2f} GB"
    else:
        return f"{size.to('terabyte').m:.2f} TB"


def with_units(obj: Any, u: Union[str, pint.Unit]) -> pint.Quantity:
    u = str(u) if isinstance(u, pint.Unit) else u
    if isinstance(obj, pint.Quantity):
        return obj.to(u)
    return obj * units[u]


"""
--------------------------------------------------------------------------------
- files
"""

"""
--------------------------------------------------------------------------------
- time
"""

TZLOCAL = get_localzone()


def as_date(
    obj: Union[Number, str, datetime.date, datetime.datetime],
    **kw,
) -> datetime.date:
    """
    Coerce a string, date/datetime object or seconds since epoch into a date.

    Parameters
    ----------
    obj
    kw

    Returns
    -------

    """
    if isinstance(obj, datetime.date):
        d = obj
    elif isinstance(obj, datetime.datetime):
        d = obj.date
    elif isinstance(obj, str):
        obj = obj.replace("/", "-")
        parts = obj.split(maxsplit=1)
        first, rest = parts[0], parts[1:]
        y, m, d = first.split("-")
        if len(m) < 2:
            m = "0" + m
        if len(d) < 2:
            d = "0" + d
        parts[0] = f"{y}-{m}-{d}"
        obj = " ".join(parts)
        d = datetime.date.fromisoformat(obj)
    else:
        d = datetime.datetime.fromtimestamp(obj).date

    tz = kw.get("tz", kw.get("tzinfo", None))
    if tz:
        if isinstance(tz, str) and tz.lower() == "sandbox":
            tz = TZLOCAL
        d = d.replace(tzinfo=tz)
    return d


def as_datetime(
    obj: Union[Number, str, datetime.date, datetime.datetime],
    **kw,
) -> datetime.datetime:
    """
    Coerce a string, date/datetime object or seconds since epoch into a datetime.

    Parameters
    ----------
    obj
    kw

    Returns
    -------

    """

    if isinstance(obj, datetime.date):
        dt = datetime.datetime.fromisoformat(str(obj))
    elif isinstance(obj, datetime.datetime):
        dt = obj
    elif isinstance(obj, str):
        obj = obj.replace("/", "-")
        parts = obj.split(maxsplit=1)
        first, rest = parts[0], parts[1:]
        y, m, d = first.split("-")
        if len(m) < 2:
            m = "0" + m
        if len(d) < 2:
            d = "0" + d
        parts[0] = f"{y}-{m}-{d}"
        obj = " ".join(parts)
        dt = datetime.datetime.fromisoformat(obj)
    else:
        dt = datetime.datetime.fromtimestamp(obj)

    tz = kw.get("tz", kw.get("tzinfo", None))
    if tz:
        if isinstance(tz, str) and tz.lower() == "sandbox":
            tz = TZLOCAL
        dt = dt.replace(tzinfo=tz)
    return dt


def format_timedelta(
    t: Union[Number, datetime.timedelta, "Quantity"],
) -> str:
    """
    Create a human-readable/informative string for a given number of seconds.
    """

    def repr_one(t: units.Quantity, n_digits: int = 0) -> str:
        suffix = str(t.u) if np.isclose(t.m, 1) else str(t.u) + "s"
        if np.isclose(t.m, int(t.m)):
            return "{} {}".format(int(t.m), suffix)
        return ("{:." + str(n_digits) + "f} {}").format(t.m, suffix)

    if isinstance(t, units.Quantity):
        t = t.to("second")
    elif isinstance(t, datetime.timedelta):
        t = t.total_seconds() * units("sec")
    else:
        t = t * units("sec")

    strings = []
    remainder = t
    for period in ("year", "month", "day", "hour", "minute"):
        val = np.floor(remainder.to(period))
        if val.m > 0 or len(strings):
            strings.append(repr_one(val))
            remainder -= val

    if len(strings):
        strings.append(repr_one(remainder))
    elif remainder.m > 1e-9:
        strings.append(repr_one(remainder.to_compact(), n_digits=3))
    else:
        strings.append(repr_one(0, n_digits=3))

    return ", ".join(strings)



"""
--------------------------------------------------------------------------------
- etc
"""


class FrozenDict(dict):

    @classmethod
    def fromkeys(cls, *args, **kw) -> "FrozenDict":
        return FrozenDict(dict.fromkeys(*args, **kw))

    def _raise_type_error(self, *args, **kw) -> None:
        raise TypeError("frozendict is read-only")

    def __repr__(self) -> str:
        return "frozendict(" + super().__repr__() + ")"

    clear = _raise_type_error
    pop = _raise_type_error
    popitem = _raise_type_error
    setdefault = _raise_type_error
    update = _raise_type_error
    __setitem__ = _raise_type_error
    __delitem__ = _raise_type_error
