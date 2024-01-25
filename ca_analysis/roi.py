import abc
import copy
import json
import os
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import h5py
import numpy as np
import xarray as xr
from skimage.morphology import binary_dilation, binary_erosion

from .common import *
from .ndimage import PixelMask

__all__ = [
    "ROI",
    "ROIGroup",
]


class ROI:
    """
    Object interface to an roi's data.
    
    ``ROI(roi)``
    ``ROI(model, index)``
    
    """

    _parent: "ROIStore"
    _name: int
    _local: Mapping
    _mask: Optional[PixelMask]

    def __init__(self, *args):

        # If initialized with another ROI...
        if len(args) == 1:
            self._parent = args[0].parent
            self._name = args[0].name

        # If initialized with a storage backend and an index...
        elif len(args) == 2:
            self._parent = args[0]
            self._name = args[1]

        else:
            raise ValueError("Invalid arguments")

        self._local = {}
        self._mask = None

    @property
    def parent(self) -> "ROIStore":
        """ROIData object containing this ROI's data.
        """
        return self._parent

    @property
    def name(self) -> int:
        """Index of ROI with respect to the associated data model.
        """
        return self._name

    @property
    def mask(self) -> PixelMask:
        """
        """
        if self._mask is None:
            x, y, z = self["xpix"], self["ypix"], self["lam"]
            self._mask = PixelMask(x, y, z)
        return self._mask

    def clear(self) -> None:
        self._local.clear()
        self._mask = None

    def __getitem__(self, key: str) -> Any:
        try:
            return self._local[key]
        except KeyError:
            pass
        out = self._parent[key].sel(roi=self.name)
        if hasattr(out, "ndim") and out.ndim == 0:
            out = out.item()
        return out

    def __setitem__(self, key: str, val: Any) -> None:
        self._local[key] = val

    def __delitem__(self, key: str) -> None:
        del self._local[key]

    def __repr__(self):
        return f"<ROI({self.name})>"


class ROIGroup:
    """
    Container for ROIs. Stores ROI instances in an ndarray.
    
    """

    _elts: np.ndarray
    _parent: Optional["ROIStore"]
    _index: Optional[np.ndarray]
    _needs_update: bool

    def __init__(self, elts: Iterable["ROI"]):

        # Store the ROIs as an ndarray.
        self._elts = np.array([obj for obj in elts], dtype=object)
        self._parent = None
        self._index = None
        self._needs_update = True

    def _update(self) -> None:

        if not self._needs_update:
            return
        self._parent = None
        self._index = None
        if len(self) > 0:
            parents = set(elt.parent for elt in self)
            if len(parents) == 1:
                self._parent = list(parents)[0]
                self._index = np.array([elt.name for elt in self])
        self._needs_update = False

    # --------------------------------------------------------------------------#
    # Container interface.

    def extend(self, other) -> None:
        self._elts = np.array(list(self._elts) + list(other))
        self._needs_update = True

    def __add__(self, other):
        return ROIGroup(list(self) + list(other))

    def __iadd__(self, other):
        self._elts += other
        self._needs_update = True

    def __contains__(self, key):
        return key in self._elts

    def __len__(self):
        return len(self._elts)

    def __iter__(self):
        return iter(self._elts)

    def __getitem__(self, key):
        result = self._elts[key]
        if isinstance(result, ROI):
            return result
        else:
            return ROIGroup(result)

    def __setitem__(self, key, value):
        self._elts[key] = value
        self._needs_update = True

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "ROIGroup: " + "".join([str(elt) for elt in self._elts])


class ROIStore(abc.ABC):
    _resources: Mapping

    def __init__(self):
        self._resources = {}
        self._n_rois = None
        self._rois = None

    @property
    def resources(self) -> Mapping:
        return self._resources

    @property
    def rois(self) -> ROIGroup:
        if self._rois is None:
            self._rois = ROIGroup([ROI(self, ix) for ix in range(self.n_rois)])
        return self._rois

    @property
    def cells(self):
        return ROIGroup([r for r in self.rois if r["iscell"]])

    def clear(self) -> None:
        self._resouces.clear()

    @abc.abstractmethod
    def load(self, name: str) -> Any:
        """

        Called when __getitem__ fails.

        Parameters
        ----------
        key

        Returns
        -------

        """

    def __getitem__(self, name: str) -> Any:
        """

        Parameters
        ----------
        key

        Returns
        -------

        """
        try:
            out = self._resources[name]
        except KeyError:
            self.load(name)
            out = self._resources[name]
        return out

    def __setitem__(self, key, val):
        self._resources[key] = val

    def __delitem__(self, key):
        del self._resources[key]


class H5ROIStore(ROIStore):
    _root: Path

    def __init__(
        self,
        path: PathLike,
        session: Optional["Session"] = None,
    ):

        self._root = Path(path)
        self._path = self._root / self._fname
        if not self._path.suffix:
            if self._path.with_suffix(".h5"):
                self._path = self._path.with_suffix(".h5")
            elif self._path.with_suffix(".hdf5"):
                self._path = self._path.with_suffix(".hdf5")
            else:
                self._path = self._path.with_suffix(".h5")

        self._session = session
        self._ds = xr.Dataset()
        self._ds.attrs["session"] = self._session

        self._n_rois = 0
        self._images = {}
        self._masks = []
        self._series = {}
        self._stats = {}

        self._item_sources = [
            "_ds",
            "_images",
            "_series",
        ]
        self._name_to_source = {}

    @property
    def root(self) -> Path:
        return self._root

    @property
    def n_rois(self) -> int:
        return self._n_rois

    @property
    def path(self) -> Path:
        return self._path

    def update_coords(self, **kw):
        coords = dict(self.ds.coords)
        coords.update(kw)
        self._ds = self._ds.assign_coords(coords)

    def load(self, name: str) -> Any:

        with h5py.File(self._path) as f:

            # load a coordinate array
            try:
                keys = list(f["Coordinates"].keys())
            except KeyError:
                keys = []
            if name in keys:
                arr = load_array(f["Coordinates"], name)
                self.update_coords({name: arr})
                return self._ds.coords[name]

            # load an image
            try:
                keys = list(f["Images"].keys())
            except KeyError:
                keys = []
            if name in keys:
                arr = load_array(f["Images"], name)
                self.add_image(name, arr)
                return self._ds[name]

            # load a series
            try:
                keys = list(f["Series"].keys())
            except KeyError:
                keys = []
            if name in keys:
                arr = load_array(f["Series"], name)
                self.add_series(name, arr)
                return self._ds[name]

        raise KeyError(name)

    def load_coords(self) -> None:

        with h5py.File(self._path, "r") as f:
            if "Coordinates" not in f:
                return
            group = f["Coordinates"]
            coords = {}
            for name in group.keys():
                arr = load_array(group, name)
                coords[name] = arr
            self.update_coords(coords)

    def save_coords(self) -> None:

        coords = dict(self.ds.coords)
        if not coords:
            return

        with h5py.File(self._path, "a") as f:
            group = f.require_group("Coordinates")
            for name, arr in coords.items():
                save_array(group, name, arr)

    def save_images(self) -> None:
        if not self._images:
            return
        with h5py.File(self._path, "a") as f:
            group = f.require_group("Images")
            for name, arr in self._images.items():
                save_array(group, name, arr)

    def save_masks(self) -> None:

        if not self._masks:
            return

        m = self._masks[0]
        ndim = len(m.arrays)

        with h5py.File(self._path, "a") as f:

            group = f.require_group("Masks")
            group.attrs["kind"] = "PIXEL_MASKS"

            # combine coordinate lists into flat arrays, and store.
            for dim in ("y", "x"):
                name = dim
                if name in group:
                    del group[name]
                data = np.hstack([m[dim] for m in self._masks])
                dset = group.create_dataset(name, data=data)
                dset.attrs['kind'] = 'RAGGED_ARRAY'
                dset.attrs['partition'] = '__partition__'

            # build and save partition map
            lengths = [len(m) for m in self._masks]
            offsets = np.r_[[0], np.cumsum(lengths)]
            bmap = np.zeros([len(offsets) - 1, 2], dtype=np.intp)
            bmap[:, 0] = offsets[:-1]
            bmap[:, 1] = offsets[1:]
            dset = group.create_dataset("__partition__", data=bmap)
            dset.attrs["kind"] = "BLOCK_MAP"

    def load_masks(self) -> None:

        with h5py.File(self._path, "r") as f:
            if "Masks" not in f:
                return
            group = f["Masks"]
            ypix = group["ypix"][:]
            xpix = group["xpix"][:]
            bmap = group["__block_map__"][:]
            starts = bmap[:, 0]
            stops = bmap[:, 1]
            masks = [PixelMask(ypix[a:b], xpix[a:b]) for a, b in zip(starts, stops)]
            self._masks = masks

    def load_series(self, name) -> None:

        with h5py.File(self._path, "r") as f:
            group = f["Series"]
            dset = group[name]

            if "Series" not in f:
                return
            group = f["Coordinates"]
            coords = {}
            for name in group.keys():
                arr = load_array(group, name)
                arr.attrs["session"] = self._session
                coords[name] = arr
            self.update_coords(coords)

    def save(self) -> None:

        root = self.root
        root.mkdir(exist_ok=True, parents=True)
        self.save_coords()
        self.save_images()
        self.save_masks()
        self.save_series()

    def add_image(
        self,
        name: str,
        arr: ArrayLike,
        dims=None,
    ) -> None:

        if isinstance(arr, xr.DataArray):
            data = arr.data
            dims = dims or arr.dims
        else:
            data = np.asarray(arr)
            if dims:
                dims = tuple(dims)
            else:
                if data.ndim == 2:
                    dims = ("y", "x")
                elif data.ndim == 3:
                    dims = ("y", "x", "c")
                else:
                    raise NotImplementedError

        arr = xr.DataArray(data, dims=dims, name=name)
        arr.attrs["session"] = self._session
        self._ds[name] = arr
        self._images.append(name)

    def add_series(
        self,
        name: str,
        arr: ArrayLike,
        dims=None,
    ) -> None:

        if isinstance(arr, xr.DataArray):
            data = arr.data
            dims = dims or arr.dims
        else:
            if dims:
                dims = tuple(dims)
            else:
                if arr.ndim == 2:
                    dims = ("t", "roi")
                else:
                    raise NotImplementedError
        arr = xr.DataArray(data, dims=dims, name=name)
        arr.attrs["session"] = self._session
        self._ds[name] = arr
        self._series.append(name)

    def add_roi(self, *args, series=None):

        mask = PixelMask(*args, dims=("y", "x"))
        self._masks.append(mask)

        series = {} if not series else series
        for key, val in series.items():
            if key not in self._series:
                self._series[key] = [None] * self._n_rois
            self._series[key].append(val)

        self._n_rois += 1


def load_array(group: h5py.Group, name: str) -> xr.DataArray:
    dset = group[name]
    data = dset[:]
    if "dims" in dset.attrs:
        dims = json.loads(dset.attrs["dims"])
        arr = xr.DataArray(data, dims=dims, name=name)

    return arr


def save_array(
    group: h5py.Group,
    name: str,
    arr: ArrayLike,
    kind: str = "",
) -> None:
    old_attrs = {}
    if name in group:
        old_attrs = dict(group[name].attrs)
        del group[name]

    if isinstance(arr, xr.DataArray):
        data = arr.data
        dims = arr.dims
    else:
        data = np.asarray(arr)
        dims = None

    dset = group.create_dataset(name, data=data)
    dset.attrs["kind"] = kind
    if dims:
        dset.attrs["dims"] = json.dumps(dims)
    return dset


def load_array_meta(dset: Union[h5py.Dataset, h5py.Group]) -> Mapping:
    attrs = dict(dset.attrs)

    meta = {
        "kind": attrs.get("kind", ""),
        "dtype": dset.dtype,
        "shape": dset.shape,
    }
    if "dims" in attrs:
        dims = json.loads(attrs["dims"])
        sizes = {a: b for a, b in zip(dims, dset.shape)}
        meta.update(dict(dims=dims, sizes=sizes))

    return meta
