"""
Intake interface for h5 datasets.
"""
from typing import (
    Mapping,
    Optional,
    Union,
)

import dask.array as da
import h5py
import numpy as np
from intake.source.base import DataSource, Schema

from .common import *
from ..common import *

__all__ = [
    "H5DatasetSource",
]

DEFAULT_STORAGE_OPTIONS = FrozenDict(
    chunks=None,
    maxshape=None,
    compression=None,
    compression_opts=None,
    scaleoffset=None,
    shuffle=False,
    fletcher32=False,
    fillvalue=0,
)


class H5DatasetSource(DataSource):
    name = "h5dataset"
    container = "ndarray"
    partition_access = False
    version = "0.0.1deva"

    def __init__(
        self,
        urlpath: Union[PathLike, URL],
        name: Optional[str] = None,
        chunks: Optional[int] = None,
        metadata: Optional[Mapping] = None,
        storage_options: Optional[Mapping] = None,
    ):
        super().__init__(metadata=metadata)

        self._url = URL(urlpath)
        if name:
            self._url.query["name"] = name
        if self._url.query.get("name", None) is None:
            raise ValueError("must provide dataset name")

        self.shape = None
        self.dtype = None
        self._attrs = None
        self.npartitions = None
        self.chunks = None
        if chunks is None:
            self._chunks_arg = -1
        elif is_int(chunks):
            self._chunks_arg = chunks
        else:
            self._chunks_arg = chunks
        self._storage_options = dict(storage_options) if storage_options else {}

        self._file = None
        self._dset = None
        self._arr = None
        self._schema = None

    @property
    def url(self) -> URL:
        return self._url

    @property
    def file(self) -> Optional[h5py.File]:
        return self._file

    @property
    def attrs(self) -> Optional[Mapping]:
        self._load_metadata()
        return self._attrs

    def open(self) -> None:
        self._load_metadata()

    def to_dataset(self) -> h5py.Dataset:
        self._load_metadata()
        return self._dset

    def to_dask(self) -> da.Array:
        self._load_metadata()
        return self._arr

    def read(self) -> np.ndarray:
        self._load_metadata()
        return self._arr.compute()

    def read_partition(self, i: int) -> np.ndarray:
        return self._get_partition(i).compute()

    def _close(self) -> None:
        self._file.close()
        self._file = None
        self._dset = None
        self._arr = None
        self._schema = None

    def _load_metadata(self) -> None:
        """load metadata only if needed"""
        if self._schema is None:
            self._schema = self._get_schema()

    def _get_schema(self) -> Schema:

        if self._arr is None:
            url = self._url
            self._file = h5py.File(url.path, "r")
            self._dset = self._file[url.query["name"]]
            self.shape = self._dset.shape
            self.dtype = self._dset.dtype
            self._attrs = dict(self._dset.attrs)
            self._storage_options = {key: getattr(self._dset, key) for
                                     key in DEFAULT_STORAGE_OPTIONS}

            if self.chunks is None:
                chunks = [-1] * len(self.shape)
                chunks[0] = self._chunks_arg
            else:
                chunks = self.chunks

            self._arr = da.from_array(self._dset, chunks=chunks)
            self.npartitions = self._arr.npartitions
            self.chunks = self._arr.chunks

        return Schema(
            shape=self.shape,
            dtype=self.dtype,
            npartitions=self.npartitions,
            chunks=self.chunks,
            storage_options=self._storage_options,
            extra_metadata=self._attrs,
        )

    def _get_partition(self, i) -> da.Array:
        self._load_metadata()
        return self._arr.blocks[i]

