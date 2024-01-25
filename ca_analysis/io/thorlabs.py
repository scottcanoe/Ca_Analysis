"""
To Do:
------
- only handles single-channel streaming t-series. expand usages.

"""
import datetime
from numbers import Number
import os
from pathlib import Path
from typing import (
    ClassVar,
    List,
    Mapping,
    Optional,
    Tuple,
)
from xml.etree import ElementTree

from intake import DataSource, Schema
from typing import (
    ClassVar,
    Container,
    Mapping,
    Optional,
    Set,
)

import fsspec
import h5py
import numpy as np
import pandas as pd

from .common import *
from ..common import *
from ..indexing import iterchunks

__all__ = [
    "read_thorimage_metadata",
    "read_thorsync_data",
    "thorimage_raw_to_h5",
    "ThorImageArraySource",
    "ThorImageMetadataSource",
    "ThorSyncSource",
]

__version__ = "0.0.1dev"

def read_thorimage_metadata(path: PathLike) -> Mapping:
    src = ThorImageMetadataSource(path)
    return src.to_dict()


def read_thorsync_data(path: PathLike) -> pd.DataFrame:
    src = ThorSyncSource(
        path,
        binary={"FrameOut", "FrameTrigger", "Strobe"},
    )
    return src.read()


def thorimage_raw_to_h5(
    path: PathLike,
    dst: PathLike,
    name: Optional[str] = None,
    mode: str = "a",
    replace: bool = False,
    write_chunks: int = 100,
    delete_on_error: bool = True,
    **kw,
) -> None:
    """
    Convert thorimage raw imaging data to an h5 dataset.
    """

    dst = URL(dst)
    dst.query["name"] = dst.query.get("name", name)
    if dst.query["name"] is None:
        dst.query["name"] = "data"
    name = dst.query["name"]
    if name is None:
        raise ValueError('must provide dataset name')

    src = ThorImageArraySource(path)
    arr = src.to_dask()
    shape, dtype = arr.shape, arr.dtype
    if kw.get("maxshape") is None:
        kw["maxshape"] = shape
    file_created = not Path(dst.path).exists()
    node_created = None
    try:
        with h5py.File(dst.path, mode) as f:
            if name in f and replace:
                del f[name]
            node_created = name not in f
            dset = f.require_dataset(name, shape=shape, dtype=dtype, **kw)
            for i, slc, chunk in iterchunks(arr, write_chunks):
                slc, chunk = slc.raw, chunk.compute()
                dset[slc] = chunk
    except Exception:
        if f and node_created and name in f and delete_on_error:
            del f[name]
        f.close()
        if file_created and delete_on_error:
            os.remove(dst.path)
        raise


class ThorImageMetadataSource(DataSource):
    """
    read() returns an xml.ElementTree
    """

    name: ClassVar[str] = "thorimagemetadata"
    container: ClassVar[str] = "python"
    version: str = __version__
    partition_access: ClassVar[bool] = True

    def __init__(
        self,
        urlpath: PathLike,
        metadata: Optional[Mapping] = None,
        pattern: Optional[str] = "Experiment.xml",
    ):
        super().__init__(metadata=metadata)
        self._path = os.fspath(urlpath)  # initial path argument.
        self.path = None  # resolved path. set once known.
        self.pattern = pattern

    def read(self) -> ElementTree:
        self._load_metadata()
        return ElementTree.parse(self._schema["path"])

    def to_dict(self) -> Mapping:

        doc = self.read()

        # Basic info
        md = {}
        md["software_version"] = doc.find("Software").attrib["version"]
        uTime = int(doc.find("Date").attrib["uTime"])
        md["datetime"] = str(datetime.datetime.fromtimestamp(uTime))

        # Handle capture mode.
        capture_mode = int(doc.find('CaptureMode').attrib['mode'])
        try:
            capture_mode = {0: "z-series", 1: "t-series"}[capture_mode]
        except KeyError:
            raise ValueError("unknown capture mode: {}".format(capture_mode))
        md["capture_mode"] = capture_mode

        # Handle modality
        modality = doc.find('Modality').attrib['name'].lower()
        if modality not in {"camera", "multiphoton", "confocal"}:
            raise ValueError("unknown modality: {}".format(modality))
        md["modality"] = modality

        if modality == "camera":
            frame = self._parse_camera(doc)
            md["frame"] = frame
        elif modality == "multiphoton":
            frame, PMTs, pockels = self._parse_multiphoton(doc)
            md["frame"] = frame
            md["PMTs"] = PMTs
            md["pockels"] = pockels

        return md

    def _get_partition(self, i):
        """Subclasses should return a container object for this partition
        This function will never be called with an out-of-range value for i.
        """
        return self.read()

    def _get_schema(self) -> Schema:

        if self.path is None:
            if os.path.isdir(self._path):
                if not self.pattern:
                    raise FileNotFoundError(self._path)
                self.path = find_file(
                    self.pattern, root_dir=self._path, absolute=True,
                )
            else:
                self.path = find_file(self._path)
        schema = Schema(
            datashape=None,
            dtype=object,
            shape=None,
            npartitions=1,
            path=self.path,
            extra_metadata={},
        )
        return schema

    def _parse_camera(self, doc: ElementTree) -> Mapping:

        # Handle frame capture info
        node = doc.find('Camera').attrib
        name = node["name"]
        shape = [int(node['height']), int(node['width'])]
        dtype = "<H"
        size = [float(node['heightUM']), float(node['widthUM'])]
        channels = 1
        zoom = 1
        averaging = 1 if node["averageMode"] == "0" else int(node['averageNum'])
        exposure = float(node['exposureTimeMS']) / 1000
        rate = (1 / exposure) / averaging
        binning = [int(node['binningY']), int(node['binningX'])]

        frame = dict(
            name=name,
            shape=shape,
            dtype=dtype,
            size=size,
            channels=channels,
            zoom=zoom,
            averaging=averaging,
            rate=rate,
            exposure=exposure,
            binning=binning,
        )
        return frame

    def _parse_multiphoton(
        self,
        doc: ElementTree,
    ) -> Tuple[Mapping, List[Mapping], List[Mapping]]:

        # Handle frame shape, FOV, and pixel size.
        node = doc.find('LSM').attrib
        name = node["name"]
        shape = [int(node['pixelX']), int(node['pixelY'])]
        dtype = "<H"
        size = [float(node['heightUM']), float(node['widthUM'])]
        channels = bin(int(doc.find('Wavelengths/ChannelEnable').get('Set')))[2:].count("1")
        zoom = int(node["pixelY"]) / shape[0]
        averaging = 1 if node["averageMode"] == "0" else int(node['averageNum'])
        rate = float(node['frameRate']) / averaging
        frame = dict(
            name=name,
            shape=shape,
            dtype=dtype,
            size=size,
            channels=channels,
            zoom=zoom,
            averaging=averaging,
            rate=rate,
        )

        # PMTs
        PMTs = []
        pmt_node = doc.find("PMT")
        for index, letter in enumerate("ABCD"):
            enabled = bool(int(pmt_node.attrib[f"enable{letter}"]))
            gain = float(pmt_node.attrib[f"gain{letter}"])
            pmt = dict(enabled=enabled, gain=gain)
            PMTs.append(pmt)

        # Pockels
        pockels = []
        pockels_nodes = doc.findall("Pockels")
        for n in pockels_nodes:
            pock = dict(start=float(n.attrib["start"]), stop=float(n.attrib["stop"]))
            pockels.append(pock)

        return frame, PMTs, pockels


class ThorImageArraySource(DataSource):
    """
    Parameters
    ----------
    path: path-like
        Location of raw image file.
    metadata_path: path-like, optional
        Location of xml metadata file. If not absolute, will look in same
        directory as the raw image file.
    shape: tuple of int, optional
        If not given, will look in metadata file.
    dtype: dtype-like, optional
        If not given, will look in metadata file.
    chunks: int, optional
        Size of chunks within a file along biggest dimension - need not
        be an exact factor of the length of that dimension.
    """

    name: ClassVar[str] = "thorimagearray"
    version: ClassVar[str] = __version__
    container: ClassVar[str] = "ndarray"
    partition_access: ClassVar[bool] = True

    def __init__(
        self,
        urlpath: PathLike,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: DTypeLike = np.dtype("<H"),
        chunks: Optional[int] = None,
        metadata: Optional[Mapping] = None,
        pattern: str = "Image*.raw",
    ):
        super().__init__(metadata=metadata)

        self._path = os.fspath(urlpath)
        self.path = None
        self.shape = shape
        self.dtype = dtype
        self.npartitions = None
        self.chunks = None
        self._chunks_arg = -1 if not chunks else chunks
        self.pattern = pattern

        self._memmap = None
        self._arr = None

    def get_schema(self) -> Schema:
        self._load_metadata()
        return self._schema

    def open(self):
        self._load_metadata()

    def to_dask(self):
        self._load_metadata()
        return self._arr

    def to_memmap(self) -> np.ndarray:
        self._load_metadata()
        return self._memmap

    def read(self) -> np.ndarray:
        self._load_metadata()
        return self._arr.compute()

    def read_partition(self, i: int) -> np.ndarray:
        return self._get_partition(i).compute()

    def _close(self) -> None:
        self._schema = None
        self._memmap = None
        self._arr = None

    def _get_partition(self, i):
        self._load_metadata()
        return self._arr.blocks[i]

    def _get_schema(self) -> Schema:
        """
        Creates and sets the `_arr` attribute -- the raw array accessor.

        Returns
        -------

        """

        import dask.array

        if self._arr is None:

            if not self.path or not os.path.exists(self.path):
                # locate raw data file
                if os.path.isdir(self._path):
                    if not self.pattern:
                        raise FileNotFoundError(self._path)
                    self.path = find_file(
                        self.pattern, root_dir=self._path, absolute=True,
                    )
                else:
                    self.path = find_file(self._path)

            if self.shape is None:
                parent_dir = Path(self.path).parent
                md = ThorImageMetadataSource(parent_dir).to_dict()
                channels = md["frame"]["channels"]
                if channels != 1:
                    raise NotImplementedError('unsupported number of channels')
                frame_shape = md["frame"]["shape"]
                dtype = np.dtype(md["frame"]["dtype"])
                frame_size = int(np.prod(frame_shape) * dtype.itemsize)
                file_size = os.stat(self.path).st_size
                self.shape = (file_size // frame_size, *frame_shape)
                extra_metadata = md
            else:
                extra_metadata = {}

            self.shape = tuple(self.shape)
            self.dtype = np.dtype(self.dtype)

            if self.chunks is None:
                self.chunks = [-1] * len(self.shape)
                self.chunks[0] = self._chunks_arg

            self._memmap = np.memmap(
                self.path,
                shape=self.shape,
                dtype=self.dtype,
                mode="r",
            )

            self._arr = dask.array.from_array(self._memmap, chunks=self.chunks)
            self.chunks = self._arr.chunks

        return Schema(
            path=self.path,
            shape=self.shape,
            dtype=self.dtype,
            chunks=self.chunks,
            npartitions=1,
            extra_metadata=extra_metadata,
        )

    def _load_metadata(self):
        """load metadata only if needed"""

        if self._schema is None:
            self._schema = self._get_schema()


class ThorSyncSource(DataSource):
    """
    Driver for ThorSync's h5 output.

    Parameters
    ----------
    path: path-like
    Path to h5 file. Usually called 'Episode001.h5'.

    binary: iterable of str
    Digital lines carrying binary data. Values are squashed into {0, 1} and the dtype is cast to np.int8.



    """

    name: ClassVar[str] = "thorsync"
    container: ClassVar[str] = "dataframe"
    version: str = __version__
    partition_access: ClassVar[bool] = False

    def __init__(
        self,
        urlpath: PathLike,
        *,
        binary: Optional[Container[str]] = None,
        clock_rate: Number = 20_000_000,
        pattern: str = "Episode*.h5",
        metadata: Optional[Mapping] = None,
    ):
        super().__init__(metadata=metadata)
        self._path = urlpath
        self.path = None
        self.binary = binary
        self.clock_rate = clock_rate
        self.pattern = pattern
        self._dataframe = None

    @property
    def binary(self) -> Set:
        return self._binary

    @binary.setter
    def binary(self, val: Optional[Container[str]]) -> None:
        self._binary = set(val) if val else set()

    def get_schema(self) -> Schema:
        self._load_metadata()
        return self._get_schema()

    def _close(self) -> None:
        self._schema = None
        self._dataframe = None

    def _get_partition(self, i):
        """Subclasses should return a container object for this partition
        This function will never be called with an out-of-range value for i.
        """
        if self._dataframe is None:
            self._dataframe = self._load_dataframe()
        return self._dataframe

    def _get_schema(self) -> Schema:

        if not self.path or not os.path.exists(self.path):
            # locate raw data file
            if os.path.isdir(self._path):
                if not self.pattern:
                    raise FileNotFoundError(self._path)
                self.path = find_file(
                    self.pattern, root_dir=self._path, absolute=True,
                )
            else:
                self.path = find_file(self._path)

        with fsspec.open_files(self.path, "rb")[0] as file:
            with h5py.File(file, "r") as f:
                clock = f['Global']['GCtr']
                length = clock.shape[0]
                dtypes = {"time": np.float64}

                AI = f["AI"]
                for name, dset in AI.items():
                    dtypes[name] = dset.dtype

                DI = f["DI"]
                for name, dset in DI.items():
                    if name in self.binary:
                        dtypes[name] = np.int8

        shape = (length, len(dtypes))
        columns = tuple(dtypes.keys())

        return Schema(
            dtype=None,
            shape=shape,
            npartitions=1,
            path=self.path,
            columns=columns,
            dtypes=dtypes,
            extra_metadata={},
        )

    def _load_metadata(self) -> None:
        if self._schema is None:
            self._schema = self._get_schema()

    def _load_dataframe(self) -> pd.DataFrame:
        """
        Load the h5 data into a dataframe and return it.
        """
        self._load_metadata()

        file = fsspec.open_files(self.path, "rb")[0]
        with file as f_inner:
            with h5py.File(f_inner, "r") as f:
                data = {}
                # Create time array from 20 kHz clock ticks. Thorsync's
                # metadata file has samplerate entries, but Thorlabs'
                # house-made matlab scripts have this value hard-corded in.
                # Also, this value isn't one of the samplerates listed in
                # the metadata file ('ThorRealTimeDataSettings.xml').
                clock_rate = self.clock_rate
                clock = f["Global"]["GCtr"][:].reshape(-1)
                data['time'] = clock / clock_rate

                # Load analog lines.
                for name, dset in f['AI'].items():
                    arr = dset[:].reshape(-1)
                    data[name] = arr

                # Load digital lines.
                for name, dset in f['DI'].items():
                    arr = dset[:].reshape(-1)
                    if name in self._binary:
                        # For some reason, some digital lines that should
                        # carry only 0s or 1s have 0s and 2s or 0s and 16s.
                        # Clip them here.
                        arr = np.clip(arr, 0, 1).astype(np.int8)
                    else:
                        # Prefer signed integers to avoid pitfalls with diff.
                        arr = arr.astype(np.int32)
                    data[name] = arr

        df = pd.DataFrame(data)

        return df
