import os
from pathlib import Path
from typing import Any, Mapping, Optional
from fs.base import FS
import numpy as np
import xarray as xr
from ca_analysis.roi import ROIStore
from ca_analysis.session import Session

__all__ = [
    "Suite2PStore",
]


class Suite2PStore(ROIStore):
    """
    Interface to suite2p output files.
    """

    _root: Path
    _resources: Mapping
    _n_rois: Optional[int]
    _session: Optional["Session"]

    def __init__(
        self,
        root: os.PathLike,
        session: Optional[Session] = None,
    ):

        self._resources = {}
        self._n_rois = None
        self._root = Path(root)
        self._session = session


    @property
    def root(self) -> Path:
        return self._root

    @property
    def n_rois(self) -> int:
        if self._n_rois is None:
            self._n_rois = np.load(self._root / "iscell.npy").shape[0]
        return self._n_rois

    @property
    def n_cells(self) -> int:
        return self["iscell"].sum().item()

    def load(self, key: str) -> Any:
        """

        Parameters
        ----------
        key

        Returns
        -------

        """
        needs_pix_coords = {
            "corrimg",
            "maximg",
            "meanimg",
        }
        needs_roi_coord = {
            "F",
            "Fneu",
            "spikes",
            "aspect_ratio",
            "compact",
            "footprint",
            "lam",
            "med",
            "overlap",
            "radius",
            "skew",
            "std",
        }

        coords = {}
        if key in needs_pix_coords or key in {"xpix", "ypix"}:
            if "ypix" not in self._resources or "xpix" not in self._resources:
                ops = self._get_ops()
                imshape = ops["meanImg"].shape
                n_ypix, n_xpix = imshape[0], imshape[1]
                ypix = xr.DataArray(np.arange(n_ypix, dtype=np.intp), dims=("ypix",))
                ypix.attrs['session'] = str(self._session)
                xpix = xr.DataArray(np.arange(n_xpix, dtype=np.intp), dims=("xpix",))
                xpix.attrs['session'] = str(self._session)
                self._resources["ypix"] = ypix
                self._resources["xpix"] = xpix

        if key in needs_pix_coords:
            coords.update({"ypix": self._resources["ypix"],
                           "xpix": self._resources["xpix"]})

        if key in needs_roi_coord or key == "roi":
            if "roi" not in self._resources:
                inds = np.arange(self.n_rois)
                self._resources["roi"] = xr.DataArray(inds, dims=("roi",))
            coords["roi"] = self._resources["roi"]

        if key in {"ypix", "xpix", "roi"}:
            return self._resources[key]

        if key == "F":
            data = np.load(self._root / "F.npy").T
            arr = xr.DataArray(data, dims=("time", "roi"), coords=coords)
            arr.attrs['session'] = str(self._session)
            self._resources["F"] = arr
            return self._resources["F"]

        if key == "Fneu":
            data = np.load(self._root / "Fneu.npy").T
            arr = xr.DataArray(data, dims=("time", "roi"), coords=coords)
            arr.attrs['session'] = str(self._session)
            self._resources["Fneu"] = arr
            return self._resources["Fneu"]

        if key == "spikes":
            data = np.load(self._root / "spks.npy").T
            arr = xr.DataArray(data, dims=("time", "roi"), coords=coords)
            arr.attrs['session'] = str(self._session)
            self._resources["spikes"] = arr
            return self._resources["spikes"]

        if key == "ops":
            return self._get_ops()

        if key == "stat":
            return self._get_stat()

        if key == "iscell":
            data = np.load(self._root / "iscell.npy")[:, 0].astype(bool)
            arr = xr.DataArray(data, dims=("roi",), coords=coords)
            arr.attrs['session'] = str(self._session)
            self._resources["iscell"] = arr
            return self._resources["iscell"]

        if key == "P_cell":
            data = np.load(self._root / "iscell.npy")[:, 1]
            arr = xr.DataArray(data, dims=("roi",), coords=coords)
            arr.attrs['session'] = str(self._session)
            self._resources["P_cell"] = arr
            return self._resources["P_cell"]


        if key in {
            "aspect_ratio",
            "compact",
            "footprint",
            "lam",
            "med",
            "overlap",
            "radius",
            "skew",
            "std",
        }:

            stat = self._get_stat()
            data = [elt[key] for elt in stat]
            if isinstance(data[0], np.ndarray):
                data = np.array(data, dtype=object)
            else:
                data = np.array(data)
            arr = xr.DataArray(data, dims=("roi",), coords=coords)
            arr.attrs['session'] = str(self._session)
            self._resources[key] = arr
            return self._resources[key]

        if key in {
            "corrimg",
            "maximg",
            "meanimg",
        }:
            ops = self._get_ops()
            if key == "corrimg":
                data = ops["Vcorr"]
            elif key == "mean":
                data = ops["meanImg"]
            else:
                data = ops["max_proj"]
            arr = xr.DataArray(data, dims=("ypix", "xpix"), coords=coords)
            arr.attrs['session'] = str(self._session)
            self._resources[key] = arr
            return self._resources[key]

        raise KeyError(key)

    def save_iscell(self) -> None:
        """
        Store iscell and P_cell.
        """
        iscell = self["iscell"].data
        P_cell = self["P_cell"].data
        data = np.zeros([len(iscell), 2])
        data[:, 0] = iscell.astype(float)
        data[:, 1] = P_cell
        np.save(self._root / "iscell.npy", data)

    def _get_ops(self) -> Mapping:
        try:
            return self._resources["ops"]
        except KeyError:
            path = self._root / "ops1.npy"
            path = path if path.exists() else self._root / "ops.npy"
            arr = np.load(path, allow_pickle=True).item()
            self._resources["ops"] = arr
            return self._resources["ops"]

    def _get_stat(self) -> np.ndarray:
        try:
            return self._resources["stat"]
        except KeyError:
            arr = np.load(self._root / "stat.npy", allow_pickle=True)
            self._resources["stat"] = arr
            return self._resources["stat"]

