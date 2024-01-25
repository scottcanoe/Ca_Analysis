import matplotlib
matplotlib.use('TkAgg')

from experiments.meso.roi_processing import *
from experiments.meso.seq_learn_3.utils import *


# sessions = [
#     # no scopolamine
#     # open_session('Thy1_0093', '2023-04-19', '1', fs=0),
#     # open_session('Thy1_0093', '2023-04-19', '2', fs=0),
#     open_session('M110', '2023-04-21', '1', fs=0),
#     open_session('M110', '2023-04-21', '2', fs=0),
#     open_session('M115', '2023-04-21', '1', fs=0),
#     open_session('M115', '2023-04-21', '2', fs=0),
# #     open_session('M150', '2023-04-21', '1', fs=0),
# #     open_session('M150', '2023-04-21', '2', fs=0),
# #     open_session('M152', '2023-04-21', '1', fs=0),
# #     open_session('M152', '2023-04-21', '2', fs=0),
# #     # open_session('M153', '2023-04-21', '1', fs=0),
# #     # open_session('M153', '2023-04-21', '2', fs=0),
# #
# #     # scopolamine
# #     open_session('Thy1_0093', '2023-04-25', '1', fs=0),
#     # open_session('Thy1_0093', '2023-04-25', '2', fs=0),
#     open_session('M110', '2023-04-25', '1', fs=0),
#     open_session('M110', '2023-04-25', '2', fs=0),
#     open_session('M115', '2023-04-25', '1', fs=0),
#     open_session('M115', '2023-04-25', '2', fs=0),
# ]

# for s in sessions:
#     if s.fs.exists('raw'):
#         s.fs.removetree('raw')
#     print(s)
#     make_splits(s)
#     #if s.fs.exists('trials.h5'):
#     #    s.fs.remove('trials.h5')
#     s.push()
#     nas_fs = s.get_fs(-1)
#     if nas_fs.exists('trials.h5'):
#         nas_fs.remove('trials.h5')


class ManualSegmentation:

    def __init__(self, session: Session, filename: str = "rois.h5"):
        super().__init__()
        self.session = session
        self.filename = self.session.fs.getsyspath(filename)

    def _prepare(self) -> None:
        roi_path = s.fs.getsyspath(self.filename)
        roi_path = Path(roi_path).with_suffix('.h5')
        with h5py.File(roi_path, 'r') as f:
            names, traces = [], []
            for key, group in f.items():
                names.append(key)
                traces.append(group[self.key][:])
        data = np.stack(traces).T
        dims = ('time', 'roi')
        coords = {'roi': names}
        self.data = xr.DataArray(data, dims=dims, coords=coords)

    def __getitem__(self, name: str) -> ROI:
        roi = ManualROI(self, name)
        return roi


class ManualROI:

    def __init__(self, parent: ManualSegmentation, name: str):
        super().__init__()
        self.parent = parent
        self.name = name
        self._ach = None
        self._ca = None
        self._x = None
        self._y = None

    @property
    def ach(self) -> SessionData:
        if self._ach is None:
            self._ach = ManualROIChannel(self, 'ach')
        return self._ach

    @property
    def ca(self) -> SessionData:
        if self._ach is None:
            self._ach = ManualROIChannel(self, 'ca')
        return self._ach

    @property
    def x(self) -> np.ndarray:
        if self._x is None:
            with h5py.File(self.parent.filename, 'r') as f:
                group = f[self.name]
                self._x = group['x'][:]
        return self._x

    @x.setter
    def x(self, pixels: ArrayLike) -> None:
        self._x = np.array(pixels)

    @property
    def y(self) -> np.ndarray:
        if self._y is None:
            with h5py.File(self.parent.filename, 'r') as f:
                group = f[self.name]
                self._y = group['y'][:]
        return self._y

    @y.setter
    def y(self, pixels: ArrayLike) -> None:
        self._y = np.array(pixels)


class ManualROIChannel(SessionData):

    def __init__(
        self,
        parent: ManualROI,
        channel: str,
    ):
        self.parent = parent
        super().__init__(session)
        self.channel = channel



# sessions = [
#     open_session('M172', '2023-05-15', '1', fs=0),
#     open_session('M172', '2023-05-19', '1', fs=0),
#     open_session('M173', '2023-05-15', '1', fs=0),
#     open_session('M173', '2023-05-19', '1', fs=0),
#     open_session('M174', '2023-05-15', '1', fs=0),
#     open_session('M174', '2023-05-19', '1', fs=0),
# ]

s = open_session('M150', '2023-04-21', '2')
print(s)
print(s.attrs['schema'])
print(s.events.schema.name)
# s.rois = ManualSegmentation(s)
#
# lh = s.rois['LH']
