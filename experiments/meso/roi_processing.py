from typing import Mapping, Optional, Tuple
import h5py
import napari
import numpy as np
from experiments.meso.main import *
from ca_analysis import *


class ROIData:

    def __init__(self):
        pass


def create_masks(
    s: Session,
    background: Optional[str] = None,
) -> "napari.Viewer":
    import napari

    # find background
    if is_str(background) or background is None:
        with h5py.File(s.fs.getsyspath("mov.h5"), "r+") as f:
            if background is None:
                background = 'ca' if 'ca' in f else 'ach'
            dset = f[background]
            arr = da.from_array(dset, chunks=(1, -1, -1))
            arr = arr[:100].compute()
            background = arr.mean(axis=0)
    else:
        print('using background')
        background = np.asarray(background)

    viewer = napari.view_image(background)
    bg_layer = viewer.layers[0]
    bg_layer.name = "background"
    win = viewer.window
    win.add_plugin_dock_widget('napari-roi')
    layer = napari.layers.Shapes(
        edge_color="red",
        face_color="transparent",
    )
    layer.name = "ROIs"
    viewer.add_layer(layer)
    viewer.window.session = s
    return viewer


def load_napari_mask_data(s: Session, name: str = "masks.h5") -> Mapping:
    dct = {'name': [], 'data': [], 'shape_type': [], 'y': [], 'x': []}
    with h5py.File(s.fs.getsyspath(name), "r") as f:
        dset_names = list(f.keys())
        for i, name in enumerate(dset_names):
            group = f[name]
            dct['name'].append(name)
            dct['data'].append(group['data'][:])
            dct['shape_type'].append(str(group.attrs['shape_type']))
            pixels = group['pixels']
            dct['y'].append(pixels[:, 0])
            dct['x'].append(pixels[:, 1])

    return dct


def process_masks(s: Session) -> Mapping[str, ROI]:
    masks = load_napari_mask_data(s)
    rois = {}
    for i in range(len(masks['name'])):
        r = ROIData()
        r.name = masks['name'][i]
        r.y = masks['y'][i]
        r.x = masks['x'][i]
        rois[r.name] = r

    # Extract traces
    with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
        channels = list(f.keys())
        for ch in channels:
            mov = f[ch][:]
            for r in rois.values():
                F = np.zeros(mov.shape[0])
                for i in range(mov.shape[0]):
                    frame = mov[i]
                    F[i] = np.mean(frame[r.y, r.x])
                setattr(r, ch, F)

    # Save masks and traces
    with h5py.File(s.fs.getsyspath('rois.h5'), 'w') as f:
        for r in rois.values():
            group = f.create_group(r.name)
            group.create_dataset('y', data=r.y)
            group.create_dataset('x', data=r.x)
            for ch in channels:
                group.create_dataset(ch, data=getattr(r, ch))

    return rois


class ROI(SessionData):

    def __init__(
        self,
        session: Session,
        roi_name: str,
        channel: str,
        filename: str = "rois.h5",
    ):
        super().__init__(session)
        self.roi_name = roi_name
        self.channel = channel
        self.filename = filename

    def _prepare(self) -> None:
        roi_path = self._session.fs.getsyspath(self.filename)
        with h5py.File(roi_path, 'r') as f:
            group = f[self.roi_name]
            data = group[self.channel][:]
        self.data = xr.DataArray(data, dims=('time',))


# class ManualROISegmentation(SessionData):
#
#     def __init__(self, session: Session, key: str, filename: str = "rois.h5"):
#         super().__init__(session)
#         self.key = key
#         self.filename = filename
#
#     def _prepare(self) -> None:
#         roi_path = s.fs.getsyspath(self.filename)
#         roi_path = Path(roi_path).with_suffix('.h5')
#         with h5py.File(roi_path, 'r') as f:
#             names, traces = [], []
#             for key, group in f.items():
#                 names.append(key)
#                 traces.append(group[self.key][:])
#         data = np.stack(traces).T
#         dims = ('time', 'roi')
#         coords = {'roi': names}
#         self.data = xr.DataArray(data, dims=dims, coords=coords)
#

def load_trials(s: Session) -> Tuple[xr.DataArray]:
    with h5py.File(s.fs.getsyspath('mean_mov.h5'), 'r') as f:
        group = f['ach']
        left = xr.DataArray(group['left'][:], dims=('trial', 'time', 'y', 'x'))
        right = xr.DataArray(group['right'][:], dims=('trial', 'time', 'y', 'x'))
    evs = [s.events.schema.get(event=0)] * left.sizes['time']
    evs[10:20] = [s.events.schema.get(event=1)] * 10
    left.coords['event'] = xr.DataArray(evs, dims=('time',))
    evs = [s.events.schema.get(event=0)] * left.sizes['time']
    evs[10:20] = [s.events.schema.get(event=2)] * 10
    right.coords['event'] = xr.DataArray(evs, dims=('time',))

    return left, right


if __name__ == '__main__':
    s = open_session('M150', '2023-03-30', '1', fs=0)
    with h5py.File(s.fs.getsyspath('mov.h5'), 'r') as f:
        data = f['ca'][:100]
        mov = xr.DataArray(data, dims=('time', 'y', 'x'))

    mov = gaussian_filter(mov, (2, 2, 2))
    background = mov.mean('time')
    #
    create_masks(s, background=background)
    # process_masks(s)
