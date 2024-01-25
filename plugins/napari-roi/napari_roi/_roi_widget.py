from pathlib import Path
import re
from typing import TYPE_CHECKING, MutableSequence, Optional, Sequence

import h5py
from napari.layers import Shapes
from napari.utils.events import Event
from napari.viewer import Viewer
from qtpy.QtCore import QEvent, QItemSelection, QObject, QPoint, QSignalBlocker, Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyle,
    QTableView,
    QWidget,
)
import numpy as np

from ._roi import ROI, ROIBase, ROIOrigin
from .qt import ROILayerAccessor, ROITableModel
from .qt.utils import MutableItemModelSequenceWrapper

if TYPE_CHECKING:
    from vispy.app.canvas import MouseEvent


class ROIWidget(QWidget):
    DEFAULT_COLUMN_WIDTHS = (120, 80, 80, 80, 80)
    ROI_LAYER_TEXT_COLOR = "red"

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):

        super(ROIWidget, self).__init__(parent=parent)
        self._initialized = False

        self._viewer = viewer
        self._roi_layer: Optional[Shapes] = None
        self._roi_layer_accessor: Optional[ROILayerAccessor] = None
        self._roi_table_model: Optional[ROITableModel] = None

        self.setMinimumHeight(200)
        self.setLayout(QGridLayout())

        self._roi_table_widget = QWidget(parent=self)
        roi_table_widget_layout = QFormLayout()
        self._roi_table_widget.setLayout(roi_table_widget_layout)
        roi_table_widget_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._roi_table_view = QTableView(parent=self._roi_table_widget)
        self._roi_table_view.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )
        self._roi_table_view.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._roi_table_view.customContextMenuRequested.connect(
            self._on_roi_table_view_context_menu_requested
        )
        roi_table_widget_layout.addRow(self._roi_table_view)
        self._roi_origin_combo_box = QComboBox(parent=self._roi_table_widget)
        self._roi_origin_combo_box.setFixedWidth(200)
        self._roi_origin_combo_box.addItems(
            (
                str(ROIOrigin.TOP_LEFT),
                # str(ROIOrigin.TOP_RIGHT),
                # str(ROIOrigin.BOTTOM_LEFT),
                # str(ROIOrigin.BOTTOM_RIGHT),
                # str(ROIOrigin.CENTER),
            )
        )
        self._roi_origin_combo_box.currentTextChanged.connect(
            self._on_roi_origin_combo_box_current_text_changed
        )
        roi_table_widget_layout.addRow("X/Y origin:", self._roi_origin_combo_box)

        self._save_widget = QWidget(parent=self)
        save_widget_layout = QGridLayout()
        self._save_widget.setLayout(save_widget_layout)
        self._roi_file_line_edit = QLineEdit(parent=self)
        self._roi_file_line_edit.setReadOnly(True)
        self._roi_file_line_edit.addAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            QLineEdit.ActionPosition.TrailingPosition,
        ).triggered.connect(self._on_roi_file_line_edit_browse_action_triggered)
        save_widget_layout.addWidget(self._roi_file_line_edit, 0, 0, 1, 2)
        self._autosave_roi_file_check_box = QCheckBox(
            "Autosave", parent=self._save_widget
        )
        self._autosave_roi_file_check_box.stateChanged.connect(
            self._on_autosave_roi_file_check_box_state_changed
        )
        save_widget_layout.addWidget(self._autosave_roi_file_check_box, 1, 0, 1, 1)
        self._save_push_button = QPushButton(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "Save",
            parent=self._save_widget,
        )
        self._save_push_button.clicked.connect(self._on_save_push_button_clicked)
        save_widget_layout.addWidget(self._save_push_button, 1, 1, 1, 1)

        self._update_layout(False)
        self.installEventFilter(self)

        if isinstance(self._viewer.layers.selection.active, Shapes):
            self._roi_layer = self._viewer.layers.selection.active
        self._on_roi_layer_changed(None)
        self._viewer.layers.selection.events.active.connect(
            self._on_active_layer_changed
        )

        self._initialized = True

    @property
    def viewer(self) -> Viewer:
        return self._viewer

    @property
    def autosave_roi_file(self) -> bool:
        if self._roi_layer_accessor is not None:
            return self._roi_layer_accessor.autosave_roi_file
        return False

    @autosave_roi_file.setter
    def autosave_roi_file(self, autosave_roi_file: bool) -> None:
        assert self._roi_layer_accessor is not None
        self._roi_layer_accessor.autosave_roi_file = autosave_roi_file
        self._autosave_roi_file_check_box.setChecked(autosave_roi_file)

    @property
    def roi_file(self) -> Optional[Path]:
        if self._roi_layer_accessor is not None:
            return self._roi_layer_accessor.roi_file
        return None

    @roi_file.setter
    def roi_file(self, roi_file: Optional[Path]) -> None:
        assert self._roi_layer_accessor is not None
        self._roi_layer_accessor.roi_file = roi_file
        self._roi_file_line_edit.setText(
            str(roi_file) if roi_file is not None else None
        )

    @property
    def roi_layer(self) -> Optional[Shapes]:
        return self._roi_layer

    @property
    def roi_layer_accessor(self) -> Optional[ROILayerAccessor]:
        return self._roi_layer_accessor

    @property
    def roi_origin(self) -> Optional[ROIOrigin]:
        if self._roi_layer_accessor is not None:
            return self._roi_layer_accessor.roi_origin
        return None

    @roi_origin.setter
    def roi_origin(self, roi_origin: ROIOrigin) -> None:
        assert self._roi_layer_accessor is not None
        self._roi_layer_accessor.roi_origin = roi_origin
        self._roi_origin_combo_box.setCurrentText(str(roi_origin))

    @property
    def roi_table_model(self) -> Optional[ROITableModel]:
        return self._roi_table_model

    @property
    def new_roi_name(self) -> Optional[str]:
        if self._roi_layer_accessor is not None:
            return self._roi_layer_accessor.new_roi_name
        return None

    @new_roi_name.setter
    def new_roi_name(self, new_roi_name: str) -> None:
        assert self._roi_layer_accessor is not None
        self._roi_layer_accessor.new_roi_name = new_roi_name
        self._new_roi_name_line_edit.setText(new_roi_name)

    @property
    def current_roi_name(self) -> Optional[str]:
        if self._roi_layer_accessor is not None:
            return self._roi_layer_accessor.current_roi_name
        return None

    @current_roi_name.setter
    def current_roi_name(self, current_roi_name: str) -> None:
        assert self._roi_layer_accessor is not None
        self._roi_layer_accessor.current_roi_name = current_roi_name

    """
    ----------------------------------------------------------------------------
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ParentChange:
            parent = self.parent()
            if isinstance(parent, QDockWidget):
                parent.dockLocationChanged.connect(self._on_dock_location_changed)
        return super(ROIWidget, self).eventFilter(obj, event)

    def get_rois(self) -> MutableSequence[ROIBase]:
        assert self._roi_layer_accessor is not None
        assert self._roi_table_model is not None
        return MutableItemModelSequenceWrapper(
            self._roi_layer_accessor, self._roi_table_model
        )

    def load_roi_file(self) -> None:
        assert self._roi_layer_accessor is not None
        assert self.roi_file is not None

        layer = self.roi_layer
        with h5py.File(self.roi_file, "r") as f:
            roi_names = np.array(list(f.keys()), dtype=str)
            for name in roi_names:
                group = f[name]
                data = group['data'][:]
                shape_type = group.attrs['shape_type']
                layer.add(data, shape_type=shape_type)
        layer.properties = dict(roi_name=roi_names)
        self._roi_layer.refresh()
        self._refresh_roi_table_widget()

    def save_roi_file(self) -> None:
        assert self._roi_layer_accessor is not None
        assert self.roi_file is not None
        path = str(self.roi_file)
        path = path.rstrip(".csv")
        path = Path(path).with_suffix('.h5')
        roi_layer = self.roi_layer
        names = roi_layer.properties['roi_name']
        masks = roi_layer.to_masks()
        with h5py.File(path, "w") as f:
            for i in range(roi_layer.nshapes):
                group = f.create_group(names[i])
                group.create_dataset('data', data=roi_layer.data[i])
                group.attrs["shape_type"] = roi_layer.shape_type[i]
                y, x = np.where(masks[i])
                pixels = np.stack([y, x]).T
                group.create_dataset('pixels', data=pixels)

    """
    ----------------------------------------------------------------------------
    """

    def _create_roi_name(self) -> str:
        assert self._roi_layer_accessor is not None
        desired_roi_name = self.new_roi_name
        existing_roi_names = [roi.name for roi in self._roi_layer_accessor]
        existing_roi_numbers = []
        regex = re.compile(rf"{desired_roi_name} \((?P<roi_number>\d+)\)")
        for existing_roi_name in existing_roi_names:
            m = regex.fullmatch(existing_roi_name)
            if m is not None:
                existing_roi_numbers.append(int(m.group("roi_number")))
        if (
                desired_roi_name not in existing_roi_names
                and len(existing_roi_numbers) == 0
        ):
            return desired_roi_name
        roi_number = 2
        if len(existing_roi_numbers) > 0:
            roi_number = max(existing_roi_numbers) + 1
        return f"{desired_roi_name} ({roi_number})"

    def _on_active_layer_changed(self, event: Event) -> None:
        old_roi_layer = self._roi_layer
        if isinstance(event.value, Shapes):
            self._roi_layer = event.value
        else:
            self._roi_layer = None
        self._on_roi_layer_changed(old_roi_layer)

    def _on_dock_location_changed(self, area: Qt.DockWidgetArea) -> None:
        horizontal = area in (
            Qt.DockWidgetArea.TopDockWidgetArea,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self._update_layout(horizontal)

    def _on_autosave_roi_file_check_box_state_changed(
        self, state: Qt.CheckState
    ) -> None:
        self.autosave_roi_file = state == Qt.CheckState.Checked
        self._refresh_save_widget()
        if self._initialized and state == Qt.CheckState.Checked:
            self.save_roi_file()

    def _on_roi_file_line_edit_browse_action_triggered(self, checked: bool) -> None:
        file_dialog = QFileDialog(
            parent=self,
            caption="Save ROI coordinates as",
            filter="HDF5 file (*.h5)",
        )
        if self.roi_file is not None:
            if self.roi_file.parent.is_dir():
                file_dialog.setDirectory(str(self.roi_file.parent))
            if self.roi_file.is_file():
                file_dialog.selectFile(str(self.roi_file))
        else:
            directory = self._viewer.window.session.fs.getsyspath("")
            file_dialog.setDirectory(directory)
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            path = Path(file_dialog.selectedFiles()[0])
            if path.suffix.lower() != ".h5":
                path = path.with_name(path.name + ".h5")
            self.roi_file = path
            self._refresh_save_widget()
            if self.roi_file.is_file():
                answer = QMessageBox.question(
                    self,
                    "Load existing ROIs",
                    "Do you want to load existing ROIs and add them to the current"
                    " layer, applying the currently selected X/Y origin?",
                    buttons=QMessageBox.StandardButton.No
                            | QMessageBox.StandardButton.Yes,
                    defaultButton=QMessageBox.StandardButton.No,
                )
                if answer == QMessageBox.StandardButton.Yes:
                    self.load_roi_file()

    def _on_roi_layer_changed(self, old_roi_layer: Optional[Shapes]) -> None:
        if old_roi_layer is not None:
            old_roi_layer.events.data.disconnect(self._on_roi_layer_data_changed)
            old_roi_layer.events.properties.disconnect(
                self._on_roi_layer_properties_changed
            )
            old_roi_layer.mouse_drag_callbacks.remove(self._on_roi_layer_mouse_drag)
        if self._roi_layer is not None:
            self._roi_layer_accessor = ROILayerAccessor(self._roi_layer)
            self._roi_table_model = ROITableModel(self._roi_layer_accessor)
            self._roi_layer.events.data.connect(self._on_roi_layer_data_changed)
            self._roi_layer.events.properties.connect(
                self._on_roi_layer_properties_changed
            )
            self._roi_layer.mouse_drag_callbacks.append(self._on_roi_layer_mouse_drag)
            self._roi_layer.text = ROILayerAccessor.ROI_NAME_FEATURES_KEY
            self._roi_layer.text.color = self.ROI_LAYER_TEXT_COLOR
            self.setEnabled(True)
        else:
            self._roi_layer_accessor = None
            self._roi_table_model = None
            self.setEnabled(False)
        old_roi_table_model = self._roi_table_view.model()
        self._roi_table_view.setModel(self._roi_table_model)
        if old_roi_table_model is None and self._roi_table_model is not None:
            for c in range(0, self._roi_table_model.columnCount()):
                self._roi_table_view.setColumnWidth(c, self.DEFAULT_COLUMN_WIDTHS[c])
            self._roi_table_view.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Interactive
            )
        selection_model = self._roi_table_view.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(
                self._on_roi_table_view_selection_changed
            )

        self._refresh_roi_table_widget()
        self._refresh_save_widget()

    def _on_roi_layer_data_changed(self, event: Event) -> None:
        self._refresh_roi_table_widget()
        if self._initialized and self.autosave_roi_file:
            self.save_roi_file()

    def _on_roi_layer_mouse_drag(self, roi_layer: Shapes, event: "MouseEvent") -> None:
        if roi_layer.mode.startswith("add_"):
            self.current_roi_name = self._create_roi_name()
        if roi_layer.mode.startswith("add_") or roi_layer.mode in (
                "select",  # move shape
                "direct",  # move vertex
                "vertex_insert",
                "vertex_remove",
        ):
            yield
            self._refresh_roi_table_widget()
            while event.type == "mouse_move":
                self._refresh_roi_table_widget(row_indices=roi_layer.selected_data)
                yield

    def _on_roi_layer_properties_changed(self, event: Event) -> None:
        self._refresh_roi_table_widget()
        if self._initialized and self.autosave_roi_file:
            self.save_roi_file()

    def _on_roi_origin_combo_box_current_text_changed(self, text: str) -> None:
        self.roi_origin = ROIOrigin(text)
        self._refresh_roi_table_widget()
        if self._initialized and self.autosave_roi_file:
            self.save_roi_file()

    def _on_roi_table_view_context_menu_requested(self, pos: QPoint) -> None:
        index = self._roi_table_view.indexAt(pos)
        if index.isValid():
            menu = QMenu()
            delete_action = menu.addAction(
                self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton),
                "Delete",
            )
            if menu.exec(self._roi_table_view.mapToGlobal(pos)) == delete_action:
                del self._roi_layer_accessor[index.row()]
                self._refresh_roi_table_widget(row_indices=[index.row()])

    def _on_roi_table_view_selection_changed(
        self, selected: QItemSelection, deselected: QItemSelection
    ) -> None:
        row_indices = set(index.row() for index in selected.indexes())
        self._roi_layer.selected_data = row_indices
        self._roi_layer.refresh()

    def _on_save_push_button_clicked(self, checked: bool) -> None:
        self.save_roi_file()

    def _refresh_save_widget(self) -> None:
        self._roi_file_line_edit.setEnabled(not self.autosave_roi_file)
        with QSignalBlocker(self._roi_file_line_edit):
            self._roi_file_line_edit.setText(str(self.roi_file or ""))
        self._autosave_roi_file_check_box.setEnabled(self.roi_file is not None)
        if self._roi_layer_accessor is not None:
            with QSignalBlocker(self._autosave_roi_file_check_box):
                self._autosave_roi_file_check_box.setChecked(self.autosave_roi_file)
        self._save_push_button.setEnabled(
            self.roi_file is not None and not self.autosave_roi_file
        )

    def _refresh_roi_table_widget(
        self, row_indices: Optional[Sequence[int]] = None
    ) -> None:
        if self._roi_table_model is not None:
            if row_indices is not None:
                self._roi_table_model.refresh_rows(row_indices)
            else:
                self._roi_table_model.reset()
        if self._roi_layer_accessor is not None:
            with QSignalBlocker(self._roi_origin_combo_box):
                self._roi_origin_combo_box.setCurrentText(str(self.roi_origin))

    def _update_layout(self, horizontal: bool) -> None:
        layout = self.layout()
        if horizontal:
            layout.addWidget(
                self._save_widget,
                1,
                0,
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            )
            layout.addWidget(self._roi_table_widget, 0, 1, 2, 1)
        else:
            layout.addWidget(self._roi_table_widget, 1, 0)
            layout.addWidget(
                self._save_widget, 2, 0, alignment=Qt.AlignmentFlag.AlignBottom
            )
