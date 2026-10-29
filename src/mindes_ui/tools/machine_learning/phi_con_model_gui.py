"""Dataset preparation workflow for phase concentration machine learning."""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QTimer, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QSplitter, QTableView,
    QTabWidget, QVBoxLayout, QWidget,
)
from ...i18n import tr
from .training_panel import TrainingPanel
from .raw_data_plot import RawDataPlotPanel
from .prediction_panel import PredictionPanel
from .fix_con_prediction.io.read_csv import ThermoCSVData, ThermoMLDataset


class CSVTableModel(QAbstractTableModel):
    """Read-only DataFrame view without creating an item for every CSV cell."""

    def __init__(self, dataframe, parent=None):
        super().__init__(parent)
        self.dataframe = dataframe

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.dataframe)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.dataframe.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self.dataframe.iat[index.row(), index.column()]
        return "" if pd.isna(value) else str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.dataframe.columns[section])
        return str(section + 1)


class PhiConModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("menu.tools.phi_con_model"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._set_adaptive_dialog_size(parent)
        self.df: pd.DataFrame | None = None
        self.csv_path: str | None = None
        self.csv_data: ThermoCSVData | None = None
        self.dataset: ThermoMLDataset | None = None
        self._close_after_training = False
        self._build_ui()

    def _set_adaptive_dialog_size(self, parent):
        screen = QGuiApplication.primaryScreen().availableGeometry()

        if parent is not None:
            pg = parent.geometry()

            # 相对父窗口放大一点，但不超过父窗口
            w = int(pg.width() * 0.9)
            h = int(pg.height() * 0.9)

            # 同时不超过屏幕可用区域
            w = min(w, screen.width() - 40)
            h = min(h, screen.height() - 60)

            # 下限，避免太小
            w = max(w, 980)
            h = max(h, 680)

            self.resize(w, h)

            # 居中到父窗口
            x = pg.x() + (pg.width() - w) // 2
            y = pg.y() + (pg.height() - h) // 2
            x = max(screen.x(), min(x, screen.right() - w))
            y = max(screen.y(), min(y, screen.bottom() - h))
            self.move(x, y)
        else:
            w = int(screen.width() * 0.7)
            h = int(screen.height() * 0.7)
            w = max(1000, min(w, 1600))
            h = max(700, min(h, 1000))
            self.resize(w, h)

    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_load = QPushButton(tr("fitter.load_csv"))
        self.btn_load.clicked.connect(self.on_load_csv)
        top.addWidget(self.btn_load)
        self.lbl_file = QLabel(tr("fitter.no_file"))
        top.addWidget(self.lbl_file, stretch=1)
        root.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)
        self.csv_group = QGroupBox(tr("phi_con.csv_data"))
        csv_layout = QVBoxLayout(self.csv_group)
        self.data_tabs = QTabWidget()
        self.csv_table = self._make_table()
        self.input_table = self._make_table()
        self.output_table = self._make_table()
        self.csv_model = self.csv_table.model()
        self.data_tabs.addTab(self.csv_table, tr("phi_con.raw_data"))
        self.data_tabs.addTab(self.input_table, tr("phi_con.inputs"))
        self.data_tabs.addTab(self.output_table, tr("phi_con.outputs"))
        self.data_tabs.setTabEnabled(1, False)
        self.data_tabs.setTabEnabled(2, False)
        self.raw_plot = RawDataPlotPanel(self)
        self.data_tabs.addTab(self.raw_plot, tr("raw_plot.tab"))
        self.data_tabs.setTabEnabled(3, False)
        csv_layout.addWidget(self.data_tabs)
        splitter.addWidget(self.csv_group)

        self.workflow_tabs = QTabWidget()
        self.create_dataset_page = QWidget()
        page_layout = QVBoxLayout(self.create_dataset_page)
        options = QGroupBox(tr("phi_con.dataset_options"))
        form = QFormLayout(options)
        self.include_temperature = QCheckBox(tr("phi_con.include_temperature"))
        self.include_temperature.setChecked(True)
        form.addRow(self.include_temperature)
        self.threshold_choice = QComboBox()
        for key, value in (("phi_con.auto", "auto"), ("phi_con.manual", "manual"), ("phi_con.none", "none")):
            self.threshold_choice.addItem(tr(key), value)
        form.addRow(tr("phi_con.threshold"), self.threshold_choice)
        self.threshold_value = QDoubleSpinBox()
        self.threshold_value.setDecimals(8)
        self.threshold_value.setRange(0.0, 0.99999999)
        self.threshold_value.setSingleStep(0.01)
        self.threshold_value.setValue(0.1)
        self.threshold_value.setEnabled(False)
        form.addRow(tr("phi_con.threshold_value"), self.threshold_value)
        self.threshold_info = QLabel()
        self.threshold_info.setWordWrap(True)
        form.addRow(self.threshold_info)
        self.threshold_help = QLabel(tr("phi_con.threshold_help"))
        self.threshold_help.setWordWrap(True)
        form.addRow(self.threshold_help)
        page_layout.addWidget(options)
        self.btn_create_dataset = QPushButton(tr("phi_con.create_dataset"))
        self.btn_create_dataset.setEnabled(False)
        self.btn_create_dataset.clicked.connect(self.on_create_dataset)
        page_layout.addWidget(self.btn_create_dataset)
        page_layout.addStretch()
        self.workflow_tabs.addTab(self.create_dataset_page, tr("phi_con.create_dataset"))
        self.training_panel = TrainingPanel(self)
        self.training_panel.busy_changed.connect(self._training_busy_changed)
        self.training_panel.settled.connect(self._training_settled)
        self.workflow_tabs.addTab(self.training_panel, tr("phi_con.train_model"))
        self.prediction_panel = PredictionPanel(self)
        self.prediction_panel.busy_changed.connect(self._prediction_busy_changed)
        self.prediction_panel.settled.connect(self._prediction_settled)
        self.workflow_tabs.addTab(self.prediction_panel, tr("phi_con.predict"))
        self.workflow_tabs.setTabEnabled(2, False)
        splitter.addWidget(self.workflow_tabs)
        splitter.setSizes([620, 400])
        self.lbl_status = QLabel(tr("phi_con.begin"))
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)
        self.include_temperature.toggled.connect(self._options_changed)
        self.threshold_choice.currentIndexChanged.connect(self._options_changed)
        self.threshold_value.valueChanged.connect(self._options_changed)

    def _make_table(self):
        table = QTableView()
        table.setModel(CSVTableModel(pd.DataFrame(), table))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setDefaultSectionSize(150)
        return table

    def _set_table_data(self, table, dataframe):
        previous = table.model()
        model = CSVTableModel(dataframe, table)
        table.setModel(model)
        for index, name in enumerate(dataframe.columns):
            table.setColumnHidden(index, name == "sample_id")
        previous.deleteLater()
        return model

    def _clear_dataset(self):
        self.dataset = None
        self.training_panel.set_dataset(None)
        self.prediction_panel.reset()
        self.workflow_tabs.setTabEnabled(2, False)
        self.data_tabs.setCurrentIndex(0)
        for index, table in ((1, self.input_table), (2, self.output_table)):
            self._set_table_data(table, pd.DataFrame())
            self.data_tabs.setTabEnabled(index, False)

    def _options_changed(self, *_):
        self.threshold_value.setEnabled(self.threshold_choice.currentData() == "manual")
        self._clear_dataset()
        if self.csv_data is not None:
            self.lbl_status.setText(tr("phi_con.ready"))

    def on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("fitter.dialog.select_csv"), "", tr("filter.csv_all")
        )
        if not path:
            return
        try:
            data = ThermoCSVData(path)
        except Exception as error:
            QMessageBox.critical(self, tr("fitter.failed_load"), str(error))
            return
        self.csv_data = data
        self.df = data.df
        self.csv_path = path
        self.csv_model = self._set_table_data(self.csv_table, self.df)
        self.raw_plot.set_data(data)
        self.data_tabs.setTabEnabled(3, True)
        self._clear_dataset()
        self.lbl_file.setText(os.path.basename(path))
        self.lbl_file.setToolTip(path)
        detected = data.detected_phase_threshold["threshold"]
        self.threshold_info.setText(
            tr("phi_con.detected", value=f"{detected:g}")
            if detected is not None else tr("phi_con.no_threshold")
        )
        self.btn_create_dataset.setEnabled(not self.df.empty)
        self.lbl_status.setText(tr("phi_con.csv_loaded", rows=len(self.df), columns=len(self.df.columns)))

    def on_create_dataset(self):
        if self.csv_data is None:
            return
        choice = self.threshold_choice.currentData()
        threshold = self.threshold_value.value() if choice == "manual" else (None if choice == "none" else "auto")
        try:
            dataset = self.csv_data.make_dataset(
                include_temperature=self.include_temperature.isChecked(),
                phase_threshold=threshold,
            )
            if not len(dataset.x):
                raise ValueError(tr("phi_con.empty_dataset"))
            if not np.isfinite(dataset.x).all() or not np.isfinite(dataset.y).all():
                raise ValueError(tr("phi_con.invalid_values"))
        except Exception as error:
            QMessageBox.critical(self, tr("phi_con.create_failed"), str(error))
            return
        self.dataset = dataset
        self.training_panel.set_dataset(dataset)
        self.prediction_panel.reset()
        self.workflow_tabs.setTabEnabled(2, False)
        self._set_table_data(self.input_table, pd.DataFrame(dataset.x, columns=dataset.metadata["input_columns"]))
        self._set_table_data(self.output_table, pd.DataFrame(dataset.y, columns=dataset.metadata["output_columns"]))
        self.data_tabs.setTabEnabled(1, True)
        self.data_tabs.setTabEnabled(2, True)
        self.data_tabs.setCurrentIndex(1)
        self.lbl_status.setText(tr("phi_con.created", rows=len(dataset.x), inputs=dataset.x.shape[1], outputs=dataset.y.shape[1]))


    @property
    def is_training(self):
        return self.training_panel.is_training or self.prediction_panel.is_predicting

    def _training_busy_changed(self, busy):
        self.btn_load.setEnabled(not busy)
        self.create_dataset_page.setEnabled(not busy)
        if busy:
            self.prediction_panel.reset()
            self.workflow_tabs.setTabEnabled(2, False)

    def _training_settled(self):
        if self._close_after_training:
            QTimer.singleShot(0, self.close)
        elif self.training_panel.trainer is not None:
            self.workflow_tabs.setTabEnabled(2, True)
            self.prediction_panel.set_model(self.dataset, self.training_panel.trainer, self.df)

    def _prediction_busy_changed(self, busy):
        self.btn_load.setEnabled(not busy)
        self.create_dataset_page.setEnabled(not busy)
        self.training_panel.setEnabled(not busy)

    def _prediction_settled(self):
        if self._close_after_training:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        if self.is_training:
            self._close_after_training = True
            self.training_panel.cancel_training()
            self.prediction_panel.cancel_prediction()
            event.ignore()
            return
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    dialog = PhiConModelDialog()
    dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
