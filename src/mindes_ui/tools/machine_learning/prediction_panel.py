"""Full-dataset inference and focused phase-concentration comparisons."""
from __future__ import annotations

import threading
import textwrap

import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.text import Text
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QLabel, QPushButton,
    QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from ...i18n import tr
from .raw_data_plot import DiscreteValueControl


def label(key, **params):
    return tr("ml_predict." + key, **params)


def output_groups(inputs, outputs):
    """Group the same component across phases, preferring the longest phase name."""
    phases = sorted((name.removeprefix("phi_") for name in inputs if name.startswith("phi")),
                    key=len, reverse=True)
    groups = {}
    for index, column in enumerate(outputs):
        component = column
        for phase in phases:
            prefix = "phase_con_" + phase
            if column == prefix or column.startswith(prefix + "_"):
                component = column[len(prefix):].lstrip("_")
                break
        groups.setdefault(component, []).append(index)
    return groups


def comparison_slice(frame, fixed):
    """Select composition/temperature while retaining all phase states in source order."""
    mask = np.ones(len(frame), dtype=bool)
    for name, value in fixed.items():
        mask &= frame[name].eq(value).to_numpy()
    rows = frame.loc[mask]
    if rows.empty:
        raise ValueError(tr("raw_plot.no_rows"))
    return rows


class PredictionWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, dataset, trainer, parent=None):
        super().__init__(parent)
        self.dataset, self.trainer = dataset, trainer
        self.cancel = threading.Event()

    def run(self):
        try:
            result = np.empty(self.dataset.y.shape, dtype=np.float32)
            for start in range(0, len(self.dataset.x), 8192):
                if self.cancel.is_set():
                    return
                stop = min(start + 8192, len(self.dataset.x))
                values = np.asarray(self.trainer.predict(self.dataset.x[start:stop]))
                if values.shape != result[start:stop].shape or not np.isfinite(values).all():
                    raise ValueError(label("invalid_result"))
                result[start:stop] = values
            if not self.cancel.is_set():
                self.completed.emit(result)
        except Exception as error:
            if not self.cancel.is_set():
                self.failed.emit(str(error))


class PredictionPanel(QWidget):
    busy_changed = Signal(bool)
    settled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset = self.trainer = self.predictions = self.worker = None
        self.frame = pd.DataFrame()
        self.fixed_controls = {}
        root = QVBoxLayout(self)
        note = QLabel(label("note"))
        note.setWordWrap(True)
        root.addWidget(note)
        self.predict_button = QPushButton(label("predict"))
        self.predict_button.clicked.connect(self.start_prediction)
        root.addWidget(self.predict_button)
        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.controls = QWidget()
        form = QFormLayout(self.controls)
        scroll.setWidget(self.controls)
        splitter.addWidget(scroll)
        self.component = QComboBox()
        form.addRow(label("component"), self.component)
        self.fixed_box = QWidget()
        self.fixed_form = QFormLayout(self.fixed_box)
        self.fixed_form.setContentsMargins(0, 0, 0, 0)
        form.addRow(self.fixed_box)
        self.show_phases = QCheckBox(label("show_phases"))
        self.show_phases.setChecked(True)
        form.addRow(self.show_phases)
        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5, 3.5), layout="none")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(200, 210)
        self.axes = self.figure.add_subplot(111)
        self.legend = None
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        splitter.addWidget(plot)
        splitter.setSizes([230, 360])
        self.plot_button = QPushButton(label("plot"))
        self.plot_button.clicked.connect(self.plot_comparison)
        root.addWidget(self.plot_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        available = {font.name for font in font_manager.fontManager.ttflist}
        self._fonts = [name for name in (self.font().family(), "Microsoft YaHei", "Yu Gothic",
            "Malgun Gothic", "Noto Sans CJK SC", "DejaVu Sans") if name in available]
        self.canvas.mpl_connect("resize_event", self._layout_plot)
        self.reset()

    @property
    def is_predicting(self):
        return self.worker is not None

    def reset(self):
        self.dataset = self.trainer = self.predictions = None
        self.frame = pd.DataFrame()
        self.controls.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.plot_button.setEnabled(False)
        self.axes.clear()
        self._draw()
        self.status.setText(label("need_model"))

    def set_model(self, dataset, trainer, raw_frame=None):
        self.reset()
        self.dataset, self.trainer = dataset, trainer
        inputs = dataset.metadata["input_columns"]
        self.outputs = dataset.metadata["output_columns"]
        self.frame = pd.DataFrame(dataset.x, columns=inputs)
        # Temperature can still define a recorded slice when excluded from model inputs.
        if "temperature" not in inputs and raw_frame is not None and "temperature" in raw_frame:
            if len(raw_frame) == len(self.frame):
                self.frame["temperature"] = raw_frame["temperature"].to_numpy()
        self.conditions = list(self.frame.columns)
        for index, name in enumerate(self.outputs):
            self.frame[name] = dataset.y[:, index]
        self.phases = [name for name in inputs if name.startswith("phi")]
        self.groups = output_groups(inputs, self.outputs)
        self.component.clear()
        for name in self.groups:
            self.component.addItem(name or label("concentration"), name)
        self.fixed_controls = {}
        while self.fixed_form.rowCount():
            self.fixed_form.removeRow(0)
        for name in self.conditions:
            if name != "temperature" and not name.startswith("con"):
                continue
            control = DiscreteValueControl(self.frame[name].to_numpy(), float(self.frame[name].iloc[0]))
            self.fixed_controls[name] = control
            self.fixed_form.addRow(tr("raw_plot.fixed", column=name), control)
        self.start_prediction()

    def start_prediction(self):
        if self.trainer is None or self.is_predicting:
            return
        self.predictions = None
        self.axes.clear()
        self._draw()
        self.controls.setEnabled(False)
        self.plot_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.status.setText(label("running", rows=len(self.dataset.x)))
        self.worker = PredictionWorker(self.dataset, self.trainer, self)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.busy_changed.emit(True)
        self.worker.start()

    def cancel_prediction(self):
        if self.worker is not None:
            self.worker.cancel.set()

    def _completed(self, predictions):
        self.predictions = predictions
        self.controls.setEnabled(True)
        self.plot_button.setEnabled(True)
        self.plot_comparison()

    def _failed(self, error):
        self.status.setText(label("failed", error=error))

    def _finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.predict_button.setEnabled(self.trainer is not None)
        self.busy_changed.emit(False)
        self.settled.emit()

    def plot_comparison(self):
        if self.predictions is None:
            return
        for control in self.fixed_controls.values():
            control._edited()
        fixed = {name: control.value() for name, control in self.fixed_controls.items()}
        indices = self.groups[self.component.currentData()]
        targets = [self.outputs[index] for index in indices]
        self.axes.clear()
        self.axes.set_axis_on()
        try:
            rows = comparison_slice(self.frame[self.conditions + targets], fixed)
        except ValueError as error:
            message = str(error)
            self.status.setText(message)
            self.axes.set_axis_off()
            self.axes.text(0.5, 0.5, message, transform=self.axes.transAxes,
                           ha="center", va="center", wrap=True, fontsize=9)
            self._draw()
            return
        x = np.arange(len(rows))
        if self.show_phases.isChecked():
            for name in self.phases:
                self.axes.plot(x, rows[name], linewidth=1, alpha=0.65, label=name)
        for index in indices:
            name = self.outputs[index]
            line, = self.axes.plot(x, rows[name], linewidth=1.6, marker="o", markersize=3,
                                  label=label("recorded", column=name))
            self.axes.plot(x, self.predictions[rows.index, index], color=line.get_color(),
                           linestyle="--", linewidth=1.6, label=label("predicted", column=name))
        self.axes.set_xlabel(label("sample_position"))
        self.axes.set_ylabel(label("value"))
        self.axes.grid(True, alpha=0.2)
        error = self.predictions[np.ix_(rows.index.to_numpy(), indices)].astype(float) - rows[targets].to_numpy()
        self.status.setText(label("summary", total=len(self.dataset.x), rows=len(rows),
                                  rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error)))))
        self._draw()

    def _draw(self):
        self._layout_plot()
        self.toolbar.update()
        self.canvas.draw_idle()

    def _layout_plot(self, event=None):
        """Reserve measured space for a figure legend, independent of axes size."""
        if self.legend is not None:
            self.legend.remove()
            self.legend = None
        width, height = self.figure.bbox.size
        renderer = self.canvas.get_renderer()
        scale = self.figure.dpi / 100
        gap = 12 * scale
        legend_height = 0
        handles, labels = self.axes.get_legend_handles_labels()
        if handles:
            # Start with two columns, falling back to wrapped single-column text
            # when the panel is narrow. The legend remains part of saved figures.
            for columns in (2, 1):
                wrapped = [textwrap.fill(text, max(12, int((width / columns - 65 * scale) / (5 * scale))))
                           for text in labels]
                self.legend = self.figure.legend(handles, wrapped, loc="lower center",
                    bbox_to_anchor=(0.5, gap / height), ncols=columns,
                    frameon=False, borderaxespad=0,
                    prop={"family": self._fonts, "size": 7})
                bounds = self.legend.get_window_extent(renderer)
                if bounds.width <= width - 2 * gap or columns == 1:
                    legend_height = bounds.height
                    break
                self.legend.remove()
        # Fixed pixel margins avoid the feedback loop between an external axes
        # legend and constrained_layout when the same plot is drawn repeatedly.
        bottom = 52 * scale + (legend_height + 2 * gap if handles else 0)
        top, left, right = 12 * scale, 65 * scale, 16 * scale
        self.canvas.setMinimumHeight(int((bottom + top + 110 * scale) / self.canvas.device_pixel_ratio))
        self.axes.set_position([left / width, bottom / height,
                                max(1, width - left - right) / width,
                                max(1, height - bottom - top) / height])
        for text in self.figure.findobj(Text):
            text.set_fontfamily(self._fonts)
        self.canvas.draw_idle()
