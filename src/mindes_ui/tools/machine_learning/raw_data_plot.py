"""Discrete thermodynamic slices and an additive raw-data plotting panel."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.text import Text
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QPushButton, QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from ...i18n import tr


def label(key, **values):
    return tr("raw_plot." + key, **values)


def slice_curve(frame, *, x, y, sweep, dependent, fixed, low, high, variables):
    """Select recorded rows only; reject underconstrained or multivalued slices."""
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError(label("invalid_range"))
    expected = set(variables) - {sweep, dependent}
    if set(fixed) != expected:
        raise ValueError(label("missing_conditions"))
    mask = frame[sweep].between(low, high)
    for column, value in fixed.items():
        mask &= frame[column].eq(value)
    rows = frame.loc[mask].copy()
    if rows.empty:
        raise ValueError(label("no_rows"))
    # A single sweep value must describe one system state and one plotted pair.
    columns = list(dict.fromkeys([sweep, *([dependent] if dependent else []), x, y]))
    numeric = rows[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(label("nonfinite"))
    distinct = numeric.drop_duplicates()
    if distinct[sweep].duplicated().any():
        raise ValueError(label("ambiguous"))
    return distinct.sort_values(sweep, kind="stable")


class DiscreteValueControl(QWidget):
    """Slider indices preserve exact source values; edits snap to the nearest."""
    changed = Signal()

    def __init__(self, values, value=None, parent=None):
        super().__init__(parent)
        values = np.asarray(values, dtype=float)
        self.values = np.unique(values[np.isfinite(values)])
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, len(self.values) - 1))
        self.edit = QLineEdit()
        self.edit.setMaximumWidth(125)
        self.edit.setToolTip(label("snap_hint"))
        row.addWidget(self.slider, 1)
        row.addWidget(self.edit)
        self.slider.valueChanged.connect(self._slider_changed)
        self.edit.editingFinished.connect(self._edited)
        self.setEnabled(bool(len(self.values)))
        self.set_value(self.values[0] if value is None and len(self.values) else value)

    def value(self):
        return float(self.values[self.slider.value()]) if len(self.values) else float("nan")

    def set_value(self, value):
        if not len(self.values):
            self.edit.clear()
            return
        if value is not None and np.isfinite(value):
            self.slider.setValue(int(np.argmin(np.abs(self.values - value))))
        self.edit.setText(format(self.value(), ".15g"))

    def _slider_changed(self, *_):
        self.edit.setText(format(self.value(), ".15g"))
        self.changed.emit()

    def _edited(self):
        try:
            value = float(self.edit.text())
        except ValueError:
            value = self.value()
        self.set_value(value)
        self.changed.emit()


class RawDataPlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = pd.DataFrame()
        self.variables = []
        self.groups = {}
        self.fixed_controls = {}
        self.curves = []
        self._condition_values = {}
        self._rebuilding = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.controls = QWidget()
        form = QFormLayout(self.controls)
        scroll.setWidget(self.controls)
        splitter.addWidget(scroll)
        self.x_axis = QComboBox()
        self.y_axis = QComboBox()
        axes = QWidget()
        row = QHBoxLayout(axes)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label("x")))
        row.addWidget(self.x_axis, 1)
        row.addWidget(QLabel(label("y")))
        row.addWidget(self.y_axis, 1)
        form.addRow(axes)
        self.sweep = QComboBox()
        self.dependent = QComboBox()
        form.addRow(label("sweep"), self.sweep)
        form.addRow(label("dependent"), self.dependent)
        self.range_box = QWidget()
        self.range_form = QFormLayout(self.range_box)
        self.range_form.setContentsMargins(0, 0, 0, 0)
        form.addRow(self.range_box)
        self.fixed_box = QWidget()
        self.fixed_form = QFormLayout(self.fixed_box)
        self.fixed_form.setContentsMargins(0, 0, 0, 0)
        form.addRow(self.fixed_box)
        self.curve_name = QLineEdit()
        self.curve_name.setPlaceholderText(label("automatic_name"))
        form.addRow(label("name"), self.curve_name)
        hint = QLabel(label("hint"))
        hint.setWordWrap(True)
        form.addRow(hint)
        self.add_button = QPushButton(label("add"))
        self.add_button.clicked.connect(self.add_curve)

        plot = QWidget()
        layout = QVBoxLayout(plot)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5, 3), layout="constrained", facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(180, 180)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, plot)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        splitter.addWidget(plot)
        splitter.setSizes([260, 340])
        root.addWidget(self.add_button)
        self.curve_list = QListWidget()
        self.curve_list.setMaximumHeight(65)
        root.addWidget(self.curve_list)
        buttons = QHBoxLayout()
        self.remove_button = QPushButton(label("remove"))
        self.clear_button = QPushButton(label("clear"))
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.clear_button)
        root.addLayout(buttons)
        self.status = QLabel(label("load_first"))
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.remove_button.clicked.connect(self.remove_curve)
        self.clear_button.clicked.connect(self.clear_curves)
        self.x_axis.currentTextChanged.connect(self._axis_changed)
        self.y_axis.currentTextChanged.connect(self._axis_changed)
        self.sweep.currentTextChanged.connect(self._sweep_changed)
        self.dependent.currentIndexChanged.connect(self._rebuild_fixed)
        self.controls.setEnabled(False)
        self.add_button.setEnabled(False)
        available = {font.name for font in font_manager.fontManager.ttflist}
        self._fonts = [name for name in (self.font().family(), "Microsoft YaHei",
            "Yu Gothic", "Malgun Gothic", "Noto Sans CJK SC", "DejaVu Sans") if name in available]
        self._redraw()

    def set_data(self, data):
        self._rebuilding = True
        self.frame = data.df
        numeric = [column for column in self.frame if pd.api.types.is_numeric_dtype(self.frame[column])
                   and np.isfinite(self.frame[column].to_numpy(dtype=float)).any()]
        self.groups = {"temperature": list(data.temperature_columns),
                       "composition": list(data.con_columns), "phase": list(data.phi_columns)}
        self.variables = [column for group in self.groups.values() for column in group if column in numeric]
        self._condition_values = {}
        for column in self.variables:
            values = self.frame[column].to_numpy(dtype=float)
            self._condition_values[column] = float(values[np.isfinite(values)][0])
        self.fixed_controls = {}
        for combo in (self.x_axis, self.y_axis):
            combo.clear()
            combo.addItems(numeric)
        self.sweep.clear()
        self.sweep.addItems(self.variables)
        default_x = next(iter(data.con_columns), next(iter(self.variables), ""))
        self.x_axis.setCurrentText(default_x)
        default_y = next(iter(data.phase_con_columns), next(iter(numeric), ""))
        self.y_axis.setCurrentText(default_y)
        self.sweep.setCurrentText(default_x)
        self.controls.setEnabled(bool(numeric and self.variables))
        self.add_button.setEnabled(bool(numeric and self.variables))
        self._rebuilding = False
        self.clear_curves()
        self._sweep_changed()

    def _axis_changed(self, *_):
        if self._rebuilding or self.curves:
            return
        for axis in (self.x_axis.currentText(), self.y_axis.currentText()):
            if axis in self.variables:
                self.sweep.setCurrentText(axis)
                break
        self._redraw()

    def _sweep_changed(self, *_):
        if self._rebuilding:
            return
        self._remember_conditions()
        self._rebuilding = True
        sweep = self.sweep.currentText()
        previous = self.dependent.currentData()
        self.dependent.clear()
        self.dependent.addItem(label("none"), None)
        candidates = []
        for group in ("composition", "phase"):
            if sweep in self.groups.get(group, []):
                candidates = [column for column in self.groups[group] if column != sweep and column in self.variables]
        for column in candidates:
            self.dependent.addItem(column, column)
        if previous in candidates:
            self.dependent.setCurrentIndex(self.dependent.findData(previous))
        elif candidates:
            self.dependent.setCurrentIndex(1)
        self.dependent.setEnabled(bool(candidates))
        while self.range_form.rowCount():
            self.range_form.removeRow(0)
        values = self.frame[sweep].to_numpy(dtype=float) if sweep else []
        finite = np.asarray(values)[np.isfinite(values)]
        self.low = DiscreteValueControl(values)
        self.high = DiscreteValueControl(values, max(finite) if len(finite) else None)
        self.range_form.addRow(label("minimum"), self.low)
        self.range_form.addRow(label("maximum"), self.high)
        self._rebuilding = False
        self._rebuild_fixed()

    def _remember_conditions(self):
        self._condition_values.update({column: control.value() for column, control in self.fixed_controls.items()})

    def _rebuild_fixed(self, *_):
        if self._rebuilding:
            return
        self._remember_conditions()
        self.fixed_controls = {}
        while self.fixed_form.rowCount():
            self.fixed_form.removeRow(0)
        excluded = {self.sweep.currentText(), self.dependent.currentData()}
        for column in self.variables:
            if column in excluded:
                continue
            control = DiscreteValueControl(self.frame[column].to_numpy(dtype=float), self._condition_values.get(column))
            self.fixed_controls[column] = control
            self.fixed_form.addRow(label("fixed", column=column), control)

    def add_curve(self):
        if self.frame.empty or not self.variables:
            return
        # Also commit edits when Add is invoked without a focus change.
        for control in [self.low, self.high, *self.fixed_controls.values()]:
            control._edited()
        x, y = self.x_axis.currentText(), self.y_axis.currentText()
        sweep, dependent = self.sweep.currentText(), self.dependent.currentData()
        fixed = {column: control.value() for column, control in self.fixed_controls.items()}
        try:
            rows = slice_curve(self.frame, x=x, y=y, sweep=sweep, dependent=dependent,
                fixed=fixed, low=self.low.value(), high=self.high.value(), variables=self.variables)
        except ValueError as error:
            self.status.setText(str(error))
            return
        conditions = ", ".join(f"{column}={value:.8g}" for column, value in fixed.items())
        name = self.curve_name.text().strip() or conditions or label("curve", number=len(self.curves) + 1)
        details = f"{sweep}: {self.low.value():.8g}–{self.high.value():.8g}"
        if dependent:
            details += "\n" + label("dependent_value", column=dependent)
        self.curves.append(dict(x=rows[x].to_numpy(copy=True), y=rows[y].to_numpy(copy=True), label=name))
        self.curve_list.addItem(name)
        self.curve_list.item(self.curve_list.count() - 1).setToolTip(details + "\n" + conditions)
        self.curve_list.setCurrentRow(self.curve_list.count() - 1)
        self._set_axis_lock()
        self._redraw()
        self.status.setText(label("added", points=len(rows), curves=len(self.curves)))

    def _set_axis_lock(self):
        for control in (self.x_axis, self.y_axis, self.sweep):
            control.setEnabled(not self.curves)

    def remove_curve(self):
        index = self.curve_list.currentRow()
        if index < 0:
            return
        self.curves.pop(index)
        self.curve_list.takeItem(index)
        self._set_axis_lock()
        self._redraw()
        self.status.setText(label("count", curves=len(self.curves)))

    def clear_curves(self):
        self.curves.clear()
        self.curve_list.clear()
        self._set_axis_lock()
        self._redraw()
        self.status.setText(label("ready"))

    def _redraw(self):
        self.axes.clear()
        self.axes.set_facecolor("white")
        self.axes.set_xlabel(self.x_axis.currentText())
        self.axes.set_ylabel(self.y_axis.currentText())
        self.axes.grid(True, alpha=0.2)
        self.axes.tick_params(direction="in", top=True, right=True)
        for curve in self.curves:
            self.axes.plot(curve["x"], curve["y"], marker="o", markersize=3,
                           linewidth=1.5, label=curve["label"])
        if self.curves:
            self.axes.legend(fontsize=8)
        for text in self.figure.findobj(Text):
            text.set_fontfamily(self._fonts)
        self.toolbar.update()
        self.canvas.draw_idle()
