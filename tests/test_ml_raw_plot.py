"""Recorded-data slice constraints, discrete controls, and additive plotting."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from mindes_ui.tools.machine_learning.fix_con_prediction.io.read_csv import ThermoCSVData
from mindes_ui.tools.machine_learning.raw_data_plot import (
    DiscreteValueControl, RawDataPlotPanel, slice_curve,
)


def raw_frame():
    rows = []
    for temperature in (1000., 1200.):
        for con_c in (0.05, 0.15):
            for con_a in (0.1, 0.2, 0.3):
                for phi_a in (0.25, 0.5):
                    rows.append(dict(sample_id=len(rows) + 1, temperature=temperature,
                        con_A=con_a, con_B=1-con_a-con_c, con_C=con_c,
                        phi_A=phi_a, phi_B=1-phi_a,
                        phase_con_A_C=con_a * temperature / 1000 + phi_a,
                        phase_con_B_C=con_c))
    return pd.DataFrame(rows)


class RawPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "raw.csv"
        raw_frame().to_csv(self.path, index=False)
        self.data = ThermoCSVData(self.path)
        self.panel = RawDataPlotPanel()
        self.addCleanup(self.panel.deleteLater)
        self.panel.set_data(self.data)

    def test_composition_slices_add_temperatures_without_changing_previous_curve(self):
        p = self.panel
        original = self.data.df.copy(deep=True)
        self.assertEqual(p.sweep.currentText(), "con_A")
        self.assertEqual(p.dependent.currentData(), "con_B")
        self.assertEqual(set(p.fixed_controls), {"temperature", "con_C", "phi_A", "phi_B"})
        p.add_curve()
        self.assertEqual(len(p.curves), 1, p.status.text())
        np.testing.assert_allclose(p.curves[0]["x"], [0.1, 0.2, 0.3])
        first_y = p.curves[0]["y"].copy()
        self.assertFalse(p.x_axis.isEnabled())
        self.assertFalse(p.y_axis.isEnabled())
        p.fixed_controls["temperature"].set_value(1200)
        p.add_curve()
        self.assertEqual(len(p.axes.lines), 2)
        np.testing.assert_array_equal(p.curves[0]["y"], first_y)
        self.assertFalse(np.array_equal(p.curves[1]["y"], first_y))
        pd.testing.assert_frame_equal(self.data.df, original)
        p.remove_curve()
        self.assertFalse(p.x_axis.isEnabled())
        p.clear_curves()
        self.assertTrue(p.x_axis.isEnabled())
        self.assertTrue(p.y_axis.isEnabled())

    def test_temperature_sweep_fixes_every_composition_and_phase_fraction(self):
        p = self.panel
        p.x_axis.setCurrentText("temperature")
        self.assertEqual(p.sweep.currentText(), "temperature")
        self.assertIsNone(p.dependent.currentData())
        self.assertFalse(p.dependent.isEnabled())
        self.assertEqual(set(p.fixed_controls), {"con_A", "con_B", "con_C", "phi_A", "phi_B"})
        p.add_curve()
        self.assertEqual(len(p.curves), 1, p.status.text())
        np.testing.assert_array_equal(p.curves[0]["x"], [1000, 1200])

    def test_property_on_x_and_variable_on_y(self):
        p = self.panel
        p.x_axis.setCurrentText("phase_con_A_C")
        p.y_axis.setCurrentText("temperature")
        self.assertEqual(p.sweep.currentText(), "temperature")
        p.add_curve()
        np.testing.assert_array_equal(p.curves[0]["y"], [1000, 1200])

    def test_discrete_values_slider_and_typed_input(self):
        control = DiscreteValueControl([1200, 1000, 1200, float("nan")])
        self.addCleanup(control.deleteLater)
        control.edit.setText("1162")
        control.edit.editingFinished.emit()
        self.assertEqual(control.value(), 1200)
        self.assertEqual(control.edit.text(), "1200")
        control.slider.setValue(0)
        self.assertEqual(control.edit.text(), "1000")
        control.edit.setText("invalid")
        control.edit.editingFinished.emit()
        self.assertEqual(control.value(), 1000)
        control.edit.setText("1e9")
        control.edit.editingFinished.emit()
        self.assertEqual(control.value(), 1200)

    def test_range_and_empty_slice_do_not_add_or_lock_axes(self):
        p = self.panel
        p.low.set_value(0.3)
        p.high.set_value(0.1)
        p.add_curve()
        self.assertEqual(len(p.curves), 0)
        self.assertTrue(p.x_axis.isEnabled())
        p.low.set_value(0.1)
        p.high.set_value(0.2)
        p.fixed_controls["phi_A"].set_value(0.5)
        # phi_B=0.75 is an existing value but cannot coexist with phi_A=0.5.
        p.add_curve()
        self.assertEqual(len(p.curves), 0)
        p.fixed_controls["phi_B"].set_value(0.5)
        p.add_curve()
        np.testing.assert_allclose(p.curves[0]["x"], [0.1, 0.2])

    def test_reload_clears_curves_and_unlocks_axes(self):
        p = self.panel
        p.add_curve()
        p.set_data(self.data)
        self.assertEqual(len(p.curves), 0)
        self.assertEqual(p.curve_list.count(), 0)
        self.assertTrue(p.x_axis.isEnabled())

    def test_ambiguous_states_rejected_and_exact_duplicates_collapsed(self):
        p = self.panel
        frame = self.data.df
        kwargs = dict(x="con_A", y="phase_con_A_C", sweep="con_A", dependent="con_B",
            fixed={key: control.value() for key, control in p.fixed_controls.items()},
            low=0.1, high=0.3, variables=p.variables)
        result = slice_curve(pd.concat([frame, frame]), **kwargs)
        self.assertEqual(len(result), 3)
        conflicting = frame.iloc[[0]].copy()
        conflicting["con_B"] = 0.7
        with self.assertRaises(ValueError):
            slice_curve(pd.concat([frame, conflicting]), **kwargs)
        conflicting = frame.iloc[[0]].copy()
        conflicting["phase_con_A_C"] = float("nan")
        with self.assertRaises(ValueError):
            slice_curve(pd.concat([frame, conflicting]), **kwargs)
        kwargs["fixed"].pop("phi_A")
        with self.assertRaises(ValueError):
            slice_curve(frame, **kwargs)

    def test_phase_sweep_has_only_one_dependent_phase(self):
        p = self.panel
        p.x_axis.setCurrentText("phi_A")
        self.assertEqual(p.dependent.currentData(), "phi_B")
        self.assertEqual(set(p.fixed_controls), {"temperature", "con_A", "con_B", "con_C"})
        p.add_curve()
        self.assertEqual(len(p.curves), 1, p.status.text())
        np.testing.assert_allclose(p.curves[0]["x"], [0.25, 0.5])
