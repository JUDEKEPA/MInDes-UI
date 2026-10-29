"""Prediction alignment, component comparisons, and worker lifecycle checks."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
import unittest

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from mindes_ui.tools.machine_learning.fix_con_prediction.io.read_csv import ThermoMLDataset
from mindes_ui.tools.machine_learning.prediction_panel import (
    PredictionPanel, PredictionWorker, comparison_slice, output_groups,
)


def data():
    x = np.array([[t, c, phi, 1-phi] for t in (1000, 1200)
                  for c in (0.3, 0.5) for phi in (0., 0.5, 1.)])
    y = np.column_stack([x[:, 1] + x[:, 2], x[:, 1] + x[:, 3],
                         x[:, 1] * x[:, 2], x[:, 1] * x[:, 3]])
    return ThermoMLDataset(x=x, y=y, metadata={
        "input_columns": ["temperature", "con_C", "phi_A", "phi_B"],
        "output_columns": ["phase_con_A_C", "phase_con_B_C", "phase_con_A_D", "phase_con_B_D"]})


class FakeTrainer:
    def __init__(self):
        self.calls = []

    def predict(self, x):
        self.calls.append(x.copy())
        return np.column_stack([x[:, 1] + x[:, 2], x[:, 1] + x[:, 3],
                                x[:, 1] * x[:, 2], x[:, 1] * x[:, 3]]) + 0.01


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = PredictionPanel()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.panel.cancel_prediction()
        if self.panel.worker is not None:
            self.panel.worker.wait(10000)
            self.app.processEvents()
        self.panel.deleteLater()

    def wait(self):
        deadline = time.monotonic() + 10
        while self.panel.is_predicting and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertFalse(self.panel.is_predicting)

    def test_full_prediction_and_component_overlays_preserve_row_alignment(self):
        dataset = data()
        original = dataset.x.copy()
        trainer = FakeTrainer()
        self.panel.set_model(dataset, trainer)
        self.wait()
        p = self.panel
        np.testing.assert_allclose(p.predictions, dataset.y + 0.01, atol=1e-7)
        np.testing.assert_array_equal(dataset.x, original)
        self.assertEqual(p.component.count(), 2)
        self.assertEqual(set(p.fixed_controls), {"temperature", "con_C"})
        self.assertEqual(len(p.axes.lines), 6)
        real, predicted = p.axes.lines[2:4]
        self.assertEqual(real.get_color(), predicted.get_color())
        self.assertEqual(predicted.get_linestyle(), "--")
        np.testing.assert_allclose(real.get_ydata(), dataset.y[:3, 0])
        p.fixed_controls["temperature"].set_value(1200)
        p.fixed_controls["con_C"].edit.setText("0.49")
        p.component.setCurrentIndex(p.component.findData("D"))
        p.plot_comparison()
        np.testing.assert_allclose(p.axes.lines[2].get_ydata(), dataset.y[-3:, 2])
        np.testing.assert_allclose(p.axes.lines[3].get_ydata(), p.predictions[-3:, 2])
        self.assertEqual(len(trainer.calls), 1)  # New slices use cached predictions.
        p.show_phases.setChecked(False)
        p.plot_comparison()
        self.assertEqual(len(p.axes.lines), 4)
        p.reset()
        self.assertIsNone(p.predictions)
        self.assertFalse(p.plot_button.isEnabled())
        self.assertEqual(len(p.axes.lines), 0)

    def test_batches_keep_all_rows_in_order(self):
        dataset = data()
        dataset.x = np.tile(dataset.x, (700, 1))
        dataset.y = np.tile(dataset.y, (700, 1))
        trainer = FakeTrainer()
        worker = PredictionWorker(dataset, trainer)
        results = []
        worker.completed.connect(results.append)
        worker.run()
        self.assertEqual([len(x) for x in trainer.calls], [8192, 208])
        np.testing.assert_allclose(results[0], dataset.y + 0.01, atol=1e-7)
        worker.cancel.set()
        worker.run()
        self.assertEqual(len(results), 1)

    def test_repeated_comparisons_keep_legend_below_axis_label(self):
        import warnings
        self.panel.set_model(data(), FakeTrainer())
        self.wait()
        p = self.panel
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            for width, height in ((5, 3.5), (3.4, 3.5), (7, 4)):
                p.figure.set_size_inches(width, height)
                positions = []
                for _ in range(8):
                    p.plot_comparison()
                    p.canvas.draw()
                    renderer = p.canvas.get_renderer()
                    bounds = p.legend.get_window_extent(renderer)
                    xlabel = p.axes.xaxis.label.get_window_extent(renderer)
                    self.assertLess(bounds.y1, xlabel.y0)
                    self.assertGreaterEqual(bounds.y0, 0)
                    self.assertEqual(len(p.figure.legends), 1)
                    self.assertGreater(p.axes.get_window_extent(renderer).height, 80)
                    positions.append(p.axes.get_position().bounds)
                np.testing.assert_allclose(positions, np.tile(positions[0], (8, 1)))

    def test_bad_prediction_fails_without_enabling_plot(self):
        class BadTrainer:
            def predict(self, x):
                return np.full((len(x), 4), np.nan)
        self.panel.set_model(data(), BadTrainer())
        self.wait()
        self.assertIsNone(self.panel.predictions)
        self.assertFalse(self.panel.plot_button.isEnabled())
        self.assertTrue(self.panel.predict_button.isEnabled())

    def test_temperature_can_be_fixed_when_excluded_from_model(self):
        dataset = data()
        temperatures = dataset.x[:, 0].copy()
        dataset.x = dataset.x[:, 1:]
        dataset.metadata["input_columns"] = dataset.metadata["input_columns"][1:]
        class Trainer:
            def predict(self, x):
                return np.zeros((len(x), 4))
        self.panel.set_model(dataset, Trainer(), pd.DataFrame({"temperature": temperatures}))
        self.wait()
        self.assertIn("temperature", self.panel.fixed_controls)
        self.assertEqual(len(self.panel.axes.lines[0].get_xdata()), 3)

    def test_long_phase_names_are_matched_before_short_names(self):
        groups = output_groups(["phi_A", "phi_A_long"],
            ["phase_con_A_C", "phase_con_A_long_C", "phase_con_A_D"])
        self.assertEqual(groups, {"C": [0, 1], "D": [2]})

    def test_independent_phase_grid_plots_all_states_at_fixed_composition(self):
        x = np.array([[0., con, a, b] for con in (0.001, 0.511)
                      for a in (0.11, 0.21, 0.31) for b in (0.11, 0.21, 0.31)])
        dataset = ThermoMLDataset(x=x, y=np.zeros((len(x), 4)), metadata=data().metadata)
        p = self.panel
        p.set_model(dataset, FakeTrainer())
        self.wait()
        self.assertEqual(set(p.fixed_controls), {"temperature", "con_C"})
        self.assertEqual(len(p.axes.lines), 6)
        np.testing.assert_array_equal(p.axes.lines[0].get_xdata(), np.arange(9))
        np.testing.assert_array_equal(p.axes.lines[0].get_ydata(), x[:9, 2])
        np.testing.assert_array_equal(p.axes.lines[1].get_ydata(), x[:9, 3])
        p.fixed_controls["con_C"].edit.setText("0.51")
        p.plot_comparison()
        self.assertEqual(p.fixed_controls["con_C"].value(), 0.511)
        np.testing.assert_allclose(p.axes.lines[3].get_ydata(), p.predictions[9:, 0])

    def test_extra_phases_are_plotted_together_without_fixed_controls(self):
        dataset = data()
        dataset.x[:, 2:] *= 0.8
        dataset.x = np.column_stack([dataset.x, np.full(len(dataset.x), 0.2)])
        dataset.metadata["input_columns"].append("phi_C")
        self.panel.set_model(dataset, FakeTrainer())
        self.wait()
        self.assertEqual(set(self.panel.fixed_controls), {"temperature", "con_C"})
        self.assertEqual(len(self.panel.axes.lines), 7)
        np.testing.assert_allclose(self.panel.axes.lines[2].get_ydata(), [0.2, 0.2, 0.2])

    def test_multiple_compositions_are_fixed_without_filtering_phi(self):
        dataset = data()
        dataset.x = np.column_stack([dataset.x, np.tile([0.1, 0.2, 0.1], 4)])
        dataset.metadata["input_columns"].append("con_D")
        p = self.panel
        p.set_model(dataset, FakeTrainer())
        self.wait()
        self.assertEqual(set(p.fixed_controls), {"temperature", "con_C", "con_D"})
        np.testing.assert_array_equal(p.axes.lines[0].get_ydata(), [0., 1.])
        p.fixed_controls["con_D"].set_value(0.2)
        p.plot_comparison()
        np.testing.assert_array_equal(p.axes.lines[0].get_ydata(), [0.5])
        np.testing.assert_allclose(p.axes.lines[3].get_ydata(), p.predictions[[1], 0])

    def test_slice_retains_all_phase_states_and_source_indices(self):
        frame = pd.DataFrame({"con_C": [0.3, 0.4, 0.3], "phi_A": [0., 1., 0.],
            "phi_B": [1., 0., 0.9], "phase_con_A_C": [0.1, 0.2, 0.3]})
        rows = comparison_slice(frame, {"con_C": 0.3})
        self.assertEqual(list(rows.index), [0, 2])
        np.testing.assert_array_equal(rows["phi_B"], [1., 0.9])
        with self.assertRaises(ValueError):
            comparison_slice(frame, {"con_C": 0.9})
