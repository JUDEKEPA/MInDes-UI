"""Integration checks for the dataset creation workflow."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest
import time
import importlib.util
import threading
from unittest.mock import patch

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from mindes_ui.tools.machine_learning.phi_con_model_gui import PhiConModelDialog


class DatasetWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "thermo.csv"
        self.path.write_text(
            "sample_id,temperature,con_C,phi_A,phi_B,phase_con_A_C,phase_con_B_C\n"
            "1,1000,0.3,0.1,0.9,0,0.4\n"
            "2,1100,0.4,0.3,0.7,0.2,0.5\n",
            encoding="utf-8",
        )
        self.dialog = PhiConModelDialog()
        self.addCleanup(self.dialog.deleteLater)
        self.addCleanup(self.dialog.close)

    def load(self):
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(self.path), "")):
            self.dialog.on_load_csv()

    def test_create_uses_backend_and_keeps_raw_table(self):
        d = self.dialog
        self.assertFalse(d.btn_create_dataset.isEnabled())
        self.load()
        original = d.df.copy()
        self.assertTrue(d.csv_table.isColumnHidden(0))
        self.assertTrue(d.data_tabs.isTabEnabled(3))
        self.assertIs(d.raw_plot.frame, d.df)
        d.include_temperature.setChecked(False)
        d.on_create_dataset()
        expected = d.csv_data.make_dataset(include_temperature=False)
        np.testing.assert_array_equal(d.dataset.x, expected.x)
        np.testing.assert_array_equal(d.dataset.y, expected.y)
        pd.testing.assert_frame_equal(d.csv_model.dataframe, original)
        self.assertEqual(list(d.input_table.model().dataframe.columns), expected.metadata["input_columns"])
        self.assertEqual(list(d.output_table.model().dataframe.columns), expected.metadata["output_columns"])
        self.assertTrue(d.data_tabs.isTabEnabled(1))
        self.assertTrue(d.data_tabs.isTabEnabled(2))
        self.assertTrue(d.workflow_tabs.isTabEnabled(1))
        self.assertFalse(d.workflow_tabs.isTabEnabled(2))

    def test_option_change_and_reload_invalidate_dataset(self):
        self.load()
        d = self.dialog
        d.on_create_dataset()
        d.threshold_choice.setCurrentIndex(d.threshold_choice.findData("none"))
        self.assertIsNone(d.dataset)
        self.assertEqual(d.input_table.model().rowCount(), 0)
        d.on_create_dataset()
        self.assertEqual(d.dataset.metadata["phase_threshold_mode"], "none")
        np.testing.assert_array_equal(d.dataset.x, d.df.loc[:, d.dataset.metadata["input_columns"]])
        self.load()
        self.assertIsNone(d.dataset)
        self.assertFalse(d.data_tabs.isTabEnabled(1))

    def test_manual_threshold_success_and_failure(self):
        self.load()
        d = self.dialog
        d.threshold_choice.setCurrentIndex(d.threshold_choice.findData("manual"))
        self.assertTrue(d.threshold_value.isEnabled())
        d.threshold_value.setValue(0.2)
        d.on_create_dataset()
        self.assertEqual(d.dataset.metadata["phase_threshold"], 0.2)
        self.assertEqual(d.dataset.x.dtype, np.dtype("float64"))
        d.threshold_value.setValue(0.95)
        with patch.object(QMessageBox, "critical") as error:
            d.on_create_dataset()
        error.assert_called_once()
        self.assertIsNone(d.dataset)
        self.assertFalse(d.data_tabs.isTabEnabled(1))

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "sklearn")), "ML dependencies required")
    def test_training_enables_prediction_and_dataset_changes_invalidate_it(self):
        self.load()
        d = self.dialog
        d.threshold_choice.setCurrentIndex(d.threshold_choice.findData("none"))
        d.on_create_dataset()
        d.training_panel.layers.setText("4")
        d.training_panel.epochs.setValue(1)
        d.training_panel.device.setCurrentIndex(d.training_panel.device.findData("cpu"))
        d.training_panel.start_training()
        deadline = time.monotonic() + 120
        while d.is_training and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertFalse(d.is_training)
        self.assertTrue(d.workflow_tabs.isTabEnabled(2))
        np.testing.assert_allclose(d.prediction_panel.predictions,
                                   d.training_panel.trainer.predict(d.dataset.x), atol=1e-7)
        self.assertTrue(d.training_panel.isEnabled())
        d.include_temperature.setChecked(False)
        self.assertIsNone(d.prediction_panel.predictions)
        self.assertFalse(d.workflow_tabs.isTabEnabled(2))

    def test_close_waits_for_prediction_and_cancels_result(self):
        self.load()
        d = self.dialog
        d.on_create_dataset()
        started, release = threading.Event(), threading.Event()

        class SlowTrainer:
            def predict(self, x):
                started.set()
                release.wait(10)
                return np.zeros((len(x), d.dataset.y.shape[1]))

        d.prediction_panel.set_model(d.dataset, SlowTrainer(), d.df)
        try:
            self.assertTrue(started.wait(5))
            self.assertFalse(d.close())
            self.assertTrue(d.prediction_panel.worker.cancel.is_set())
        finally:
            release.set()
            d.prediction_panel.worker.wait(10000)
            self.app.processEvents()
        self.assertFalse(d.is_training)
        self.assertIsNone(d.prediction_panel.predictions)
