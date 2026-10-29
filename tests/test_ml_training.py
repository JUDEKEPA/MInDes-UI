"""Training options, worker execution, and lifecycle regression checks."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import importlib.util
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox
from mindes_ui.tools.machine_learning.training_panel import TrainingPanel
from mindes_ui.tools.machine_learning.training_worker import TrainingWorker
from mindes_ui.tools.machine_learning.fix_con_prediction.io.read_csv import ThermoMLDataset

HAS_TRAINING = all(importlib.util.find_spec(name) is not None for name in ("torch", "sklearn"))
HAS_SEARCH = HAS_TRAINING and importlib.util.find_spec("hyperopt") is not None


def dataset():
    x = np.linspace(0, 1, 40).reshape(20, 2)
    y = x[:, :1] * 0.5
    return ThermoMLDataset(x=x, y=y, metadata={"include_temperature": False,
        "input_columns": ["con_A", "con_B"], "output_columns": ["phase_con_A"]})


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = TrainingPanel()
        self.addCleanup(self.panel.deleteLater)
        self.panel.set_dataset(dataset())
        self.panel.layers.setText("4, 4")
        self.panel.epochs.setValue(2)
        self.panel.batch_size.setValue(8)
        self.panel.device.setCurrentIndex(self.panel.device.findData("cpu"))

    def wait_finished(self):
        deadline = time.monotonic() + 120
        while self.panel.is_training and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        if self.panel.worker is not None:
            self.panel.worker.request_cancel()
            self.panel.worker.wait(30000)
            self.app.processEvents()
        self.assertFalse(self.panel.is_training)

    def test_options_and_validation(self):
        options = self.panel.options()
        self.assertEqual(options["hidden_layers"], (4, 4))
        self.assertEqual(options["test_size"], 0.2)
        self.assertEqual(options["seed"], 42)
        self.assertFalse(self.panel.normalize_temperature.isEnabled())
        self.panel.layers.setText("-4, 0")
        with self.assertRaises(ValueError):
            self.panel.options()
        self.panel.method.setCurrentIndex(1)
        options = self.panel.options()
        self.assertEqual(options["search_ranges"]["layer1"], (20, 300, 40))
        self.panel.search_fields["epochs"].setText("20, 1, 0")
        with self.assertRaises(ValueError):
            self.panel.options()

    def test_missing_dependencies_does_not_start_worker(self):
        with patch("importlib.util.find_spec", return_value=None), patch.object(QMessageBox, "warning") as warning:
            self.panel.start_training()
        warning.assert_called_once()
        self.assertFalse(self.panel.is_training)

    def test_live_plot_keeps_epochs_and_resets_between_runs_and_datasets(self):
        def progress(run, points):
            epoch, loss, val_loss = points[-1]
            self.panel._progress(dict(run=run, runs=4, epoch=epoch, epochs=3,
                loss=loss, val_loss=val_loss, lr=0.001, points=points))
        progress(1, [(1, 0.8, 0.9), (2, 0.4, 0.5)])
        progress(1, [(3, 0.2, 0.3)])
        np.testing.assert_array_equal(self.panel.train_line.get_xdata(), [1, 2, 3])
        np.testing.assert_allclose(self.panel.test_line.get_ydata(), [0.9, 0.5, 0.3])
        progress(2, [(1, 0.6, 0.7)])
        np.testing.assert_array_equal(self.panel.train_line.get_xdata(), [1])
        self.panel.set_dataset(None)
        self.assertEqual(len(self.panel.train_line.get_xdata()), 0)

    def test_log_loss_axis_handles_zero_and_preserves_history(self):
        import warnings
        p = self.panel
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            p.loss_log_scale.setChecked(True)
            p.loss_canvas.draw()
            p._progress(dict(run=1, runs=1, epoch=2, epochs=2, loss=0.,
                val_loss=0., lr=0.001, points=[(1, 0., 0.), (2, 0., 0.)]))
            p.loss_canvas.draw()
            self.assertEqual(p.loss_axes.get_yscale(), "log")
            self.assertGreater(p.loss_axes.get_ylim()[0], 0)
            p._progress(dict(run=2, runs=2, epoch=2, epochs=2, loss=0.,
                val_loss=1e-6, lr=0.001, points=[(1, 0.1, 0.2), (2, 0., 1e-6)]))
            p.loss_canvas.draw()
            self.assertTrue(np.isnan(p.train_line.get_ydata()[1]))
            self.assertEqual(p._train_losses, [0.1, 0.])
            p.loss_log_scale.setChecked(False)
            p.loss_canvas.draw()
            self.assertEqual(p.loss_axes.get_yscale(), "linear")
            np.testing.assert_array_equal(p.train_line.get_ydata(), [0.1, 0.])
            p.loss_log_scale.setChecked(True)
            p.set_dataset(None)
            p.loss_canvas.draw()
            self.assertEqual(p.loss_axes.get_yscale(), "log")

    def test_cancellation_settles_worker_and_unlocks_options(self):
        def waiting(worker):
            while not worker._cancel.wait(0.01):
                pass
            raise RuntimeError("cancel requested")
        with patch("importlib.util.find_spec", return_value=object()), patch.object(TrainingWorker, "train", waiting):
            self.panel.start_training()
            self.assertFalse(self.panel.options_widget.isEnabled())
            self.panel.cancel_training()
            self.wait_finished()
        self.assertIsNone(self.panel.trainer)
        self.assertTrue(self.panel.options_widget.isEnabled())
        self.assertTrue(self.panel.start_button.isEnabled())

    @unittest.skipUnless(HAS_TRAINING, "Optional ML dependencies are not installed")
    def test_manual_training_completes_and_preserves_dataset(self):
        original = self.panel.dataset.x.copy()
        self.panel.start_training()
        self.wait_finished()
        self.assertIsNotNone(self.panel.trainer, self.panel.status.text())
        self.assertEqual(self.panel.trainer.hidden_layers, (4, 4))
        self.assertEqual(len(self.panel.trainer.history["loss"]), 2)
        self.assertTrue(np.isfinite(self.panel.trainer.training_metadata["final_val_loss"]))
        np.testing.assert_array_equal(self.panel.dataset.x, original)
        np.testing.assert_allclose(self.panel.train_line.get_ydata(), self.panel.trainer.history["loss"])
        np.testing.assert_allclose(self.panel.test_line.get_ydata(), self.panel.trainer.history["val_loss"])

    @unittest.skipUnless(HAS_SEARCH, "Optional Bayesian dependencies are not installed")
    def test_bayesian_training_retains_selected_metadata(self):
        self.panel.method.setCurrentIndex(1)
        for key, field in self.panel.search_fields.items():
            field.setText("1, 1, 1" if key == "epochs" else "4, 4, 1")
        self.panel.max_evals.setValue(1)
        self.panel.start_training()
        self.wait_finished()
        trainer = self.panel.trainer
        self.assertIsNotNone(trainer, self.panel.status.text())
        self.assertEqual(trainer.hidden_layers, (4, 4, 4))
        self.assertEqual(trainer.training_metadata["bayesian_search"]["max_evals"], 1)
        self.assertEqual(trainer.training_metadata["epoch"], 1)
        np.testing.assert_allclose(self.panel.test_line.get_ydata(), trainer.history["val_loss"])

    @unittest.skipUnless(HAS_TRAINING, "Optional ML dependencies are not installed")
    def test_worker_batches_without_dropping_epochs(self):
        options = self.panel.options()
        options["epochs"] = 5
        worker = TrainingWorker(self.panel.dataset, options)
        updates = []
        worker.progress.connect(updates.append)
        with patch("mindes_ui.tools.machine_learning.training_worker.time.monotonic", return_value=1):
            trainer = worker.train()
        points = [point for update in updates for point in update["points"]]
        self.assertEqual([point[0] for point in points], [1, 2, 3, 4, 5])
        self.assertEqual(len(updates), 2)
        np.testing.assert_allclose([point[1] for point in points], trainer.history["loss"])

    @unittest.skipUnless(HAS_TRAINING, "Optional ML dependencies are not installed")
    def test_temperature_scaling_matches_training_and_predict(self):
        from mindes_ui.tools.machine_learning.fix_con_prediction.core.phase_con_pred_model import RegressionTasks
        import torch
        data = dataset()
        data.x[:, 0] = data.x[:, 0] * 200 + 1000
        data.metadata["include_temperature"] = True
        data.metadata["input_columns"][0] = "temperature"
        trainer = RegressionTasks(device="cpu", random_state=42)
        trainer.set_temperature_range(1000, 1200)
        trainer.initialise(data, hidden_layers=(4,), test_size=0.2)
        train, validation, _ = trainer.model_training(epoch=1, batch_size=8)
        with torch.no_grad():
            x = data.x.copy().astype(np.float32)
            x[:, 0] = (x[:, 0] - 1000) / 200
            expected = trainer.model(torch.as_tensor(x)).numpy()
        np.testing.assert_allclose(trainer.predict(data.x), expected, atol=1e-7)
        self.assertTrue(np.isfinite(train) and np.isfinite(validation))

    @unittest.skipUnless(HAS_TRAINING, "Optional ML dependencies are not installed")
    def test_backend_cancellation_is_checked_between_batches(self):
        from mindes_ui.tools.machine_learning.fix_con_prediction.core.phase_con_pred_model import RegressionTasks, TrainingCanceled
        trainer = RegressionTasks(device="cpu", random_state=42)
        trainer.initialise(dataset(), hidden_layers=(4,), test_size=0.2)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 3
        with self.assertRaises(TrainingCanceled):
            trainer.model_training(epoch=10, batch_size=2, cancel_check=cancel)
