"""Dataset threshold behavior and export regression checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mindes_ui.tools.machine_learning.fix_con_prediction.io.read_csv import ThermoCSVData


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "thermo.csv"
        self.path.write_text(
            "temperature,con_C,phi_A,phi_B,phi_C,phase_con_A_C,phase_con_B_C,phase_con_C_C\n"
            "1000,0.3,0.1,0.3,0.6,0,0.2,0.4\n"
            "1100,0.4,0.2,0.3,0.5,0.1,0.3,0.5\n",
            encoding="utf-8",
        )

    def test_threshold_transforms_all_rows_and_preserves_targets(self):
        data = ThermoCSVData(self.path, phase_threshold=0.1)
        dataset = data.make_dataset(include_temperature=False)
        self.assertEqual(dataset.x.shape, (2, 4))
        np.testing.assert_allclose(dataset.x[:, 1:], [[0, 1 / 3, 2 / 3], [0.2, 0.3, 0.5]])
        np.testing.assert_array_equal(dataset.y, data.df.loc[:, data.output_columns])
        self.assertEqual(dataset.metadata["phase_threshold_mode"], "transform")
        self.assertTrue(dataset.metadata["phase_fraction_preprocessing"]["enabled"])

    def test_none_disables_transformation_even_with_detected_threshold(self):
        data = ThermoCSVData(self.path)
        dataset = data.make_dataset(phase_threshold=None)
        np.testing.assert_array_equal(dataset.x, data.df.loc[:, data.input_columns])
        self.assertEqual(dataset.metadata["phase_threshold_mode"], "none")
        self.assertFalse(dataset.metadata["phase_fraction_preprocessing"]["enabled"])

    def test_auto_threshold_and_no_detectable_threshold(self):
        data = ThermoCSVData(self.path)
        self.assertEqual(data.phase_threshold, 0.1)
        self.assertEqual(data.make_dataset().metadata["phase_threshold_mode"], "transform")
        data.df.loc[:, "phase_con_A_C"] = 0.2
        data.df.to_csv(self.path, index=False)
        dataset = ThermoCSVData(self.path).make_dataset()
        self.assertIsNone(dataset.metadata["phase_threshold"])
        self.assertEqual(dataset.metadata["phase_threshold_mode"], "none")

    def test_no_surviving_phase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no phase above the threshold"):
            ThermoCSVData(self.path, phase_threshold=0.9).make_dataset()

    def test_saved_arrays_are_float64_and_metadata_matches(self):
        data = ThermoCSVData(self.path, phase_threshold=None)
        prefix = Path(self.temp.name) / "dataset"
        data.save_dataset(prefix)
        with np.load(prefix.with_suffix(".npz")) as saved:
            self.assertEqual(saved["x"].dtype, np.dtype("float64"))
            self.assertEqual(saved["y"].dtype, np.dtype("float64"))
        metadata = json.loads((prefix.parent / "dataset_metadata.json").read_text())
        self.assertEqual(metadata["dtype"], "float64")
        self.assertEqual(metadata["phase_threshold_mode"], "none")
