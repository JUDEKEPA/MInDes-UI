"""Read thermodynamic CSV data for machine-learning datasets.

The CSV exported by thermodynamic calculations can contain a variable number
of concentration, phase-fraction, and phase-concentration columns. Column names
are expected to use these simple groups: ``temperature``, ``con*``, ``phi*``,
and ``phase_con*``. Temperature can be included in, or excluded from, the ML
input matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
import csv
import json
import re
from typing import Any

import numpy as np
import pandas as pd


class ThermoMLDataset:
    """ML arrays and metadata created from a thermodynamic CSV table."""

    def __init__(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        self.x = x
        self.y = y
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"ThermoMLDataset(x_shape={self.x.shape}, "
            f"y_shape={self.y.shape}, metadata_keys={list(self.metadata)})"
        )


class ThermoCSVData:
    """Thermodynamic table prepared for machine-learning workflows.

    Parameters
    ----------
    path:
        CSV file path.
    The column names are detected by this convention:
    ``temperature`` can be used as an input, ``con*`` columns are concentration
    inputs, ``phi*`` columns are phase-fraction inputs, and ``phase_con*``
    columns are outputs.
    read_kwargs:
        Extra keyword arguments passed to :func:`read_thermo_csv`.

    Notes
    -----
    The ML inputs are ``con + phi`` columns, plus ``temperature`` when
    ``include_temperature`` is true. The ML outputs are all ``phase_con``
    columns.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        include_temperature: bool = True,
        phase_threshold: float | str | None = "auto",
        **read_kwargs: Any,
    ) -> None:
        self.path = Path(path)
        self.include_temperature = include_temperature
        self.dataframe = read_thermo_csv(self.path, **read_kwargs)
        self.columns = tuple(self.dataframe.columns)

        self.temperature_columns = _columns_equal(self.columns, "temperature")
        self.con_columns = _columns_starting_with(self.columns, "con")
        self.phi_columns = _columns_starting_with(self.columns, "phi")
        self.phase_con_columns = _columns_starting_with(self.columns, "phase_con")

        self.input_columns = self._input_columns()
        self.output_columns = self.phase_con_columns

        self._validate_column_groups()
        self.detected_phase_threshold = detect_phase_threshold_setting(
            self.dataframe,
            phi_columns=self.phi_columns,
            phase_con_columns=self.phase_con_columns,
        )
        self.phase_threshold = self._resolve_phase_threshold(phase_threshold)

    @property
    def df(self) -> pd.DataFrame:
        """Alias for the underlying :class:`pandas.DataFrame`."""
        return self.dataframe

    @property
    def input(self) -> np.ndarray:
        """ML input array using the default temperature setting."""
        return self.features()

    @property
    def output(self) -> np.ndarray:
        """ML output array: all ``phase_con`` columns."""
        return self.targets()

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the underlying table."""
        return self.dataframe.shape

    def features(
        self,
        *,
        include_temperature: bool | None = None,
        phase_threshold: float | str | None = "default",
    ) -> np.ndarray:
        """Return the ML input matrix."""
        input_columns = self._input_columns(include_temperature)
        dataframe = self.dataset_dataframe(
            phase_threshold=phase_threshold,
        )
        return dataframe.loc[:, input_columns].to_numpy(dtype=np.float64)

    def targets(
        self,
        *,
        phase_threshold: float | str | None = "default",
    ) -> np.ndarray:
        """Return the ML output matrix."""
        dataframe = self.dataset_dataframe(
            phase_threshold=phase_threshold,
        )
        return dataframe.loc[:, self.output_columns].to_numpy(dtype=np.float64)

    def features_targets(
        self,
        *,
        include_temperature: bool | None = None,
        phase_threshold: float | str | None = "default",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(input, output)`` arrays for model training."""
        dataframe = self.dataset_dataframe(
            phase_threshold=phase_threshold,
        )
        input_columns = self._input_columns(include_temperature)
        return (
            dataframe.loc[:, input_columns].to_numpy(dtype=np.float64),
            dataframe.loc[:, self.output_columns].to_numpy(dtype=np.float64),
        )

    def describe_dataset(
        self,
        *,
        include_temperature: bool | None = None,
        phase_threshold: float | str | None = "default",
    ) -> dict[str, Any]:
        """Describe which columns will become ML inputs and outputs."""
        input_columns = self._input_columns(include_temperature)
        dataframe = self.dataset_dataframe(
            phase_threshold=phase_threshold,
        )
        resolved_phase_threshold = self._resolve_phase_threshold(phase_threshold)

        return {
            "source_path": str(self.path),
            "source_rows": len(self.dataframe),
            "dataset_rows": len(dataframe),
            "all_columns": list(self.columns),
            "include_temperature": self._include_temperature(include_temperature),
            "phase_threshold": resolved_phase_threshold,
            "phase_threshold_mode": (
                "transform" if resolved_phase_threshold is not None else "none"
            ),
            "phase_fraction_transform": (
                "phi <= threshold becomes zero; surviving phi values are normalized"
                if resolved_phase_threshold is not None
                else None
            ),
            "phase_fraction_preprocessing": {
                "enabled": (
                    resolved_phase_threshold is not None
                ),
                "threshold": resolved_phase_threshold,
                "comparison": "less_than_or_equal",
                "inactive_value": 0.0,
                "normalize_survivors": True,
                "zero_inactive_phase_outputs": True,
            },
            "detected_phase_threshold": self.detected_phase_threshold,
            "input_columns": list(input_columns),
            "output_columns": list(self.output_columns),
            "column_groups": {
                "temperature": list(self.temperature_columns),
                "con": list(self.con_columns),
                "phi": list(self.phi_columns),
                "phase_con": list(self.phase_con_columns),
            },
        }

    def print_dataset_description(
        self,
        *,
        include_temperature: bool | None = None,
        phase_threshold: float | str | None = "default",
    ) -> None:
        """Print a compact summary for checking dataset construction."""
        metadata = self.describe_dataset(
            include_temperature=include_temperature,
            phase_threshold=phase_threshold,
        )
        print(f"Source: {metadata['source_path']}")
        print(f"Rows: {metadata['dataset_rows']}")
        print(f"Include temperature: {metadata['include_temperature']}")
        print(f"Phase threshold: {metadata['phase_threshold']}")
        print(f"Phase threshold mode: {metadata['phase_threshold_mode']}")
        print(f"Inputs: {metadata['input_columns']}")
        print(f"Outputs: {metadata['output_columns']}")

    def make_dataset(
        self,
        *,
        include_temperature: bool | None = None,
        phase_threshold: float | str | None = "default",
    ) -> ThermoMLDataset:
        """Create float64 ML arrays and metadata from the full current table."""
        input_columns = self._input_columns(include_temperature)
        dataframe = self.dataset_dataframe(
            phase_threshold=phase_threshold,
        )
        x = dataframe.loc[:, input_columns].to_numpy(dtype=np.float64)
        y = dataframe.loc[:, self.output_columns].to_numpy(dtype=np.float64)
        metadata = self.describe_dataset(
            include_temperature=include_temperature,
            phase_threshold=phase_threshold,
        )
        metadata.update(
            {
                "x_shape": list(x.shape),
                "y_shape": list(y.shape),
                "dtype": "float64",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "creator": "ThermoCSVData.make_dataset",
            }
        )
        return ThermoMLDataset(x=x, y=y, metadata=metadata)

    def save_dataset(
        self,
        output_prefix: str | Path,
        *,
        include_temperature: bool | None = None,
        phase_threshold: float | str | None = "default",
    ) -> ThermoMLDataset:
        """Save ``X``/``y`` arrays to NPZ and creation metadata to JSON.

        Files are written as ``<output_prefix>.npz`` and
        ``<output_prefix>_metadata.json``.
        """
        dataset = self.make_dataset(
            include_temperature=include_temperature,
            phase_threshold=phase_threshold,
        )
        output_prefix = Path(output_prefix)
        npz_path = output_prefix.with_suffix(".npz")
        metadata_path = output_prefix.with_name(
            f"{output_prefix.stem}_metadata.json"
        )

        np.savez_compressed(npz_path, x=dataset.x, y=dataset.y)
        metadata = dict(dataset.metadata)
        metadata["array_file"] = str(npz_path)
        metadata["metadata_file"] = str(metadata_path)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

        return ThermoMLDataset(x=dataset.x, y=dataset.y, metadata=metadata)

    def dataset_dataframe(
        self,
        *,
        phase_threshold: float | str | None = "default",
    ) -> pd.DataFrame:
        """Transform phase fractions when a threshold is set; otherwise keep raw data.

        Every row is retained. With a threshold, fractions at or below it become
        zero and surviving fractions are normalized to sum to one.
        """
        resolved_phase_threshold = self._resolve_phase_threshold(phase_threshold)
        if resolved_phase_threshold is None:
            return self.dataframe.copy()
        return transform_phase_fractions(
            self.dataframe,
            phi_columns=self.phi_columns,
            phase_threshold=resolved_phase_threshold,
        )

    def extract_fixed(
        self,
        fixed: Mapping[str, float | int] | None = None,
        *,
        tolerance: float = 1e-8,
        columns: Sequence[str] | None = None,
        as_numpy: bool = False,
    ) -> pd.DataFrame | np.ndarray:
        """Extract rows after fixing one or more variables.

        Parameters
        ----------
        fixed:
            Mapping from column name to fixed value, for example
            ``{"temperature": 0, "con_CON_0": 0.271}``. Use one entry to
            obtain a 2-D slice and two entries to obtain a 1-D curve.
        tolerance:
            Floating point tolerance used by :func:`numpy.isclose`.
        columns:
            Optional subset of columns to return.
        as_numpy:
            Return a NumPy array instead of a DataFrame.
        """
        if fixed is None:
            fixed = {}

        selected_columns = list(columns) if columns is not None else None
        self._check_columns(tuple(fixed), role="fixed")
        if selected_columns is not None:
            self._check_columns(selected_columns, role="selected")

        mask = np.ones(len(self.dataframe), dtype=bool)
        for column, value in fixed.items():
            values = self.dataframe[column].to_numpy()
            if np.issubdtype(values.dtype, np.number):
                mask &= np.isclose(values, value, atol=tolerance, rtol=0.0)
            else:
                mask &= values == value

        if selected_columns is None:
            result = self.dataframe.loc[mask, :].copy()
        else:
            result = self.dataframe.loc[mask, selected_columns].copy()
        if as_numpy:
            return result.to_numpy()
        return result

    def extract_xy(
        self,
        *,
        fixed: Mapping[str, float | int] | None = None,
        include_temperature: bool | None = None,
        tolerance: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(input, output)`` arrays after fixing variables."""
        data = self.extract_fixed(fixed, tolerance=tolerance)
        assert isinstance(data, pd.DataFrame)
        input_columns = self._input_columns(include_temperature)
        x = data.loc[:, input_columns].to_numpy(dtype=np.float64)
        y = data.loc[:, self.output_columns].to_numpy(dtype=np.float64)
        return x, y

    def unique_values(self, column: str) -> np.ndarray:
        """Return sorted unique values for one column."""
        self._check_columns((column,), role="unique")
        return np.sort(self.dataframe[column].dropna().unique())

    def _validate_column_groups(self) -> None:
        self._check_columns(self.input_columns + self.output_columns, role="ML")
        if not self.temperature_columns:
            raise ValueError(
                "No temperature column detected. The CSV must contain a "
                "'temperature' column."
            )
        if not self.con_columns:
            raise ValueError(
                "No concentration input columns detected. The CSV must contain "
                "at least one 'con*' column."
            )
        if not self.phi_columns:
            raise ValueError(
                "No phase-fraction input columns detected. The CSV must contain "
                "at least one 'phi*' column."
            )
        if not self.output_columns:
            raise ValueError(
                "No phase concentration output columns detected. The CSV must "
                "contain at least one 'phase_con*' column."
            )

    def _include_temperature(self, include_temperature: bool | None = None) -> bool:
        if include_temperature is None:
            return self.include_temperature
        return include_temperature

    def _input_columns(
        self,
        include_temperature: bool | None = None,
    ) -> tuple[str, ...]:
        columns = self.con_columns + self.phi_columns
        if self._include_temperature(include_temperature):
            return self.temperature_columns + columns
        return columns

    def _check_columns(self, columns: Sequence[str], *, role: str) -> None:
        missing = [column for column in columns if column not in self.dataframe]
        if missing:
            raise KeyError(f"Unknown {role} column(s): {missing}")

    def _resolve_phase_threshold(
        self,
        phase_threshold: float | str | None = "default",
    ) -> float | None:
        if phase_threshold == "default":
            return getattr(self, "phase_threshold", None)
        if phase_threshold == "auto":
            return self.detected_phase_threshold["threshold"]
        if phase_threshold is None:
            return None

        threshold = float(phase_threshold)
        if threshold < 0.0 or threshold >= 1.0:
            raise ValueError("phase_threshold must be in [0, 1)")
        return threshold

    def __getitem__(self, name: str) -> pd.Series:
        """Return a column by name, allowing ``data["temperature"]`` access."""
        return self.dataframe[name]

    def __repr__(self) -> str:
        return (
            f"ThermoCSVData(rows={len(self.dataframe)}, "
            f"inputs={list(self.input_columns)!r}, "
            f"outputs={list(self.output_columns)!r})"
        )


def transform_phase_fractions(
    dataframe: pd.DataFrame,
    *,
    phi_columns: Sequence[str] | None = None,
    phase_threshold: float,
) -> pd.DataFrame:
    """Zero small phase fractions and normalize every surviving phase.

    This operation is independent of the number of phases. Values satisfying
    ``phi <= phase_threshold`` become zero. Remaining values retain their
    relative proportions and are normalized to sum to one in each row.
    """
    if phase_threshold < 0.0 or phase_threshold >= 1.0:
        raise ValueError("phase_threshold must be in [0, 1)")
    if phi_columns is None:
        phi_columns = _columns_starting_with(tuple(dataframe.columns), "phi")
    phi_columns = tuple(phi_columns)
    if not phi_columns:
        raise ValueError("No phi columns were found")

    transformed = dataframe.copy()
    phi_values = transformed.loc[:, phi_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(phi_values)):
        raise ValueError("Phi values must all be finite")

    phi_values[phi_values <= phase_threshold] = 0.0
    phi_sums = phi_values.sum(axis=1, keepdims=True)
    invalid_rows = np.flatnonzero(phi_sums[:, 0] <= 0.0)
    if len(invalid_rows):
        raise ValueError(
            "Cannot normalize rows with no phase above the threshold: "
            f"{invalid_rows.tolist()}"
        )

    transformed.loc[:, phi_columns] = phi_values / phi_sums
    return transformed


def detect_phase_threshold_setting(
    dataframe: pd.DataFrame,
    *,
    phi_columns: Sequence[str] | None = None,
    phase_con_columns: Sequence[str] | None = None,
    zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Infer the raw-data phase threshold from zeroed phase concentrations.

    The export rule is assumed to be: when a phase fraction is at or below the
    threshold, the corresponding ``phase_con`` value is written as zero. For
    each ``phi_*`` column, this function finds associated ``phase_con_*``
    columns and records the largest phi value where those outputs are zero.
    The inferred dataset threshold is the largest such value across phases.
    """
    columns = tuple(dataframe.columns)
    if phi_columns is None:
        phi_columns = _columns_starting_with(columns, "phi")
    if phase_con_columns is None:
        phase_con_columns = _columns_starting_with(columns, "phase_con")

    per_phase = {}
    threshold_candidates = []
    for phi_column in phi_columns:
        matching_phase_con_columns = _matching_phase_con_columns(
            phi_column,
            phase_con_columns,
        )
        zero_phi_values = []
        for phase_con_column in matching_phase_con_columns:
            zero_mask = np.isclose(
                dataframe[phase_con_column].to_numpy(dtype=float),
                0.0,
                atol=zero_tolerance,
                rtol=0.0,
            )
            if np.any(zero_mask):
                max_zero_phi = float(
                    np.nanmax(dataframe.loc[zero_mask, phi_column].to_numpy(dtype=float))
                )
                zero_phi_values.append(max_zero_phi)

        phase_threshold = max(zero_phi_values) if zero_phi_values else None
        if phase_threshold is not None:
            threshold_candidates.append(phase_threshold)
        per_phase[phi_column] = {
            "matching_phase_con_columns": list(matching_phase_con_columns),
            "threshold": phase_threshold,
        }

    threshold = max(threshold_candidates) if threshold_candidates else None
    return {
        "threshold": threshold,
        "zero_tolerance": zero_tolerance,
        "criterion": (
            "Largest phi value where the corresponding phase_con column is zero; "
            "dataset threshold is the largest value across phi columns."
        ),
        "per_phase": per_phase,
    }


def read_thermo_csv(
    path: str | Path,
    *,
    coerce_numeric: bool = True,
    repair_leading_thousands: bool = True,
) -> pd.DataFrame:
    """Read a thermodynamic CSV file into a DataFrame.

    Some exported files write ``sample_id`` values as ``1,000`` without quotes,
    which makes those rows wider than the header. When
    ``repair_leading_thousands`` is true, extra leading fields are joined back
    into the first column.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV file is empty: {path}") from error

        width = len(header)
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            row = _repair_row_width(
                row,
                width=width,
                line_number=line_number,
                repair_leading_thousands=repair_leading_thousands,
            )
            rows.append(row)

    frame = pd.DataFrame(rows, columns=header)
    if coerce_numeric:
        frame = _coerce_numeric_columns(frame)
    return frame


def read_csv(path: str | Path, **kwargs: Any) -> ThermoCSVData:
    """Read a thermodynamic CSV file as :class:`ThermoCSVData`."""
    return ThermoCSVData(path, **kwargs)


def _repair_row_width(
    row: list[str],
    *,
    width: int,
    line_number: int,
    repair_leading_thousands: bool,
) -> list[str]:
    if len(row) == width:
        return row

    if len(row) > width and repair_leading_thousands:
        extra = len(row) - width
        leading_fields = row[: extra + 1]
        if all(_is_integer_text(field) for field in leading_fields):
            return ["".join(leading_fields)] + row[extra + 1 :]

    if len(row) < width:
        return row + [""] * (width - len(row))

    raise ValueError(
        f"CSV row {line_number} has {len(row)} fields, but the header has {width}."
    )


def _columns_equal(columns: Sequence[str], name: str) -> tuple[str, ...]:
    return tuple(column for column in columns if column == name)


def _columns_starting_with(columns: Sequence[str], prefix: str) -> tuple[str, ...]:
    return tuple(column for column in columns if column.startswith(prefix))


def _matching_phase_con_columns(
    phi_column: str,
    phase_con_columns: Sequence[str],
) -> tuple[str, ...]:
    phase_name = phi_column.removeprefix("phi_")
    prefix = f"phase_con_{phase_name}"
    return tuple(
        column
        for column in phase_con_columns
        if column == prefix or column.startswith(f"{prefix}_")
    )


def _coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted:
        try:
            converted[column] = pd.to_numeric(converted[column])
        except (TypeError, ValueError):
            pass
    return converted


def _is_integer_text(value: str) -> bool:
    return re.fullmatch(r"\s*[+-]?\d+\s*", value) is not None
