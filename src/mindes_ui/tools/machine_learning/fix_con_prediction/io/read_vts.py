"""Read ASCII VTK StructuredGrid (``.vts``) files."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import keyword
import re
from typing import Literal, overload
import xml.etree.ElementTree as ET

import numpy as np


_VTK_DTYPES = {
    "Float32": np.float32,
    "Float64": np.float64,
    "Int8": np.int8,
    "UInt8": np.uint8,
    "Int16": np.int16,
    "UInt16": np.uint16,
    "Int32": np.int32,
    "UInt32": np.uint32,
    "Int64": np.int64,
    "UInt64": np.uint64,
}


class VTSData:
    """Data read from an ASCII VTK StructuredGrid file.

    Named arrays are available through both ``fields[name]`` and, when the
    VTK name is a valid non-conflicting Python identifier, ``data.name``.
    Scalar arrays from a 2-D grid have shape ``(ny, nx)``. Arrays from a 3-D
    grid have shape ``(nz, ny, nx)``.
    """

    def __init__(
        self,
        *,
        extent: tuple[int, int, int, int, int, int],
        coordinates: np.ndarray,
        fields: dict[str, np.ndarray],
    ) -> None:
        self.extent = extent
        self.coordinates = coordinates
        self.fields = fields

        xmin, xmax, ymin, ymax, zmin, zmax = extent
        self.shape = (
            zmax - zmin + 1,
            ymax - ymin + 1,
            xmax - xmin + 1,
        )
        self.field_names = tuple(fields)

        reserved_names = set(vars(self)) | set(dir(type(self)))
        for name, values in fields.items():
            if (
                name.isidentifier()
                and not keyword.iskeyword(name)
                and name not in reserved_names
            ):
                setattr(self, name, values)

    def __getitem__(self, name: str) -> np.ndarray:
        """Return a named field, allowing ``data["field_name"]`` access."""
        return self.fields[name]

    def __repr__(self) -> str:
        return (
            f"VTSData(shape={self.shape}, fields={list(self.field_names)!r}, "
            f"coordinates_shape={self.coordinates.shape})"
        )


class VTSSeries:
    """An ordered collection of serial VTS files.

    Parameters
    ----------
    directory:
        Folder containing files whose names include ``step<number>``.
    mode:
        ``"eager"`` reads every file during construction. ``"lazy"`` reads
        a file only when it is accessed, then caches the resulting VTSData.
    pattern:
        Glob pattern used to find files in ``directory``.

    Notes
    -----
    Integer and slice indexing are positional after numeric step sorting.
    Use :meth:`at_step` when addressing the simulation step itself.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        mode: Literal["eager", "lazy"] = "lazy",
        pattern: str = "*.vts",
    ) -> None:
        if mode not in {"eager", "lazy"}:
            raise ValueError("mode must be either 'eager' or 'lazy'")

        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise NotADirectoryError(f"VTS directory does not exist: {directory}")

        step_and_file = [
            (self._step_from_path(path), path)
            for path in self.directory.glob(pattern)
            if path.is_file()
        ]
        if not step_and_file:
            raise FileNotFoundError(
                f"No VTS files matching {pattern!r} found in {self.directory}"
            )
        step_and_file.sort(key=lambda item: item[0])

        steps = [item[0] for item in step_and_file]
        if len(set(steps)) != len(steps):
            raise ValueError(f"Duplicate simulation steps found: {steps}")

        self.mode = mode
        self.pattern = pattern
        self.steps = tuple(steps)
        self.files = tuple(item[1] for item in step_and_file)
        self._step_to_index = {step: index for index, step in enumerate(self.steps)}
        self._cache: dict[int, VTSData] = {}

        if mode == "eager":
            for index in range(len(self.files)):
                self._load(index)

    @staticmethod
    def _step_from_path(path: Path) -> int:
        match = re.search(r"step(-?\d+)", path.stem, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(
                f"Cannot find 'step<number>' in VTS filename: {path.name}"
            )
        return int(match.group(1))

    def _load(self, index: int) -> VTSData:
        if index not in self._cache:
            self._cache[index] = read_vts(self.files[index])
        return self._cache[index]

    def __len__(self) -> int:
        return len(self.files)

    @overload
    def __getitem__(self, index: int) -> VTSData: ...

    @overload
    def __getitem__(self, index: slice) -> list[VTSData]: ...

    def __getitem__(self, index: int | slice) -> VTSData | list[VTSData]:
        if isinstance(index, slice):
            selected = range(*index.indices(len(self)))
            return [self._load(position) for position in selected]

        if not isinstance(index, int):
            raise TypeError("VTSSeries indices must be integers or slices")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("VTSSeries index out of range")
        return self._load(index)

    def __iter__(self) -> Iterator[VTSData]:
        for index in range(len(self)):
            yield self._load(index)

    def at_step(self, step: int) -> VTSData:
        """Return data for an exact simulation step such as 100 or 500."""
        try:
            index = self._step_to_index[step]
        except KeyError as error:
            raise KeyError(
                f"Step {step} is unavailable; available steps are {self.steps}"
            ) from error
        return self._load(index)

    @property
    def loaded_steps(self) -> tuple[int, ...]:
        """Simulation steps currently resident in memory."""
        return tuple(self.steps[index] for index in sorted(self._cache))

    def clear_cache(self) -> None:
        """Release loaded data in lazy mode."""
        if self.mode == "eager":
            raise RuntimeError("Cannot clear the cache of an eager VTSSeries")
        self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"VTSSeries(directory={str(self.directory)!r}, mode={self.mode!r}, "
            f"steps={len(self)}, loaded={len(self._cache)})"
        )


def _read_ascii_array(element: ET.Element) -> np.ndarray:
    """Convert one ASCII VTK DataArray element to a NumPy array."""
    data_format = element.get("format", "ascii").lower()
    if data_format != "ascii":
        raise ValueError(
            f"Unsupported DataArray format {data_format!r}; only ASCII is supported"
        )

    vtk_type = element.get("type", "Float64")
    try:
        dtype = _VTK_DTYPES[vtk_type]
    except KeyError as error:
        raise ValueError(f"Unsupported VTK data type: {vtk_type!r}") from error

    return np.fromstring(element.text or "", dtype=dtype, sep=" ")


def read_vts(path: str | Path) -> VTSData:
    """Read all point fields and coordinates from an ASCII ``.vts`` file."""
    path = Path(path)
    root = ET.parse(path).getroot()

    structured_grid = root.find("StructuredGrid")
    if structured_grid is None:
        raise ValueError(f"{path} does not contain a StructuredGrid element")

    extent_text = structured_grid.get("WholeExtent")
    if extent_text is None:
        raise ValueError("StructuredGrid is missing the WholeExtent attribute")

    extent_values = tuple(int(value) for value in extent_text.split())
    if len(extent_values) != 6:
        raise ValueError(f"WholeExtent must contain 6 integers, got {extent_text!r}")
    extent = extent_values

    xmin, xmax, ymin, ymax, zmin, zmax = extent
    nx = xmax - xmin + 1
    ny = ymax - ymin + 1
    nz = zmax - zmin + 1
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid grid extent: {extent}")
    number_of_points = nx * ny * nz

    point_data = structured_grid.find("PointData")
    if point_data is None:
        raise ValueError("StructuredGrid does not contain PointData")

    fields: dict[str, np.ndarray] = {}
    for element in point_data.findall("DataArray"):
        name = element.get("Name")
        if not name:
            raise ValueError("A PointData DataArray is missing its Name attribute")
        if name in fields:
            raise ValueError(f"Duplicate PointData array name: {name!r}")

        components = int(element.get("NumberOfComponents", "1"))
        values = _read_ascii_array(element)
        expected_values = number_of_points * components
        if values.size != expected_values:
            raise ValueError(
                f"Field {name!r} contains {values.size} values; "
                f"expected {expected_values}"
            )

        if components == 1:
            values = values.reshape(nz, ny, nx)
            if nz == 1:
                values = values[0]
        else:
            values = values.reshape(nz, ny, nx, components)
            if nz == 1:
                values = values[0]
        fields[name] = values

    points = structured_grid.find("Points")
    if points is None:
        raise ValueError("StructuredGrid does not contain Points")
    coordinate_array = points.find("DataArray")
    if coordinate_array is None:
        raise ValueError("Points does not contain a DataArray")

    coordinate_components = int(
        coordinate_array.get("NumberOfComponents", "3")
    )
    coordinates = _read_ascii_array(coordinate_array)
    expected_coordinates = number_of_points * coordinate_components
    if coordinates.size != expected_coordinates:
        raise ValueError(
            f"Coordinate array contains {coordinates.size} values; "
            f"expected {expected_coordinates}"
        )
    coordinates = coordinates.reshape(number_of_points, coordinate_components)

    return VTSData(extent=extent, coordinates=coordinates, fields=fields)
