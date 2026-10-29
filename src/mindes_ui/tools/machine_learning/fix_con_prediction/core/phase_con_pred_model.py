"""PyTorch regression model for phase concentration prediction."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
import json
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class TrainingCanceled(Exception):
    """Cooperative cancellation requested by the caller."""


class PhaseConcentrationNet(nn.Module if nn is not None else object):
    """Fully connected tanh network for phase concentration regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Iterable[int] = (28, 56, 28),
    ) -> None:
        _require_torch()
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for layer_size in hidden_layers:
            layers.append(nn.Linear(previous_dim, int(layer_size)))
            layers.append(nn.Tanh())
            previous_dim = int(layer_size)
        layers.append(nn.Linear(previous_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RegressionTasks:
    """Train and evaluate a PyTorch phase concentration regression model."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        random_state: int | None = None,
    ) -> None:
        _require_torch()

        self.device = torch.device(device or _default_device())
        self.random_state = random_state

        self.temperature_max = None
        self.temperature_min = None

        self.input_dim = None
        self.output_dim = None
        self.hidden_layers = None
        self.dataset_metadata = None
        self.split_metadata = None
        self.training_metadata = None
        self.model_metadata = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.model = None
        self.history = None

    def set_temperature_range(
        self,
        temperature_min: float,
        temperature_max: float,
    ) -> None:
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

    def import_dataset(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Normalize optional temperature input and split train/test arrays."""
        inputs = np.asarray(inputs, dtype=np.float32).copy()
        outputs = np.asarray(outputs, dtype=np.float32).copy()

        if inputs.ndim != 2:
            raise ValueError(f"inputs must be a 2-D array, got shape {inputs.shape}")
        if outputs.ndim != 2:
            raise ValueError(f"outputs must be a 2-D array, got shape {outputs.shape}")
        if len(inputs) != len(outputs):
            raise ValueError("inputs and outputs must contain the same number of rows")

        if self.temperature_min is not None and self.temperature_max is not None:
            denominator = self.temperature_max - self.temperature_min
            if denominator == 0:
                raise ValueError("temperature_max and temperature_min must be different")
            inputs[:, 0] = (inputs[:, 0] - self.temperature_min) / denominator

        if "random_state" not in kwargs and self.random_state is not None:
            kwargs["random_state"] = self.random_state
        self.split_metadata = dict(kwargs)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            inputs,
            outputs,
            **kwargs,
        )
        self.input_dim = self.x_train.shape[1]
        self.output_dim = self.y_train.shape[1]
        self.dataset_metadata = None

    def initialise(
        self,
        dataset: Any | None = None,
        *,
        inputs: np.ndarray | None = None,
        outputs: np.ndarray | None = None,
        hidden_layers: Iterable[int] = (28, 56, 28),
        **split_kwargs: Any,
    ) -> None:
        """Import data if supplied, then build a model with matching dimensions.

        Parameters
        ----------
        dataset:
            Optional object with ``x`` and ``y`` arrays, such as the
            ``ThermoMLDataset`` created by ``read_csv.py``.
        inputs, outputs:
            Optional raw arrays. Use these instead of ``dataset`` when the data
            is already separated.
        hidden_layers:
            Hidden layer sizes for the neural network.
        split_kwargs:
            Extra keyword arguments passed to :func:`train_test_split` when
            ``dataset`` or ``inputs``/``outputs`` are supplied.
        """
        if dataset is not None:
            if inputs is not None or outputs is not None:
                raise ValueError("Pass either dataset or inputs/outputs, not both")
            inputs = dataset.x
            outputs = dataset.y
            self.dataset_metadata = getattr(dataset, "metadata", None)

        if inputs is not None or outputs is not None:
            if inputs is None or outputs is None:
                raise ValueError("inputs and outputs must be passed together")
            self.import_dataset(inputs, outputs, **split_kwargs)
            if dataset is not None:
                self.dataset_metadata = getattr(dataset, "metadata", None)

        self._require_dataset()
        self.input_dim = self.x_train.shape[1]
        self.output_dim = self.y_train.shape[1]
        self.hidden_layers = tuple(int(layer) for layer in hidden_layers)
        self.model = PhaseConcentrationNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
        ).to(self.device)
        self.model_metadata = self.describe_model()

    def initialize(
        self,
        dataset: Any | None = None,
        *,
        inputs: np.ndarray | None = None,
        outputs: np.ndarray | None = None,
        hidden_layers: Iterable[int] = (28, 56, 28),
        **split_kwargs: Any,
    ) -> None:
        """US spelling alias for :meth:`initialise`."""
        self.initialise(
            dataset,
            inputs=inputs,
            outputs=outputs,
            hidden_layers=hidden_layers,
            **split_kwargs,
        )

    def model_generate(
        self,
        first_layer_neuron: int = 28,
        second_layer_neuron: int = 56,
        third_layer_neuron: int = 28,
    ) -> None:
        """Generate a three-hidden-layer PyTorch model."""
        self.initialise(
            hidden_layers=(
                first_layer_neuron,
                second_layer_neuron,
                third_layer_neuron,
            )
        )

    def model_training(
        self,
        epoch: int = 100,
        batch_size: int = 32000,
        learning_rate: float = 1e-3,
        lr_factor: float = 0.1,
        lr_patience: int = 2000,
        min_lr: float = 1e-6,
        *,
        progress_callback=None,
        cancel_check=None,
    ) -> tuple[float, float, dict[str, list[float]]]:
        """Train and validate the PyTorch model."""
        self._require_dataset()
        if self.model is None:
            self.model_generate()

        train_loader = self._make_loader(batch_size=batch_size, shuffle=True)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )

        history = {"loss": [], "val_loss": [], "lr": []}
        for epoch_index in range(int(epoch)):
            if cancel_check is not None and cancel_check():
                raise TrainingCanceled()
            train_loss = self._train_one_epoch(
                train_loader, loss_function, optimizer, cancel_check=cancel_check
            )
            val_loss = self._evaluate_loss(self.x_test, self.y_test, loss_function)
            scheduler.step(train_loss)

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            if progress_callback is not None:
                progress_callback(epoch_index + 1, int(epoch), train_loss, val_loss, history["lr"][-1])

        if cancel_check is not None and cancel_check():
            raise TrainingCanceled()
        # Split arrays are already prepared (including temperature scaling).
        train_loss = self._evaluate_loss(self.x_train, self.y_train, loss_function)
        val_loss = self._evaluate_loss(self.x_test, self.y_test, loss_function)

        self.history = history
        self.training_metadata = {
            "epoch": int(epoch),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "lr_factor": float(lr_factor),
            "lr_patience": int(lr_patience),
            "min_lr": float(min_lr),
            "final_train_loss": float(train_loss),
            "final_val_loss": float(val_loss),
            "best_train_loss": _min_or_none(history["loss"]),
            "best_val_loss": _min_or_none(history["val_loss"]),
            "final_lr": None if not history["lr"] else float(history["lr"][-1]),
        }
        self.model_metadata = self.describe_model()
        return train_loss, val_loss, history

    def bayesian_opt_model_training(
        self,
        first_layer: list[int] | None = None,
        second_layer: list[int] | None = None,
        third_layer: list[int] | None = None,
        epoch: list[int] | None = None,
        max_evals: int = 10,
    ) -> None:
        """Tune layer sizes and epoch count with hyperopt, if installed."""
        try:
            from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "hyperopt is required for bayesian_opt_model_training"
            ) from error

        def objective(params: dict[str, float]) -> dict[str, float | str]:
            self.model_generate(
                int(params["first_layer"]),
                int(params["second_layer"]),
                int(params["third_layer"]),
            )
            _, val_loss, _ = self.model_training(int(params["epoch"]))
            print("Test mse:", val_loss)
            return {"loss": val_loss, "status": STATUS_OK}

        if first_layer is None:
            first_layer = [20, 300, 40]
        if second_layer is None:
            second_layer = [20, 300, 40]
        if third_layer is None:
            third_layer = [20, 300, 40]
        if epoch is None:
            epoch = [300, 2000, 400]

        space = {
            "first_layer": hp.quniform(
                "first_layer", first_layer[0], first_layer[1], first_layer[2]
            ),
            "second_layer": hp.quniform(
                "second_layer", second_layer[0], second_layer[1], second_layer[2]
            ),
            "third_layer": hp.quniform(
                "third_layer", third_layer[0], third_layer[1], third_layer[2]
            ),
            "epoch": hp.quniform("epoch", epoch[0], epoch[1], epoch[2]),
        }

        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
        )
        print(best)

        model_temp = []
        history_temp = []
        loss_temp = []
        for _ in range(3):
            self.model_generate(
                int(best["first_layer"]),
                int(best["second_layer"]),
                int(best["third_layer"]),
            )
            _, val_loss, history = self.model_training(epoch=int(best["epoch"]))

            model_temp.append(self.model)
            history_temp.append(history)
            loss_temp.append(val_loss)

        best_index = loss_temp.index(min(loss_temp))
        self.model = model_temp[best_index]
        self.history = history_temp[best_index]

    def manually_model_training(
        self,
        first_layer: int,
        second_layer: int,
        third_layer: int,
        epoch: int,
        batch_size: int = 32000,
    ) -> tuple[float, float, dict[str, list[float]]]:
        self.model_generate(first_layer, second_layer, third_layer)
        return self.model_training(epoch=epoch, batch_size=batch_size)

    def prepare_prediction_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Apply metadata-defined multiphase preprocessing before prediction."""
        prepared = np.asarray(inputs, dtype=np.float32).copy()
        if prepared.ndim != 2:
            raise ValueError(f"inputs must be a 2-D array, got shape {prepared.shape}")

        preprocessing = _phase_preprocessing_from_metadata(self.dataset_metadata)
        if preprocessing is None:
            return prepared

        input_columns = preprocessing["input_columns"]
        if prepared.shape[1] != len(input_columns):
            raise ValueError(
                "Input width does not match dataset metadata: "
                f"expected {len(input_columns)}, got {prepared.shape[1]}"
            )

        phi_indices = preprocessing["phi_indices"]
        phi_values = prepared[:, phi_indices]
        phi_values[phi_values <= preprocessing["threshold"]] = 0.0
        phi_sums = phi_values.sum(axis=1, keepdims=True)
        invalid_rows = np.flatnonzero(phi_sums[:, 0] <= 0.0)
        if len(invalid_rows):
            raise ValueError(
                "Cannot normalize prediction rows with no surviving phase: "
                f"{invalid_rows.tolist()}"
            )
        prepared[:, phi_indices] = phi_values / phi_sums
        return prepared

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Preprocess inputs, predict, and zero outputs for inactive phases."""
        if self.model is None:
            raise RuntimeError("Call model_generate or model_training before predict")

        inputs = np.asarray(inputs, dtype=np.float32)
        single_input = inputs.ndim == 1
        if single_input:
            inputs = inputs.reshape(1, -1)
        original_inputs = inputs.copy()
        prepared_inputs = self.prepare_prediction_inputs(inputs)
        if self.temperature_min is not None and self.temperature_max is not None:
            denominator = self.temperature_max - self.temperature_min
            if denominator <= 0:
                raise ValueError("Temperature maximum must exceed minimum")
            prepared_inputs[:, 0] = (prepared_inputs[:, 0] - self.temperature_min) / denominator

        self.model.eval()
        inputs_tensor = torch.as_tensor(
            prepared_inputs,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            result = self.model(inputs_tensor).cpu().numpy()
        result = self.zero_inactive_phase_outputs(original_inputs, result)
        return result[0] if single_input else result

    def zero_inactive_phase_outputs(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
    ) -> np.ndarray:
        """Force each inactive phase's concentration outputs to exactly zero."""
        result = np.asarray(outputs, dtype=np.float32).copy()
        preprocessing = _phase_preprocessing_from_metadata(self.dataset_metadata)
        if preprocessing is None:
            return result

        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim != 2 or result.ndim != 2:
            raise ValueError("inputs and outputs must be 2-D arrays")
        if len(inputs) != len(result):
            raise ValueError("inputs and outputs must contain the same number of rows")

        for phi_index, output_indices in preprocessing["phase_output_indices"]:
            inactive_rows = inputs[:, phi_index] <= preprocessing["threshold"]
            for output_index in output_indices:
                result[inactive_rows, output_index] = 0.0
        return result

    def describe_model(self) -> dict[str, Any]:
        """Return metadata describing the current model and training setup."""
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_class": "PhaseConcentrationNet",
            "framework": "PyTorch",
            "torch_version": torch.__version__ if torch is not None else None,
            "device": str(self.device),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": list(self.hidden_layers or ()),
            "activation": "tanh",
            "output_activation": "linear",
            "train_rows": None if self.x_train is None else len(self.x_train),
            "test_rows": None if self.x_test is None else len(self.x_test),
            "split": self.split_metadata,
            "normalization": {
                "temperature_min": self.temperature_min,
                "temperature_max": self.temperature_max,
                "temperature_normalized": (
                    self.temperature_min is not None
                    and self.temperature_max is not None
                ),
                "output_normalized": False,
            },
            "dataset": self.dataset_metadata,
            "training": self.training_metadata,
        }
        return _json_ready(metadata)

    def save_model_metadata(self, path: str | Path) -> dict[str, Any]:
        """Write model metadata to a JSON file and return it."""
        metadata = self.describe_model()
        path = Path(path)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        self.model_metadata = metadata
        return metadata

    def save_model(self, path: str | Path) -> dict[str, Any]:
        """Save model weights and metadata to a PyTorch checkpoint."""
        if self.model is None:
            raise RuntimeError("Call initialise or model_training before save_model")

        metadata = self.describe_model()
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "metadata": metadata,
        }
        path = Path(path)
        torch.save(checkpoint, path)
        self.model_metadata = metadata
        return metadata

    def save_model_folder(
        self,
        folder: str | Path,
        *,
        checkpoint_name: str = "model_checkpoint.pt",
        metadata_name: str = "metadata.json",
        torchscript_name: str = "model_scripted.pt",
        example_input: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Save model and metadata together in one folder.

        The folder contains a normal Python checkpoint, a standalone JSON
        metadata file, and a TorchScript model that can be loaded from C++ with
        ``torch::jit::load``.
        """
        if self.model is None:
            raise RuntimeError("Call initialise or model_training before save_model_folder")

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        checkpoint_path = folder / checkpoint_name
        metadata_path = folder / metadata_name
        torchscript_path = folder / torchscript_name

        metadata = self.describe_model()
        metadata.update(
            {
                "export_folder": str(folder),
                "checkpoint_file": checkpoint_name,
                "metadata_file": metadata_name,
                "torchscript_file": torchscript_name,
                "cpp_load_recommendation": (
                    "Use torch::jit::load on model_scripted.pt, and read "
                    "metadata.json for input/output columns and dimensions."
                ),
            }
        )

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "metadata": metadata,
            },
            checkpoint_path,
        )
        self._save_torchscript_model(torchscript_path, example_input=example_input)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_ready(metadata), handle, indent=2)
            handle.write("\n")

        self.model_metadata = metadata
        return _json_ready(metadata)

    def _save_torchscript_model(
        self,
        path: Path,
        *,
        example_input: np.ndarray | None = None,
    ) -> None:
        self.model.eval()
        if example_input is None:
            if self.input_dim is None:
                raise RuntimeError("input_dim is unknown; pass example_input")
            example_input = np.zeros((1, int(self.input_dim)), dtype=np.float32)
        example_tensor = torch.as_tensor(
            example_input,
            dtype=torch.float32,
            device=self.device,
        )
        scripted = torch.jit.trace(self.model, example_tensor)
        scripted.save(str(path))

    def _make_loader(self, *, batch_size: int, shuffle: bool) -> DataLoader:
        x_tensor = torch.as_tensor(self.x_train, dtype=torch.float32)
        y_tensor = torch.as_tensor(self.y_train, dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        cancel_check=None,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        sample_count = 0
        for x_batch, y_batch in train_loader:
            if cancel_check is not None and cancel_check():
                raise TrainingCanceled()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            predictions = self.model(x_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = x_batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        return total_loss / sample_count

    def _evaluate_loss(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        loss_function: nn.Module,
    ) -> float:
        self.model.eval()
        x_tensor = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(outputs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = self.model(x_tensor)
            loss = loss_function(predictions, y_tensor)
        return float(loss.item())

    def _require_dataset(self) -> None:
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call import_dataset before generating or training a model")


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required by phase_con_pred_model.py. "
            "Install torch before using RegressionTasks."
        )


def _min_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(min(values))


def _phase_preprocessing_from_metadata(
    dataset_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not dataset_metadata:
        return None

    input_columns = list(dataset_metadata.get("input_columns") or [])
    output_columns = list(dataset_metadata.get("output_columns") or [])
    phi_indices = [
        index
        for index, column in enumerate(input_columns)
        if column.startswith("phi")
    ]
    if not phi_indices:
        return None

    settings = dataset_metadata.get("phase_fraction_preprocessing")
    threshold = None
    enabled = False
    if isinstance(settings, dict):
        enabled = bool(settings.get("enabled", False))
        threshold = settings.get("threshold")

    if threshold is None and "training_phase_threshold" in dataset_metadata:
        threshold = dataset_metadata.get("training_phase_threshold")
        enabled = threshold is not None

    if threshold is None and dataset_metadata.get("phase_threshold_mode") == "transform":
        threshold = dataset_metadata.get("phase_threshold")
        enabled = threshold is not None

    if not enabled or threshold is None:
        return None

    threshold = float(threshold)
    if threshold < 0.0 or threshold >= 1.0:
        raise ValueError("Prediction phase threshold must be in [0, 1)")

    phase_output_indices = []
    for phi_index in phi_indices:
        phase_name = input_columns[phi_index].removeprefix("phi_")
        output_prefix = f"phase_con_{phase_name}"
        matching_outputs = [
            index
            for index, column in enumerate(output_columns)
            if column == output_prefix or column.startswith(f"{output_prefix}_")
        ]
        if not matching_outputs:
            raise ValueError(
                f"No phase_con output matches phi input {input_columns[phi_index]!r}"
            )
        phase_output_indices.append((phi_index, matching_outputs))
    return {
        "threshold": threshold,
        "input_columns": input_columns,
        "phi_indices": phi_indices,
        "phase_output_indices": phase_output_indices,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
