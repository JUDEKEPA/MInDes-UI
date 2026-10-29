"""Background training using the phase-concentration backend."""
from copy import deepcopy
import threading
import time

import numpy as np
from PySide6.QtCore import QThread, Signal


class TrainingWorker(QThread):
    progress = Signal(object)
    completed = Signal(object)
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, dataset, options, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.options = options
        self._cancel = threading.Event()
        self._last_progress = 0.0

    def request_cancel(self):
        self._cancel.set()

    def run(self):
        try:
            self.completed.emit(self.train())
        except Exception as error:
            if self._cancel.is_set():
                self.canceled.emit()
            else:
                self.failed.emit(str(error))

    def train(self):
        from .fix_con_prediction.core.phase_con_pred_model import RegressionTasks, TrainingCanceled
        import torch

        options = self.options
        seed = options["seed"]
        if seed is not None:
            torch.manual_seed(seed)
        trainer = RegressionTasks(device=options["device"], random_state=seed)
        if options["temperature_range"] is not None:
            trainer.set_temperature_range(*options["temperature_range"])
        trainer.initialise(
            self.dataset, hidden_layers=options["hidden_layers"],
            test_size=options["test_size"], shuffle=options["shuffle"],
        )
        total_runs = 1 if options["method"] == "manual" else options["max_evals"] + 3
        run_index = 0

        def fit(layers, epochs):
            nonlocal run_index
            if self._cancel.is_set():
                raise TrainingCanceled()
            run_index += 1
            trainer.initialise(hidden_layers=layers)
            pending = []
            latest = None

            def flush():
                if pending:
                    self.progress.emit(dict(latest, points=list(pending)))
                    pending.clear()

            def report(epoch, total, loss, val_loss, lr):
                nonlocal latest
                pending.append((epoch, loss, val_loss))
                latest = dict(run=run_index, runs=total_runs,
                    epoch=epoch, epochs=total, loss=loss, val_loss=val_loss, lr=lr)
                now = time.monotonic()
                if now - self._last_progress >= 0.1 or epoch == 1 or epoch == total:
                    self._last_progress = now
                    flush()

            try:
                return trainer.model_training(
                    epoch=epochs, batch_size=options["batch_size"],
                    learning_rate=options["learning_rate"], lr_factor=options["lr_factor"],
                    lr_patience=options["lr_patience"], min_lr=options["min_lr"],
                    progress_callback=report, cancel_check=self._cancel.is_set,
                )
            finally:
                flush()

        if options["method"] == "manual":
            fit(options["hidden_layers"], options["epochs"])
        else:
            from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
            space = {key: hp.choice(key, list(range(low, high + 1, step)))
                     for key, (low, high, step) in options["search_ranges"].items()}

            def objective(params):
                layers = tuple(int(params[key]) for key in ("layer1", "layer2", "layer3"))
                _, loss, _ = fit(layers, int(params["epochs"]))
                return {"loss": loss, "status": STATUS_OK}

            best = space_eval(space, fmin(
                fn=objective, space=space, algo=tpe.suggest, trials=Trials(),
                max_evals=options["max_evals"], rstate=np.random.default_rng(seed),
                show_progressbar=False, verbose=False,
            ))
            layers = tuple(int(best[key]) for key in ("layer1", "layer2", "layer3"))
            best_loss = float("inf")
            selected = None
            for _ in range(3):
                _, loss, _ = fit(layers, int(best["epochs"]))
                if loss < best_loss:
                    best_loss = loss
                    selected = (deepcopy(trainer.model.state_dict()), deepcopy(trainer.history),
                                deepcopy(trainer.training_metadata))
            trainer.model.load_state_dict(selected[0])
            trainer.history, trainer.training_metadata = selected[1:]
            trainer.training_metadata["bayesian_search"] = {
                "max_evals": options["max_evals"], "best_parameters": best,
                "search_ranges": options["search_ranges"], "final_retrains": 3,
            }
            trainer.model_metadata = trainer.describe_model()
        if self._cancel.is_set():
            raise TrainingCanceled()
        return trainer
