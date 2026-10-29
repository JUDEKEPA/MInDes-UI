"""Training settings and progress panel for the ML workflow."""
import importlib.util
import math

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import font_manager
from matplotlib.text import Text

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)
from ...i18n import tr
from .training_worker import TrainingWorker


def label(key, **params):
    return tr("ml_train." + key, **params)


class TrainingPanel(QWidget):
    busy_changed = Signal(bool)
    settled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset = None
        self.trainer = None
        self.worker = None
        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.options_widget = QWidget()
        content = QVBoxLayout(self.options_widget)
        scroll.setWidget(self.options_widget)
        root.addWidget(scroll, 1)

        model = QGroupBox(label("model"))
        form = QFormLayout(model)
        self.method = QComboBox()
        self.method.addItem(label("manual"), "manual")
        self.method.addItem(label("bayesian"), "bayesian")
        form.addRow(label("method"), self.method)
        self.layers = QLineEdit("28, 56, 28")
        form.addRow(label("layers"), self.layers)
        self.epochs = self._integer(1, 1000000, 100)
        form.addRow(label("epochs"), self.epochs)
        self.batch_size = self._integer(1, 10000000, 32000)
        form.addRow(label("batch_size"), self.batch_size)
        self.device = QComboBox()
        for text, value in ((label("auto"), None), ("CPU", "cpu"), ("CUDA", "cuda"), ("MPS", "mps")):
            self.device.addItem(text, value)
        form.addRow(label("device"), self.device)
        self.architecture_info = QLabel(label("architecture"))
        self.architecture_info.setWordWrap(True)
        form.addRow(self.architecture_info)
        content.addWidget(model)

        self.search_group = QGroupBox(label("search"))
        search = QFormLayout(self.search_group)
        self.search_fields = {}
        for key, title, default in (("layer1", "search_layer1", "20, 300, 40"),
                                    ("layer2", "search_layer2", "20, 300, 40"),
                                    ("layer3", "search_layer3", "20, 300, 40"),
                                    ("epochs", "search_epochs", "300, 2000, 400")):
            field = QLineEdit(default)
            self.search_fields[key] = field
            search.addRow(label(title), field)
        self.max_evals = self._integer(1, 10000, 10)
        search.addRow(label("max_evals"), self.max_evals)
        hint = QLabel(label("search_hint"))
        hint.setWordWrap(True)
        search.addRow(hint)
        content.addWidget(self.search_group)

        split = QGroupBox(label("split"))
        form = QFormLayout(split)
        self.test_percent = self._number(0.1, 99.9, 20, 1)
        self.test_percent.setSuffix(" %")
        form.addRow(label("validation"), self.test_percent)
        self.shuffle = QCheckBox(label("shuffle"))
        self.shuffle.setChecked(True)
        form.addRow(self.shuffle)
        seed_row = QWidget()
        row = QHBoxLayout(seed_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.use_seed = QCheckBox(label("use_seed"))
        self.use_seed.setChecked(True)
        self.seed = self._integer(0, 2147483647, 42)
        row.addWidget(self.use_seed)
        row.addWidget(self.seed)
        self.use_seed.toggled.connect(self.seed.setEnabled)
        form.addRow(seed_row)
        content.addWidget(split)

        optimization = QGroupBox(label("optimization"))
        form = QFormLayout(optimization)
        self.learning_rate = self._number(1e-10, 1, 1e-3, 10)
        self.lr_factor = self._number(0.0001, 0.9999, 0.1, 4)
        self.lr_patience = self._integer(0, 1000000, 2000)
        self.min_lr = self._number(0, 1, 1e-6, 10)
        for key in ("learning_rate", "lr_factor", "lr_patience", "min_lr"):
            form.addRow(label(key), getattr(self, key))
        content.addWidget(optimization)

        temperature = QGroupBox(label("temperature"))
        form = QFormLayout(temperature)
        self.normalize_temperature = QCheckBox(label("normalize_temperature"))
        form.addRow(self.normalize_temperature)
        self.temperature_min = self._number(-1e9, 1e9, 0, 6)
        self.temperature_max = self._number(-1e9, 1e9, 1, 6)
        form.addRow(label("temperature_min"), self.temperature_min)
        form.addRow(label("temperature_max"), self.temperature_max)
        self.normalize_temperature.toggled.connect(self._temperature_changed)
        content.addWidget(temperature)
        content.addStretch()

        self.loss_log_scale = QCheckBox(label("loss_log_scale"))
        self.loss_log_scale.setToolTip(label("loss_log_hint"))
        self.loss_log_scale.toggled.connect(self._draw_loss_plot)
        root.addWidget(self.loss_log_scale)
        self.loss_canvas = FigureCanvas(Figure(figsize=(4, 2.5), layout="constrained"))
        self.loss_canvas.setMinimumHeight(210)
        self.loss_axes = self.loss_canvas.figure.add_subplot(111)
        self.loss_axes.set_xlabel(label("loss_epoch"))
        self.loss_axes.set_ylabel(label("loss_mse"))
        self.train_line, = self.loss_axes.plot([], [], color="#2878b5", label=label("loss_train"))
        self.test_line, = self.loss_axes.plot([], [], color="#e07b23", label=label("loss_test"))
        self.loss_axes.legend(loc="upper right", fontsize=8)
        self.loss_axes.grid(True, alpha=0.25)
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        self._loss_fonts = [name for name in (
            self.font().family(), "Microsoft YaHei", "Yu Gothic", "Malgun Gothic",
            "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
        ) if name in available_fonts]
        self._reset_loss_plot()
        root.addWidget(self.loss_canvas, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        root.addWidget(self.progress)
        self.status = QLabel(label("need_dataset"))
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        buttons = QHBoxLayout()
        self.start_button = QPushButton(label("start"))
        self.cancel_button = QPushButton(label("cancel"))
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        root.addLayout(buttons)
        self.start_button.clicked.connect(self.start_training)
        self.cancel_button.clicked.connect(self.cancel_training)
        self.method.currentIndexChanged.connect(self._method_changed)
        self._method_changed()
        self._temperature_changed()

    @staticmethod
    def _integer(low, high, value):
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    @staticmethod
    def _number(low, high, value, decimals):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    @property
    def is_training(self):
        return self.worker is not None

    def _method_changed(self, *_):
        searching = self.method.currentData() == "bayesian"
        self.search_group.setVisible(searching)
        self.layers.setEnabled(not searching)
        self.epochs.setEnabled(not searching)

    def _temperature_changed(self, *_):
        enabled = self.normalize_temperature.isChecked() and self.normalize_temperature.isEnabled()
        self.temperature_min.setEnabled(enabled)
        self.temperature_max.setEnabled(enabled)

    def set_dataset(self, dataset):
        if self.is_training:
            return
        self.dataset = dataset
        self.trainer = None
        self._reset_loss_plot()
        self.progress.setValue(0)
        self.start_button.setEnabled(dataset is not None)
        self.status.setText(label("need_dataset") if dataset is None else label("ready", rows=len(dataset.x), inputs=dataset.x.shape[1], outputs=dataset.y.shape[1]))
        has_temperature = dataset is not None and dataset.metadata.get("include_temperature", False)
        self.normalize_temperature.setEnabled(has_temperature)
        self.normalize_temperature.setChecked(False)
        if has_temperature:
            self.temperature_min.setValue(float(dataset.x[:, 0].min()))
            self.temperature_max.setValue(float(dataset.x[:, 0].max()))
        self._temperature_changed()

    def options(self):
        method = self.method.currentData()
        layers = (28, 56, 28)
        if method == "manual":
            try:
                layers = tuple(int(value.strip()) for value in self.layers.text().split(","))
            except ValueError:
                raise ValueError(label("invalid_layers")) from None
            if not layers or any(value <= 0 for value in layers):
                raise ValueError(label("invalid_layers"))
        ranges = {}
        if method == "bayesian":
            for key, field in self.search_fields.items():
                try:
                    low, high, step = (int(value.strip()) for value in field.text().split(","))
                except ValueError:
                    raise ValueError(label("invalid_range")) from None
                if low <= 0 or high < low or step <= 0 or (high-low)//step > 100000:
                    raise ValueError(label("invalid_range"))
                ranges[key] = (low, high, step)
        if self.min_lr.value() > self.learning_rate.value():
            raise ValueError(label("invalid_lr"))
        temperature = None
        if self.normalize_temperature.isEnabled() and self.normalize_temperature.isChecked():
            temperature = (self.temperature_min.value(), self.temperature_max.value())
            if temperature[0] >= temperature[1]:
                raise ValueError(label("invalid_temperature"))
        test_size = self.test_percent.value() / 100
        if self.dataset is None or len(self.dataset.x) - math.ceil(len(self.dataset.x) * test_size) < 1:
            raise ValueError(label("invalid_split"))
        return dict(method=method, hidden_layers=layers, epochs=self.epochs.value(),
            batch_size=self.batch_size.value(), device=self.device.currentData(),
            test_size=test_size, shuffle=self.shuffle.isChecked(),
            seed=self.seed.value() if self.use_seed.isChecked() else None,
            learning_rate=self.learning_rate.value(), lr_factor=self.lr_factor.value(),
            lr_patience=self.lr_patience.value(), min_lr=self.min_lr.value(),
            temperature_range=temperature, search_ranges=ranges, max_evals=self.max_evals.value())

    def start_training(self):
        if self.is_training or self.dataset is None:
            return
        try:
            options = self.options()
            dependencies = ["torch", "sklearn"] + (["hyperopt"] if options["method"] == "bayesian" else [])
            missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
            if missing:
                raise ValueError(label("missing_dependencies", names=", ".join(missing)))
        except Exception as error:
            QMessageBox.warning(self, label("cannot_start"), str(error))
            return
        self.trainer = None
        self._reset_loss_plot()
        self.worker = TrainingWorker(self.dataset, options, self)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.canceled.connect(self._canceled)
        self.worker.finished.connect(self._finished)
        self.options_widget.setEnabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.status.setText(label("starting"))
        self.busy_changed.emit(True)
        self.worker.start()

    def cancel_training(self):
        if self.worker is not None:
            self.worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText(label("canceling"))

    def _progress(self, values):
        if self._loss_run != values["run"]:
            self._reset_loss_plot()
            self._loss_run = values["run"]
        for epoch, loss, val_loss in values["points"]:
            self._loss_epochs.append(epoch)
            self._train_losses.append(loss)
            self._test_losses.append(val_loss)
        self.loss_axes.set_title(label("loss_run", run=values["run"], runs=values["runs"]), fontsize=10)
        self._draw_loss_plot()
        self.progress.setValue(int(1000 * (values["run"] - 1 + values["epoch"] / values["epochs"]) / values["runs"]))
        if self.cancel_button.isEnabled():
            self.status.setText(label("progress", **values))

    def _completed(self, trainer):
        self.trainer = trainer
        self._loss_epochs = list(range(1, len(trainer.history["loss"]) + 1))
        self._train_losses = list(trainer.history["loss"])
        self._test_losses = list(trainer.history["val_loss"])
        self.loss_axes.set_title(label("loss_retained"), fontsize=10)
        self._draw_loss_plot()
        self.progress.setValue(1000)
        self.status.setText(label("completed", train=trainer.training_metadata["final_train_loss"], validation=trainer.training_metadata["final_val_loss"], layers=", ".join(map(str, trainer.hidden_layers))))

    def _failed(self, message):
        self.status.setText(label("failed", error=message))

    def _reset_loss_plot(self):
        self._loss_run = None
        self._loss_epochs = []
        self._train_losses = []
        self._test_losses = []
        self.loss_axes.set_title(label("loss_title"), fontsize=10)
        self.loss_axes.set_xlim(0, 1)
        self.loss_axes.set_ylim((0.001, 1) if self.loss_log_scale.isChecked() else (0, 1))
        self._draw_loss_plot()

    def _draw_loss_plot(self):
        logarithmic = self.loss_log_scale.isChecked()
        for line, losses in ((self.train_line, self._train_losses), (self.test_line, self._test_losses)):
            displayed = [value if math.isfinite(value) and (value > 0 or not logarithmic)
                         else float("nan") for value in losses]
            line.set_data(self._loss_epochs, displayed)
        self.loss_axes.relim()
        scale = "log" if logarithmic else "linear"
        if self.loss_axes.get_yscale() != scale:
            self.loss_axes.set_yscale(scale)
        self.loss_axes.autoscale(enable=True, axis="both")
        if logarithmic and not any(math.isfinite(value) and value > 0
                                   for losses in (self._train_losses, self._test_losses) for value in losses):
            self.loss_axes.set_ylim(0.001, 1)
        for text in self.loss_canvas.figure.findobj(Text):
            text.set_fontfamily(self._loss_fonts)
        self.loss_canvas.draw_idle()

    def _canceled(self):
        self.status.setText(label("canceled"))

    def _finished(self):
        worker = self.worker
        self.worker = None
        worker.deleteLater()
        self.options_widget.setEnabled(True)
        self.start_button.setEnabled(self.dataset is not None)
        self.cancel_button.setEnabled(False)
        self.busy_changed.emit(False)
        self.settled.emit()
