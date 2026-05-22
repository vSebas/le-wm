import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
import json
import csv
from pathlib import Path

if not hasattr(pa, "PyExtensionType"):
    pa.PyExtensionType = pa.ExtensionType

from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    def resize_fn(x):
        shape = x.shape
        x = x.reshape(-1, *shape[-3:])
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        return x.reshape(*shape[:-2], img_size, img_size)

    resize = dt.transforms.WrapTorchTransform(resize_fn, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str, row_indices=None):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    if row_indices is not None:
        col_data = col_data[row_indices]
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


class BestModelObjectCallback(Callback):
    """Save the best model object according to a logged validation metric."""

    def __init__(
        self,
        dirpath,
        filename="model_object",
        monitor="validate/pred_loss_epoch",
        mode="min",
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.best_score = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if trainer.sanity_checking:
            return

        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return

        score = float(score.detach().cpu()) if torch.is_tensor(score) else float(score)
        is_better = self.best_score is None
        if self.best_score is not None:
            if self.mode == "min":
                is_better = score < self.best_score
            elif self.mode == "max":
                is_better = score > self.best_score
            else:
                raise ValueError(f"Unsupported mode={self.mode!r}")

        if not is_better:
            return

        self.best_score = score
        epoch = int(trainer.current_epoch)
        object_path = self.dirpath / f"{self.filename}_best_pred_object.ckpt"
        meta_path = self.dirpath / f"{self.filename}_best_pred_object.json"
        self.dirpath.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.model, object_path)
        meta_path.write_text(
            json.dumps(
                {
                    "monitor": self.monitor,
                    "mode": self.mode,
                    "best_score": score,
                    "logged_epoch": epoch,
                    "object_checkpoint": object_path.name,
                },
                indent=2,
            )
            + "\n"
        )


class EpochMetricsCallback(Callback):
    """Write compact epoch-level metrics for plotting and checkpoint selection."""

    def __init__(
        self,
        dirpath,
        filename="epoch_metrics.csv",
        best_filename="best_metrics.json",
        monitor="validate/pred_loss_epoch",
        mode="min",
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.best_filename = best_filename
        self.monitor = monitor
        self.mode = mode
        self.rows = {}
        self.best_score = None
        self.best_epoch = None

    def on_fit_start(self, trainer, pl_module):
        path = self.dirpath / self.filename
        if not path.exists():
            return
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("epoch") not in (None, ""):
                    self.rows[int(row["epoch"])] = row
                    score = _as_float(row.get(self.monitor))
                    if score is not None and self._is_better(score):
                        self.best_score = score
                        self.best_epoch = int(row["epoch"])

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        epoch = int(trainer.current_epoch)
        row = {"epoch": epoch}
        for key, value in trainer.callback_metrics.items():
            if key.endswith("_step"):
                continue
            scalar = _metric_to_float(value)
            if scalar is not None:
                row[key] = scalar

        row["lr"] = _current_lr(trainer)
        score = _as_float(row.get(self.monitor))
        is_best = score is not None and self._is_better(score)
        if is_best:
            self.best_score = score
            self.best_epoch = epoch
        row["is_best"] = bool(is_best)
        row["best_epoch"] = self.best_epoch
        row["best_score"] = self.best_score

        self.rows[epoch] = row
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self._write_csv()
        self._write_best_json()

    def _is_better(self, score):
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        if self.mode == "max":
            return score > self.best_score
        raise ValueError(f"Unsupported mode={self.mode!r}")

    def _write_csv(self):
        path = self.dirpath / self.filename
        fields = ["epoch"]
        dynamic_fields = sorted(
            {
                key
                for row in self.rows.values()
                for key in row
                if key != "epoch"
            }
        )
        fields.extend(dynamic_fields)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for epoch in sorted(self.rows):
                writer.writerow(self.rows[epoch])

    def _write_best_json(self):
        path = self.dirpath / self.best_filename
        path.write_text(
            json.dumps(
                {
                    "monitor": self.monitor,
                    "mode": self.mode,
                    "best_epoch": self.best_epoch,
                    "best_score": self.best_score,
                    "metrics_csv": self.filename,
                },
                indent=2,
            )
            + "\n"
        )


def _metric_to_float(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _current_lr(trainer):
    optimizers = getattr(trainer, "optimizers", None) or []
    if not optimizers:
        return None
    param_groups = getattr(optimizers[0], "param_groups", [])
    if not param_groups:
        return None
    return float(param_groups[0].get("lr"))
