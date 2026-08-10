"""
Local YOLO training manager.

Runs training in a background thread, streams progress via callback.
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import yaml

from eyve.core.project_manager import Project
from eyve.core.logger import log
from eyve.training.dataset_builder import build_dataset


@dataclass
class TrainConfig:
    base_model: str = "yolov8n.pt"   # yolov8n/s/m/l/x.pt
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    device: str = "auto"             # "auto" | "cpu" | "0" (cuda)
    conf_default: float = 0.50
    patience: int = 20               # early stopping

    def device_arg(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


@dataclass
class TrainProgress:
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    map50: float = 0.0
    elapsed_s: float = 0.0
    eta_s: float = 0.0
    status: str = "idle"   # idle | preparing | training | done | error
    message: str = ""
    best_model: Optional[str] = None


ProgressCallback = Callable[[TrainProgress], None]


class TrainManager:
    def __init__(self, project: Project):
        self._project = project
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._progress = TrainProgress()
        self._callbacks: list[ProgressCallback] = []

    def add_callback(self, cb: ProgressCallback) -> None:
        self._callbacks.append(cb)

    def _emit(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self._progress, k, v)
        p = TrainProgress(**self._progress.__dict__)
        for cb in self._callbacks:
            try:
                cb(p)
            except Exception:
                pass

    def start(self, config: TrainConfig) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run, args=(config,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self, cfg: TrainConfig) -> None:
        start_time = time.time()
        try:
            self._emit(status="preparing", message="Preparing dataset…", epoch=0)

            yaml_path = build_dataset(self._project)

            self._emit(status="training", message="Starting YOLO training…",
                       total_epochs=cfg.epochs)

            from ultralytics import YOLO
            from ultralytics.utils.callbacks.base import BaseCallback
            from eyve.core.model_manager import local_path_or_name

            model_path = local_path_or_name(cfg.base_model)
            log.debug(f"Loading base model: {model_path}")
            model = YOLO(model_path)

            mgr = self
            epoch_times: list[float] = []
            epoch_start = [time.time()]

            class _EpveCallback(BaseCallback):
                def on_train_epoch_end(self, trainer):
                    elapsed = time.time() - start_time
                    epoch_times.append(time.time() - epoch_start[0])
                    epoch_start[0] = time.time()
                    ep = trainer.epoch + 1
                    total = trainer.epochs
                    avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 1
                    remaining = avg_ep * (total - ep)
                    metrics = trainer.metrics or {}
                    mgr._emit(
                        epoch=ep,
                        total_epochs=total,
                        loss=float(getattr(trainer, "loss", 0) or 0),
                        map50=float(metrics.get("metrics/mAP50(B)", 0)),
                        elapsed_s=elapsed,
                        eta_s=remaining,
                        status="training",
                        message=f"Epoch {ep}/{total}",
                    )
                    if mgr._stop_flag.is_set():
                        raise KeyboardInterrupt("Training stopped by user")

            model.add_callback("on_train_epoch_end", _EpveCallback().on_train_epoch_end)

            device = cfg.device_arg()
            results = model.train(
                data=str(yaml_path),
                epochs=cfg.epochs,
                imgsz=cfg.imgsz,
                batch=cfg.batch,
                device=device,
                patience=cfg.patience,
                project=str(self._project.paths.runs),
                name="train",
                exist_ok=True,
                verbose=False,
            )

            # copy best model to project/models/
            run_dir = self._project.paths.runs / "train"
            best_src = run_dir / "weights" / "best.pt"
            last_src = run_dir / "weights" / "last.pt"
            models_dir = self._project.paths.models
            models_dir.mkdir(parents=True, exist_ok=True)

            best_dst = models_dir / "best.pt"
            last_dst = models_dir / "last.pt"
            if best_src.exists():
                import shutil
                shutil.copy2(best_src, best_dst)
            if last_src.exists():
                import shutil
                shutil.copy2(last_src, last_dst)

            # save metadata
            meta = {
                "date": datetime.now().isoformat(),
                "dataset": str(yaml_path),
                "classes": self._project.class_names,
                "base_model": cfg.base_model,
                "epochs": cfg.epochs,
                "imgsz": cfg.imgsz,
                "batch": cfg.batch,
                "device": device,
                "best_model": str(best_dst) if best_dst.exists() else "",
                "status": "done",
            }
            with open(self._project.paths.training_metadata, "w", encoding="utf-8") as f:
                yaml.safe_dump(meta, f, allow_unicode=True)

            # update project active model
            if best_dst.exists():
                self._project.active_model = str(best_dst)
                self._project.save()

            elapsed = time.time() - start_time
            self._emit(
                status="done",
                message=str(best_dst) if best_dst.exists() else "Training complete.",
                best_model=str(best_dst) if best_dst.exists() else None,
                elapsed_s=elapsed,
                eta_s=0,
            )
            log.info(f"Training done. Best model: {best_dst}")

        except KeyboardInterrupt:
            self._emit(status="idle", message="Training stopped.")
        except Exception as e:
            log.error(f"Training error: {e}", exc_info=True)
            self._emit(status="error", message=str(e))
