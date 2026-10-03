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
    base_model: str = "yolov8n.pt"   # stock name (yolov8n/s/m/l.pt) OR absolute path to a .pt
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    device: str = "auto"             # "auto" | "cpu" | "0" (cuda)
    conf_default: float = 0.50
    patience: int = 20               # early stopping
    # Fine-tuning: only labels written/edited after this moment go into the
    # dataset ("solo etiquetas nuevas desde el último entrenamiento").
    only_new_since: Optional[datetime] = None

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


def last_training_date(project: Project) -> Optional[datetime]:
    """When the project was last trained (from training_metadata.yaml), or None."""
    try:
        meta = yaml.safe_load(project.paths.training_metadata.read_text(encoding="utf-8")) or {}
        return datetime.fromisoformat(str(meta["date"]))
    except Exception:
        return None


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

            yaml_path = build_dataset(self._project, since=cfg.only_new_since)

            self._emit(status="training", message="Starting YOLO training…",
                       total_epochs=cfg.epochs)

            from ultralytics import YOLO
            from eyve.core.model_manager import local_path_or_name

            # A path (project best.pt, or any .pt the user picked) is used as-is:
            # that is how "keep improving the same model" works.  Stock names go
            # through the offline-aware resolver.
            base = Path(cfg.base_model)
            if base.is_absolute() and base.exists():
                model_path = str(base)
            else:
                model_path = local_path_or_name(cfg.base_model)
            log.debug(f"Loading base model: {model_path}")
            model = YOLO(model_path)

            # Ultralytics 8.x callbacks are plain functions — the old
            # BaseCallback class was removed from the public API (importing
            # it raised ImportError and killed every training run: BUG-02).
            mgr = self
            epoch_times: list[float] = []
            epoch_start = [time.time()]

            def _on_epoch_end(trainer):
                elapsed = time.time() - start_time
                epoch_times.append(time.time() - epoch_start[0])
                epoch_start[0] = time.time()
                ep = trainer.epoch + 1
                total = trainer.epochs
                avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 1
                remaining = avg_ep * (total - ep)
                metrics = trainer.metrics or {}
                raw_loss = getattr(trainer, "loss", 0)
                try:
                    raw_loss = float(raw_loss.detach())   # tensor with grad
                except AttributeError:
                    raw_loss = float(raw_loss or 0)
                mgr._emit(
                    epoch=ep,
                    total_epochs=total,
                    loss=raw_loss,
                    map50=float(metrics.get("metrics/mAP50(B)", 0)),
                    elapsed_s=elapsed,
                    eta_s=remaining,
                    status="training",
                    message=f"Epoch {ep}/{total}",
                )
                if mgr._stop_flag.is_set():
                    raise KeyboardInterrupt("Training stopped by user")

            model.add_callback("on_train_epoch_end", _on_epoch_end)

            device = cfg.device_arg()
            results = model.train(
                data=str(yaml_path),
                epochs=cfg.epochs,
                imgsz=cfg.imgsz,
                batch=cfg.batch,
                device=device,
                patience=cfg.patience,
                # Ruta ABSOLUTA a proposito: ultralytics resuelve un
                # project= relativo contra su propio directorio de runs, asi
                # que un proyecto abierto con ruta relativa terminaba
                # entrenando en runs/detect/<ruta>/runs/train y la copia
                # posterior a models/ no encontraba nada.
                project=str(self._project.paths.runs.resolve()),
                name="train",
                exist_ok=True,
                verbose=False,
            )

            # copy best model to project/models/
            run_dir = self._project.paths.runs.resolve() / "train"
            best_src = run_dir / "weights" / "best.pt"
            last_src = run_dir / "weights" / "last.pt"
            if not best_src.exists():
                # Si los pesos no estan donde se esperaban, el entrenamiento
                # NO termino bien aunque ultralytics haya devuelto resultados.
                # Decirlo: la version anterior se reportaba "listo" y dejaba
                # el modelo viejo en su lugar.
                raise FileNotFoundError(
                    f"el entrenamiento no dejo pesos en {best_src}")
            models_dir = self._project.paths.models
            models_dir.mkdir(parents=True, exist_ok=True)

            best_dst = models_dir / "best.pt"
            last_dst = models_dir / "last.pt"
            if best_src.exists():
                import shutil
                # Preserve the previous model before overwriting — without
                # this there is no way back if the new training came out
                # worse (PRD §10.2).  best.pt stays as "the active one".
                if best_dst.exists():
                    stamp = datetime.now().strftime("%Y%m%d_%H%M")
                    backup = models_dir / f"best_{stamp}.pt"
                    if not backup.exists():
                        shutil.copy2(best_dst, backup)
                        log.info(f"Previous model preserved as {backup.name}")
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
                "only_new_since": cfg.only_new_since.isoformat() if cfg.only_new_since else None,
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
            # str(e) alone can be cryptic (an ImportError shows just the
            # symbol name).  Include the exception type so the UI message
            # gives the user/support something actionable.
            self._emit(status="error", message=f"{type(e).__name__}: {e}")
