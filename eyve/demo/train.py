"""
Genera el dataset de la demo y entrena el modelo, en un comando.

    python -m eyve.demo.train --out "projects/Demo_Textil" --frames 900 --epochs 60

Usa el TrainManager de Eyve, no una ruta aparte: el modelo que termina en
projects/Demo_Textil/models/best.pt es el mismo que produciría un usuario
entrenando desde la pantalla de Entrenamiento. Eso importa para la expo —
lo que se enseña es el producto, no una maqueta.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from eyve.core.logger import log
from eyve.demo.dataset import build_dataset
from eyve.training.train_manager import TrainConfig, TrainManager


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Dataset + entrenamiento del modelo de la demo")
    ap.add_argument("--out", default="projects/Demo_Textil")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip-dataset", action="store_true",
                    help="reentrenar con el dataset que ya existe")
    args = ap.parse_args(argv)

    out = Path(args.out)

    if args.skip_dataset:
        from eyve.core.project_manager import load_project
        # resolve(): con --out relativo, todas las rutas del proyecto salen
        # relativas al directorio desde donde se lanzo el comando
        proj = load_project(out.resolve())
        print(f"Reusando dataset: {proj.raw_image_count()} imagenes")
    else:
        print(f"Generando {args.frames} frames en {out} ...")

        def show(done, total):
            print(f"\r  {done}/{total}", end="", flush=True)

        proj = build_dataset(out, frames=args.frames, seed=args.seed,
                             progress=show)
        print(f"\n  listo: {proj.raw_image_count()} imagenes, "
              f"clases {proj.class_names}")

    cfg = TrainConfig(base_model=args.base, epochs=args.epochs,
                      imgsz=args.imgsz, batch=args.batch, device=args.device)
    print(f"Entrenando ({args.epochs} epocas, device={cfg.device_arg()}) ...")

    mgr = TrainManager(proj)
    last = {"epoch": -1}

    def on_progress(p):
        if p.status == "error":
            print(f"\n  ERROR: {p.message}")
        elif p.epoch != last["epoch"]:
            last["epoch"] = p.epoch
            print(f"\r  epoca {p.epoch}/{p.total_epochs}  "
                  f"loss={p.loss:.4f}  mAP50={p.map50:.3f}  "
                  f"ETA {int(p.eta_s)}s   ", end="", flush=True)

    mgr.add_callback(on_progress)
    mgr.start(cfg)

    while mgr._progress.status in ("preparing", "training", "idle"):
        time.sleep(2.0)
        if mgr._thread is not None and not mgr._thread.is_alive():
            break

    print()
    p = mgr._progress
    if p.status == "error":
        print(f"Fallo el entrenamiento: {p.message}")
        return 1

    best = proj.paths.best_model
    print(f"Modelo: {best}  (existe={best.exists()})")
    print(f"mAP50 final: {p.map50:.3f}")
    log.info(f"Demo model trained: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
