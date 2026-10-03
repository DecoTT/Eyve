"""
¿La textura del tejido le cuesta a YOLO?

No es una pregunta de opinión: se mide.  Se genera un conjunto de
evaluación nuevo —que el modelo no vio— con el mismo número de defectos por
ligamento, y se reporta precisión y recuperación POR LIGAMENTO.  Si la
sarga le costara más que la tela lisa, se vería aquí.

No se usa el mAP de ultralytics a propósito: haría falta armar un dataset
por ligamento y correr val() cinco veces.  Lo que importa responder es más
simple — de los defectos que hay, ¿cuántos encuentra, y cuántas cosas marca
que no estaban?

    python tests/eval_weave_impact.py [--frames 60]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from eyve.demo.textile import TextilePattern, MOTIFS, WEAVES, YOLO_CLASSES


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1e-6)


def frame_con_defectos(rng: random.Random, weave: str):
    """Un frame de evaluación y sus cajas verdaderas."""
    p = TextilePattern(width=960, height=540, axis=rng.choice(("x", "y")),
                       motif=rng.choice(MOTIFS), weave=weave,
                       weave_amp=rng.uniform(0.04, 0.11),
                       speed=rng.uniform(60, 190),
                       tilt_deg=rng.uniform(-3.5, 3.5),
                       seed=rng.randrange(1 << 30))
    p.offset = rng.uniform(0, p.fabric_len)
    diag = (p.width ** 2 + p.height ** 2) ** 0.5
    for _ in range(rng.randint(1, 3)):
        sx = rng.randint(70, p.width - 70)
        sy = rng.randint(70, p.height - 70)
        fx, fy = p.screen_to_fabric(sx, sy)
        if rng.random() < 0.55:
            p.streak(fx, fy, length=int(rng.triangular(45, diag * 0.5, 140)),
                     thickness=rng.randint(3, 11))
        else:
            p.blob(fx, fy, size=int(rng.triangular(24, 170, 60)))
    return p.frame(), p.visible_labels()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Impacto del tejido en YOLO")
    ap.add_argument("--frames", type=int, default=60,
                    help="frames de evaluacion POR ligamento")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args(argv)

    pesos = REPO / "projects" / "Demo_Textil" / "models" / "best.pt"
    if not pesos.exists():
        pesos = (REPO / "projects" / "Demo_Textil" / "runs" / "train"
                 / "weights" / "best.pt")
    if not pesos.exists():
        print(f"No hay modelo en {pesos}.\n"
              f"  python -m eyve.demo.train --out \"projects/Demo_Textil\"")
        return 1

    from ultralytics import YOLO
    model = YOLO(str(pesos))
    names = list(model.names.values())
    print(f"modelo: {pesos}   clases: {names}")
    print(f"evaluacion: {args.frames} frames nuevos por ligamento, "
          f"conf {args.conf}\n")

    # Antes de concluir nada: comprobar que el tejido SI cambia la imagen.
    # Sin esto, un "no afecta" podria significar simplemente que el
    # parametro no hace nada, y seria una verificacion que no puede fallar.
    base = None
    print("cuanto cambia la imagen cada ligamento (vs tela lisa):")
    for weave in WEAVES:
        rng = random.Random(args.seed)
        f, _ = frame_con_defectos(rng, weave)
        if weave == "ninguno":
            continue
        rng2 = random.Random(args.seed)
        f0, _ = frame_con_defectos(rng2, "ninguno")
        dif = float(np.abs(f.astype(int) - f0.astype(int)).mean())
        print(f"   {weave:>12}  {dif:5.2f} niveles de diferencia media")
        if base is None or dif < base:
            base = dif
    if base is None or base < 0.5:
        print("\nEL TEJIDO NO ESTA CAMBIANDO LA IMAGEN: "
              "la comparacion de abajo no probaria nada.")
        return 1
    print()

    print(f"{'ligamento':>12} {'defectos':>9} {'encontr.':>9} "
          f"{'recall':>8} {'precision':>10} {'falsos':>7}")
    filas = []
    for weave in WEAVES:
        rng = random.Random(args.seed)      # mismo material y defectos
        verdaderos = aciertos = detecciones = 0
        for _ in range(args.frames):
            frame, gt = frame_con_defectos(rng, weave)
            gt = [g for g in gt if g[0] in YOLO_CLASSES]
            verdaderos += len(gt)
            res = model.predict(frame, conf=args.conf, imgsz=512, verbose=False)
            cajas = []
            for r in res:
                for b in r.boxes:
                    xy = b.xyxy[0].tolist()
                    cajas.append((names[int(b.cls[0])],
                                  tuple(int(v) for v in xy)))
            detecciones += len(cajas)
            usadas = set()
            for cls, x1, y1, x2, y2 in gt:
                mejor, mejor_i = 0.0, -1
                for i, (dcls, caja) in enumerate(cajas):
                    if i in usadas or dcls != cls:
                        continue
                    v = iou((x1, y1, x2, y2), caja)
                    if v > mejor:
                        mejor, mejor_i = v, i
                if mejor >= 0.3:
                    usadas.add(mejor_i)
                    aciertos += 1
        recall = aciertos / max(1, verdaderos)
        prec = aciertos / max(1, detecciones)
        falsos = detecciones - aciertos
        filas.append((weave, recall, prec))
        print(f"{weave:>12} {verdaderos:>9} {aciertos:>9} "
              f"{recall:>7.1%} {prec:>9.1%} {falsos:>7}")

    liso = dict((w, r) for w, r, _ in filas).get("ninguno")
    print()
    if liso is not None:
        peor = min((r for w, r, _ in filas if w != "ninguno"), default=liso)
        caida = liso - peor
        print(f"recall en tela lisa: {liso:.1%}")
        print(f"peor recall con tejido: {peor:.1%}   "
              f"diferencia: {caida:+.1%}")
        if caida > 0.05:
            print("-> el tejido SI le cuesta: conviene mas material con "
                  "textura en el entrenamiento")
        else:
            print("-> el tejido NO le cuesta de forma apreciable: entrenar "
                  "con variedad lo volvio invariante")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
