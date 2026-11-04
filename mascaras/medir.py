# -*- coding: utf-8 -*-
"""
Antes y despues de la mascara, sobre los MISMOS cuadros.

Es la medicion del video 6 (medir_linea_base.py: 4 estampados x 5 tejidos x 3
fallos, siguiendo cada fallo mientras cruza, misma calibracion y seguimiento
que la demo), con tres seguimientos corriendo a la vez sobre cada cuadro:

    antes    Patron solo                 lo que hay hoy
    mascara  la mascara sola             lo que aporta
    despues  mascara + Patron            lo que veria el operador

Separacion de datos:
  - la mascara se aprende SIEMPRE en el rollo 1, que no se usa para probar
    (rollos.py: otro hilo, misma impresion).
  - conjunto "dev" (rollo 3): el unico con el que se afino.
  - conjunto "video" (rollo 0, semilla 2027): los mismos 240 fallos del video.
    "antes" debe dar exactamente lo que dio linea_base.json.
  - conjunto "nuevo" (rollo 4, semilla 4242): fallos nunca vistos, en un
    rollo que no toca nadie mas.  El rollo 2 es el de tests/test_mascara.py,
    y ahi aparecio una falsa alarma que obligo a subir el umbral: ya no cuenta
    como dato no visto.

    python medir.py video resultados/video.json --workers 6
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from eyve.demo.textile import MOTIFS, WEAVES, PRINT_FAULTS
from eyve.inference.tracker import InstanceTracker, TrackerConfig
from eyve.modules import PatternModule
from eyve.modules.mascara import MascaraAprendida, fusionar
from rollos import rollo, cuadros_buenos, W, H

DT = 1 / 16.0
CONJUNTOS = {"video": (2027, 0, 4), "nuevo": (4242, 4, 8), "dev": (3001, 3, 2)}
ROLLO_APRENDER = 1


def toca(a, b, halo=0):
    return not (a[2] + halo < b[0] or b[2] + halo < a[0] or
                a[3] + halo < b[1] or b[3] + halo < a[1])


def plan(semilla: int, K: int) -> dict:
    """Los fallos de cada material, con las mismas llamadas al rng que el
    guion del video: con semilla 2027 y K=4 son sus mismos 240."""
    rng = random.Random(semilla)
    out = {}
    for motif in MOTIFS:
        for weave in WEAVES:
            d = dict(motif=motif, weave=weave,
                     limpia=rng.randrange(1 << 30), fallos=[])
            for fault in PRINT_FAULTS:
                for k in range(K):
                    f = dict(fault=fault, k=k, seed=rng.randrange(1 << 30))
                    f["sx"] = rng.randint(150, W - 150)
                    f["sy"] = rng.randint(110, H - 110)
                    if fault == "fantasma":
                        f["size"] = rng.randint(130, 200)
                    elif fault == "offset":
                        f["size"] = rng.randint(140, 220)
                    else:
                        f["size"] = rng.randint(120, 200)
                        f["sev"] = rng.uniform(0.5, 1.0)
                    d["fallos"].append(f)
            out[f"{motif}|{weave}"] = d
    return out


def perturbar(f, spec):
    """
    Lo que en una linea real cambia despues de aprender.  Solo se aplica a
    los cuadros de prueba: la mascara se aprendio con la tela sin tocar.

      rampa:0.04      la luz cae 4 % de un lado al otro del cuadro
      ganancia:0.9    la lampara baja 10 %
      escala:1.005    la tela llega 0.5 % estirada
      desliza:12      la tela se corre 12 px hacia un lado, despacio
    """
    if not spec:
        return f
    que, val = spec.split(":")
    val = float(val)
    h, w = f.shape[:2]
    if que == "rampa":
        r = np.linspace(-val / 2, val / 2, w, dtype=np.float32)[None, :, None]
        return np.clip(f * (1.0 + r), 0, 255).astype(np.uint8)
    if que == "ganancia":
        return np.clip(f.astype(np.float32) * val, 0, 255).astype(np.uint8)
    if que == "escala":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, val)
        return cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    if que == "desliza":
        M = np.float32([[1, 0, 0], [0, 1, val]])
        return cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    raise ValueError(spec)


def tracker():
    return InstanceTracker(config=TrackerConfig(
        iou_match=0.20, max_lost_frames=18, min_confirm_frames=2,
        process_conf_min=0.30))


def nuevo(motif, weave, numero, seed, **kw):
    p = rollo(motif, weave, numero, seed=seed, **kw)
    m = PatternModule()
    m.enabled = True
    m.sensitivity = 70.0
    for _ in range(10):
        m.calibrate(p.frame())
        p.advance(0.07)
    return p, m


def aprender(motif, weave, sens, kw=None, sin_mascara=False):
    masc = MascaraAprendida()
    masc.sensibilidad = sens
    if sin_mascara:
        return masc, masc.informe
    inf = masc.aprender(cuadros_buenos(
        rollo(motif, weave, ROLLO_APRENDER, seed=5, **(kw or {}))))
    if masc.estado == "por_aprobar":
        masc.aprobar("medir.py")
    return masc, inf


def correr(tarea):
    (d, numero, cuadros_limpia, sens, kw_prueba, sin_mascara, pert,
     fuerza) = tarea
    cv2.setNumThreads(1)
    motif, weave = d["motif"], d["weave"]
    masc, inf = aprender(motif, weave, sens, sin_mascara=sin_mascara)
    r = dict(motif=motif, weave=weave, avisos=inf.avisos, estado=masc.estado,
             enganche=inf.enganche, residuo_pct=inf.residuo_pct,
             a1=inf.a1, a2=inf.a2, fallos=[])

    # ── tela limpia ───────────────────────────────────────────────────────
    p, m = nuevo(motif, weave, numero, d["limpia"], **kw_prueba)
    ta, tm, tf = tracker(), tracker(), tracker()
    fa = dict(antes=0, mascara=0, despues=0)
    fa64 = dict(antes=0)
    sin_enganche = 0
    ms = ms_pat = 0.0
    for i in range(cuadros_limpia):
        p.advance(DT)
        f = perturbar(p.frame(), pert)
        t0 = time.perf_counter()
        m.analyze(f)
        t1 = time.perf_counter()
        masc.analizar(f)
        ms += time.perf_counter() - t1
        ms_pat += t1 - t0
        sin_enganche += not masc.enganchada
        da = m.as_detections()
        a = any(t.confirmed for t in ta.update(da, f))
        b = any(t.confirmed for t in tm.update(masc.as_detections(), f))
        c = any(t.confirmed for t in tf.update(fusionar(masc, m), f))
        fa["antes"] += a
        fa["mascara"] += b
        fa["despues"] += c
        if i < 64:
            fa64["antes"] += a
    r["limpia"] = dict(cuadros=cuadros_limpia, falsas=fa, falsas64=fa64,
                       sin_enganche=sin_enganche,
                       ms=1000 * ms / max(1, cuadros_limpia),
                       ms_patron=1000 * ms_pat / max(1, cuadros_limpia))

    # ── cada fallo, mientras cruza ────────────────────────────────────────
    for fl in d["fallos"]:
        p, m = nuevo(motif, weave, numero, fl["seed"], **kw_prueba)
        ta, tm, tf = tracker(), tracker(), tracker()
        fx, fy = p.screen_to_fabric(fl["sx"], fl["sy"])
        if fl["fault"] == "fantasma":
            p.ghost(fx, fy, size=fl["size"], strength=fuerza)
        elif fl["fault"] == "offset":
            p.misregister(fx, fy, size=fl["size"])
        else:
            p.ink_starved(fx, fy, size=fl["size"], severity=fl["sev"])
        v = 0
        conf = dict(antes=0, mascara=0, despues=0)
        lejos = dict(antes=0, mascara=0, despues=0)
        tipos = {}
        for _ in range(60):
            f = perturbar(p.frame(), pert)
            m.analyze(f)
            masc.analizar(f)
            for rg, tp in zip(masc.regiones, masc.tipos):
                tipos[tp] = tipos.get(tp, 0) + 1
            salidas = dict(
                antes=[t for t in ta.update(m.as_detections(), f) if t.confirmed],
                mascara=[t for t in tm.update(masc.as_detections(), f) if t.confirmed],
                despues=[t for t in tf.update(fusionar(masc, m), f) if t.confirmed])
            gt = [g[1:] for g in p.visible_labels()]
            todo = [g[1:] for g in p.visible_labels(min_side=1, min_visible=0.0)]
            if gt:
                v += 1
            for k, tks in salidas.items():
                if gt and any(toca(t.bbox, g) for t in tks for g in gt):
                    conf[k] += 1
                # Marca lejos del fallo: falsa alarma aunque la tela traiga
                # un fallo.  El halo perdona la caja holgada, nada mas.
                if any(not any(toca(t.bbox, g, 40) for g in todo) for t in tks):
                    lejos[k] += 1
            p.advance(DT)
            if v and not gt:
                break
        r["fallos"].append(dict(fault=fl["fault"], k=fl["k"], sx=fl["sx"],
                                sy=fl["sy"], size=fl["size"], visibles=v,
                                conf=conf, lejos=lejos, tipos=tipos))
    print(f"{motif} {weave} listo", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conjunto", choices=sorted(CONJUNTOS))
    ap.add_argument("salida")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limpia", type=int, default=560,
                    help="cuadros de tela limpia: 560 = una vuelta al rollo")
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--sens", type=float, default=50.0)
    ap.add_argument("--solo", default="", help="materiales motif|weave,...")
    ap.add_argument("--tilt", type=float, default=None,
                    help="inclinacion de la tela de prueba (robustez)")
    ap.add_argument("--perturba", default="", help="ver perturbar()")
    ap.add_argument("--fuerza", type=float, default=0.45,
                    help="intensidad del fantasma; 0.45 es la de la demo")
    ap.add_argument("--solo-fantasma", action="store_true")
    ap.add_argument("--sin-mascara", action="store_true",
                    help="solo 'antes': para reproducir la linea base")
    a = ap.parse_args()
    semilla, numero, K = CONJUNTOS[a.conjunto]
    K = a.K or K
    p = plan(semilla, K)
    kw = {} if a.tilt is None else dict(tilt_deg=a.tilt)
    if a.solo_fantasma:
        for d in p.values():
            d["fallos"] = [f for f in d["fallos"] if f["fault"] == "fantasma"]
    tareas = [(d, numero, a.limpia, a.sens, kw, a.sin_mascara, a.perturba,
               a.fuerza)
              for k, d in p.items()
              if not a.solo or k in a.solo.split(",")]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(correr, tareas, chunksize=1)
    json.dump(dict(conjunto=a.conjunto, K=K, sens=a.sens, tilt=a.tilt,
                   perturba=a.perturba, fuerza=a.fuerza,
                   segundos=time.time() - t0, materiales=res),
              open(a.salida, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print(f"listo en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
