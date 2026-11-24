"""
Máscara y Patrón sobre la tela realista (`eyve.demo.tela_real`).

    python mascaras/medir_tela_real.py salida.json [--workers 6] [--sens 50]
           [--realismo 1.0] [--limpia 240] [--por-tipo 2]

Por material (4 motivos x 3 tejidos, con las tres paletas en rotación):
  - la máscara aprende de 24 cuadros buenos del ROLLO 1;
  - se prueba en el ROLLO 2: mismo diseño, otro hilo, otro arco de trama y
    otra densidad de tinta, como un rollo nuevo en planta;
  - tela limpia: `--limpia` cuadros, cuenta los que tienen algo confirmado;
  - cada falla real (9 tipos, `--por-tipo` de cada una) entra por la derecha y
    cruza; se cuenta si la falla se marcó (alguna vez y en qué % de los cuadros
    en que se ve) y las marcas lejos de toda falla.
"""
from __future__ import annotations

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

import cv2  # noqa: E402

from eyve.demo.tela_real import FALLAS_REALES, PALETAS, TelaRealista  # noqa: E402
from eyve.modules import PatternModule  # noqa: E402
from eyve.modules.mascara import MascaraAprendida, MascaraColor, fusionar  # noqa: E402
from medir import toca, tracker  # noqa: E402

DT = 1 / 16.0
MOTIVOS = ("diamantes", "flores", "rayas", "puntos")
TEJIDOS = ("sarga", "tafetan", "canasta")
#: tamaños para medir: las fallas largas se acortan para que crucen a tiempo
ARGS = {"raya_racla": dict(largo=600), "franja_sin_tinta": dict(largo=700),
        "doble_impresion": dict(largo=500)}


def tela(motivo, tejido, paleta, rollo, seed, realismo):
    return TelaRealista(motif=motivo, weave=tejido, paleta=paleta, rollo=rollo,
                        seed=seed, realismo=realismo)


def patron(p):
    m = PatternModule()
    m.enabled = True
    m.sensitivity = 70.0
    for _ in range(10):
        m.calibrate(p.frame())
        p.advance(0.07)
    return m


def correr(t):
    motivo, tejido, paleta, sens, realismo, limpia, por_tipo, semilla, color, calibrar = t
    cv2.setNumThreads(1)
    rng = random.Random(semilla)
    r = dict(motivo=motivo, tejido=tejido, paleta=paleta, fallas=[])

    # aprender, rollo 1
    p = tela(motivo, tejido, paleta, 1, 5, realismo)
    buenos = []
    for _ in range(24):
        buenos.append(p.frame())
        p.advance(0.33)
    masc = MascaraColor() if color else MascaraAprendida()
    masc.sensibilidad = sens
    inf = masc.aprender(buenos)
    r.update(estado=masc.estado, enganche=inf.enganche,
             residuo_pct=inf.residuo_pct, avisos=inf.avisos)
    if masc.estado == "por_aprobar":
        if calibrar:
            # unos segundos más de tela buena del MISMO rollo, después de los de
            # aprender: lo que haría el operador antes de aprobar
            cal = []
            for _ in range(80):
                p.advance(DT * 2)
                cal.append(p.frame())
            r["umbral"] = masc.calibrar(cal)
        masc.aprobar("medir_tela_real")

    # tela limpia, rollo 2
    p = tela(motivo, tejido, paleta, 2, rng.randrange(1 << 30), realismo)
    m = patron(p)
    ta, tm = tracker(), tracker()
    fa = dict(patron=0, mascara=0)
    suelta = 0
    ms_m = ms_p = 0.0
    for _ in range(limpia):
        p.advance(DT)
        f = p.frame()
        t0 = time.perf_counter()
        m.analyze(f)
        t1 = time.perf_counter()
        masc.analizar(f)
        ms_m += time.perf_counter() - t1
        ms_p += t1 - t0
        suelta += not masc.enganchada
        fa["patron"] += any(x.confirmed for x in ta.update(m.as_detections(), f))
        fa["mascara"] += any(x.confirmed for x in tm.update(masc.as_detections(), f))
    r["limpia"] = dict(cuadros=limpia, falsas=fa, sin_enganche=suelta,
                       ms_mascara=1000 * ms_m / limpia, ms_patron=1000 * ms_p / limpia)

    # cada falla real, entrando por la derecha
    for tipo in FALLAS_REALES:
        for k in range(por_tipo):
            p = tela(motivo, tejido, paleta, 2, rng.randrange(1 << 30), realismo)
            m = patron(p)
            fx = int(p.offset + p.width + 380)
            fy = rng.randint(110, p.height - 110)
            d = p.falla(tipo, fx, fy, **ARGS.get(tipo, {}))
            ta, tm, tf = tracker(), tracker(), tracker()
            vis = 0
            conf = dict(patron=0, mascara=0, juntos=0)
            lejos = dict(patron=0, mascara=0, juntos=0)
            for _ in range(200):
                p.advance(DT)
                f = p.frame()
                m.analyze(f)
                masc.analizar(f)
                sal = dict(
                    patron=[x for x in ta.update(m.as_detections(), f) if x.confirmed],
                    mascara=[x for x in tm.update(masc.as_detections(), f) if x.confirmed],
                    juntos=[x for x in tf.update(fusionar(masc, m), f) if x.confirmed])
                gt = [g[1:] for g in p.visible_labels(min_side=6, min_visible=0.0)]
                if gt:
                    vis += 1
                for kk, tks in sal.items():
                    if gt and any(toca(x.bbox, g) for x in tks for g in gt):
                        conf[kk] += 1
                    if any(not any(toca(x.bbox, g, 40) for g in gt) for x in tks):
                        lejos[kk] += 1
                if vis and not gt:
                    break
            r["fallas"].append(dict(tipo=tipo, k=k, fy=fy, visibles=vis,
                                    conf=conf, lejos=lejos,
                                    caja=[d.x1, d.y1, d.x2, d.y2]))
    print(f"{motivo} {tejido} {paleta} listo", flush=True)
    return r


def resumen(res):
    tipos = list(FALLAS_REALES)
    print("\n| Falla | Patrón (instancias) | Máscara | Juntos | cuadros juntos |")
    print("|---|---|---|---|---|")
    for t in tipos:
        fs = [f for r in res for f in r["fallas"] if f["tipo"] == t and f["visibles"]]
        n = len(fs)
        pi = sum(f["conf"]["patron"] > 0 for f in fs)
        mi = sum(f["conf"]["mascara"] > 0 for f in fs)
        ji = sum(f["conf"]["juntos"] > 0 for f in fs)
        vis = sum(f["visibles"] for f in fs) or 1
        jc = sum(f["conf"]["juntos"] for f in fs)
        print(f"| {t} | {pi}/{n} | {mi}/{n} | {ji}/{n} | {100 * jc / vis:.0f} % |")
    lim = sum(r["limpia"]["cuadros"] for r in res)
    fp = sum(r["limpia"]["falsas"]["patron"] for r in res)
    fm = sum(r["limpia"]["falsas"]["mascara"] for r in res)
    se = sum(r["limpia"]["sin_enganche"] for r in res)
    lj = {k: sum(f["lejos"][k] for r in res for f in r["fallas"]) for k in ("patron", "mascara")}
    print(f"\nFalsas en limpia: Patrón {fp}/{lim}, máscara {fm}/{lim}; sin enganche {se}")
    print(f"Marcas lejos de la falla: Patrón {lj['patron']}, máscara {lj['mascara']}")
    for r in res:
        print(f"  {r['motivo']:9} {r['tejido']:8} {r['paleta']:5} {r['estado']:12} "
              f"enganche {r['enganche']:.3f}  falsas m {r['limpia']['falsas']['mascara']:3} "
              f"p {r['limpia']['falsas']['patron']:3}  suelta {r['limpia']['sin_enganche']:3}  "
              f"ms m {r['limpia']['ms_mascara']:.0f} p {r['limpia']['ms_patron']:.0f}  umbral {r.get('umbral', '-')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("salida")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sens", type=float, default=50.0)
    ap.add_argument("--realismo", type=float, default=1.0)
    ap.add_argument("--limpia", type=int, default=240)
    ap.add_argument("--por-tipo", type=int, default=2)
    ap.add_argument("--semilla", type=int, default=2311)
    ap.add_argument("--solo", default="", help="motivo|tejido,...")
    ap.add_argument("--color", action="store_true", help="MascaraColor (gris + color)")
    ap.add_argument("--calibrar", action="store_true", help="umbral calibrado con tela buena")
    a = ap.parse_args()
    paletas = list(PALETAS)
    tareas = []
    i = 0
    for mo in MOTIVOS:
        for te in TEJIDOS:
            if a.solo and f"{mo}|{te}" not in a.solo.split(","):
                continue
            tareas.append((mo, te, paletas[i % len(paletas)], a.sens, a.realismo,
                           a.limpia, a.por_tipo, a.semilla + i, a.color, a.calibrar))
            i += 1
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(correr, tareas)
    Path(a.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    resumen(res)
    print(f"listo en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
