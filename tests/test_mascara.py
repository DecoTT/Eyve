# -*- coding: utf-8 -*-
"""
La mascara aprendida: que vea lo que Patron deja pasar, sin marcar tela buena.

Tres cosas deciden si sirve, y cada una tiene aqui su caso malo:

  - el fantasma que a Patron se le escapa se marca, y la caja cae sobre el
    fallo.  Caja corrida 300 px: el mismo chequeo tiene que fallar.
  - la tela limpia de un rollo que no se vio no marca nada.
  - sin aprobacion no marca nada, y una mascara alterada pierde la aprobacion.

Se aprende en el rollo 1 y se prueba en otros: TextilePattern siembra su hilo
con semillas fijas, asi que dos telas iguales son la misma tela pixel a pixel
aunque cambie `seed`.  Ver mascaras/rollos.py.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mascaras"))

import cv2
import numpy as np

from eyve.demo.textile import TextilePattern
from eyve.inference.tracker import InstanceTracker, TrackerConfig
from eyve.modules.mascara import (APROBADA, KIND_EXTRA, KIND_FALTA,
                                  POR_APROBAR, RECHAZADA, SIN_APRENDER,
                                  MascaraAprendida, fusionar, _reducir)
from eyve.modules.pattern_module import PatternModule
from medir import DT, nuevo, plan, toca, tracker
from rollos import cuadros_buenos, rollo

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FALLA'}  {name}: got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


def check_true(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


_cache = {}


def mascara(motif, weave, aprobar=True):
    k = (motif, weave)
    if k not in _cache:
        m = MascaraAprendida()
        m.aprender(cuadros_buenos(rollo(motif, weave, 1, seed=5)))
        _cache[k] = m
    m = _cache[k]
    if aprobar and m.estado != APROBADA:
        m.aprobar("prueba")
    return m


def cruce(masc, motif, weave, numero, fl):
    """Sigue un fallo mientras cruza: (patron, mascara, cajas de la mascara)."""
    p, pat = nuevo(motif, weave, numero, fl["seed"])
    fx, fy = p.screen_to_fabric(fl["sx"], fl["sy"])
    if fl["fault"] == "fantasma":
        p.ghost(fx, fy, size=fl["size"])
    elif fl["fault"] == "offset":
        p.misregister(fx, fy, size=fl["size"])
    else:
        p.ink_starved(fx, fy, size=fl["size"], severity=fl["sev"])
    ta, tm = tracker(), tracker()
    vio_p = vio_m = 0
    pares = []
    v = 0
    for _ in range(60):
        f = p.frame()
        pat.analyze(f)
        masc.analizar(f)
        a = [t for t in ta.update(pat.as_detections(), f) if t.confirmed]
        b = [t for t in tm.update(masc.as_detections(), f) if t.confirmed]
        gt = [g[1:] for g in p.visible_labels()]
        if gt:
            v += 1
            vio_p += any(toca(t.bbox, g) for t in a for g in gt)
            vio_m += any(toca(t.bbox, g) for t in b for g in gt)
            pares += [(t.bbox, gt[0]) for t in b]
        p.advance(DT)
        if v and not gt:
            break
    return vio_p, vio_m, pares


print("\n[1] Los rollos de prueba no son el rollo aprendido")
a = rollo("flores", "canasta", 0, seed=1).frame()
b = rollo("flores", "canasta", 1, seed=1).frame()
c = rollo("flores", "canasta", 0, seed=2).frame()
check("misma tela con otra semilla: identica (por eso hacen falta rollos)",
      int(np.abs(a.astype(int) - c.astype(int)).max()), 0)
check_true("otro rollo: otro hilo", float(np.abs(a.astype(int) - b.astype(int)).mean()) > 1.0)
check_true("otro rollo: mismo estampado",
           float(np.corrcoef(cv2.GaussianBlur(a, (0, 0), 3).ravel(),
                             cv2.GaussianBlur(b, (0, 0), 3).ravel())[0, 1]) > 0.97)
d = TextilePattern(width=960, height=540, axis="x", motif="flores",
                   weave="canasta", speed=110.0, seed=1).frame()
check("el rollo 0 es la tela del video", int(np.abs(a.astype(int) - d.astype(int)).max()), 0)

print("\n[2] El plan reproduce los fallos de la medicion del video")
pl = plan(2027, 4)
check("20 materiales", len(pl), 20)
check("12 fallos por material", {len(v["fallos"]) for v in pl.values()}, {12})
primero = pl["diamantes|sarga"]["fallos"][0]
check("mismo orden de fallos", primero["fault"], "falta_impresion")

print("\n[3] Aprender, ver y aprobar")
m0 = MascaraAprendida()
check("recien hecha", m0.estado, SIN_APRENDER)
inf = m0.aprender([rollo("flores", "sarga").frame()] * 3)
check("con 3 cuadros no aprende", m0.estado, SIN_APRENDER)
check_true("y lo dice", bool(inf.avisos), str(inf.avisos))
liso = [np.full((540, 960, 3), 200, np.uint8)] * 12
m1 = MascaraAprendida()
m1.aprender(liso)
check("material liso: no aprende", m1.estado, SIN_APRENDER)
_r = np.random.default_rng(5)
unica = [cv2.GaussianBlur(_r.integers(0, 255, (540, 960, 3), dtype=np.uint8),
                          (0, 0), 9) for _ in range(12)]
m1 = MascaraAprendida()
m1.aprender(unica)
check("material que no se repite: no aprende", m1.estado, SIN_APRENDER)

m = mascara("flores", "canasta", aprobar=False)
check("aprender deja la mascara por aprobar", m.estado, POR_APROBAR)
check_true("la celda es la retícula de la tela (48 px de trabajo)",
           abs(min(np.hypot(*m.informe.a1), np.hypot(*m.informe.a2)) - 48.0) < 0.5,
           f"a1={m.informe.a1} a2={m.informe.a2}")
vista = m.vista(rollo("flores", "canasta", 2, seed=9).frame())
check_true("hay algo que ver", vista.ndim == 3 and float(vista.std()) > 20)

malo = rollo("flores", "canasta", 2, seed=11)
malo.advance(2.0)
fx, fy = malo.screen_to_fabric(480, 270)
malo.ghost(fx, fy, size=190, strength=0.6)
fm = malo.frame()
check("sin aprobar no marca, aunque haya fallo", m.analizar(fm), [])
m.rechazar()
check("rechazada", m.estado, RECHAZADA)
check("rechazada no marca", m.analizar(fm), [])
m.aprobar("prueba")
check("aprobada", m.estado, APROBADA)
check_true("aprobada SI marca ese mismo cuadro", len(m.analizar(fm)) > 0)

with tempfile.TemporaryDirectory() as td:
    ruta = Path(td) / "m.npz"
    m.guardar(ruta)
    m2 = MascaraAprendida.cargar(ruta)
    check("guardada conserva la aprobacion", m2.estado, APROBADA)
    check("y marca igual", m2.analizar(fm), m.analizar(fm))
    # lo que importa: una mascara alterada no hereda la aprobacion
    d_ = dict(np.load(ruta, allow_pickle=False))
    d_["celda"] = d_["celda"] * 0.9
    np.savez_compressed(ruta, **d_)
    m3 = MascaraAprendida.cargar(ruta)
    check("archivo alterado: pierde la aprobacion", m3.estado, POR_APROBAR)
    check("y no marca", m3.analizar(fm), [])
copia = mascara("flores", "canasta")
guardada = copia._celda.copy()
copia._celda = copia._celda * 0.9
check("alterada en memoria: no esta lista", copia.lista, False)
check("y no marca", copia.analizar(fm), [])
copia._celda = guardada
check("restaurada, vuelve a estar lista", copia.lista, True)

print("\n[4] La reticula torcida se reduce")
r = _reducir(np.array([[48.0, 0.0], [118.0, 52.0]]))
check_true("misma area", abs(abs(np.linalg.det(r)) - 48 * 52) < 1e-6)
check_true("vectores cortos", max(np.hypot(*r[0]), np.hypot(*r[1])) < 60, str(r))

print("\n[5] Los tres fantasmas que se escaparon en el video")
for mat, k in (("flores|sarga_fina", None), ("flores|canasta", None),
               ("puntos|canasta", None)):
    motif, weave = mat.split("|")
    masc = mascara(motif, weave)
    fans = [f for f in pl[mat]["fallos"] if f["fault"] == "fantasma"]
    res = [(f, *cruce(masc, motif, weave, 0, f)) for f in fans]
    esc = [x for x in res if x[1] == 0]
    check(f"{mat}: Patron deja pasar exactamente uno", len(esc), 1)
    for f, vp, vm, pares in esc:
        check_true(f"{mat}: la mascara lo marca", vm > 0, f"cuadros={vm}")
        # Que la caja caiga sobre el fallo, y que el mismo chequeo falle con
        # la caja corrida.  Sin esto, «toca» pasaria con una caja cualquiera.
        n = sum(toca(c, g) for c, g in pares)
        lejos = sum(toca((c[0] + 300, c[1], c[2] + 300, c[3]), g)
                    or toca((c[0] - 300, c[1], c[2] - 300, c[3]), g)
                    for c, g in pares)
        check_true(f"{mat}: caja sobre el fallo", n == len(pares) and n > 0,
                   f"{n}/{len(pares)}")
        check_true(f"{mat}: corrida 300 px ya casi no toca", lejos < 0.25 * len(pares),
                   f"{lejos}/{len(pares)}")

print("\n[6] Tela limpia de un rollo que no se vio")
for motif, weave in (("flores", "canasta"), ("puntos", "canasta"),
                     ("rayas", "sarga"), ("diamantes", "tafetan")):
    masc = mascara(motif, weave)
    p = rollo(motif, weave, 2, seed=3)
    n = 0
    for _ in range(90):
        p.advance(0.6)
        n += len(masc.analizar(p.frame()))
    check(f"{motif}·{weave}: 0 regiones en 90 cuadros por todo el rollo", n, 0)

print("\n[7] Signo: de que lado esta el fallo")
masc = mascara("diamantes", "sarga")
p = rollo("diamantes", "sarga", 2, seed=4)
p.advance(1.0)
fx, fy = p.screen_to_fabric(480, 270)
p.ghost(fx, fy, size=190)
masc.analizar(p.frame())
check("fantasma = tinta de mas", set(masc.tipos), {KIND_EXTRA})
p = rollo("diamantes", "sarga", 2, seed=4)
p.advance(1.0)
fx, fy = p.screen_to_fabric(480, 270)
p.ink_starved(fx, fy, size=180, severity=0.9)
masc.analizar(p.frame())
check_true("falta de tinta = falta", KIND_FALTA in masc.tipos, str(masc.tipos))

print("\n[8] Mascara de otro material: no calza, y lo dice en vez de marcar")
masc = mascara("flores", "sarga")
p = rollo("diamantes", "sarga", 2, seed=8)
n = 0
sueltas = 0
for _ in range(20):
    p.advance(0.5)
    n += len(masc.analizar(p.frame()))
    sueltas += not masc.enganchada
check("ninguna region", n, 0)
check("20 de 20 cuadros sin enganche", sueltas, 20)

print("\n[9] Un fallo entre los cuadros buenos no se aprende")
p = rollo("flores", "sarga", 1, seed=21)
fx, fy = p.screen_to_fabric(700, 270)
p.ghost(fx, fy, size=190, strength=0.6)
sucia = MascaraAprendida()
# Los cuadros de aprendizaje traen el fantasma cruzando
sucia.aprender(cuadros_buenos(p))
sucia.aprobar("prueba")
q = rollo("flores", "sarga", 2, seed=12)
q.advance(2.0)
fx, fy = q.screen_to_fabric(480, 270)
q.ghost(fx, fy, size=190, strength=0.6)
check_true("sigue viendo un fantasma despues", len(sucia.analizar(q.frame())) > 0)
limpia = rollo("flores", "sarga", 2, seed=13)
n = 0
for _ in range(40):
    limpia.advance(0.6)
    n += len(sucia.analizar(limpia.frame()))
check("y la tela limpia sigue limpia", n, 0)

print("\n[10] Fusion: lo que la mascara marco, Patron no lo repite")
masc = mascara("diamantes", "sarga")
p, pat = nuevo("diamantes", "sarga", 2, 77)
fx, fy = p.screen_to_fabric(480, 270)
p.misregister(fx, fy, size=190)
f = p.frame()
pat.analyze(f)
masc.analizar(f)
check_true("los dos lo ven", len(pat.regions) > 0 and len(masc.regiones) > 0)
juntas = fusionar(masc, pat)
check("una sola deteccion por fallo", len(juntas), len(masc.regiones))

print("\n[10b] Lo que cambia despues de aprender no es un fallo")
masc = mascara("diamantes", "sarga")


def rampa(f, val=0.04):
    r = np.linspace(-val / 2, val / 2, f.shape[1], dtype=np.float32)[None, :, None]
    return np.clip(f * (1.0 + r), 0, 255).astype(np.uint8)


def recorrer(tela, cambio=lambda f: f, n=30):
    c = 0
    for _ in range(n):
        tela.advance(0.6)
        c += len(masc.analizar(cambio(tela.frame())))
    return c


check("rampa de luz de 4 %: 0 regiones",
      recorrer(rollo("diamantes", "sarga", 2, seed=31), rampa), 0)
check("tela 0.3 grados mas inclinada: 0 regiones",
      recorrer(rollo("diamantes", "sarga", 2, seed=32, tilt_deg=2.1)), 0)
girada = rollo("diamantes", "sarga", 2, seed=33, tilt_deg=2.1)
girada.advance(1.0)
fx, fy = girada.screen_to_fabric(480, 270)
girada.ghost(fx, fy, size=190)
check_true("y un fantasma en esa tela si se marca", len(masc.analizar(girada.frame())) > 0)

print("\n[11] La ventana de revision decide")
import tkinter as tk

from revisar_mascara import Revision

raiz = tk.Tk()
raiz.withdraw()
m = MascaraAprendida()
m.aprender(cuadros_buenos(rollo("puntos", "sarga", 1, seed=5)))
with tempfile.TemporaryDirectory() as td:
    ruta = Path(td) / "v.npz"
    w = Revision(raiz, m, rollo("puntos", "sarga", 2, seed=1).frame(), ruta, "Ana")
    w.top.withdraw()
    raiz.update()
    check("antes del boton sigue por aprobar", m.estado, POR_APROBAR)
    w.btn_aprobar.invoke()
    check("Aprobar aprueba", m.estado, APROBADA)
    check("y queda quien", MascaraAprendida.cargar(ruta).aprobada_por, "Ana")
    w = Revision(raiz, m, None, ruta)
    w.top.withdraw()
    w.btn_rechazar.invoke()
    check("Rechazar rechaza, en el archivo", MascaraAprendida.cargar(ruta).estado, RECHAZADA)
raiz.destroy()

print()
if fails:
    print(f"FALLARON {len(fails)}: {fails}")
    sys.exit(1)
print("Todo bien.")
