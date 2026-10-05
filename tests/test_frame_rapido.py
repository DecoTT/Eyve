# -*- coding: utf-8 -*-
"""
La tela optimizada tiene que verse igual que la de antes.

Acelerar `frame()` no sirve de nada si cambia lo que se ve: el modelo se
entreno con esta tela y el modulo Patron se calibra sobre ella.  Asi que
esto no mide velocidad, mide IGUALDAD: carga la version anterior de
`textile.py` desde git y compara pixel a pixel.

Se compara con los 4 ligamentos y los 4 estampados, con y sin defectos, y
con varias semillas de defectos — no con una sola configuracion, porque el
atajo del recorte depende de cuanta capa de defectos haya y una sola
medida no distingue.

    .venv\\Scripts\\python.exe tests\\test_frame_rapido.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from eyve.demo.textile import MOTIFS, WEAVES, TextilePattern

#: Caracterizado, no supuesto: la diferencia contra la version anterior es
#: SIEMPRE +0, +1 o +2 (hasta +3 donde la rotacion interpola), nunca
#: negativa, y vale 0.87 niveles de media sobre 255 — un 0.34 % de brillo,
#: uniforme.  Es el cambio de truncar (numpy .astype) a redondear al mas
#: cercano (cv2.multiply), que es aritmetica mas correcta, no otra imagen.
#:
#: Exigir que NUNCA sea negativa es mas severo que una tolerancia
#: simetrica: un cambio de verdad en la tela —otro orden de efectos, otro
#: modulador— movería pixeles en los dos sentidos, y eso se caza aqui.
TOLERANCIA = 3
TOLERANCIA_MEDIA = 1.0

#: Contra QUE se compara: el commit justo anterior a optimizar la tela,
#: no HEAD.  Con HEAD la prueba se vuelve trivial en cuanto se hace
#: commit —compararia el codigo consigo mismo y daria 0 siempre— y
#: dejaria de proteger nada.  Pasó: por eso esta fijo.
REF_ANTES = "d2f67c9"

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


def comparar(name, a, b):
    """
    La nueva tela contra la anterior.

    Tres condiciones a la vez: ningun pixel se mueve mas de TOLERANCIA, la
    media se queda por debajo de un nivel, y —la que de verdad distingue—
    ningun pixel se vuelve MAS OSCURO, porque el redondeo solo puede subir.
    """
    d = b.astype(np.int16) - a.astype(np.int16)
    mx, md, mn = int(np.abs(d).max()), float(np.abs(d).mean()), int(d.min())
    ok = mx <= TOLERANCIA and md <= TOLERANCIA_MEDIA and mn >= 0
    check(name, ok, f"max={mx} media={md:.3f} min={mn:+d}")
    return mx


def cargar_version_anterior(ref: str):
    """Importa el textile.py de *ref* como modulo aparte."""
    src = subprocess.run(
        ["git", "show", f"{ref}:eyve/demo/textile.py"], cwd=str(REPO),
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if src.returncode != 0 or not src.stdout.strip():
        return None
    tmp = Path(tempfile.gettempdir()) / "eyve_textile_anterior.py"
    tmp.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("textile_anterior", tmp)
    mod = importlib.util.module_from_spec(spec)
    # Registrarlo ANTES de ejecutarlo: el @dataclass del modulo busca su
    # propio modulo en sys.modules mientras se construye la clase.
    sys.modules["textile_anterior"] = mod
    spec.loader.exec_module(mod)
    return mod


def ensuciar(p, n, semilla):
    rng = np.random.default_rng(semilla)
    for _ in range(n):
        fx = int(rng.integers(50, p.fw - 50))
        fy = int(rng.integers(50, p.fh - 50))
        if rng.random() < 0.5:
            p.streak(fx, fy, length=int(rng.integers(90, 280)),
                     thickness=int(rng.integers(4, 9)))
        else:
            p.blob(fx, fy, size=int(rng.integers(50, 130)))


def construir(modulo, motif, weave, n_defectos, semilla, pasos):
    # seed= FIJO, y no por gusto: TextilePattern siembra su generador con
    # None por defecto, asi que dos telas sin semilla sacan defectos de
    # formas distintas y la comparacion mediria eso en vez de medir el
    # cambio del codigo. Pasaba: salian 200 niveles de diferencia.
    p = modulo.TextilePattern(width=960, height=540, axis="x", motif=motif,
                              weave=weave, speed=110.0, tilt_deg=1.8,
                              seed=1234 + semilla)
    ensuciar(p, n_defectos, semilla)
    for _ in range(pasos):
        p.advance(1 / 30)
    return p.frame()


def main() -> int:
    anterior = cargar_version_anterior(REF_ANTES)
    if anterior is None:
        print("  no se pudo leer el textile.py de HEAD; nada que comparar")
        return 1
    check("la version anterior se carga",
          hasattr(anterior, "TextilePattern"))

    print("\n[1] Los 4 ligamentos, con tela limpia")
    peor = 0
    for w in WEAVES:
        a = construir(anterior, "diamantes", w, 0, 1, 5)
        b = construir(sys.modules["eyve.demo.textile"], "diamantes", w, 0, 1, 5)
        peor = max(peor, comparar(f"ligamento {w}", a, b))

    print("\n[2] Los 4 estampados, con tela limpia")
    for m in MOTIFS:
        a = construir(anterior, m, "sarga", 0, 1, 5)
        b = construir(sys.modules["eyve.demo.textile"], m, "sarga", 0, 1, 5)
        peor = max(peor, comparar(f"estampado {m}", a, b))

    print("\n[3] Con defectos, 8 semillas — que es donde cambia el atajo")
    # El recorte del compuesto alfa depende de donde caigan los defectos:
    # con una sola semilla podria acertar por casualidad.
    for semilla in range(1, 9):
        a = construir(anterior, "diamantes", "sarga", 5, semilla, 7)
        b = construir(sys.modules["eyve.demo.textile"], "diamantes", "sarga",
                      5, semilla, 7)
        peor = max(peor, comparar(f"semilla {semilla}", a, b))

    print("\n[4] Tela saturada: el caso en que el atajo NO ahorra nada")
    a = construir(anterior, "flores", "canasta", 40, 99, 3)
    b = construir(sys.modules["eyve.demo.textile"], "flores", "canasta",
                  40, 99, 3)
    peor = max(peor, comparar("40 defectos", a, b))

    print("\n[5] La tela guardada NO se ensucia al generar frames")
    # Las capas se desplazan por REBANADAS, que son vistas de la tela
    # guardada.  Si no se copiara antes de componer los defectos encima,
    # cada frame los pintaria DENTRO de self._base y la tela se iria
    # ensuciando sola, un poco mas en cada vuelta, para siempre.  Es el
    # tipo de fallo que no se ve en un minuto y arruina una tarde de expo.
    nuevo = sys.modules["eyve.demo.textile"]
    p = nuevo.TextilePattern(width=960, height=540, axis="x",
                             motif="diamantes", weave="sarga", speed=110.0,
                             tilt_deg=1.8, seed=55)
    p.streak(400, 250, length=220, thickness=8)
    p.blob(900, 300, size=110)
    base_al_empezar = p._base.copy()
    for _ in range(120):
        p.frame()
        p.advance(1 / 30)
    dif = int(np.abs(p._base.astype(np.int16)
                     - base_al_empezar.astype(np.int16)).max())
    check("la tela guardada no cambio tras 120 frames", dif == 0,
          f"diferencia maxima {dif}")

    # Y el mismo desplazamiento tiene que dar el mismo frame: si la tela se
    # ensuciara poco a poco, esto cambiaria aunque lo de arriba no lo pille.
    p.offset = 0.0
    a1 = p.frame()
    for _ in range(30):
        p.advance(1 / 30)
    p.offset = 0.0
    a2 = p.frame()
    dif = int(np.abs(a1.astype(np.int16) - a2.astype(np.int16)).max())
    check("el mismo desplazamiento da el mismo frame", dif == 0,
          f"diferencia maxima {dif}")

    print("\n[6] Da la vuelta al final del rollo sin costuras")
    # Cuando la ventana cruza el final hay que pegar dos trozos: es el
    # camino que NO son vistas, y el que se olvida al probar porque el
    # desplazamiento casi siempre cae en medio.
    p2 = nuevo.TextilePattern(width=960, height=540, axis="x", motif="rayas",
                              weave="tafetan", speed=110.0, tilt_deg=1.8,
                              seed=3)
    v2 = anterior.TextilePattern(width=960, height=540, axis="x",
                                 motif="rayas", weave="tafetan", speed=110.0,
                                 tilt_deg=1.8, seed=3)
    for despl in (p2.fw - 1500, p2.fw - 500, p2.fw - 100, p2.fw - 1, 0, 5):
        p2.offset = float(despl)
        v2.offset = float(despl)
        comparar(f"desplazamiento {despl} de {p2.fw}", v2.frame(), p2.frame())

    print("\n[7] Y la comprobacion que impide que esto pase por casualidad")
    # Si la comparacion estuviera mal montada —por ejemplo comparando un
    # frame consigo mismo— todo saldria 0 y no probaria nada.  Dos telas
    # DISTINTAS tienen que salir muy distintas.
    a = construir(sys.modules["eyve.demo.textile"], "rayas", "sarga", 0, 1, 5)
    b = construir(sys.modules["eyve.demo.textile"], "puntos", "sarga", 0, 1, 5)
    d = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
    check("dos telas distintas SI difieren", d > 50, f"max dif = {d}")

    print(f"\nmaxima diferencia en todo: {peor} niveles sobre 255 "
          f"(tolerancia {TOLERANCIA})")

    print("\n" + "=" * 60)
    if fails:
        print(f"FALLARON {len(fails)}:")
        for f in fails:
            print("  -", f)
        return 1
    print("LA TELA OPTIMIZADA SE VE IGUAL QUE LA DE ANTES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
