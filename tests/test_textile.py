# -*- coding: utf-8 -*-
"""
Prueba de la tela sintetica.

Lo que realmente importa: que las etiquetas caigan ENCIMA del defecto. Si
screen_to_fabric y fabric_to_screen no son inversas exactas, el dataset
sale corrido y el modelo aprende a detectar el lugar equivocado — y eso no
se nota hasta la expo. Asi que se prueba de dos formas independientes:

  (a) ida y vuelta de coordenadas en muchos desplazamientos
  (b) comprobacion por PIXEL: el defecto pintado debe estar DENTRO de la
      caja que reporta visible_labels, y la tela fuera de la caja debe
      seguir limpia. Esta no puede pasar si la proyeccion esta mal.
"""
import sys
import tempfile
from pathlib import Path

# El repo es el padre de tests/: nada de rutas absolutas, para que las
# pruebas corran en cualquier maquina y desde cualquier carpeta.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Salidas de las pruebas (hojas de contacto, proyectos temporales).
SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

import numpy as np
import cv2
from eyve.demo.textile import (TextilePattern, DEFECT_CLASSES,
                               YOLO_CLASSES, PRINT_FAULTS)

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


print("\n[1] La tela se construye y el frame tiene la forma correcta")
for axis in ("x", "y"):
    p = TextilePattern(width=960, height=540, axis=axis, seed=1)
    f = p.frame()
    check(f"forma del frame (axis={axis})", f.shape, (540, 960, 3))
    check(f"dtype (axis={axis})", f.dtype, np.dtype("uint8"))
    check_true(f"largo multiplo del periodo (axis={axis})",
               p.fabric_len % p.period == 0, f"len={p.fabric_len} p={p.period}")

print("\n[2] Viaje infinito: sin costura al dar la vuelta")
p = TextilePattern(width=320, height=240, axis="x", tilt_deg=0.0, seed=2)
span = p.fw
p.offset = 0.0
f0 = p.frame()
p.offset = float(span)          # una vuelta completa
f1 = p.frame()
check_true("una vuelta completa vuelve al mismo frame",
           np.array_equal(f0, f1))
p.offset = float(span) - 1
fa = p.frame()
p.offset = 0.0
fb = p.frame()
# columnas contiguas a traves de la costura deben parecerse
# La costura real cae DENTRO del frame en offset=span-1: su columna 0 es la
# ultima de la tela y su columna 1 es la primera. Comparar fa[:,-1] con
# fb[:,0] no eran columnas contiguas — ese era el error del test anterior.
p.offset = float(span) - 1
seam = p.frame().astype(int)
d = float(np.mean(np.abs(seam[:, 0] - seam[:, 1])))
ref = float(np.mean(np.abs(seam[:, 40] - seam[:, 41])))
check_true("la costura no se distingue de cualquier otra columna",
           d < max(12.0, ref * 2.5), f"costura={d:.1f} tipico={ref:.1f}")

print("\n[3] Ida y vuelta pantalla -> tela -> pantalla")
# En el interior del encuadre la ida y vuelta debe ser exacta. En las
# esquinas con inclinacion alta el punto cae FUERA de la tela (como una
# camara que ve mas alla de la orilla del rollo): ahi se recorta, y lo que
# se exige es que NO se teletransporte al extremo opuesto.
for axis in ("x", "y"):
    for tilt in (0.0, 1.8, -4.0, 8.0):
        p = TextilePattern(width=960, height=540, axis=axis,
                           tilt_deg=tilt, seed=3)
        worst = 0
        for off in (0, 137, 999, 2048, 3839):
            p.offset = float(off % (p.fw if axis == "x" else p.fh))
            for sx, sy in [(480, 270), (200, 150), (760, 390),
                           (300, 420), (650, 120)]:
                fx, fy = p.screen_to_fabric(sx, sy)
                bx, by = p.fabric_to_screen(fx, fy)
                worst = max(worst, abs(bx - sx), abs(by - sy))
        check_true(f"ida y vuelta exacta en el interior "
                   f"(axis={axis}, tilt={tilt})",
                   worst <= 2, f"peor error={worst}px")

        # esquinas: el punto se recorta a la orilla, nunca salta al otro lado
        worst_corner = 0
        p.offset = 500.0
        for sx, sy in [(0, 0), (959, 0), (0, 539), (959, 539)]:
            fx, fy = p.screen_to_fabric(sx, sy)
            bx, by = p.fabric_to_screen(fx, fy)
            worst_corner = max(worst_corner, abs(bx - sx), abs(by - sy))
        check_true(f"las esquinas se recortan, no se teletransportan "
                   f"(axis={axis}, tilt={tilt})",
                   worst_corner < 90, f"peor error en esquina={worst_corner}px")

print("\n[4] Comprobacion por PIXEL: la caja cae sobre el defecto")
# Sin rotacion primero, para aislar: la caja debe contener la tinta.
for axis in ("x", "y"):
    for tilt in (0.0, 1.8):
        p = TextilePattern(width=640, height=480, axis=axis,
                           tilt_deg=tilt, motif="puntos", seed=4)
        p.offset = 523.0
        limpio = p.frame().astype(np.int16)
        # pintar un defecto en el centro de la ventana
        fx, fy = p.screen_to_fabric(320, 240)
        p.blob(fx, fy, size=70, cls="mancha")
        sucio = p.frame().astype(np.int16)

        diff = np.abs(sucio - limpio).sum(axis=2)
        ink = diff > 40                       # pixeles que cambiaron
        check_true(f"el defecto se ve en el frame (axis={axis}, tilt={tilt})",
                   ink.sum() > 200, f"px={int(ink.sum())}")

        labels = p.visible_labels()
        check(f"una etiqueta (axis={axis}, tilt={tilt})", len(labels), 1)
        if not labels:
            continue
        cls, x1, y1, x2, y2 = labels[0]
        check(f"clase correcta (axis={axis}, tilt={tilt})", cls, "mancha")

        inside = np.zeros_like(ink)
        inside[y1:y2, x1:x2] = True
        dentro = int((ink & inside).sum())
        fuera = int((ink & ~inside).sum())
        frac = dentro / max(1, dentro + fuera)
        check_true(f"la tinta esta DENTRO de la caja (axis={axis}, tilt={tilt})",
                   frac > 0.93, f"{frac:.1%} dentro, {fuera} px fuera")

        # y la caja no debe ser absurdamente mas grande que la tinta
        ys, xs = np.where(ink)
        tight = (xs.min(), ys.min(), xs.max(), ys.max())
        holgura = max(x1 - tight[0], y1 - tight[1],
                      tight[2] - x2, tight[3] - y2)
        area_caja = (x2 - x1) * (y2 - y1)
        area_tinta = (tight[2]-tight[0]+1) * (tight[3]-tight[1]+1)
        check_true(f"caja ajustada (axis={axis}, tilt={tilt})",
                   area_caja < area_tinta * 2.6,
                   f"caja={area_caja} tinta={area_tinta} holgura={holgura}")

def tinta_bbox(limpio, sucio, thr=40):
    """Donde quedo realmente la tinta en el frame renderizado."""
    d = np.abs(sucio.astype(int) - limpio.astype(int)).sum(axis=2)
    ys, xs = np.where(d > thr)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def desfase_maximo(invertir_signo=False):
    """
    Peor desfase entre el centro de la tinta y el centro de la etiqueta,
    barriendo posiciones lejos del centro del encuadre.

    invertir_signo: aplica a proposito el error de rotacion que tuvo esta
    clase, para comprobar que la prueba lo detecta.
    """
    peor = 0
    for axis in ("x", "y"):
        for tilt in (-3.0, 1.8, 4.0):
            for off in (0, 311, 1500):
                for sx, sy in [(480, 80), (480, 460), (90, 270), (870, 270),
                               (140, 110), (820, 430)]:
                    for ang in (0.0, 1.57, 0.8):
                        q = TextilePattern(width=960, height=540, axis=axis,
                                           tilt_deg=tilt, motif="puntos",
                                           seed=11)
                        if invertir_signo:
                            orig = q.fabric_to_screen

                            def girado(fx, fy, _q=q, _o=orig):
                                # reflejo del punto respecto del centro en
                                # el eje de la rotacion = doble giro
                                import math as _m
                                sxx, syy = _o(fx, fy)
                                cx, cy = _q.width / 2.0, _q.height / 2.0
                                a = _m.radians(2 * _q.tilt_deg)
                                dx, dy = sxx - cx, syy - cy
                                return (int(dx * _m.cos(a) - dy * _m.sin(a) + cx),
                                        int(dx * _m.sin(a) + dy * _m.cos(a) + cy))
                            q.fabric_to_screen = girado
                        q.offset = float(off)
                        limpio = q.frame()
                        fx, fy = q.screen_to_fabric(sx, sy)
                        q.streak(fx, fy, length=170, angle=ang, thickness=5)
                        ib = tinta_bbox(limpio, q.frame())
                        lab = q.visible_labels()
                        if ib is None or not lab:
                            continue
                        _, lx1, ly1, lx2, ly2 = lab[0]
                        dx_ = ((lx1 + lx2) // 2) - ((ib[0] + ib[2]) // 2)
                        dy_ = ((ly1 + ly2) // 2) - ((ib[1] + ib[3]) // 2)
                        peor = max(peor, abs(dx_), abs(dy_))
    return peor


print("\n[4b] La ETIQUETA cae donde quedo la TINTA, lejos del centro")
# Esta es la prueba que la ida y vuelta no podia hacer: componer +a y -a se
# cancela con cualquier signo, asi que un signo invertido pasaba inadvertido
# y las etiquetas salian giradas el doble del angulo.
d_ok = desfase_maximo()
check_true("desfase tinta/etiqueta despreciable", d_ok <= 8,
           f"peor desfase={d_ok}px")

d_mal = desfase_maximo(invertir_signo=True)
check_true("la prueba DETECTA el signo invertido", d_mal > 15,
           f"con el error inyectado el desfase es {d_mal}px")

print("\n[5] El defecto VIAJA con la tela (no se queda pegado a la pantalla)")
p = TextilePattern(width=640, height=480, axis="x", tilt_deg=0.0,
                   motif="puntos", speed=200.0, seed=5)
fx, fy = p.screen_to_fabric(500, 240)
p.blob(fx, fy, size=60, cls="mancha")
pos = []
for _ in range(6):
    lab = p.visible_labels()
    pos.append(lab[0][1] if lab else None)
    p.advance(0.25)       # 50 px por paso
xs_ = [q for q in pos if q is not None]
check_true("la caja se mueve con la tela", len(set(xs_)) > 1, f"x={xs_}")
check_true("se mueve en contra del viaje (la tela entra por la derecha)",
           all(b <= a for a, b in zip(xs_, xs_[1:])), f"x={xs_}")

print("\n[6] Defecto medio fuera del encuadre se descarta")
# punto de radio exacto (blob() es aleatorio y el caso quedaba justo en el
# umbral: el test salia flaky, no el codigo)
p = TextilePattern(width=640, height=480, axis="x", tilt_deg=0.0, seed=6)
p.offset = 0.0
p.begin_stroke("mancha")
p.paint(*p.screen_to_fabric(635, 240), radius=60)   # 54% visible
p.end_stroke()
check("defecto cortado por el borde no se etiqueta",
      len(p.visible_labels(min_visible=0.55)), 0)

# el mismo defecto, completo dentro del encuadre, SI se etiqueta
p.clear_defects()
p.begin_stroke("mancha")
p.paint(*p.screen_to_fabric(320, 240), radius=60)
p.end_stroke()
check("el mismo defecto completo si se etiqueta",
      len(p.visible_labels(min_visible=0.55)), 1)

print("\n[7] Las tres clases pintan y etiquetan")
p = TextilePattern(width=800, height=600, axis="x", tilt_deg=0.0,
                   motif="diamantes", seed=7)
p.streak(*p.screen_to_fabric(180, 150), length=120, cls="rayon")
p.blob(*p.screen_to_fabric(420, 300), size=70, cls="mancha")
p.missing_print(*p.screen_to_fabric(640, 450), size=80)
labels = p.visible_labels()
check("tres defectos etiquetados", len(labels), 3)
check("las tres clases pintadas estan presentes",
      sorted(set(l[0] for l in labels)),
      sorted(["rayon", "mancha", "falta_impresion"]))

print("\n[7b] Las primitivas de fallo de impresion dejan marca")
# Fantasma y offset no son clases entrenables, pero el generador tiene que
# saber hacerlos: son el material con el que se prueba el modulo Patron y
# lo que el modo automatico de la demo le ensena a la gente.
for nombre, hacer in [("fantasma", lambda q, f: q.ghost(*f, size=150)),
                      ("offset", lambda q, f: q.misregister(*f, size=160))]:
    q = TextilePattern(width=640, height=480, axis="x", tilt_deg=0.0,
                       motif="diamantes", seed=20)
    limpio = q.frame().astype(int)
    hacer(q, q.screen_to_fabric(320, 240))
    d_ = np.abs(q.frame().astype(int) - limpio).sum(axis=2)
    check_true(f"{nombre} cambia la imagen", (d_ > 25).sum() > 400,
               f"px={(d_ > 25).sum()}")
    check(f"{nombre} queda registrado como defecto", len(q.defects), 1)

check_true("las clases entrenables son solo dos", len(YOLO_CLASSES) == 2,
           str(YOLO_CLASSES))
check_true("y los fallos de impresion van aparte", len(PRINT_FAULTS) == 3,
           str(PRINT_FAULTS))

print("\n[8] falta_impresion borra el motivo (deja tela desnuda)")
p = TextilePattern(width=400, height=300, axis="x", tilt_deg=0.0,
                   motif="diamantes", seed=8)
antes = p.frame()
fx, fy = p.screen_to_fabric(200, 150)
p.missing_print(fx, fy, size=90)
desp = p.frame()
roi_a = antes[110:190, 160:240]
roi_d = desp[110:190, 160:240]
check_true("el motivo desaparece en el parche",
           roi_d.std() < roi_a.std() * 0.6,
           f"std antes={roi_a.std():.1f} despues={roi_d.std():.1f}")
check_true("el parche queda claro (tela, no tinta)",
           roi_d.mean() > 215, f"media={roi_d.mean():.0f}")

print("\n[9] clear_defects deja la tela como nueva")
p = TextilePattern(width=400, height=300, axis="x", tilt_deg=0.0, seed=9)
limpio = p.frame()
p.blob(*p.screen_to_fabric(200, 150), size=60)
check_true("hay defecto", len(p.visible_labels()) == 1)
p.clear_defects()
check("sin defectos", len(p.defects), 0)
check_true("el frame vuelve a ser identico al limpio",
           np.array_equal(p.frame(), limpio))

print("\n[10] Un trazo continuo es UN defecto, no uno por punto")
p = TextilePattern(width=640, height=480, axis="x", tilt_deg=0.0, seed=10)
p.begin_stroke("rayon")
prev = p.screen_to_fabric(100, 240)
for sx in range(100, 400, 10):
    cur = p.screen_to_fabric(sx, 240)
    p.paint_segment(prev, cur, radius=6)
    prev = cur
p.end_stroke()
check("un trazo = un defecto", len(p.defects), 1)
check("una sola etiqueta", len(p.visible_labels()), 1)

print("\n[11] Clase desconocida no se pinta en silencio")
p = TextilePattern(width=320, height=240, seed=11)
try:
    p.begin_stroke("no_existe")
    fails.append("begin_stroke acepto una clase invalida")
    print("  FALLA  begin_stroke acepto una clase invalida")
except ValueError:
    print("  OK    begin_stroke rechaza clase invalida")
p.paint(10, 10, 5, "no_existe")
check("paint con clase invalida no crea defecto", len(p.defects), 0)

# muestras visuales
out = SCRATCH
for motif in ("diamantes", "flores", "rayas", "puntos"):
    pp = TextilePattern(width=640, height=360, motif=motif, seed=42)
    pp.streak(*pp.screen_to_fabric(150, 110), length=120, cls="rayon")
    pp.blob(*pp.screen_to_fabric(360, 200), size=64, cls="mancha")
    pp.missing_print(*pp.screen_to_fabric(520, 280), size=70)
    img = pp.frame().copy()
    for cls, x1, y1, x2, y2 in pp.visible_labels():
        col = {"rayon": (0, 215, 255), "mancha": (255, 190, 0),
               "falta_impresion": (120, 255, 120)}[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, cls, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(SCRATCH / f"textil_{motif}.png"), img)
print("\n  muestras escritas: textil_<motivo>.png")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("TELA SINTETICA: TODAS LAS PRUEBAS PASARON")
