# -*- coding: utf-8 -*-
"""
Módulo Arnés con conectores de DOS filas (JST PUD 2×4 y 2×5).

JST numera los circuitos impares en una fila y los pares en la otra (1 y 2
en la misma columna). Desde una cara se ve una fila; desde la otra, la otra:
cada cara trae la mitad de los pines y sin las dos no hay veredicto.

Las imágenes salen del mismo dibujante de test_harness_module: cada cara se
dibuja con los hilos de SU fila. Cada caso que se dice detectar tiene una
entrada que TIENE que fallar.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):       # reusar su dibujante, sin su salida
    import test_harness_module as T1
from eyve.modules.harness_module import HarnessModule, nombre_de

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


# pin 1..8: impares en la fila de la cara A, pares en la de la B
P8 = [("azul", None), ("amarillo", None), ("café", None), ("rojo", None),
      ("verde", None), ("blanco", "rojo"), ("negro", None), ("naranja", None)]
COD8 = "1 AZ · 2 AM · 3 CF · 4 RJ · 5 VE · 6 BL/RJ · 7 NE · 8 NA"
P10 = P8 + [("gris", None), ("rosa", None)]
COD10 = COD8 + " · 9 GR · 10 RS"


def fila(P, cara, fila_a="impares"):
    """
    Los hilos que se ven desde *cara*, en el orden que espera dibujar():
    para la A, de izquierda a derecha; para la B, en orden de pin (dibujar
    la espeja). Cara A con fila_a="impares" ve 1, 3, 5…; la B ve 2, 4, 6…
    """
    impar = (cara == "a") == (fila_a == "impares")
    return [P[k] for k in range(0 if impar else 1, len(P), 2)]


def dibujar(P, cara="a", clase="pud", **kw):
    img, dets = T1.dibujar(fila(P, cara), cara=cara, **kw)
    dets[0].class_name = f"{clase}_{cara}"
    return img, dets


def aprendido(P=P8, cod=COD8, verificar=True, caras=("a", "b")):
    m = HarnessModule()
    m.enabled = True
    m.filas = 2
    for cara in caras:
        for s in range(6):
            img, dets = dibujar(P, cara, seed=s)
            m.learn(img, dets, n=len(P))
        m.finish_learning(cara)
    m.codigo = cod
    if verificar:
        m.verificar("Calidad - turno 1")
    return m


def juzgar(m, P, cara="a", t0=0.0, **kw):
    img, dets = dibujar(P, cara, **kw)
    t = t0
    for _ in range(30):
        m.update(img, dets, now=t)
        t += 0.05
    return t


mover, salir = T1.mover, T1.salir


print("\n[1] Mapa de pines: impares en una fila, pares en la otra")
m = HarnessModule()
m.filas = 2
check("cara A (impares), de izquierda a derecha: 1 3 5 7",
      [m._pin("a", i, 4) for i in range(4)] == [1, 3, 5, 7])
check("cara B (la pieza girada): 8 6 4 2",
      [m._pin("b", i, 4) for i in range(4)] == [8, 6, 4, 2])
check("2×5: A 1…9, B 10…2", [m._pin("a", i, 5) for i in range(5)] == [1, 3, 5, 7, 9]
      and [m._pin("b", i, 5) for i in range(5)] == [10, 8, 6, 4, 2])
m.fila_a = "pares"
check("si la cara A queda del lado de los pares: A 2 4 6 8, B 7 5 3 1",
      [m._pin("a", i, 4) for i in range(4)] == [2, 4, 6, 8]
      and [m._pin("b", i, 4) for i in range(4)] == [7, 5, 3, 1])
m1 = HarnessModule()
check("una fila no cambia: A 1…6, B 6…1", [m1._pin("a", i, 6) for i in range(6)] == [1, 2, 3, 4, 5, 6]
      and [m1._pin("b", i, 6) for i in range(6)] == [6, 5, 4, 3, 2, 1])
check("pareja de columna: 1↔2, 4↔3, 9↔10", (m._pareja(1), m._pareja(4), m._pareja(9)) == (2, 3, 10))

print("\n[2] Aprender: cada cara su fila; sin las dos no se verifica")
m = aprendido(verificar=False, caras=("a",))
check("la cara A aprende 4 hilos de 8", m.golden["a"]["n"] == 4 and m.n_total == 8,
      (m.golden["a"]["n"], m.n_total))
check("falta la B", m.faltan_caras() == ["b"], m.faltan_caras())
check("no se deja verificar con una cara", not m.verificar("Calidad - turno 1"))
check("el cotejo lo dice", any("falta la cara B" in a and "pares" in a for a in m.cotejar()), m.cotejar())
check("y el estado también", "Falta aprender la cara B" in m.state_text(), m.state_text())
m = aprendido(verificar=False)
check("con las dos: A = 1 3 5 7", m.nombres("a") == [P8[0], P8[2], P8[4], P8[6]], m.nombres("a"))
check("B = 2 4 6 8 (la traza del 6 incluida)", m.nombres("b") == [P8[1], P8[3], P8[5], P8[7]],
      m.nombres("b"))
check("cotejo contra la codificación de 8: sin diferencias", m.cotejar() == [], m.cotejar())
m.codigo = COD8.replace("4 RJ", "4 CF")
check("cotejo: el pin 4 escrito mal se avisa con su número (cara B)",
      any("cara B pin 4" in a for a in m.cotejar()), m.cotejar())
m.codigo = "1 AZ · 2 AM · 3 CF · 4 RJ · 5 VE · 6 BL/RJ"
check("cotejo: un texto de 6 hilos para un conector de 8 se avisa",
      any("8" in a for a in m.cotejar()), m.cotejar())
m.codigo = COD8
check("verifica con las dos caras", m.verificar("Calidad - turno 1") and m.verified)
h1 = m.huella()
m.fila_a = "pares"
check("cambiar qué fila ve la A invalida la verificación", m.huella() != h1 and not m.verified)
m.fila_a = "impares"
n_ = HarnessModule().learn_n
mm = HarnessModule(); mm.filas = 2
check("dos filas con un número impar de hilos no se aprende", mm.learn_n(7) == 0 and mm.learn_n(8) == 4)

print("\n[3] Pieza buena: A, voltea, B → OK")
m = aprendido()
t = juzgar(m, P8, "a")
check("tras la A pide voltear: faltan los pares", m.state == "voltea" and m.verdict is None, m.state)
t = juzgar(m, P8, "b", t0=mover(m, t))
check("con la B: OK", m.verdict == "OK", (m.state, m.defects))
t = salir(m, t)
check("cuenta 1 OK", m.counts["ok"] == 1, m.counts)
m = aprendido()
t = juzgar(m, P8, "b")
check("empezar por la B también: pide la A", m.state == "voltea", m.state)

print("\n[4] Cruzados en la misma fila, otro color: con el número de pin real")
mal = list(P8)
mal[2], mal[4] = mal[4], mal[2]            # 3 y 5: misma fila (cara A)
m = aprendido()
juzgar(m, mal, "a")
check("cruzados 3↔5 en la cara A", m.defects.get("a") == ["cruzados 3↔5"], m.defects)
mal = list(P8)
mal[5] = ("rosa", None)                     # va blanco/rojo
m = aprendido()
t = juzgar(m, mal, "a")
check("PREVIO: la cara A no ve el pin 6 y sale bien", m.face_ok.get("a") is True, m.defects)
juzgar(m, mal, "b", t0=mover(m, t))
check("la B ve el pin 6 de otro color", m.verdict == "NOT_OK"
      and any(d.startswith("pin 6: va blanco/rojo, hay rosa") for d in m.defects.get("b", [])),
      m.defects)
mal = list(P8)
mal[7] = ("verde", None)                    # pin 8: va naranja
m = aprendido()
juzgar(m, mal, "b")
check("el pin 8 (orilla de la B) se nombra 8", any(d.startswith("pin 8:") for d in m.defects.get("b", [])),
      m.defects)

print("\n[5] Hilo encimado: se reacomoda, no se rechaza")
# el hilo del pin 4 (rojo, fila de abajo) se subió y tapa al 3 en la cara A
enc = list(P8)
enc[2] = P8[3]
m = aprendido()
t = juzgar(m, enc, "a")
check("cara A: 'pin 3: hilo encimado, se ve el del pin 4'",
      m.defects.get("a") == ["pin 3: hilo encimado, se ve el del pin 4"], m.defects)
check("la acción es reacomodar, no rechazo", "reacomoda" in m.action_text()
      and "RECHAZO" not in m.action_text(), m.action_text())
t = juzgar(m, P8, "a", t0=mover(m, t))      # los acomodan
t = juzgar(m, P8, "b", t0=mover(m, t))
check("reacomodados y con las dos caras: OK", m.verdict == "OK", m.defects)
t = salir(m, t)
check("cuenta como corregida", m.counts["corregida"] == 1, m.counts)
mal = list(P8)
mal[2] = ("rosa", None)                     # rosa no es el hilo del 4
m = aprendido()
juzgar(m, mal, "a")
check("un color que no es el de la otra fila NO se llama encimado",
      any(d.startswith("pin 3: va café, hay rosa") for d in m.defects.get("a", [])), m.defects)

print("\n[6] Cruzados ENTRE filas (3↔4): lo dicen las dos caras juntas")
mal = list(P8)
mal[2], mal[3] = mal[3], mal[2]
m = aprendido()
t = juzgar(m, mal, "a")
check("PREVIO: con una cara parece encimado", m.encimados(), m.defects)
juzgar(m, mal, "b", t0=mover(m, t))
check("con la B: 'cruzados 3↔4 (entre filas)'", "cruzados 3↔4 (entre filas)" in m.defects_text()
      and "encimado" not in m.defects_text(), m.defects)
check("y eso sí es corregir o rechazo", "RECHAZO" in m.action_text(), m.action_text())

print("\n[7] Traza en la fila de abajo")
m = aprendido()
t = juzgar(m, P8, "a")
juzgar(m, P8, "b", t0=mover(m, t), sin_traza_en_a=False)
check("PREVIO: la traza del 6 se vio en la B → OK", m.verdict == "OK", m.defects)
mal = list(P8)
mal[5] = ("blanco", None)                   # blanco liso donde va blanco/rojo
m = aprendido()
t = juzgar(m, mal, "a")
juzgar(m, mal, "b", t0=mover(m, t))
check("blanco liso en el 6 → NOT OK por traza del pin 6", m.verdict == "NOT_OK"
      and "pin 6" in m.defects_text(), m.defects)

print("\n[8] 2×5: diez pines, mismo módulo")
m = aprendido(P=P10, cod=COD10)
check("5 por cara y 10 en total", m.golden["a"]["n"] == 5 and m.n_total == 10)
check("cotejo de 10: sin diferencias", m.cotejar() == [], m.cotejar())
t = juzgar(m, P10, "a")
t = juzgar(m, P10, "b", t0=mover(m, t))
check("pieza buena de 10: OK", m.verdict == "OK", m.defects)
mal = list(P10)
mal[9] = ("negro", None)                    # pin 10: va rosa
m = aprendido(P=P10, cod=COD10)
juzgar(m, mal, "b")
check("pin 10 de otro color: se nombra 10", any(d.startswith("pin 10:") for d in m.defects.get("b", [])),
      m.defects)
m8 = aprendido()
juzgar(m8, P10, "a")
check("una pieza de 10 en el job de 8 → NOT OK", m8.verdict == "NOT_OK", (m8.state, m8.defects))

print("\n[9] Tarjeta Visto / Va con los 8 pines")
m = aprendido()
mal = list(P8)
mal[4] = ("rosa", None)
juzgar(m, mal, "a")
img = np.zeros((720, 1280, 3), np.uint8)
m.draw(img)
check("se dibuja sin errores y pinta algo", img.any())
anchos = []
for N, P, cod in ((8, P8, COD8), (10, P10, COD10)):
    mm = aprendido(P=P, cod=cod)
    juzgar(mm, P, "a")
    im = np.zeros((720, 1280, 3), np.uint8)
    mm._draw_pines(im, 1.0, 0, 0)               # solo la tarjeta, en la esquina
    fil = np.where(im.any(axis=2).any(axis=0))[0]
    anchos.append(int(fil.max() - fil.min()) if len(fil) else 0)
check("la de 10 es más ancha que la de 8 (todos los pines, no solo los de la cara)",
      anchos[1] > anchos[0] > 0, anchos)

print("\n[10] Job: filas, fila de la A y clases se guardan")
m = aprendido()
m.target_class = "pud"
cfg = m.config()
m2 = HarnessModule()
m2.apply_config(cfg)
check("filas, fila_a y prefijo vuelven", (m2.filas, m2.fila_a, m2.target_class) == (2, "impares", "pud"))
check("y sigue verificado", m2.verified)
juzgar(m2, P8, "a")
check("el cargado juzga igual", m2.face_ok.get("a") is True, m2.defects)
m3 = HarnessModule()
m3.apply_config({k: v for k, v in cfg.items() if k not in ("filas", "fila_a", "target_class")})
check("un job viejo (sin 'filas') carga como una fila", m3.filas == 1 and m3.target_class == "")
check("los hilos de la pieza van en el job (8)", cfg.get("n_hilos") == 8, cfg.get("n_hilos"))
mv = HarnessModule(); mv.filas = 2; mv.n_hilos = 10
mv2 = HarnessModule(); mv2.apply_config(mv.config())
check("y sin golden también (10)", mv2.n_hilos == 10 and mv2.filas == 2, (mv2.n_hilos, mv2.filas))
juzgar(aprendido(), P8, "a", clase="conector")
mc = aprendido()
juzgar(mc, P8, "a", clase="pud")
check("sin prefijo vale pud_a igual que conector_a", mc.face_ok.get("a") is True, mc.state)

print("\n[11] Traza del color del tapete (azul sobre blanco, tapete azul)")
# El tapete se ve entre hilos y esas columnas no cuentan como hilo. Una traza
# de un color parecido al tapete NO es tapete: va dentro del hilo, lejos de
# la frontera entre hilos. Antes se borraba: la pieza buena salía "no se ve
# la traza" según un píxel de la caja, y la mala (blanco liso) podía pasar.
AZUL_TAPETE = (176, 74, 32)                # BGR: el azul del hilo a media luz
PT = list(P8)
PT[5] = ("blanco", "azul")


def aprendido_t(P):
    m = HarnessModule()
    m.enabled, m.filas = True, 2
    for cara in ("a", "b"):
        for s_ in range(6):
            img, dets = dibujar(P, cara, seed=s_, tapete=AZUL_TAPETE)
            m.learn(img, dets, n=len(P))
        m.finish_learning(cara)
    m.codigo = COD8.replace("6 BL/RJ", "6 BL/AZ")
    return m


def juzgar_t(m, P, cara, t0=0.0):
    img, dets = dibujar(P, cara, tapete=AZUL_TAPETE)
    t = t0
    for _ in range(30):
        m.update(img, dets, now=t)
        t += 0.05
    return t


m = aprendido_t(PT)
gt = m._golden_pin(6)
check("PREVIO: el golden del 6 aprendió la traza azul", gt is not None and gt["traza"] is not None
      and nombre_de(gt["traza"]) == "azul", gt and gt["traza"])
m.verificar("Calidad - turno 1")
t = juzgar_t(m, PT, "a")
juzgar_t(m, PT, "b", t0=mover(m, t))
check("la pieza buena con esa traza: OK", m.verdict == "OK", (m.verdict, m.defects))
PL = list(PT)
PL[5] = ("blanco", None)
m2 = HarnessModule()
m2.apply_config(m.config())
m2.enabled = True
t = juzgar_t(m2, PL, "a")
juzgar_t(m2, PL, "b", t0=mover(m2, t))
check("y el blanco liso en su lugar sale NOT OK por la traza del 6", m2.verdict == "NOT_OK"
      and "pin 6" in m2.defects_text(), m2.defects)

print("\n" + "=" * 60)
print("ARNÉS: DOS FILAS (JST PUD 2×4 y 2×5)")
if fails:
    print(f"FALLARON {len(fails)}: {fails}")
    sys.exit(1)
print("todo bien")
