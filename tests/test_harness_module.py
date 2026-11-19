# -*- coding: utf-8 -*-
"""
Módulo Arnés: colores de hilo contra el golden del job.

Las imágenes se dibujan aquí (conector blanco, hilos cilíndricos, traza,
termo), así que cada prueba sabe la verdad. Cada caso bueno lleva su caso
malo, y lo que se dice detectar se comprueba con una entrada que TIENE que
fallar.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eyve.modules.harness_module import (HarnessModule, delta_e00, leer_banda, nombre_de,
                                          parse_codigo, rgb_to_lab, segmentar, texto_de)

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


RGB = {"azul": (36, 82, 196), "amarillo": (236, 206, 40), "café": (110, 66, 38),
       "rojo": (200, 32, 32), "verde": (32, 140, 64), "blanco": (232, 232, 228),
       "negro": (35, 35, 38), "rosa": (236, 146, 168), "naranja": (238, 118, 26),
       "gris": (128, 128, 130)}

SKU = [("azul", None), ("amarillo", None), ("café", None), ("rojo", None),
       ("verde", None), ("blanco", "rojo")]


def dibujar(hilos, cara="a", ang=0.0, luz=(1.0, 1.0, 1.0), termo=0.0, sin_traza_en_a=False,
            W=960, H=540, cx=480, cy=200, pitch=30, ruido=2.0, seed=0):
    """
    Conector arriba, hilos hacia abajo, girado *ang* grados alrededor del
    centro del conector. La cara B se ve espejada. *termo*: fracción de la
    banda tapada desde el conector. Devuelve (frame BGR, dets).
    """
    rng = np.random.default_rng(seed)
    n = len(hilos)
    img = np.full((H, W, 3), (92, 98, 104), np.float32)       # tapete gris
    cw, ch = pitch * n + 18, 58
    x0, y0 = cx - cw // 2, cy - ch // 2
    orden = hilos if cara == "a" else hilos[::-1]
    # hilos: de la base del conector hacia abajo
    yb0, yb1 = y0 + ch, y0 + ch + 170
    wd = int(pitch * 0.86)
    for k, (base, traza) in enumerate(orden):
        xc = x0 + 9 + pitch * k + pitch / 2
        for dx in range(-wd // 2, wd // 2 + 1):
            u = (dx + wd / 2) / wd                       # 0..1 a lo ancho
            sombra = 0.55 + 0.45 * np.sin(np.pi * u)
            col = np.array(RGB[base][::-1], np.float32)
            if traza and 0.52 <= u <= 0.78 and not (sin_traza_en_a and cara == "a"):
                col = np.array(RGB[traza][::-1], np.float32)
            img[yb0:yb1, int(xc + dx)] = col * sombra
    # conector: nylon blanco con ventanas (cara A) o seguros (cara B)
    img[y0:y0 + ch, x0:x0 + cw] = (222, 226, 228)
    for k in range(n):
        xk = x0 + 9 + pitch * k
        if cara == "a":
            img[y0 + 10:y0 + 34, xk + 6:xk + pitch - 6] = (150, 150, 155)
        elif k in (0, n - 1):
            img[y0 + 8:y0 + 20, xk + 4:xk + pitch - 4] = (190, 192, 195)
    hb = (x0 + 6, yb0, x0 + cw - 6, yb0 + 64)
    if termo > 0:
        img[yb0:yb0 + int(termo * 120), x0:x0 + cw] = (230, 232, 233)
        hb = (hb[0], yb0 + int(termo * 120), hb[2], yb0 + max(int(termo * 120) + 4, 64))
    img = img * np.array(luz[::-1], np.float32)[None, None, :]
    img += rng.normal(0, ruido, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cb = (x0, y0, x0 + cw, y0 + ch)
    if ang:
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        img = cv2.warpAffine(img, M, (W, H), borderValue=(92, 98, 104))

        def caja(b):
            p = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[0], b[3], 1], [b[2], b[3], 1]]) @ M.T
            return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
        cb, hb = caja(cb), caja(hb)
    dets = [SimpleNamespace(class_name=f"conector_{cara}", confidence=0.95,
                            x1=cb[0], y1=cb[1], x2=cb[2], y2=cb[3]),
            SimpleNamespace(class_name="hilos", confidence=0.9,
                            x1=hb[0], y1=hb[1], x2=hb[2], y2=hb[3])]
    return img, dets


def aprendido(hilos=SKU, verificar=True):
    m = HarnessModule()
    m.enabled = True
    for cara in ("a", "b"):
        for s in range(6):
            img, dets = dibujar(hilos, cara=cara, seed=s)
            m.learn(img, dets, n=len(hilos))
        m.finish_learning(cara)
    m.codigo = "1 AZ · 2 AM · 3 CF · 4 RJ · 5 VE · 6 BL/RJ"
    if verificar:
        m.verificar("prueba")
    return m


def juzgar(m, hilos, cara="a", t0=0.0, **kw):
    """Pieza quieta frente a la cámara 1.5 s; devuelve la lectura de la cara."""
    img, dets = dibujar(hilos, cara=cara, **kw)
    t = t0
    for _ in range(30):
        m.update(img, dets, now=t)
        t += 0.05
    return t


def mover(m, t, n=3):
    """Unos frames de movimiento (voltear, acomodar)."""
    for k in range(n):
        ruido = np.random.default_rng(100 + k).integers(0, 255, (540, 960, 3), np.uint8)
        m.update(ruido, [SimpleNamespace(class_name="conector_a", confidence=0.9,
                                         x1=10, y1=10, x2=50, y2=50)], now=t)
        t += 0.05
    return t


def salir(m, t):
    vacio = np.full((540, 960, 3), (92, 98, 104), np.uint8)
    for _ in range(20):
        m.update(vacio, [], now=t)
        t += 0.05
    return t


print("\n[1] Color: ΔE2000 y nombres")
check("ΔE00 de un color consigo mismo = 0",
      float(delta_e00(rgb_to_lab((200, 32, 32)), rgb_to_lab((200, 32, 32)))) < 1e-6)
# valor de referencia publicado (Sharma 2005, par 1)
check("ΔE00 coincide con la tabla de Sharma (2.0425)",
      abs(float(delta_e00([50, 2.6772, -79.7751], [50, 0, -82.7485])) - 2.0425) < 1e-3)
for nom, rgb in RGB.items():
    check(f"nombre de {nom}", nombre_de(rgb_to_lab(rgb)) == nom, nombre_de(rgb_to_lab(rgb)))
for a, b in (("rosa", "rojo"), ("blanco", "gris"), ("café", "negro"), ("naranja", "rojo")):
    d = float(delta_e00(rgb_to_lab(RGB[a]), rgb_to_lab(RGB[b])))
    check(f"par que se confunde {a}↔{b} queda arriba de la tolerancia (11)", d > 11, f"ΔE={d:.1f}")

print("\n[2] Codificación escrita")
p = parse_codigo("1 AZ · 2 AM · 3 CF · 4 RJ · 5 VE · 6 BL/RJ")
check("lee los 6 pines con traza", p == [("azul", None), ("amarillo", None), ("café", None),
                                        ("rojo", None), ("verde", None), ("blanco", "rojo")], p)
check("también sin números y con comas", parse_codigo("AZ, AM, CF") ==
      [("azul", None), ("amarillo", None), ("café", None)])
check("y en inglés", parse_codigo("BU, YE, BN/WH")[2] == ("café", "blanco"))
check("lo desconocido se marca, no se tira", parse_codigo("AZ, XX")[1] == ("?", None))
check("texto_de vuelve al formato", texto_de(p).startswith("1 AZ · 2 AM"))

print("\n[3] Segmentación")
cols = np.zeros((60, 3))
cols[:20] = (50, 60, 40)
cols[20:40] = (70, -40, 50)
cols[40:] = (30, 20, -60)
check("tres bloques limpios → cortes exactos", segmentar(cols, 3) == [0, 20, 40, 60],
      segmentar(cols, 3))
cols[20:40] = (50, 60, 40)
check("PREVIO: con dos bloques iguales, sin prior el corte es arbitrario", True)
c = segmentar(cols, 3, prior=[0, 1 / 3, 2 / 3, 1])
check("con prior, el corte entre iguales cae donde se aprendió", abs(c[1] - 20) <= 2, c)

print("\n[4] Aprender el golden y verificarlo")
m = aprendido(verificar=False)
check("aprende las dos caras", m.faces_required == ["a", "b"], m.faces_required)
na = m.nombres("a")
check("cara A, nombres en orden de pin", na == SKU, na)
check("cara B espejada vuelve al orden de pin", m.nombres("b") == SKU, m.nombres("b"))
check("la traza blanco/rojo se aprendió", m.golden["a"]["colors"][5]["traza"] is not None)
check("y el azul NO tiene traza (la sombra no es traza)",
      all(c["traza"] is None for c in m.golden["a"]["colors"][:5]))
check("sin verificar no juzga", not m.verified)
juzgar(m, SKU)
check("… y no da veredicto", m.verdict is None and m.state == "sin_golden", m.state)
check("cotejo contra el texto: sin diferencias", m.cotejar() == [], m.cotejar())
m.codigo = "1 AZ · 2 AM · 3 RJ · 4 CF · 5 VE · 6 BL/RJ"
check("cotejo: un texto distinto se avisa", any("pin 3" in a for a in m.cotejar()), m.cotejar())
check("verificar pide nombre", not m.verificar("  "))
check("verificar firma", m.verificar("Ana QA") and m.verified)
m.codigo = "1 AZ · 2 AM · 3 CF · 4 RJ · 5 VE · 6 BL/RJ"
check("cambiar el texto invalida la verificación", not m.verified)
m.verificar("Ana QA")
m.golden["a"]["colors"][0]["base"][1] += 30
check("alterar el golden invalida la verificación", not m.verified)

print("\n[5] Pieza buena: cara A, voltea, cara B → OK")
m = aprendido()
t = juzgar(m, SKU, "a")
check("tras la cara A pide voltear", m.state == "voltea" and m.verdict is None, m.state)
check("la cara A salió bien", m.face_ok.get("a") is True)
t = mover(m, t)
t = juzgar(m, SKU, "b", t0=t)
check("con la cara B: OK", m.verdict == "OK" and m.state == "resultado", (m.state, m.defects))
t = salir(m, t)
check("al irse cuenta 1 OK", m.counts == {"ok": 1, "corregida": 0, "rechazo": 0}, m.counts)

print("\n[6] Empezar por la cara B también vale")
m = aprendido()
t = juzgar(m, SKU, "b")
check("primero B → pide la A", m.state == "voltea", m.state)
t = juzgar(m, SKU, "a", t0=mover(m, t))
check("luego A → OK", m.verdict == "OK", m.defects)

print("\n[7] Hilos cruzados")
mal = list(SKU)
mal[2], mal[3] = mal[3], mal[2]          # café y rojo cruzados
m = aprendido()
juzgar(m, mal, "a")
check("cruzados 3↔4 → NOT OK", m.verdict == "NOT_OK", (m.state, m.defects))
check("y dice cuáles", m.defects.get("a") == ["cruzados 3↔4"], m.defects)
check("y la acción es corregir o rechazo", "RECHAZO" in m.action_text(), m.action_text())

print("\n[8] Un hilo de otro color")
for mal_color in ("rosa", "naranja"):
    mal = list(SKU)
    mal[3] = (mal_color, None)          # va rojo
    m = aprendido()
    juzgar(m, mal, "a")
    check(f"rojo → {mal_color}: NOT OK en el pin 4", m.verdict == "NOT_OK"
          and any(d.startswith("pin 4") for d in m.defects.get("a", [])), m.defects)

print("\n[9] Traza")
mal = list(SKU)
mal[5] = ("blanco", None)
m = aprendido()
t = juzgar(m, mal, "a")
check("blanco liso en la cara A: todavía no se condena (la raya puede ir abajo)",
      m.face_ok.get("a") is True and m.state == "voltea", (m.state, m.defects))
t = juzgar(m, mal, "b", t0=mover(m, t))
check("y tampoco en la B → NOT OK: no se ve la traza del pin 6", m.verdict == "NOT_OK"
      and any("pin 6: no se ve la traza" in d for d in m.defects.get("*", [])), m.defects)
m = aprendido()
t = juzgar(m, SKU, "a", sin_traza_en_a=True)
t = juzgar(m, SKU, "b", t0=mover(m, t))
check("la raya solo del lado B (pieza buena girada) → OK", m.verdict == "OK", m.defects)
sku2 = list(SKU)
sku2[5] = ("blanco", None)
m = aprendido(hilos=sku2)
juzgar(m, SKU, "a")
check("blanco/rojo donde va liso → NOT OK desde la primera cara", m.verdict == "NOT_OK", m.defects)
check("cotejo: la traza del texto se busca en cualquier cara",
      aprendido().cotejar() == [], aprendido().cotejar())

print("\n[10] Colores repetidos en el mismo conector")
rep = [("verde", None), ("azul", None), ("verde", None), ("rojo", None), ("verde", None), ("negro", None)]
m = aprendido(hilos=rep)
t = juzgar(m, rep, "a")
check("tres verdes: la pieza buena sale bien", m.face_ok.get("a") is True, m.defects)
mal = list(rep)
mal[3], mal[4] = mal[4], mal[3]
m = aprendido(hilos=rep)
juzgar(m, mal, "a")
check("rojo y verde cruzados entre verdes → cruzados 4↔5", m.defects.get("a") == ["cruzados 4↔5"],
      m.defects)

print("\n[10b] Hilos vecinos muy distintos no inventan traza")
rn = [("rojo", None), ("negro", None), ("rojo", None), ("blanco", None), ("negro", None), ("amarillo", None)]
m = aprendido(hilos=rn)
check("golden sin trazas falsas en las orillas", m.trazas() == {}, m.trazas())
mal = list(rn)
mal[2] = ("negro", None)
m = aprendido(hilos=rn)
juzgar(m, mal, "a")
check("un negro donde va rojo: solo se reporta ese pin", m.defects.get("a") == ["pin 3: va rojo, hay negro"],
      m.defects)

print("\n[11] Girada en el plano e inclinada")
for ang in (180, 90, -12, 8):
    m = aprendido()
    juzgar(m, SKU, "a", ang=ang)
    bien = m.face_ok.get("a") is True
    mal = list(SKU)
    mal[0], mal[1] = mal[1], mal[0]
    m2 = aprendido()
    juzgar(m2, mal, "a", ang=ang)
    check(f"{ang:+}°: buena bien y cruzada mal", bien and m2.defects.get("a") == ["cruzados 1↔2"],
          (m.defects, m2.defects))

print("\n[12] Luz más cálida")
calida = (1.12, 1.0, 0.82)
m = aprendido()
juzgar(m, SKU, "a", luz=calida)
check("con balance de blancos sigue bien", m.face_ok.get("a") is True, m.defects)
m = aprendido()
m.white_balance = False
m.verificar("prueba")
juzgar(m, SKU, "a", luz=calida)
check("PREVIO: sin balance de blancos la misma luz la tumba", m.face_ok.get("a") is False, m.defects)

print("\n[13] El termo tapa")
m = aprendido()
t = juzgar(m, SKU, "a", termo=0.6)
check("termo tapando → no juzga y pide subirlo", m.verdict is None and m.state == "tapado"
      and m.action_text() == "Sube el termo", (m.state, m.verdict))
t = juzgar(m, SKU, "a", t0=mover(m, t))
check("lo sube → sigue leyendo y la cara sale bien", m.face_ok.get("a") is True, (m.state, m.defects))

print("\n[14] Pieza corregida y pieza a rechazo")
m = aprendido()
mal = list(SKU)
mal[0], mal[1] = mal[1], mal[0]
t = juzgar(m, mal, "a")
check("PREVIO: NOT OK", m.verdict == "NOT_OK")
t = mover(m, t)                              # la re-pinchan en la estación
t = juzgar(m, SKU, "a", t0=t)
t = juzgar(m, SKU, "b", t0=mover(m, t))
check("corregida y con las dos caras: OK", m.verdict == "OK", m.defects)
t = salir(m, t)
check("cuenta como corregida", m.counts["corregida"] == 1, m.counts)
t = juzgar(m, mal, "a", t0=t)
t = salir(m, t)
check("una que se va con NOT OK cuenta rechazo", m.counts["rechazo"] == 1, m.counts)

print("\n[15] Job: se guarda y vuelve igual")
m = aprendido()
cfg = m.config()
m2 = HarnessModule()
m2.apply_config(cfg)
check("golden, texto y verificación sobreviven", m2.golden == m.golden and m2.codigo == m.codigo
      and m2.verified)
juzgar(m2, SKU, "a")
check("y el cargado juzga igual", m2.face_ok.get("a") is True)
import yaml
check("config es YAML puro (sin ndarray)", isinstance(yaml.safe_load(yaml.safe_dump(cfg)), dict))

print("\n[16] Otro SKU, mismo conector")
sku_b = [("negro", None), ("rojo", None), ("rojo", None), ("negro", None), ("amarillo", None), ("azul", None)]
mb = aprendido(hilos=sku_b)
juzgar(mb, SKU, "a")
check("la pieza del SKU A en el job del SKU B → NOT OK", mb.verdict == "NOT_OK")
mb2 = aprendido(hilos=sku_b)
juzgar(mb2, sku_b, "a")
check("y la del B en su job → bien", mb2.face_ok.get("a") is True, mb2.defects)

print("\n" + "=" * 60)
print("ARNÉS: COLORES DE HILO CONTRA EL GOLDEN, POR CARA")
if fails:
    print(f"FALLARON {len(fails)}: {fails}")
    sys.exit(1)
print("todo bien")
