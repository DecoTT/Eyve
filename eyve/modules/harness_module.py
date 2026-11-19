"""
Arnés — colores de los hilos contra el golden del job, por cara del conector.

Para qué
────────
El probador eléctrico dice que cada pin llega a donde debe. No dice si el
hilo que llega es del color que pide el SKU: un arnés con el amarillo y el
café cruzados en los dos extremos pasa la prueba eléctrica. Este módulo
revisa eso, mirando los hilos justo donde salen del conector, donde todavía
van en el orden de los pines.

Reparto del trabajo
───────────────────
  YOLO   → lo nombrable: el conector y QUÉ CARA se ve (`<familia>_a` /
           `<familia>_b`), y la banda por donde salen los hilos (`hilos`).
  Arnés  → los colores, sin entrenar: un golden aprendido de una pieza
           buena, por cara, y verificado por una persona contra la
           codificación escrita del SKU.

Mismo conector, otro SKU, otros colores = otro job, mismo modelo.

Cómo se lee una cara
────────────────────
  1. La dirección de los hilos sale de la caja del conector a la de la
     banda, y se afina con el tensor de estructura de la banda (los hilos
     son líneas paralelas). Se endereza la banda para que los hilos queden
     verticales, saliendo del conector hacia abajo; así el orden de lectura
     no cambia aunque la pieza esté girada en el plano.
  2. Balance de blancos con el nylon del conector (siempre está ahí): lo
     claro y sin color de su caja es la referencia de blanco del frame.
  3. Una franja delgada pegada al conector se reduce a un perfil por
     columna (Lab) y se parte en N tramos con programación dinámica: mínima
     varianza dentro de cada tramo, con un precio por alejarse de donde el
     golden dice que va cada corte.
  4. Por tramo: color base (mediana de las columnas mayoritarias) y, si una
     minoría de columnas difiere en tono, color de traza.
  5. ΔE2000 contra el golden de esa posición. Si un tramo se parece más al
     golden de otra posición (de color distinto) y viceversa: «cruzados».

Pieza con volteo
────────────────
Como Ensamble, el veredicto es por pieza y se pausa con movimiento:

    VACIO → ASENTANDO → INSPECCIONA LA CARA QUE SE VE → (falta otra cara:
    VOLTEA) → ASENTANDO → INSPECCIONA LA OTRA → RESULTADO

La cara la dice YOLO, no el orden: si el operador empieza por B, o voltea
dos veces, sigue bien. Cada cara se aprende como se ve, así que no hay que
espejar nada.

Límites, dichos de frente:
  - Con la luz controlada (difusa, temperatura fija). Un brillo especular
    fuerte sobre un hilo de color puede leerse como traza.
  - La traza se ve si queda del lado que mira a la cámara; por eso se pide
    la otra cara.
  - Los hilos tienen que salir en orden de pines junto al conector. Si se
    enciman (cara del seguro de algunos conectores) y no salen N tramos
    limpios, no se juzga: se pide reacomodar.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from eyve.modules.base import InspectionModule, ModuleVerdict

_VERDE, _ROJO, _GRIS, _AMBAR = (60, 190, 60), (40, 40, 230), (134, 126, 120), (20, 160, 240)

#: paleta para NOMBRAR (solo para mostrar y para cotejar con el texto); la
#: decisión se toma contra el golden, nunca contra estos valores. sRGB.
PALETA = {
    "negro":    (35, 35, 38),
    "café":     (110, 66, 38),
    "rojo":     (200, 32, 32),
    "naranja":  (238, 118, 26),
    "amarillo": (236, 206, 40),
    "verde":    (32, 140, 64),
    "azul":     (36, 82, 196),
    "violeta":  (124, 60, 168),
    "gris":     (128, 128, 130),
    "blanco":   (232, 232, 228),
    "rosa":     (236, 146, 168),
}

#: códigos de la codificación escrita (dibujo del cliente) → nombre
CODIGOS = {
    "NE": "negro", "BK": "negro", "NEGRO": "negro", "BLACK": "negro",
    "CF": "café", "BN": "café", "CAFE": "café", "CAFÉ": "café", "BROWN": "café", "MARRON": "café",
    "RJ": "rojo", "RD": "rojo", "ROJO": "rojo", "RED": "rojo",
    "NA": "naranja", "OR": "naranja", "NARANJA": "naranja", "ORANGE": "naranja",
    "AM": "amarillo", "YE": "amarillo", "YL": "amarillo", "AMARILLO": "amarillo", "YELLOW": "amarillo",
    "VE": "verde", "VD": "verde", "GN": "verde", "VERDE": "verde", "GREEN": "verde",
    "AZ": "azul", "BU": "azul", "BL_EN": "azul", "AZUL": "azul", "BLUE": "azul",
    "VI": "violeta", "VT": "violeta", "VIOLETA": "violeta", "MORADO": "violeta", "VIOLET": "violeta",
    "GR": "gris", "GY": "gris", "GRIS": "gris", "GRAY": "gris", "GREY": "gris",
    "BL": "blanco", "BCO": "blanco", "WH": "blanco", "BLANCO": "blanco", "WHITE": "blanco",
    "RS": "rosa", "PK": "rosa", "ROSA": "rosa", "PINK": "rosa",
}
#: abreviatura con la que se muestra cada nombre
ABREV = {"negro": "NE", "café": "CF", "rojo": "RJ", "naranja": "NA", "amarillo": "AM",
         "verde": "VE", "azul": "AZ", "violeta": "VI", "gris": "GR", "blanco": "BL",
         "rosa": "RS"}


# ── color ─────────────────────────────────────────────────────────────────
def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (…,3) → CIE Lab float (L 0-100), D65."""
    a = np.asarray(bgr, np.float32) / 255.0
    lab = cv2.cvtColor(a.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).reshape(a.shape)
    return lab


def rgb_to_lab(rgb) -> np.ndarray:
    r, g, b = rgb
    return bgr_to_lab(np.array([[[b, g, r]]], np.uint8))[0, 0]


def delta_e00(lab1, lab2) -> np.ndarray:
    """CIEDE2000, vectorizado sobre el último eje."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, np.float64), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lm, Cmp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hm = np.where(C1p * C2p == 0, hs,
                  np.where(np.abs(h1p - h2p) <= 180, hs / 2,
                           np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hm - 30)) + 0.24 * np.cos(np.radians(2 * hm))
         + 0.32 * np.cos(np.radians(3 * hm + 6)) - 0.20 * np.cos(np.radians(4 * hm - 63)))
    dth = 30 * np.exp(-((hm - 275) / 25) ** 2)
    Rc = 2 * np.sqrt(Cmp ** 7 / (Cmp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lm - 50) ** 2 / np.sqrt(20 + (Lm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cmp, 1 + 0.015 * Cmp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


_PAL_LAB = {k: rgb_to_lab(v) for k, v in PALETA.items()}


def nombre_de(lab) -> str:
    """El nombre de la paleta más cercano (solo para mostrar)."""
    if lab is None:
        return ""
    return min(_PAL_LAB, key=lambda k: float(delta_e00(lab, _PAL_LAB[k])))


def lab_to_bgr(lab) -> tuple[int, int, int]:
    a = np.array([[lab]], np.float32)
    bgr = cv2.cvtColor(a, cv2.COLOR_Lab2BGR)[0, 0]
    return tuple(int(np.clip(v * 255, 0, 255)) for v in bgr)


def parse_codigo(texto: str) -> list[tuple[str, Optional[str]]]:
    """
    «1 AZ · 2 AM · 3 BL/RJ» o «AZ, AM, BL/RJ» → [(base, traza|None), …] en
    orden de pin. Lo que no se reconoce queda como ("?", None) para que la
    persona lo vea, no se descarta.
    """
    out = []
    for tok in re.split(r"[,;·|\n]+|\s+(?=\d+\s*[:.\-)]?\s*[A-Za-zÁÉÍÓÚáéíóú])", texto or ""):
        tok = re.sub(r"^\s*\d+\s*[:.\-)]?\s*", "", tok).strip()
        if not tok:
            continue
        partes = [p.strip().upper() for p in re.split(r"[/+]", tok) if p.strip()]
        base = CODIGOS.get(partes[0], "?") if partes else "?"
        traza = CODIGOS.get(partes[1], "?") if len(partes) > 1 else None
        out.append((base, traza))
    return out


def texto_de(nombres: list[tuple[str, Optional[str]]]) -> str:
    return " · ".join(f"{i + 1} " + ABREV.get(b, "?") + (f"/{ABREV.get(t, '?')}" if t else "")
                      for i, (b, t) in enumerate(nombres))


# ── lectura de una cara ───────────────────────────────────────────────────
def _box(d):
    return (float(d.x1), float(d.y1), float(d.x2), float(d.y2))


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def white_gains(frame: np.ndarray, conn_box) -> Optional[np.ndarray]:
    """Ganancias BGR que vuelven neutro el nylon del conector, o None."""
    x1, y1, x2, y2 = [int(round(v)) for v in conn_box]
    h, w = frame.shape[:2]
    roi = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    if roi.size < 30:
        return None
    px = roi.reshape(-1, 3).astype(np.float32)
    lum = px.mean(1)
    chroma = px.max(1) - px.min(1)
    claros = px[(lum >= np.percentile(lum, 70)) & (chroma <= np.percentile(chroma, 60))]
    if len(claros) < 10:
        return None
    ref = np.median(claros, 0)
    if ref.mean() < 60:
        return None
    g = ref.mean() / np.maximum(ref, 1.0)
    return np.clip(g, 0.6, 1.6)


def leer_banda(frame: np.ndarray, conn_box, band_box, gains=None,
               cols_por_hilo: int = 30, n_hint: int = 6) -> Optional[dict]:
    """
    La banda enderezada: hilos verticales saliendo del conector hacia abajo.

    Devuelve {"cols": Lab por columna (W,3), "px": Lab de la franja (H,W,3),
    "angle": grados, "depth_rel": profundidad visible / ancho del conector,
    "white_frac": fracción de columnas blancas sin color} o None.
    """
    cc, bc = _center(conn_box), _center(band_box)
    d0 = np.array([bc[0] - cc[0], bc[1] - cc[1]], np.float64)
    if np.hypot(*d0) < 1:
        return None
    d0 /= np.hypot(*d0)

    # ROI que cubre conector y banda, con margen para girar
    xs = [conn_box[0], conn_box[2], band_box[0], band_box[2]]
    ys = [conn_box[1], conn_box[3], band_box[1], band_box[3]]
    pad = 0.25 * max(max(xs) - min(xs), max(ys) - min(ys))
    h, w = frame.shape[:2]
    X1, Y1 = int(max(min(xs) - pad, 0)), int(max(min(ys) - pad, 0))
    X2, Y2 = int(min(max(xs) + pad, w)), int(min(max(ys) + pad, h))
    roi = frame[Y1:Y2, X1:X2].astype(np.float32)
    if roi.shape[0] < 8 or roi.shape[1] < 8:
        return None
    if gains is not None:
        roi = np.clip(roi * gains[None, None, :], 0, 255)

    # dirección fina: los hilos son líneas paralelas → tensor de estructura
    bx1, by1, bx2, by2 = [int(round(v)) for v in band_box]
    sub = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    sub = sub[max(by1 - Y1, 0):max(by2 - Y1, 1), max(bx1 - X1, 0):max(bx2 - X1, 1)]
    ang = np.degrees(np.arctan2(d0[1], d0[0]))
    if sub.shape[0] > 6 and sub.shape[1] > 6:
        gx = cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3)
        jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
        if jxx + jyy > 1e-3:
            # orientación del gradiente dominante; los hilos van perpendiculares
            th_grad = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
            th_hilo = th_grad + 90.0
            # de las dos direcciones del hilo, la que va como d0
            cand = [th_hilo, th_hilo + 180.0]
            fino = min(cand, key=lambda a: abs(((a - ang + 180) % 360) - 180))
            if abs(((fino - ang + 180) % 360) - 180) < 35:
                ang = fino

    # girar para que d apunte hacia abajo (+y)
    cx, cy = cc[0] - X1, cc[1] - Y1
    M = cv2.getRotationMatrix2D((cx, cy), ang - 90.0, 1.0)
    big = int(np.hypot(*roi.shape[:2])) + 4
    M[0, 2] += big / 2 - cx
    M[1, 2] += big / 2 - cy
    rot = cv2.warpAffine(roi, M, (big, big), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def tr(pts):
        p = np.hstack([np.asarray(pts, np.float64) - [X1, Y1], np.ones((len(pts), 1))])
        return p @ M.T

    # Las cajas de YOLO van alineadas a los ejes: de una pieza inclinada
    # salen más grandes que la pieza y meten fondo en la franja. Se
    # recupera el rectángulo de verdad con el ángulo (la caja de un
    # rectángulo a×b girado t mide a·cos t + b·sin t por b·cos t + a·sin t).
    def lados(box):
        t = np.radians(((ang + 45) % 90) - 45)
        c, s = abs(np.cos(t)), abs(np.sin(t))
        Wb, Hb = box[2] - box[0], box[3] - box[1]
        den = max(c * c - s * s, 0.2)
        a = max((Wb * c - Hb * s) / den, 1.0)      # el lado más cerca del eje x
        b = max((Hb * c - Wb * s) / den, 1.0)
        dx = abs(np.cos(np.radians(ang))) >= abs(np.sin(np.radians(ang)))
        return (b, a) if dx else (a, b)            # (a lo ancho, a lo largo)

    conn_w, conn_l = lados(conn_box)
    band_w, band_l = lados(band_box)
    bcu = tr([bc])[0]
    across0, across1 = bcu[0] - band_w / 2, bcu[0] + band_w / 2
    y_conn = big / 2 + conn_l / 2                  # el centro del conector va al centro
    y_band_end = bcu[1] + band_l / 2
    depth = y_band_end - y_conn
    if band_w < 6 or depth < 2:
        return None
    # franja: del 12 % al 50 % de la profundidad visible
    ya = int(round(y_conn + 0.12 * depth))
    yb = int(round(y_conn + max(0.5 * depth, 3)))
    xa, xb = int(round(across0)), int(round(across1))
    strip = rot[max(ya, 0):max(yb, ya + 2), max(xa, 0):max(xb, xa + 2)]
    if strip.size == 0 or strip.shape[0] < 2 or strip.shape[1] < 4:
        return None
    W = max(cols_por_hilo * max(n_hint, 1), 24)
    H = max(6, min(24, strip.shape[0]))
    strip = cv2.resize(strip, (W, H), interpolation=cv2.INTER_AREA)
    lab = bgr_to_lab(np.clip(strip, 0, 255).astype(np.uint8))
    cols = np.median(lab, axis=0)
    C = np.hypot(cols[:, 1], cols[:, 2])
    white = ((cols[:, 0] > 78) & (C < 9)).mean()

    # El fondo (el tapete) se ve ENTRE los hilos: un AWG 26 mide 1.35 mm y
    # el paso es 2.0. Se mide a los lados de la banda, a la misma altura y
    # con la misma luz, y las columnas de ese color no cuentan como hilo.
    p = band_w / max(n_hint, 1)
    lados_px = []
    for x0, x1 in ((xa - 0.8 * p, xa - 0.25 * p), (xb + 0.25 * p, xb + 0.8 * p)):
        x0, x1 = int(round(max(x0, 0))), int(round(min(x1, rot.shape[1])))
        if x1 > x0:
            blk = rot[max(ya, 0):max(yb, ya + 2), x0:x1].reshape(-1, 3)
            blk = blk[blk.sum(1) > 3]               # fuera del cuadro girado = negro puro
            if len(blk):
                lados_px.append(blk)
    fondo = None
    peso = np.ones(len(cols))
    if lados_px:
        fondo = bgr_to_lab(np.clip(np.median(np.vstack(lados_px), 0), 0, 255)
                           .astype(np.uint8)[None, None, :])[0, 0]
        peso = (delta_e00(cols, fondo) > 10.0).astype(np.float64)
        if peso.sum() < 0.3 * len(cols):           # el «fondo» era otra cosa: no se usa
            peso = np.ones(len(cols))
            fondo = None
        else:
            peso = _huecos_en_frontera(peso, n_hint)
    return {"cols": cols, "px": lab, "angle": float(ang), "peso": peso,
            "fondo": None if fondo is None else [float(v) for v in fondo],
            "depth_rel": float(depth / max(conn_w, 1.0)), "white_frac": float(white),
            "aspect": float(conn_w / max(conn_l, 1.0)),
            "strip_bgr": strip.astype(np.uint8)}


def segmentar(cols: np.ndarray, n: int, prior: Optional[list] = None,
              lam: float = 1500.0, min_frac: float = 0.6,
              peso: Optional[np.ndarray] = None) -> list[int]:
    """
    Cortes [0, c1, …, W] que parten el perfil en *n* tramos de color
    parejo. *prior*: cortes relativos (0..1) de donde se espera cada uno;
    alejarse cuesta lam·(desvío en tramos)². Programación dinámica.

    El precio del prior está en la escala del costo de color (Lab², por
    columna): una columna de otro color dentro de un tramo cuesta cientos
    o miles; desviarse un tercio de tramo cuesta ~170. Así el prior no le
    gana a un borde de color de verdad, pero desempata entre dos hilos del
    mismo color, donde el color no dice nada.
    """
    W = len(cols)
    if n <= 0 or W < n:
        return []
    # la luz pesa menos que el tono: la sombra del cilindro no es un hilo
    x = cols.astype(np.float64) / np.array([2.0, 1.0, 1.0])
    w = np.ones(W) if peso is None else np.asarray(peso, np.float64)
    # las columnas de fondo pesan 0: no ensucian ningún tramo, y un corte en
    # el hueco entre dos hilos cuesta lo mismo en cualquier punto del hueco
    s0 = np.concatenate([[0.0], np.cumsum(w)])
    s1 = np.vstack([np.zeros((1, 3)), np.cumsum(x * w[:, None], 0)])
    s2 = np.concatenate([[0.0], np.cumsum((x * x).sum(1) * w)])

    def coste(i, j):          # varianza total de cols[i:j], vectorizado en i
        m = s0[j] - s0[i]
        sx = s1[j] - s1[i]
        return s2[j] - s2[i] - (sx * sx).sum(-1) / np.maximum(m, 1e-9)

    paso = W / n
    minw = max(1, int(min_frac * paso))
    INF = 1e18
    # Si el tapete se ve entre los hilos, los cortes van EN los huecos: sin
    # esto, un hilo con traza junto a uno del color de la traza (blanco/rojo
    # al lado de rojo) pierde su raya, que se va con el vecino. Solo si hay
    # huecos para todos los cortes; si los hilos se tocan, como antes.
    fuera_hueco = np.zeros(W + 1)
    con_huecos = False
    if peso is not None and (w == 0).any():
        hueco = w == 0
        en_hueco = np.zeros(W + 1, bool)
        en_hueco[1:W] = hueco[:-1] | hueco[1:]
        orillas = np.flatnonzero(np.diff(hueco.astype(np.int8)) == 1)   # hilo → hueco
        interiores = [k for k in orillas if 0 < k and hueco[k + 1:].size and (~hueco[k + 1:]).any()]
        if len(interiores) >= n - 1:
            fuera_hueco = np.where(en_hueco, 0.0, 1e12)
            con_huecos = True
    idx = np.arange(W + 1)

    def resolver(ventana):
        best = np.full((n + 1, W + 1), INF)
        arg = np.zeros((n + 1, W + 1), np.int64)
        best[0, 0] = 0.0
        for k in range(1, n + 1):
            for j in (range(k * minw, W + 1) if k < n else (W,)):
                if k < n and ventana is not None and abs(j - ventana[k]) > ventana_cols:
                    continue
                i = idx[(k - 1) * minw: j - minw + 1]
                if len(i) == 0:
                    continue
                c = best[k - 1, i] + coste(i, j)
                if prior is not None and k < n:
                    c = c + lam * ((j - prior[k] * W) / paso) ** 2
                if k < n:
                    c = c + fuera_hueco[j]
                m = int(np.argmin(c))
                best[k, j] = c[m]
                arg[k, j] = i[m]
        cortes = [W]
        j = W
        for k in range(n, 0, -1):
            j = int(arg[k, j])
            cortes.append(j)
        return best[n, W], cortes[::-1]

    # Hilos que se tocan (sin tapete entre ellos): cada hilo ocupa su paso,
    # porque sale del conector al paso de los pines. Sin esto la varianza de
    # color manda y un hilo con traza junto a uno del color de la traza
    # (blanco/rojo pegado al rojo) se queda angosto: su raya se va con el
    # vecino, que sale "rojo/gris". Cada corte se mueve poco alrededor del
    # suyo, y todos juntos un poco (la caja de la banda no es exacta).
    if prior is None or con_huecos or n < 2:
        return resolver(None)[1]
    ventana_cols = 0.15 * paso
    mejor = None
    for corr in np.linspace(-0.2, 0.2, 9):
        v = [prior[k] * W + corr * paso for k in range(n + 1)]
        r = resolver(v)
        if r[0] < INF / 2 and (mejor is None or r[0] < mejor[0]):
            mejor = r
    return (mejor or resolver(None))[1]


def _huecos_en_frontera(peso: np.ndarray, n: int, lejos: float = 0.3) -> np.ndarray:
    """
    El tapete entre hilos cae en la FRONTERA entre dos hilos (a múltiplos del
    paso). Una racha "de fondo" lejos de toda frontera está dentro de un
    hilo: es una traza de un color parecido al tapete (azul oscuro sobre un
    tapete azul gris), no tapete. Se devuelve como hilo; si no, la traza se
    borra y la pieza buena sale "no se ve la traza", o la mala pasa.
    """
    peso = np.asarray(peso, np.float64).copy()
    W = len(peso)
    pc = W / max(n, 1)
    i = 0
    while i < W:
        if peso[i] > 0:
            i += 1
            continue
        j = i
        while j < W and peso[j] == 0:
            j += 1
        centro = (i + j) / 2.0                    # en columnas
        d = abs(centro / pc - round(centro / pc)) * pc
        if i > 0 and j < W and d > lejos * pc:
            peso[i:j] = 1.0
        i = j
    return peso


def color_tramo(px: np.ndarray, cols: np.ndarray, a: int, b: int,
                umbral_traza: float = 16.0, peso: Optional[np.ndarray] = None) -> dict:
    """
    Color base y traza de un tramo [a, b).

    Solo cuentan las columnas de hilo: ni el fondo entre hilos ni las dos
    columnas de transición a cada lado de un hueco (mitad hilo, mitad
    tapete: leídas solas parecen una traza del color del tapete). Una traza
    es un grupo de columnas que cambia de TONO (o mucho de luz), no la
    sombra suave del cilindro.

    ¿Cuál es la base y cuál la traza? La que cubre más CIRCUNFERENCIA, no
    más columnas: vista de frente, una raya centrada ocupa muchas columnas
    del centro con poco arco, y la base, pocas columnas de orilla con mucho
    arco. Cada columna pesa 1/cos del ángulo de la superficie en ese punto.
    """
    idx = np.arange(a, b)
    sel = idx
    if peso is not None:
        pw = np.asarray(peso) > 0
        hilo = idx[pw[a:b]]
        if len(hilo) >= 2:
            idx = np.arange(hilo[0], hilo[-1] + 1)
        # erosión: fuera las columnas a menos de 2 de un hueco
        ero = pw.copy()
        for k in range(1, max(2, int(round(0.08 * (b - a)))) + 1):
            ero[k:] &= pw[:-k]
            ero[:-k] &= pw[k:]
        sel = idx[ero[idx]]
        if len(sel) < 3:
            sel = idx[pw[idx]] if pw[idx].sum() >= 2 else idx
    w = len(idx)
    sub = cols[sel]
    u = (sel - idx[0] + 0.5) / max(w, 1)              # 0..1 a lo ancho del hilo
    arco = 1.0 / np.sqrt(np.maximum(1.0 - (2 * u - 1) ** 2, 0.12))
    base = np.median(sub, 0)

    def dist(c, ref):
        return np.sqrt(((c[..., 0] - ref[0]) / 2.5) ** 2 + (c[..., 1] - ref[1]) ** 2
                       + (c[..., 2] - ref[2]) ** 2)

    lejos = dist(sub, base) > umbral_traza
    traza = None
    frac = float(lejos.mean())
    if 0.1 <= frac <= 0.75 and lejos.sum() >= 2 and (~lejos).sum() >= 2:
        c_lejos = np.median(sub[lejos], 0)
        c_cerca = np.median(sub[~lejos], 0)
        # una «traza» del mismo color que la base es la sombra del cilindro:
        # en un arnés la traza siempre es de otro color
        if (float(dist(c_lejos[None], c_cerca)[0]) > umbral_traza
                and nombre_de(c_lejos) != nombre_de(c_cerca)):
            if arco[lejos].sum() > arco[~lejos].sum():
                base, traza, es_traza = c_lejos, c_cerca, ~lejos
                frac = 1.0 - frac
            else:
                base, traza, es_traza = c_cerca, c_lejos, lejos
            # una traza va DENTRO del hilo: si solo aparece en la orilla es
            # la mezcla con el hilo de al lado (rojo junto a negro = «café»)
            r_ = np.arange(len(sel)) / max(len(sel) - 1, 1)
            if (es_traza & (r_ >= 0.15) & (r_ <= 0.85)).sum() < 2:
                traza, frac = None, 0.0
        else:
            frac = 0.0
    else:
        frac = 0.0
    return {"base": [float(v) for v in base],
            "traza": [float(v) for v in traza] if traza is not None else None,
            "frac": round(frac, 3)}


# ── módulo ────────────────────────────────────────────────────────────────
class HarnessModule(InspectionModule):
    name = "Arnés"
    key = "harness"
    pro = True

    def __init__(self) -> None:
        super().__init__()
        #: clases del modelo. Las caras se reconocen por sufijo (conector_a,
        #: pud_b…). Sin prefijo vale cualquier clase con sufijo de cara.
        self.target_class = ""
        self.band_class = "hilos"
        #: filas de pines del conector. Con dos filas cada cara ve UNA fila:
        #: JST numera impares en una y pares en la otra; *fila_a* dice cuál
        #: queda del lado de la cara A ("impares" o "pares").
        self.filas = 1
        self.fila_a = "impares"
        #: hilos de la pieza que pidió el job (antes de tener golden)
        self.n_hilos = 0
        self.face_suffix = {"a": "_a", "b": "_b"}
        #: golden por cara: {"a": {"n", "cuts", "colors": [{base, traza, frac}], "depth"}}
        self.golden: dict = {}
        #: codificación escrita, en orden de pin
        self.codigo: str = ""
        #: verificación de una persona: {"por", "en", "huella"}
        self.verificado: dict = {}
        self.tolerance = 11.0                   # ΔE2000
        self.traza_pen = 14.0                   # tener/no tener traza
        self.aspect_tol = 0.12                  # ancho/largo del conector contra el golden
        self.white_balance = True
        self.min_conf = 0.30
        self.settle_s = 0.35
        self.window_s = 0.6
        self.gone_s = 0.6
        self.motion_min = 0.004
        self._learn: dict = {}
        self._fonts: dict = {}
        self.counts = {"ok": 0, "corregida": 0, "rechazo": 0}
        self.reset()

    # ── estado ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.state = "vacio"
        self.face: Optional[str] = None
        self.verdict: Optional[str] = None
        self.defects: dict = {}                  # cara → [textos]
        self.face_ok: dict = {}                  # cara → bool
        self.traza_vista: set = set()            # pines cuya traza ya se vio
        self._traza_cnt: dict = {}
        self.last_closed: Optional[str] = None
        self.last_read: Optional[dict] = None
        self.last_eval: Optional[list] = None
        self._history: list = []
        self._votes: list = []
        self._still_since = None
        self._absent_since = None
        self._window_t0 = 0.0
        self._prev = None
        self._conn = None
        self._band = None

    def reset_counts(self) -> None:
        self.counts = {"ok": 0, "corregida": 0, "rechazo": 0}
        self.reset()

    @property
    def calibrated(self) -> bool:
        return bool(self.golden)

    @property
    def faces_required(self) -> list[str]:
        if self.filas == 2:
            return ["a", "b"]           # cada cara trae la mitad de los pines
        return sorted(self.golden)

    @property
    def n_total(self) -> int:
        """Hilos de la pieza (las dos filas)."""
        if not self.golden:
            return 0
        return next(iter(self.golden.values()))["n"] * self.filas

    def faltan_caras(self) -> list[str]:
        return [f for f in self.faces_required if f not in self.golden]

    @property
    def paused(self) -> bool:
        return self.state in ("movimiento", "asentando", "vacio")

    # ── huella y verificación ─────────────────────────────────────────────
    def huella(self) -> str:
        g = {f: {"n": v["n"], "colors": [[round(x, 1) for x in c["base"]]
                                         + ([round(x, 1) for x in c["traza"]] if c["traza"] else [])
                                         for c in v["colors"]]}
             for f, v in sorted(self.golden.items())}
        d = {"g": g, "t": self.codigo.strip()}
        if self.filas == 2:          # (una fila: la huella de siempre)
            d["filas"] = [2, self.fila_a]
        s = json.dumps(d, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @property
    def verified(self) -> bool:
        return bool(self.golden) and self.verificado.get("huella") == self.huella()

    def verificar(self, por: str) -> bool:
        """Una persona revisó golden y codificación: se firma con su nombre."""
        if not self.golden or not (por or "").strip() or self.faltan_caras():
            return False
        self.verificado = {"por": por.strip(), "en": datetime.now().isoformat(timespec="minutes"),
                           "huella": self.huella()}
        return True

    def pines_de(self, cara: str) -> list[int]:
        """Pines que se ven desde la cara, en orden de pin."""
        g = self.golden.get(cara)
        if not g:
            return []
        return sorted(self._pin(cara, i, g["n"]) for i in range(g["n"]))

    def nombres(self, cara: str) -> list[tuple[str, Optional[str]]]:
        """Nombres sugeridos de la cara, en orden de PIN (la B va espejada)."""
        g = self.golden.get(cara)
        if not g:
            return []
        esp = self.trazas()
        n = g["n"]
        por_pin = {}
        for i, c in enumerate(g["colors"]):
            p = self._pin(cara, i, n)
            por_pin[p] = (nombre_de(c["base"]), nombre_de(esp[p]) if p in esp else None)
        return [por_pin[p] for p in sorted(por_pin)]

    def cotejar(self) -> list[str]:
        """
        Diferencias entre lo que Eyve aprendió y la codificación escrita.
        Es ayuda para la persona que verifica, no un veto: los nombres
        dependen de la luz; el golden, no.
        """
        txt = parse_codigo(self.codigo)
        avisos = []
        esp = self.trazas()
        for f in self.faltan_caras():
            avisos.append(f"falta la cara {f.upper()}: sin ella no se revisan los pines "
                          + ("pares" if (f == "b") == (self.fila_a == "impares") else "impares"))
        for cara in sorted(self.golden):
            vistos = self.nombres(cara)
            pines = self.pines_de(cara)
            n_txt = len(vistos) * self.filas
            if txt and len(txt) != n_txt:
                avisos.append(f"cara {cara.upper()}: el texto tiene {len(txt)} hilos y Eyve ve {n_txt}")
                continue
            for p, (vb, _) in zip(pines, vistos):
                if p > len(txt):
                    continue
                tb = txt[p - 1][0]
                if tb != vb:
                    avisos.append(f"cara {cara.upper()} pin {p}: texto {ABREV.get(tb, tb)}"
                                  f", Eyve ve {ABREV.get(vb, vb)}")
        if txt and self.n_total and len(txt) != self.n_total:
            return avisos
        # la traza se coteja por pin: basta con que una cara la haya visto
        for i, (_, tt) in enumerate(txt):
            vt = nombre_de(esp[i + 1]) if (i + 1) in esp else None
            if (tt or None) != vt:
                avisos.append(f"pin {i + 1}: traza en el texto {ABREV.get(tt, tt) if tt else 'ninguna'}"
                              f", Eyve ve {ABREV.get(vt, vt) if vt else 'ninguna'}")
        return avisos

    # ── detecciones ───────────────────────────────────────────────────────
    def _face_of(self, cls: str) -> Optional[str]:
        if cls == self.band_class or not cls.startswith(self.target_class or ""):
            return None
        for f, suf in self.face_suffix.items():
            if cls.endswith(suf):
                return f
        return None

    def _split(self, detections):
        ok = [d for d in detections if getattr(d, "confidence", 1.0) >= self.min_conf]
        conns = [(d, self._face_of(d.class_name)) for d in ok]
        conns = [(d, f) for d, f in conns if f]
        if not conns:
            return None, None, None
        d, face = max(conns, key=lambda p: p[0].confidence)
        cb = _box(d)
        cw, ch = cb[2] - cb[0], cb[3] - cb[1]
        best = None
        for b in ok:
            if b.class_name != self.band_class:
                continue
            bb = _box(b)
            cx, cy = _center(bb)
            if (cb[0] - 1.2 * cw <= cx <= cb[2] + 1.2 * cw
                    and cb[1] - 1.2 * ch <= cy <= cb[3] + 1.2 * ch):
                if best is None or b.confidence > best[1]:
                    best = (bb, b.confidence)
        return cb, face, (best[0] if best else None)

    # ── leer y comparar ───────────────────────────────────────────────────
    def read_face(self, frame, conn_box, band_box, n: int, prior=None) -> Optional[dict]:
        gains = white_gains(frame, conn_box) if self.white_balance else None
        r = leer_banda(frame, conn_box, band_box, gains=gains, n_hint=n)
        if r is None:
            return None
        if prior is None:          # aprender: los hilos salen al paso de los pines
            prior = [k / n for k in range(n + 1)]
        cuts = segmentar(r["cols"], n, prior=prior, peso=r["peso"])
        if len(cuts) != n + 1:
            return None
        r["cuts"] = cuts
        r["colors"] = [color_tramo(r["px"], r["cols"], cuts[i], cuts[i + 1], peso=r["peso"])
                       for i in range(n)]
        return r

    def _pin(self, cara: str, i: int, n: int) -> int:
        """
        Posición i (de izquierda a derecha en la cara) → número de pin; *n*
        es cuántos hilos ve esa cara. La B es la pieza girada 180° sobre el
        eje de los hilos: el orden de las columnas se invierte.

        Dos filas (JST PUD y similares): impares en una fila y pares en la
        otra, el 1 y el 2 en la misma columna. La cara A ve la fila
        *fila_a*; la B, la otra.
        """
        if self.filas != 2:
            return i + 1 if cara == "a" else n - i
        col = i + 1 if cara == "a" else n - i            # columna 1..n
        impar = (cara == "a") == (self.fila_a == "impares")
        return 2 * col - 1 if impar else 2 * col

    @staticmethod
    def _pareja(p: int) -> int:
        """El pin de la otra fila en la misma columna (1↔2, 3↔4…)."""
        return p + 1 if p % 2 else p - 1

    def _golden_pin(self, p: int) -> Optional[dict]:
        for cara in sorted(self.golden):
            g = self.golden[cara]
            for i, c in enumerate(g["colors"]):
                if self._pin(cara, i, g["n"]) == p:
                    return c
        return None

    def trazas(self) -> dict:
        """
        pin → color de traza esperado. Una traza se espera si se vio al
        aprender en CUALQUIER cara: en un conector de una fila la raya puede
        quedar del lado de abajo, y por eso se pide la otra cara.
        """
        out = {}
        for cara in sorted(self.golden):
            g = self.golden[cara]
            for i, c in enumerate(g["colors"]):
                if c["traza"]:
                    out.setdefault(self._pin(cara, i, g["n"]), c["traza"])
        return out

    def _dist(self, c, g, traza_esp=None) -> float:
        """
        Distancia de un tramo leído *c* al golden *g* de una posición cuyo
        pin espera *traza_esp*. Ver una traza que no va cuesta; NO verla
        cuando va no cuesta aquí (puede estar del otro lado): se cobra al
        cerrar la pieza si ninguna cara la vio.
        """
        d = float(delta_e00(c["base"], g["base"]))
        if c["traza"] is not None:
            if traza_esp is None:
                d += self.traza_pen
            else:
                d += 0.5 * float(delta_e00(c["traza"], traza_esp))
        return d

    def compare(self, cara: str, leido: dict) -> list[str]:
        """Defectos de una cara leída contra su golden (vacío = OK)."""
        # Mismo conector, otro tamaño (un 2×5 en el job del 2×4): los hilos
        # se reparten en N tramos y el color podría cuadrar por casualidad.
        # El ancho del conector contra su largo lo delata (8 → 10 pines: +20 %).
        ga = self.golden[cara].get("aspect") or 0.0
        la = leido.get("aspect") or 0.0
        if ga > 0 and la > 0 and abs(la / ga - 1.0) > self.aspect_tol:
            self.last_eval = []
            return [f"el conector es más {'ancho' if la > ga else 'angosto'} que el del job"
                    f" (¿otro número de parte?)"]
        g = self.golden[cara]["colors"]
        c = leido["colors"]
        n = len(g)
        pin = lambda i: self._pin(cara, i, n)            # noqa: E731
        esp = self.trazas()
        tz = [esp.get(pin(j)) for j in range(n)]
        D = np.array([[self._dist(c[i], g[j], tz[j]) for j in range(n)] for i in range(n)])

        def dg(i, j):
            d = float(delta_e00(g[i]["base"], g[j]["base"]))
            if (tz[i] is None) != (tz[j] is None):
                d += self.traza_pen
            elif tz[i] is not None:
                d += 0.5 * float(delta_e00(tz[i], tz[j]))
            return d
        G = np.array([[dg(i, j) for j in range(n)] for i in range(n)])
        malos = [i for i in range(n) if D[i, i] > self.tolerance]
        self.last_eval = [(i, float(D[i, i])) for i in range(n)]
        for i in range(n):
            if c[i]["traza"] is not None and tz[i] is not None and D[i, i] <= self.tolerance:
                self._traza_cnt[pin(i)] = self._traza_cnt.get(pin(i), 0) + 1
        out, usados = [], set()
        for i in malos:
            if i in usados:
                continue
            par = [j for j in malos if j != i and j not in usados
                   and D[i, j] <= self.tolerance and D[j, i] <= self.tolerance
                   and G[i, j] > self.tolerance]
            if par:
                j = par[0]
                usados |= {i, j}
                a, b = sorted((pin(i), pin(j)))
                out.append(f"cruzados {a}↔{b}")
            else:
                usados.add(i)
                if self.filas == 2:
                    # ¿es el hilo de la otra fila, misma columna? Encimado
                    # (se reacomoda) o cruzado entre filas: lo dice la otra cara
                    q = self._pareja(pin(i))
                    gq = self._golden_pin(q)
                    if gq is not None and self._dist(c[i], gq, esp.get(q)) <= self.tolerance:
                        out.append(f"pin {pin(i)}: hilo encimado, se ve el del pin {q}")
                        continue
                vb = nombre_de(c[i]["base"]) + (f"/{nombre_de(c[i]['traza'])}" if c[i]["traza"] else "")
                gb = nombre_de(g[i]["base"]) + (f"/{nombre_de(g[i]['traza'])}" if g[i]["traza"] else "")
                out.append(f"pin {pin(i)}: va {gb}, hay {vb}")
        return out

    # ── aprender ──────────────────────────────────────────────────────────
    def learn(self, frame, detections, n: int) -> tuple[Optional[str], int]:
        """Acumula una lectura de la cara que se ve; devuelve (cara, cuántas)."""
        cb, face, bb = self._split(detections)
        n = self.learn_n(n)
        if cb is None or bb is None or n <= 0:
            return face, len(self._learn.get(face, [])) if face else 0
        r = self.read_face(frame, cb, bb, n)
        if r is None:
            return face, len(self._learn.get(face, []))
        self._learn.setdefault(face, []).append(r)
        return face, len(self._learn[face])

    def finish_learning(self, face: Optional[str] = None) -> list[str]:
        """Cierra el aprendizaje de las caras acumuladas; devuelve las aprendidas."""
        hechas = []
        for f, lecturas in list(self._learn.items()):
            if face is not None and f != face:
                continue
            if len(lecturas) < 3:
                continue
            n = len(lecturas[0]["colors"])
            W = len(lecturas[0]["cols"])
            cuts = np.median([np.array(r["cuts"]) / len(r["cols"]) for r in lecturas], 0)
            colors = []
            for i in range(n):
                bases = np.array([r["colors"][i]["base"] for r in lecturas])
                trz = [r["colors"][i]["traza"] for r in lecturas if r["colors"][i]["traza"]]
                tiene = len(trz) >= len(lecturas) / 2
                colors.append({"base": [float(v) for v in np.median(bases, 0)],
                               "traza": [float(v) for v in np.median(trz, 0)] if tiene else None,
                               "frac": float(np.median([r["colors"][i]["frac"] for r in lecturas]))})
            self.golden[f] = {"n": n, "cuts": [float(v) for v in cuts], "colors": colors,
                              "depth": float(np.median([r["depth_rel"] for r in lecturas])),
                              "aspect": float(np.median([r.get("aspect", 0.0) for r in lecturas])),
                              "cols": W}
            hechas.append(f)
            del self._learn[f]
        if hechas:
            self.reset()
        return hechas

    def forget(self) -> None:
        self.golden = {}
        self._learn = {}
        self.reset()

    def learn_n(self, n_total: int) -> int:
        """Hilos por cara para *n_total* hilos de la pieza."""
        if self.filas == 2:
            return n_total // 2 if n_total % 2 == 0 else 0
        return n_total

    # ── señales ───────────────────────────────────────────────────────────
    def _moving(self, frame) -> bool:
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        moving = False
        if self._prev is not None:
            moving = (cv2.absdiff(gray, self._prev) > 12).mean() >= self.motion_min
        self._prev = gray
        return moving

    def _tapado(self, leido: Optional[dict], cara: str) -> bool:
        """¿El termofit (u otra cosa blanca) tapa la banda?"""
        g = self.golden.get(cara)
        if g is None:
            return False
        if leido is None:
            return True
        if leido["depth_rel"] < 0.35 * g.get("depth", 1.0):
            return True
        blancos_g = np.mean([1.0 if nombre_de(c["base"]) in ("blanco", "gris") else 0.0
                             for c in g["colors"]])
        return leido["white_frac"] > 0.6 and blancos_g < 0.4

    # ── máquina de estados ────────────────────────────────────────────────
    def update(self, frame, detections, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self.last_closed = None
        cb, face, bb = self._split(detections)
        self._conn, self._band = cb, bb
        moving = self._moving(frame)

        if cb is None:
            if self._absent_since is None:
                self._absent_since = now
            if now - self._absent_since >= self.gone_s:
                self._close_piece()
                self.state, self._still_since = "vacio", None
            return
        self._absent_since = None

        if moving:
            self.state, self._still_since, self._votes = "movimiento", None, []
            self.face = face
            return
        if not self.calibrated or not self.verified:
            self.state, self.face = "sin_golden", face
            return
        if face not in self.golden:
            self.state, self.face = "cara_desconocida", face
            return

        if self.state in ("voltea", "resultado") and face == self.face:
            return          # quieta y la misma cara: nada nuevo que leer
        if self.state not in ("asentando", "inspeccionando", "tapado"):
            self.state, self._still_since = "asentando", now
        self.face = face
        if self.state == "asentando" and now - self._still_since >= self.settle_s:
            self.state, self._votes, self._window_t0 = "inspeccionando", [], now
            self._traza_cnt = {}
        if self.state in ("inspeccionando", "tapado"):
            g = self.golden[face]
            prior = g["cuts"]
            leido = None if bb is None else self.read_face(frame, cb, bb, g["n"], prior=prior)
            self.last_read = leido
            if self._tapado(leido, face):
                self.state, self._votes = "tapado", []
                return
            if self.state == "tapado":
                self.state, self._votes, self._window_t0 = "inspeccionando", [], now
            d = self.compare(face, leido)
            self._votes.append(tuple(d))
            if now - self._window_t0 >= self.window_s and len(self._votes) >= 3:
                ok_votos = [v for v in self._votes if not v]
                for p_, k_ in self._traza_cnt.items():
                    if 3 * k_ >= len(self._votes):
                        self.traza_vista.add(p_)
                if len(ok_votos) * 2 >= len(self._votes):
                    self.face_ok[face], self.defects[face] = True, []
                else:
                    self.face_ok[face] = False
                    self.defects[face] = list(Counter(v for v in self._votes if v)
                                              .most_common(1)[0][0])
                self._settle_verdict()

    _ENCIMADO = re.compile(r"pin (\d+): hilo encimado, se ve el del pin (\d+)")

    def _settle_verdict(self) -> None:
        faltan = [f for f in self.faces_required if f not in self.face_ok]
        self.defects.pop("*", None)
        if self.filas == 2:
            # pin p muestra el hilo de q en una cara y q el de p en la otra:
            # no es un hilo encimado, están cruzados entre filas
            vistos = {}
            for f in ("a", "b"):
                for d in self.defects.get(f, []):
                    m = self._ENCIMADO.match(d)
                    if m:
                        vistos[(int(m.group(1)), int(m.group(2)))] = (f, d)
            for (p, q), (f, d) in list(vistos.items()):
                if p < q and (q, p) in vistos:
                    f2, d2 = vistos[(q, p)]
                    self.defects[f] = [x for x in self.defects[f] if x != d]
                    self.defects[f2] = [x for x in self.defects[f2] if x != d2]
                    self.defects.setdefault("*", []).append(f"cruzados {p}↔{q} (entre filas)")
        if any(v is False for v in self.face_ok.values()):
            self.verdict = "NOT_OK"
        elif faltan:
            self.verdict = None
            self.state = "voltea"
            return
        else:
            # con todas las caras vistas: la traza que va, ¿se vio en alguna?
            esp = self.trazas()
            sin = sorted(p for p in esp if p not in self.traza_vista)
            if sin:
                self.verdict = "NOT_OK"
                self.defects.setdefault("*", [])
                self.defects["*"] += [
                    f"pin {p}: no se ve la traza (va {nombre_de(self._base_pin(p))}/{nombre_de(esp[p])})"
                    for p in sin]
            else:
                self.verdict = "OK"
        self._history.append(self.verdict == "OK")
        self.state = "resultado"

    def _close_piece(self) -> None:
        if self._history:
            if not self._history[-1] or self.verdict == "NOT_OK":
                out = "rechazo"
            elif len(self._history) > 1 and not all(self._history):
                out = "corregida"
            else:
                out = "ok"
            self.counts[out] += 1
            self.last_closed = out
        self._history = []
        self.verdict = None
        self.defects = {}
        self.face_ok = {}
        self.traza_vista = set()
        self.face = None

    # ── texto ─────────────────────────────────────────────────────────────
    def _base_pin(self, p: int):
        c = self._golden_pin(p)
        return c["base"] if c else None

    def encimados(self) -> bool:
        """¿Todo lo malo son hilos encimados? Entonces se reacomoda, no se rechaza."""
        ds = [d for v in self.defects.values() for d in v]
        return bool(ds) and all(self._ENCIMADO.match(d) for d in ds)

    def defects_text(self) -> str:
        partes = []
        for f in sorted(self.defects):
            for d in self.defects[f]:
                partes.append(d if f == "*" else f"cara {f.upper()}: {d}")
        return "; ".join(partes)

    def summary(self) -> str:
        c = self.counts
        return f"OK {c['ok']} · Corregidas {c['corregida']} · Rechazo {c['rechazo']}"

    def state_text(self) -> str:
        if not self.calibrated:
            return "Sin golden · muestra una pieza buena y pulsa Aprender cara"
        if self.faltan_caras():
            return f"Falta aprender la cara {', '.join(f.upper() for f in self.faltan_caras())}"
        if not self.verified:
            return "Golden sin verificar · una persona debe revisarlo"
        faltan = [f.upper() for f in self.faces_required if f not in self.face_ok]
        return {"vacio": "Esperando arnés",
                "movimiento": "Arnés en movimiento · en pausa",
                "asentando": f"Leyendo cara {(self.face or '?').upper()}…",
                "inspeccionando": f"Leyendo cara {(self.face or '?').upper()}…",
                "tapado": "Hilos tapados",
                "voltea": f"Voltea el arnés · falta cara {', '.join(faltan)}",
                "cara_desconocida": f"Cara {(self.face or '?').upper()} sin golden en este job",
                "sin_golden": "Golden sin verificar",
                "resultado": ""}.get(self.state, "")

    def action_text(self) -> str:
        if self.state == "tapado":
            return "Sube el termofit"
        if self.state == "voltea":
            return "Voltea el arnés"
        if self.verdict is None:
            return ""
        if self.verdict == "OK":
            return ("Corregida → sigue a la línea" if self._history and not all(self._history)
                    else "OK → sigue a la línea")
        txt = self.defects_text() or "NOT OK"
        if self.encimados():
            return f"{txt[0].upper() + txt[1:]} → reacomoda los hilos"
        return f"{txt[0].upper() + txt[1:]} → corregir o a RECHAZO"

    # ── dibujo ────────────────────────────────────────────────────────────
    def _f(self, n):
        if n not in self._fonts:
            from eyve.modules.assembly_module import _font
            self._fonts[n] = _font(n)
        return self._fonts[n]

    def draw(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        s = w / 1280.0
        pills = []
        y = int(22 * s)
        if self.verdict and self.state == "resultado":
            col = _VERDE if self.verdict == "OK" else _ROJO
            pills.append((self.verdict.replace("_", " "), int(58 * s), col, (int(24 * s), y)))
            y += int(92 * s)
        if self.face and self.state not in ("vacio",):
            def estado(f):
                return ("bien" if self.face_ok.get(f) else "mal") if f in self.face_ok else "—"
            caras = "  ·  ".join(f"{f.upper()} {estado(f)}" for f in self.faces_required)
            pills.append((f"Cara {self.face.upper()}   |   {caras}", int(24 * s), _GRIS,
                          (int(24 * s), y)))
            y += int(52 * s)
        st = self.state_text()
        if st:
            col = _AMBAR if self.state in ("tapado", "voltea") else _GRIS
            pills.append((st, int(24 * s), col, (int(24 * s), y)))
            y += int(52 * s)
        txt = self.action_text()
        if txt and self.state != "inspeccionando":
            col = (_VERDE if self.verdict == "OK" else
                   _AMBAR if self.state in ("tapado", "voltea") else _ROJO)
            pills.append((txt, int(30 * s), col, None))
        pills.append((self.summary(), int(24 * s), None, "tr"))
        from eyve.modules.assembly_module import AssemblyModule
        AssemblyModule._pills(self, img, pills)
        # la lectura, pin por pin: lo que se ve contra lo que va
        if (self.last_read is not None and self.face in self.golden
                and self.state in ("inspeccionando", "resultado")
                and len(self.last_read.get("colors", [])) == self.golden[self.face]["n"]):
            self._draw_pines(img, s, int(24 * s), y + int(4 * s))

    def _draw_pines(self, img, s, x0, y0) -> None:
        """
        Tarjeta con dos filas de cuadros, en orden de pin: arriba lo que se
        LEYÓ en esta cara, abajo lo que VA según el golden. El pin que no
        cuadra va enmarcado en rojo. Así se ve de un vistazo qué hilo está
        mal y por cuál hay que cambiarlo.
        """
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return
        g = self.golden[self.face]
        n = g["n"]
        N = n * self.filas                 # pines de la pieza
        malos = {i for i, d in (self.last_eval or []) if d > self.tolerance}
        # la traza que ninguna cara vio es un defecto del PIN: también se marca
        if self.state == "resultado":
            for d_ in self.defects.get("*", []):
                m_ = re.match(r"pin (\d+):", d_)
                if m_:
                    p_ = int(m_.group(1))
                    malos |= {i for i in range(n) if self._pin(self.face, i, n) == p_}
        esp = self.trazas()
        lado = int(max(18, min(34 * s, 520 * s / max(N, 1) - 8 * s)))
        paso = lado + int(8 * s)
        etq = int(54 * s)
        pad = int(12 * s)
        f_num, f_etq = self._f(int(17 * s)), self._f(int(18 * s))
        if f_num is None:
            return
        cab = int(22 * s)
        fila_txt = ""
        if self.filas == 2:
            vis = self.pines_de(self.face)
            fila_txt = (f"Cara {self.face.upper()}: se ve la fila de los "
                        f"{'impares' if vis and vis[0] % 2 else 'pares'}")
            cab += int(26 * s)
        ancho = max(pad * 2 + etq + paso * N, pad * 2 + int(len(fila_txt) * 9.5 * s))
        alto = pad * 2 + cab + lado * 2 + int(10 * s)
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle([x0, y0, x0 + ancho, y0 + alto], radius=int(10 * s),
                            fill=(20, 22, 26, 205))
        ytop = y0 + pad + cab
        ybot = ytop + lado + int(10 * s)
        if fila_txt:
            d.text((x0 + pad, y0 + pad + int(9 * s)), fila_txt, font=f_etq,
                   fill=(235, 200, 90), anchor="lm")
        ynum = ytop - int(13 * s)
        d.text((x0 + pad, ytop + lado / 2), "Visto", font=f_etq, fill=(230, 230, 230), anchor="lm")
        d.text((x0 + pad, ybot + lado / 2), "Va", font=f_etq, fill=(170, 170, 170), anchor="lm")

        def rgb(lab):
            b_, g_, r_ = lab_to_bgr(lab)
            return (r_, g_, b_)

        def cuadro(x, y, base, traza):
            d.rectangle([x, y, x + lado, y + lado], fill=rgb(base))
            if traza is not None:
                t0 = x + int(lado * 0.42)
                d.rectangle([t0, y, t0 + max(3, lado // 5), y + lado], fill=rgb(traza))
            d.rectangle([x, y, x + lado, y + lado], outline=(235, 235, 235, 160), width=1)
        if self.filas == 2:
            # los pines de la otra fila: lo que va, apagado; arriba, sin lectura
            vis = set(self.pines_de(self.face))
            for p_ in range(1, N + 1):
                if p_ in vis:
                    continue
                x = x0 + pad + etq + (p_ - 1) * paso
                d.rectangle([x, ytop, x + lado, ytop + lado], outline=(110, 110, 110), width=1)
                gp = self._golden_pin(p_)
                if gp is not None:
                    cuadro(x, ybot, gp["base"], esp.get(p_))
                    d.rectangle([x, ybot, x + lado, ybot + lado], fill=(20, 22, 26, 150))
                d.text((x + lado / 2, ynum), str(p_), font=f_num, fill=(120, 120, 120), anchor="mm")
        for i in range(n):
            p_ = self._pin(self.face, i, n)
            x = x0 + pad + etq + (p_ - 1) * paso
            c = self.last_read["colors"][i]
            cuadro(x, ytop, c["base"], c["traza"])
            cuadro(x, ybot, g["colors"][i]["base"], esp.get(p_))
            d.text((x + lado / 2, ynum), str(p_), font=f_num,
                   fill=(230, 230, 230), anchor="mm")
            if i in malos:
                m_ = int(4 * s)
                d.rectangle([x - m_, ytop - m_, x + lado + m_, ybot + lado + m_],
                            outline=(230, 40, 40), width=max(2, int(3 * s)))
        img[:] = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    # ── contrato InspectionModule ─────────────────────────────────────────
    def process(self, frame, detections, annotated) -> ModuleVerdict:
        try:
            self.update(frame, detections)
            self.draw(annotated)
            return ModuleVerdict(ok=self.verdict != "NOT_OK", label=self.summary(),
                                 triggered_by=self.defects_text(),
                                 analyzed=1 if self._conn is not None else 0,
                                 details={"state": self.state, "counts": dict(self.counts)})
        except Exception as e:   # nunca lanzar desde process()
            return ModuleVerdict(ok=True, label=f"arnés: {e}")

    def config(self) -> dict:
        c = super().config()
        c.update(golden=json.loads(json.dumps(self.golden)), codigo=self.codigo,
                 verificado=dict(self.verificado), tolerance=self.tolerance,
                 band_class=self.band_class, white_balance=bool(self.white_balance),
                 filas=int(self.filas), fila_a=self.fila_a, target_class=self.target_class,
                 n_hilos=int(self.n_total or self.n_hilos or 0),
                 settle_s=self.settle_s, window_s=self.window_s, gone_s=self.gone_s)
        return c

    def apply_config(self, cfg: dict) -> None:
        super().apply_config(cfg)
        g = cfg.get("golden")
        if isinstance(g, dict):
            try:
                self.golden = {str(f): {"n": int(v["n"]), "cuts": [float(x) for x in v["cuts"]],
                                        "colors": [{"base": [float(x) for x in c["base"]],
                                                    "traza": ([float(x) for x in c["traza"]]
                                                              if c.get("traza") else None),
                                                    "frac": float(c.get("frac", 0.0))}
                                                   for c in v["colors"]],
                                        "depth": float(v.get("depth", 1.0)),
                                        "aspect": float(v.get("aspect", 0.0)),
                                        "cols": int(v.get("cols", 0))}
                               for f, v in g.items()}
            except (KeyError, TypeError, ValueError):
                self.golden = {}
        self.codigo = str(cfg.get("codigo") or "")
        v = cfg.get("verificado")
        self.verificado = dict(v) if isinstance(v, dict) else {}
        if cfg.get("band_class"):
            self.band_class = str(cfg["band_class"])
        self.target_class = str(cfg.get("target_class") or "")
        try:
            self.filas = 2 if int(cfg.get("filas", 1)) == 2 else 1
        except (TypeError, ValueError):
            self.filas = 1
        self.fila_a = "pares" if cfg.get("fila_a") == "pares" else "impares"
        try:
            self.n_hilos = int(cfg.get("n_hilos") or 0)
        except (TypeError, ValueError):
            self.n_hilos = 0
        if "white_balance" in cfg:
            self.white_balance = bool(cfg["white_balance"])
        for k in ("tolerance", "settle_s", "window_s", "gone_s"):
            if k in cfg:
                try:
                    setattr(self, k, float(cfg[k]))
                except (TypeError, ValueError):
                    pass
        self.reset()
