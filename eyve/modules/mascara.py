"""
Máscara aprendida — prototipo (rama mascaras-proto, no registrado en
MODULE_REGISTRY).

Por qué existe
──────────────
Patrón compara el material contra copias de sí mismo, y un fantasma es
justamente una copia pálida del estampado.  La máscara cambia la pregunta:
en vez de «¿esto se parece a su vecino?» pregunta «¿esto es lo que
aprobamos?».  Una celda del estampado se aprende de cuadros buenos, el
operador la ve y la aprueba, y después cada cuadro se compara contra esa
celda puesta en su lugar.

    aprender(cuadros buenos) → una celda, su tolerancia y sus zonas
    vista()                  → lo que el operador mira
    aprobar()                → sin esto, analizar() no marca nada
    analizar(cuadro)         → regiones, con signo: tinta de más / falta

Lo que hace que el fantasma se vea:

  - la DIFERENCIA CON SIGNO.  El fantasma solo oscurece y el hilo del tejido
    sube y baja por igual: promediada a la escala de una celda, la del hilo
    se cancela y la del fantasma se acumula.  Patrón toma valor absoluto
    para poder quedarse con la menor de varias direcciones, y ahí el hilo
    suma en vez de cancelarse.
  - la LUZ se aprende, no se quita cuadro por cuadro.  La división por un
    fondo borroso que usa Patrón se come justo lo que el fantasma tiene de
    baja frecuencia.  La cámara no se mueve: la viñeta es la misma siempre.
  - la TOLERANCIA por punto de la celda.  En la orilla de un motivo una
    décima de píxel da una diferencia enorme; en la tela limpia no.  Cada
    punto pesa por lo quieto que estuvo en los cuadros buenos.
  - las ZONAS, que son lo que hace de esto una máscara: lo que debe llevar
    tinta y lo que debe quedar limpio se juzgan por separado.  Juntas se
    cancelan: una falta de tinta aclara el motivo y oscurece la tela (la
    cruda es más oscura que la impresa), y en un promedio casi no queda nada.

La retícula se trabaja en coordenadas del cuadro con dos vectores, sin
enderezar la imagen: ni interpolación de más ni cajas que girar de regreso.

Lo que NO hace, dicho de frente: necesita que el material se repita con la
misma geometría con la que se aprendió.  Si no calza, avisa (`enganchada`) y
no juzga: mejor no decir nada que marcar un cuadro entero.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: mismo ancho de análisis que Patrón
_WORK_W = 480
#: píxeles de trabajo sin juzgar en la orilla: ahí caen las cuñas de la
#: inclinación, que no son material
_MARGEN = 10
#: celdas por píxel de trabajo.  A 1x, una fase fraccionaria interpola sobre
#: una celda ya muestreada y la orilla del motivo queda borrosa.
_SOBRE = 2
#: pico mínimo de la retícula respecto del mejor, en la autocorrelación
_PICO_REL = 0.5
#: desfase mínimo que se acepta como retícula: por debajo cae el hilo
_LAG_MIN = 14

#: cuántas dispersiones puede estar el pico de un cuadro bueno
_PICO_DISP = 8.0
#: la energía es más ruidosa que la media: pide más
_EXTRA_ENERGIA = 2.5

#: Enganche mínimo: correlación entre el cuadro y la celda puesta en su lugar,
#: como fracción de la que dieron los cuadros buenos.
_ENGANCHE_REL = 0.75

KIND_EXTRA = "tinta de más"
KIND_FALTA = "falta tinta"

SIN_APRENDER = "sin_aprender"
POR_APROBAR = "por_aprobar"
APROBADA = "aprobada"
RECHAZADA = "rechazada"


@dataclass
class Informe:
    """Lo que el operador necesita saber antes de aprobar."""
    cuadros: int = 0
    a1: Tuple[float, float] = (0.0, 0.0)
    a2: Tuple[float, float] = (0.0, 0.0)
    #: correlación media de los cuadros de control contra la máscara
    enganche: float = 0.0
    #: residuo típico de los cuadros de control, en % de la tela limpia
    residuo_pct: float = 0.0
    #: cuadros descartados por no calzar con el resto
    descartados: int = 0
    #: fracción de la celda en cada zona
    zona_tinta: float = 0.0
    zona_tela: float = 0.0
    zona_borde: float = 0.0
    avisos: List[str] = field(default_factory=list)


def _gris(frame: np.ndarray) -> np.ndarray:
    """Gris en coordenadas de trabajo, 0..1, con el desenfoque de Patrón."""
    h, w = frame.shape[:2]
    if w > _WORK_W:
        frame = cv2.resize(frame, (_WORK_W, max(1, int(h * _WORK_W / float(w)))),
                           interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return cv2.GaussianBlur(g, (0, 0), 1.0)


def _dispersion(z: np.ndarray) -> float:
    """Escala robusta de un mapa: un fallo local casi no la mueve."""
    return float(np.median(np.abs(z - float(np.median(z))))) * 1.4826


def _pico_subpixel(ac: np.ndarray, y: int, x: int) -> Tuple[float, float]:
    """Posición del máximo con ajuste cuadrático por eje."""
    fy, fx = float(y), float(x)
    if 0 < y < ac.shape[0] - 1:
        a, b, c = ac[y - 1, x], ac[y, x], ac[y + 1, x]
        d = a - 2 * b + c
        if d < 0:
            fy += float(np.clip(0.5 * (a - c) / d, -1, 1))
    if 0 < x < ac.shape[1] - 1:
        a, b, c = ac[y, x - 1], ac[y, x], ac[y, x + 1]
        d = a - 2 * b + c
        if d < 0:
            fx += float(np.clip(0.5 * (a - c) / d, -1, 1))
    return fy, fx


def _picos_reticula(grises: Sequence[np.ndarray]) -> np.ndarray:
    """
    Picos (dx, dy) de la autocorrelación media, sin sesgo por traslape.

    Se rellena con ceros y se divide entre el área de traslape: la
    autocorrelación circular mezcla la orilla derecha con la izquierda, y
    sin dividir, los picos lejanos salen más bajos que los cercanos y se
    corren hacia el centro.
    """
    h, w = grises[0].shape
    ph, pw = 2 * h, 2 * w
    acc = None
    for g in grises:
        x = g - cv2.GaussianBlur(g, (0, 0), 40)
        x = x - float(x.mean())
        f = np.fft.rfft2(x, (ph, pw))
        p = (f * np.conj(f)).real
        acc = p if acc is None else acc + p
    ac = np.fft.fftshift(np.fft.irfft2(acc, (ph, pw)))
    cy, cx = ph // 2, pw // 2
    r = min(h, w) // 2
    ac = ac[cy - r:cy + r + 1, cx - r:cx + r + 1]
    lags = np.arange(-r, r + 1, dtype=np.float64)
    trasl = np.outer(h - np.abs(lags), w - np.abs(lags))
    ac = (ac / trasl).astype(np.float32)
    # Sin textura no hay nada que repetir: dividir ruido de redondeo entre
    # casi cero daba picos perfectos, y un cuadro liso «aprendía» una retícula.
    if float(ac[r, r]) < 1e-7:
        return np.zeros((0, 3))
    ac /= float(ac[r, r])

    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    d2 = yy * yy + xx * xx
    dil = cv2.dilate(ac, np.ones((5, 5), np.uint8))
    es_pico = (ac >= dil) & (d2 >= _LAG_MIN ** 2) & (d2 <= (r - 2) ** 2)
    # medio plano: cada pico tiene su opuesto
    es_pico &= (yy > 0) | ((yy == 0) & (xx > 0))
    ys, xs = np.nonzero(es_pico)
    if ys.size == 0:
        return np.zeros((0, 3))
    vals = ac[ys, xs]
    # Un pico flojo significa que el material no se repite de verdad: el mismo
    # criterio que Patrón.
    if float(vals.max()) < 0.25:
        return np.zeros((0, 3))
    keep = vals >= _PICO_REL * float(vals.max())
    out = []
    for y, x, v in zip(ys[keep], xs[keep], vals[keep]):
        fy, fx = _pico_subpixel(ac, int(y), int(x))
        out.append((fx - r, fy - r, float(v)))
    return np.array(out, dtype=np.float64)


def _bases_candidatas(picos: np.ndarray, maximo: int = 6) -> list:
    """Pares no colineares de picos, de la celda más chica a la más grande."""
    if len(picos) < 2:
        return []
    orden = np.argsort(np.hypot(picos[:, 0], picos[:, 1]))
    p = picos[orden][:12, :2]
    cand = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            area = abs(p[i, 0] * p[j, 1] - p[i, 1] * p[j, 0])
            n = np.hypot(*p[i]) * np.hypot(*p[j])
            if area < 0.3 * n:          # casi colineares
                continue
            cand.append((area, i, j))
    cand.sort()
    bases, vistas = [], []
    for area, i, j in cand:
        if any(abs(area - a) < 0.02 * a for a in vistas):
            continue
        vistas.append(area)
        bases.append(_reducir(_ajustar_base(np.array([p[i], p[j]]),
                                            picos[:, :2])))
        if len(bases) >= maximo:
            break
    return bases


def _reducir(b: np.ndarray) -> np.ndarray:
    """
    La misma retícula con sus dos vectores más cortos (Gauss).

    La autocorrelación puede dar (48, 0) y (118, 52), que genera la misma
    retícula que (48, 0) y (24, 48) pero con una celda torcida: los puntos de
    la celda quedan estirados y aprende peor.
    """
    a, c = b[0].copy(), b[1].copy()
    for _ in range(32):
        if a @ a > c @ c:
            a, c = c, a
        m = round(float(a @ c) / float(a @ a))
        if m == 0:
            break
        c = c - m * a
    return np.array([a, c])


def _luz_suave(img: np.ndarray, grado: int = 4) -> np.ndarray:
    """
    Campo de luz como polinomio.

    Un desenfoque refleja en la orilla y aplana la caída de la viñeta justo
    donde más cae: las esquinas salían oscuras contra lo aprendido, en todos
    los materiales, y ese era el pico de la tela limpia.
    """
    h, w = img.shape
    ys, xs = np.mgrid[_MARGEN:h - _MARGEN:4, _MARGEN:w - _MARGEN:4]
    terminos = [(i, j) for i in range(grado + 1) for j in range(grado + 1 - i)]

    def diseno(x, y):
        return np.stack([x ** i * y ** j for i, j in terminos], axis=-1)

    x = xs.astype(np.float64) / (w - 1) * 2 - 1
    y = ys.astype(np.float64) / (h - 1) * 2 - 1
    a = diseno(x.ravel(), y.ravel())
    sol, *_ = np.linalg.lstsq(a, img[ys, xs].ravel().astype(np.float64),
                              rcond=None)
    gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    luz = (diseno(gx.ravel(), gy.ravel()) @ sol).reshape(h, w)
    return (luz / float(luz.mean())).astype(np.float32)


def _ajustar_base(base: np.ndarray, picos: np.ndarray) -> np.ndarray:
    """
    Mínimos cuadrados sobre todos los picos que caen en la retícula.

    Un pico solo se ubica a una décima de píxel; a la orilla del cuadro, cinco
    celdas más allá, ese error ya es medio píxel.  Con los picos lejanos en el
    ajuste, el brazo es largo y el error se divide.
    """
    b = base.copy()
    for _ in range(2):
        nm = picos @ np.linalg.inv(b)
        r = np.round(nm)
        ok = np.hypot(*(picos - r @ b).T) < 1.5
        ok &= np.abs(r).sum(axis=1) > 0
        if ok.sum() < 2:
            break
        sol, *_ = np.linalg.lstsq(r[ok], picos[ok], rcond=None)
        if abs(np.linalg.det(sol)) < 1.0:
            break
        b = sol
    return b


class MascaraAprendida:
    """Una celda aprendida del material bueno, que el operador aprueba."""

    ANOMALY_LABEL = "anomalía"

    def __init__(self) -> None:
        #: 0..100, como en Patrón: mueve el múltiplo del piso aprendido
        self.sensibilidad: float = 50.0
        #: área mínima de una región, en % del cuadro
        self.min_area_pct: float = 0.5
        #: seguir los cambios suaves de luz cuadro por cuadro
        self.seguir_luz: bool = True
        self.estado: str = SIN_APRENDER
        self.aprobada_por: str = ""
        self.aprobada_en: str = ""
        self._huella_aprobada: str = ""
        self.informe = Informe()

        self._base: Optional[np.ndarray] = None    # filas a1, a2 (px trabajo)
        self._celda: Optional[np.ndarray] = None   # lo esperado
        self._peso: Optional[np.ndarray] = None    # 1/varianza, media 1
        self._sigma: Optional[np.ndarray] = None   # tolerancia por punto
        self._luz: Optional[np.ndarray] = None     # viñeta aprendida
        self._tinta: Optional[np.ndarray] = None   # zona de tinta, 0 o 1
        self._var: Optional[np.ndarray] = None     # varianza por punto
        #: piso de ruido por zona, medido en cuadros que la celda no vio
        self._piso: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._enganche_ref: float = 0.0
        self._geom = None
        self._f0 = None

        self._regiones: list = []
        self._tipos: list = []
        self._enganchada: bool = True
        self._ultimo_z: Optional[np.ndarray] = None
        self._ultima_ncc: float = 0.0

    # ── geometría de la celda ─────────────────────────────────────────────
    def _preparar(self, shape: Tuple[int, int]) -> None:
        """Lo que no cambia de cuadro a cuadro."""
        h, w = shape
        inv = np.linalg.inv(self._base)
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                             np.arange(h, dtype=np.float64))
        u = (xs * inv[0, 0] + ys * inv[1, 0]).astype(np.float32)
        v = (xs * inv[0, 1] + ys * inv[1, 1]).astype(np.float32)
        margen = np.zeros((h, w), np.float32)
        margen[_MARGEN:h - _MARGEN, _MARGEN:w - _MARGEN] = 1.0
        area = abs(np.linalg.det(self._base))
        k = max(5, int(round(math.sqrt(area))) | 1)
        # Malla para seguir la luz de cada cuadro, lejos de la orilla: ahí la
        # ventana ya no está llena.
        b = _MARGEN + k // 2
        gy, gx = np.mgrid[b:h - b:8, b:w - b:8]
        fx, fy = (xs / (w - 1) * 2 - 1), (ys / (h - 1) * 2 - 1)
        plano = np.stack([np.ones_like(fx), fx, fy, fx * fx, fx * fy, fy * fy])
        cob = cv2.boxFilter(margen, -1, (k, k), borderType=cv2.BORDER_CONSTANT)
        # Coordenadas centradas, en mitades de ancho: con ellas los términos
        # de la pose se miden en píxeles a la orilla del cuadro.
        cx = ((xs - (w - 1) / 2.0) / (w / 2.0)).astype(np.float32)
        cy = ((ys - (h - 1) / 2.0) / (w / 2.0)).astype(np.float32)
        self._geom = dict(shape=(h, w), inv=inv, u=u, v=v, margen=margen, k=k,
                          t=None, uv=None, gy=gy, gx=gx, cob=cob, cx=cx, cy=cy,
                          plano=plano.astype(np.float32),
                          malla=plano[:, gy, gx].reshape(6, -1).T)
        self._f0 = None

    def _uv(self, t) -> Tuple[np.ndarray, np.ndarray]:
        """
        Coordenadas de celda de cada píxel, para una pose.

        La pose es (tx, ty) o (tx, ty, a, b, c, d): el material se corre
        tx + a·x + b·y a lo ancho y ty + c·x + d·y a lo alto.  Con la
        traslación sola, 0.3 grados de inclinación desde que se aprendió
        marcaban todos los cuadros: dos píxeles de deriva en la orilla.
        """
        g = self._geom
        t = tuple(float(x) for x in t) + (0.0,) * (6 - len(t))
        if g["t"] != t:
            i = g["inv"]
            tx, ty, a, b, c, d = t
            u = g["u"] - np.float32(tx * i[0, 0] + ty * i[1, 0])
            v = g["v"] - np.float32(tx * i[0, 1] + ty * i[1, 1])
            if a or b or c or d:
                x, y = g["cx"], g["cy"]
                u = u - (np.float32(a * i[0, 0] + c * i[1, 0]) * x +
                         np.float32(b * i[0, 0] + d * i[1, 0]) * y)
                v = v - (np.float32(a * i[0, 1] + c * i[1, 1]) * x +
                         np.float32(b * i[0, 1] + d * i[1, 1]) * y)
            g["uv"] = (u - np.floor(u), v - np.floor(v))
            g["t"] = t
        return g["uv"]

    def _render(self, celda: np.ndarray, t: Tuple[float, float]) -> np.ndarray:
        """La celda puesta en el cuadro con la fase t."""
        th, tw = celda.shape
        u, v = self._uv(t)
        pad = 3
        c = np.pad(celda, pad, mode="wrap")
        mx = (u * tw + pad - 0.5).astype(np.float32)
        my = (v * th + pad - 0.5).astype(np.float32)
        return cv2.remap(c, mx, my, cv2.INTER_LINEAR)

    def _doblar(self, g: np.ndarray, t, tam, peso=None):
        """Acumula un cuadro sobre la celda: suma y cuenta por punto."""
        th, tw = tam
        u, v = self._uv(t)
        bu = np.minimum((u * tw).astype(np.int32), tw - 1)
        bv = np.minimum((v * th).astype(np.int32), th - 1)
        idx = (bv * tw + bu).ravel()
        w_ = self._geom["margen"].ravel() if peso is None else peso.ravel()
        s = np.bincount(idx, weights=(g.ravel() * w_).astype(np.float64),
                        minlength=th * tw)
        n = np.bincount(idx, weights=w_.astype(np.float64), minlength=th * tw)
        return s.reshape(th, tw), n.reshape(th, tw)

    @staticmethod
    def _llenar(s: np.ndarray, n: np.ndarray, sigma: float = 0.8) -> np.ndarray:
        """Promedio por punto, llenando con sus vecinos los puntos sin dato."""
        p = 6
        sp = cv2.GaussianBlur(np.pad(s, p, mode="wrap").astype(np.float32),
                              (0, 0), sigma)
        npd = cv2.GaussianBlur(np.pad(n, p, mode="wrap").astype(np.float32),
                               (0, 0), sigma)
        return (sp / np.maximum(npd, 1e-6))[p:-p, p:-p]

    def _fase(self, g: np.ndarray, celda: np.ndarray):
        """
        Dónde está la retícula en este cuadro: la pose.

        Primero la correlación completa, que un defecto apenas mueve porque
        el cuadro trae decenas de celdas.  Después Gauss-Newton para la
        fracción de píxel y la deformación suave, sin los píxeles que no
        calzan: si no, un fallo grande jala la pose hacia sí mismo y se
        perdona solo.
        """
        if self._f0 is None:
            e0 = self._render(celda, (0.0, 0.0))
            self._f0 = np.conj(np.fft.rfft2(e0 - float(e0.mean())))
        f = np.fft.rfft2(g - float(g.mean()))
        c = np.fft.irfft2(f * self._f0, g.shape)
        y, x = np.unravel_index(int(np.argmax(c)), c.shape)
        h, w = g.shape
        p = np.array([float(x if x < w // 2 else x - w),
                      float(y if y < h // 2 else y - h),
                      0.0, 0.0, 0.0, 0.0])

        geo = self._geom
        sub = (slice(_MARGEN, h - _MARGEN, 2), slice(_MARGEN, w - _MARGEN, 2))
        cx, cy = geo["cx"][sub], geo["cy"][sub]
        gs = g[sub]
        # De grueso a fino.  La orilla de un motivo solo «jala» a un par de
        # píxeles: con 0.8 grados de más, a la orilla del cuadro la deriva son
        # 3 px y Gauss-Newton se quedaba en el mínimo equivocado.  Borroso,
        # el alcance crece; después la imagen real da la precisión.
        gb = cv2.GaussianBlur(g, (0, 0), 3.0)[sub]
        for vuelta in range(4):
            e = self._render(celda, p)
            borroso = vuelta < 2
            if borroso:
                e = cv2.GaussianBlur(e, (0, 0), 3.0)
            ex = cv2.Sobel(e, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)[sub]
            ey = cv2.Sobel(e, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)[sub]
            r = (gb if borroso else gs) - e[sub]
            r = r - float(np.median(r))
            mad = float(np.median(np.abs(r))) * 1.4826 + 1e-4
            ok = np.abs(r) < (5.0 if borroso else 3.0) * mad
            exo, eyo, xo, yo, ro = ex[ok], ey[ok], cx[ok], cy[ok], r[ok]
            j = np.stack([exo, eyo, exo * xo, exo * yo, eyo * xo, eyo * yo],
                         axis=1)
            a = (j.T @ j).astype(np.float64)
            b = (j.T @ ro).astype(np.float64)
            # Con rayas el sistema casi no tiene rango: a lo largo de la raya
            # no hay nada que medir.  El término de Tikhonov deja quieta esa
            # dirección en vez de mandarla lejos.
            a += np.eye(6) * (1e-3 * float(np.trace(a)) / 6.0 + 1e-9)
            d = np.linalg.solve(a, b)
            # e(x) = celda(x - pose): subir la pose mueve la imagen, de ahí
            # el signo
            p -= np.clip(d, -2.0, 2.0)
        return tuple(float(x) for x in p)

    # ── aprender ──────────────────────────────────────────────────────────
    def aprender(self, cuadros: Sequence[np.ndarray]) -> Informe:
        """
        Aprende la máscara de cuadros BUENOS.

        Tres de cada cuatro cuadros forman la celda; el cuarto solo se usa
        para medir el piso de ruido.  Medir el piso sobre los mismos cuadros
        de los que salió la celda lo daría más bajo que en material nuevo:
        la celda ya absorbió el ruido de esos cuadros.

        Aprender deja la máscara POR APROBAR: no marca nada hasta que el
        operador la vea.
        """
        self.__init_estado()
        inf = Informe(cuadros=len(cuadros))
        if len(cuadros) < 8:
            inf.avisos.append("hacen falta al menos 8 cuadros buenos")
            self.informe = inf
            return inf

        grises = [_gris(c) for c in cuadros]
        h, w = grises[0].shape
        control = grises[3::4]
        celda_de = [g for i, g in enumerate(grises) if i % 4 != 3]

        # La viñeta se saca primero a lo bruto: promediados, los cuadros
        # pierden el motivo y queda la luz.  Luego se refina con la celda.
        luz = _luz_suave(np.mean(celda_de, axis=0))

        picos = _picos_reticula([g / luz for g in celda_de[:6]])
        bases = _bases_candidatas(picos)
        if not bases:
            inf.avisos.append("el material no se repite: no hay retícula")
            self.informe = inf
            return inf

        # Las bases se prueban de verdad y gana la celda más chica que explica
        # el material casi igual que la mejor.  Una celda de más (dos motivos
        # en vez de uno) es correcta pero aprende peor; una de menos no calza.
        resultados = []
        for base in bases:
            self._base = base
            self._preparar((h, w))
            r = self._armar_celda(celda_de[:4], luz, vueltas=1)
            resultados.append((r[0], abs(np.linalg.det(base)), base))
        mejor = min(x[0] for x in resultados)
        utiles = [x for x in resultados if x[0] <= 1.15 * mejor]
        _, _, base = min(utiles, key=lambda x: x[1])
        self._base = base
        self._preparar((h, w))
        _, celda, sigma, luz, _ = self._armar_celda(celda_de, luz)
        self._f0 = None
        self._celda, self._sigma, self._luz = celda, sigma, luz

        # La tolerancia pesa la diferencia.  El piso de la tolerancia evita
        # que unos puntos muy quietos acaparen todo el peso.
        s0 = float(np.median(sigma))
        peso = 1.0 / (sigma ** 2 + (0.5 * s0) ** 2)
        self._peso = (peso / float(peso.mean())).astype(np.float32)
        tela = float(np.percentile(celda, 95))
        self._tinta = (celda < tela - 0.06).astype(np.float32)
        self._var = (sigma ** 2).astype(np.float32)

        # ── piso de ruido, en cuadros que la celda no vio ─────────────────
        pt, pc, pe, nccs, residuos = [], [], [], [], []
        dt, dc = [], []
        descartados = 0
        for g in control:
            zt, zc, ze, ncc, res = self._mapa(g)
            if zt is None:
                descartados += 1
                continue
            pt.append(float(np.abs(zt).max()))
            pc.append(float(np.abs(zc).max()))
            pe.append(float(ze.max()))
            dt.append(_dispersion(zt))
            dc.append(_dispersion(zc))
            nccs.append(ncc)
            residuos.append(res)
        if not pt:
            inf.avisos.append("ningún cuadro de control calzó con la celda")
            self.informe = inf
            self.estado = SIN_APRENDER
            return inf
        # El piso es la MEDIANA de los picos, y además no puede pasar de
        # _PICO_DISP dispersiones del propio cuadro.  Con el máximo, un fallo
        # que se coló entre los cuadros buenos subía el piso y la máscara
        # quedaba ciega justo para ese fallo (prueba [9]): la dispersión
        # casi no se entera de un fallo local, y el pico sí.
        cota_t = _PICO_DISP * float(np.median(dt))
        cota_c = _PICO_DISP * float(np.median(dc))
        med_t, med_c = float(np.median(pt)), float(np.median(pc))
        if med_c > 1.5 * cota_c or med_t > 1.5 * cota_t:
            inf.avisos.append("parece haber un fallo entre los cuadros "
                              "buenos: revisar antes de aprobar")
        # Un material sin zona de tinta (o sin tela limpia) deja una zona
        # vacía: su piso no puede ser cero, o cualquier cosa la rebasaría.
        self._piso = (max(min(med_t, cota_t), 1e-4),
                      max(min(med_c, cota_c), 1e-4),
                      max(float(np.median(pe)), 1e-4))
        self._enganche_ref = float(np.median(nccs))

        tinta = self._tinta > 0
        borde = sigma > 2.5 * s0
        inf.a1 = (float(base[0, 0]), float(base[0, 1]))
        inf.a2 = (float(base[1, 0]), float(base[1, 1]))
        inf.enganche = self._enganche_ref
        inf.residuo_pct = 100.0 * float(np.median(residuos))
        inf.descartados = descartados
        inf.zona_borde = float(borde.mean())
        inf.zona_tinta = float((tinta & ~borde).mean())
        inf.zona_tela = float((~tinta & ~borde).mean())
        if self._enganche_ref < 0.8:
            inf.avisos.append("la celda explica mal el material: "
                              "revisar antes de aprobar")
        self.informe = inf
        self.estado = POR_APROBAR
        return inf

    def __init_estado(self) -> None:
        self.estado = SIN_APRENDER
        self.aprobada_por = ""
        self.aprobada_en = ""
        self._huella_aprobada = ""
        self._celda = self._peso = self._sigma = self._luz = None
        self._tinta = self._var = None
        self._base = None
        self._geom = None
        self._f0 = None

    def _armar_celda(self, grises, luz, vueltas: int = 3):
        """
        Dobla los cuadros sobre una celda, en dos vueltas.

        En la segunda se dejan fuera los píxeles que no calzan: es lo que
        impide que un defecto que se coló en los cuadros buenos quede
        aprendido como parte del estampado.
        """
        h, w = grises[0].shape
        a = self._geom
        n1 = np.hypot(*self._base[0])
        n2 = np.hypot(*self._base[1])
        tam = (max(8, int(round(n2 * _SOBRE))), max(8, int(round(n1 * _SOBRE))))

        s, n = self._doblar(grises[0] / luz, (0.0, 0.0), tam)
        celda = self._llenar(s, n)
        self._f0 = None

        for vuelta in range(vueltas):
            fases = []
            sumas = np.zeros(tam)
            cuenta = np.zeros(tam)
            razones = []
            for g in grises:
                gl = g / luz
                t = self._fase(gl, celda)
                fases.append(t)
                e = self._render(celda, t)
                r = gl - e
                m = a["margen"] > 0
                r0 = r - float(np.median(r[m]))
                mad = float(np.median(np.abs(r0[m]))) * 1.4826 + 1e-4
                peso = a["margen"] * (np.abs(r0) < 4.0 * mad)
                s, n = self._doblar(gl, t, tam, peso.astype(np.float32))
                sumas += s
                cuenta += n
                razones.append(g / np.maximum(e, 0.05))
            celda = self._llenar(sumas, cuenta)
            self._f0 = None
            luz = _luz_suave(np.median(razones, axis=0))

        # tolerancia por punto: cuánto se mueve cada punto de un cuadro a otro
        s2 = np.zeros(tam)
        cuenta = np.zeros(tam)
        res = []
        m = a["margen"] > 0
        for g, t in zip(grises, fases):
            gl = g / luz
            e = self._render(celda, t)
            r = gl - e
            r = r - float(np.median(r[m]))
            res.append(float(np.median(np.abs(r[m]))))
            s, n = self._doblar(r * r, t, tam)
            s2 += s
            cuenta += n
        sigma = np.sqrt(np.maximum(self._llenar(s2, cuenta, 1.2), 1e-8))
        return float(np.mean(res)), celda.astype(np.float32), \
            sigma.astype(np.float32), luz, fases

    # ── aplicar ───────────────────────────────────────────────────────────
    def _mapa(self, g: np.ndarray):
        """
        Desviación por zona: (tinta, tela, energía, ncc, residuo).

        Tinta y tela llevan signo: negativo es más oscuro que lo aprobado.
        (None, None, None, ncc, 0) si el cuadro no calza con la máscara.
        """
        if self._geom is None or g.shape != self._geom["shape"]:
            self._preparar(g.shape)
        gl = g / self._luz
        t = self._fase(gl, self._celda)
        e = self._render(self._celda, t)
        m = self._geom["margen"] > 0
        ge, ee = gl[m], e[m]
        ncc = float(np.corrcoef(ge, ee)[0, 1])
        if self._enganche_ref and ncc < _ENGANCHE_REL * self._enganche_ref:
            return None, None, None, ncc, 0.0
        # ganancia global: si la lámpara sube o baja, no es un defecto
        gan = float(np.median(ge / np.maximum(ee, 0.05)))
        r = gl / max(gan, 1e-3) - e
        res = float(np.median(np.abs(r[m])))
        marg = self._geom["margen"]
        w = self._render(self._peso, t) * marg
        k = self._geom["k"]
        if self.seguir_luz:
            # La luz multiplica: la misma rampa mueve más la tela clara que la
            # tinta.  Restarla pareja dejaba cada zona con su propio resto.
            s = self._rampa(cv2.boxFilter(r * marg, -1, (k, k),
                                          borderType=cv2.BORDER_CONSTANT))
            r = (r + e) / (1.0 + s / max(float(ee.mean()), 0.05)) - e
        tinta = self._render(self._tinta, t)
        rw = r * w
        rt = rw * tinta
        # Tres preguntas.  Las dos primeras, con signo, cancelan el hilo y
        # acumulan lo que es parejo.  La tercera, energía de más, ve lo que
        # el promedio diluye: en un estampado ralo el eco son puntos chicos
        # de buen contraste, y a la escala de una celda casi no mueven la
        # media (puntos sin tejido: un fantasma entero en 0.35 %).
        v = self._render(self._var, t)
        zs = [cv2.boxFilter(x, -1, (k, k), borderType=cv2.BORDER_CONSTANT)
              for x in (rt, rw - rt, (r * r - v) * marg)]
        # Cada zona queda con su propio nivel —la ganancia es una sola para
        # las dos— y ese nivel parejo no es un fallo.  Se resta la mediana, que
        # no se mueve por un fallo local.  Por la cobertura: en la orilla la
        # ventana no está llena y el nivel tampoco.
        g_ = self._geom
        cob = g_["cob"]
        zs = [z - float(np.median(z[g_["gy"], g_["gx"]])) * cob for z in zs]
        return zs[0], zs[1], zs[2], ncc, res

    def _rampa(self, z: np.ndarray) -> np.ndarray:
        """
        Lo que cambió la luz DESPUÉS de aprender, como superficie.

        La viñeta aprendida sirve mientras nadie mueva la lámpara.  Un cambio
        de luz cubre todo el cuadro y es suave; un fallo es local.  Se ajusta
        una superficie de grado 2 a la desviación, y el fallo se deja fuera
        DESDE EL PRIMER AJUSTE: si entra, la superficie se dobla hacia él,
        lo perdona, y del otro lado del cuadro aparece una falsa alarma de
        signo contrario.  Medido: 26 -> 313 cuadros con marca lejos del fallo.
        """
        g = self._geom
        d = g["malla"]
        y = z[g["gy"], g["gx"]].ravel().astype(np.float64)
        e = y - float(np.median(y))
        mad = float(np.median(np.abs(e))) * 1.4826 + 1e-6
        ok = np.abs(e) < 2.5 * mad
        c = np.zeros(d.shape[1])
        for _ in range(5):
            if ok.sum() < 12:
                break
            c, *_ = np.linalg.lstsq(d[ok], y[ok], rcond=None)
            e = y - d @ c
            mad = float(np.median(np.abs(e[ok]))) * 1.4826 + 1e-6
            ok = np.abs(e) < 2.5 * mad
        return np.tensordot(c.astype(np.float32), g["plano"], axes=1)

    @property
    def umbral(self) -> float:
        """En múltiplos del piso aprendido: 6.5x a 2.5x, 4.5x por defecto."""
        s = float(np.clip(self.sensibilidad, 0, 100))
        return 6.5 - 0.04 * s

    @property
    def enganchada(self) -> bool:
        """False si el último cuadro no calzó con la máscara."""
        return self._enganchada

    @property
    def lista(self) -> bool:
        """Aprobada y sin tocar desde que se aprobó."""
        return (self.estado == APROBADA and
                self._huella_aprobada == self.huella())

    def analizar(self, frame: np.ndarray) -> list:
        """Regiones contra lo aprobado, en coordenadas del CUADRO."""
        self._regiones, self._tipos = [], []
        self._enganchada = True
        if not self.lista or frame is None or frame.size == 0:
            return []
        g = _gris(frame)
        zt, zc, ze, ncc, _ = self._mapa(g)
        self._ultima_ncc = ncc
        if zt is None:
            self._ultimo_z = None
            self._enganchada = False
            return []
        st = zt / self._piso[0]
        s = np.maximum(np.maximum(np.abs(st), np.abs(zc) / self._piso[1]),
                       ze / (_EXTRA_ENERGIA * self._piso[2]))
        self._ultimo_z = s
        escala = frame.shape[1] / float(g.shape[1])
        gh, gw = g.shape
        min_area = (self.min_area_pct / 100.0) * gh * gw
        alto = self.umbral
        if float(s.max()) <= alto:
            return []
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mk = cv2.morphologyEx((s > 0.6 * alto).astype(np.uint8),
                              cv2.MORPH_OPEN, kern)
        n, et, stats, _ = cv2.connectedComponentsWithStats(mk, 8)
        for i in range(1, n):
            x, y, w_, h_, area = stats[i]
            if area < min_area:
                continue
            sel = et == i
            # histéresis: la región se dibuja hasta el umbral bajo, pero solo
            # existe si en algún punto pasó el alto
            if float(s[sel].max()) <= alto:
                continue
            self._regiones.append((int(x * escala), int(y * escala),
                                   int((x + w_) * escala),
                                   int((y + h_) * escala)))
            # Lo único que aclara la zona de tinta es que falte tinta.
            self._tipos.append(KIND_FALTA if float(st[sel].max()) > alto
                               else KIND_EXTRA)
        return list(self._regiones)

    @property
    def regiones(self) -> list:
        return list(self._regiones)

    @property
    def tipos(self) -> list:
        return list(self._tipos)

    def as_detections(self) -> list:
        """Las regiones como detecciones para el tracker, igual que Patrón."""
        from eyve.inference.tracker import RawDetection
        return [RawDetection(self.ANOMALY_LABEL, 1.0, *r)
                for r in self._regiones]

    # ── lo que ve y decide el operador ────────────────────────────────────
    def huella(self) -> str:
        """Identifica el contenido: una máscara cambiada ya no está aprobada."""
        if self._celda is None:
            return ""
        h = hashlib.sha256()
        for a in (self._celda, self._peso, self._tinta, self._luz, self._base):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(repr(tuple(round(p, 9) for p in self._piso)).encode())
        return h.hexdigest()[:16]

    def aprobar(self, quien: str = "operador") -> None:
        if self.estado not in (POR_APROBAR, RECHAZADA) or self._celda is None:
            raise ValueError("no hay máscara que aprobar")
        self.estado = APROBADA
        self.aprobada_por = quien
        self.aprobada_en = time.strftime("%Y-%m-%d %H:%M:%S")
        self._huella_aprobada = self.huella()

    def rechazar(self) -> None:
        if self._celda is None:
            raise ValueError("no hay máscara que rechazar")
        self.estado = RECHAZADA
        self._huella_aprobada = ""

    def vista(self, frame: Optional[np.ndarray] = None,
              ancho: int = 960) -> np.ndarray:
        """
        Lo que el operador aprueba: lo aprendido, sus zonas y el material.

        Izquierda, el estampado aprendido, como se ve en el cuadro: la celda
        en sí es un rombo y, mostrada cuadrada, un motivo sale partido en las
        cuatro esquinas.  Centro, las zonas: gris = tinta, blanco = tela
        limpia, naranja = orilla que casi no se juzga.  Derecha, si hay
        cuadro, lo que NO calza: gris parejo es que la máscara cae en su
        lugar; un relieve en cada motivo es que está corrida; una mancha es
        un fallo.
        """
        if self._celda is None:
            raise ValueError("no hay máscara")
        lado = ancho // 3
        if self._geom is None:
            self._preparar(self._luz.shape)
        h, w = self._geom["shape"]
        n = min(h, 144)
        y0, x0 = (h - n) // 2, (w - n) // 2
        trozo = (slice(y0, y0 + n), slice(x0, x0 + n))

        e = self._render(self._celda, (0.0, 0.0))[trozo]
        c = np.clip(e / max(float(e.max()), 1e-3), 0, 1)
        izq = cv2.cvtColor(
            cv2.resize((c * 255).astype(np.uint8), (lado, lado),
                       interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)

        s0 = float(np.median(self._sigma))
        zonas = np.full((n, n, 3), 255, np.uint8)
        zonas[self._render(self._tinta, (0.0, 0.0))[trozo] > 0.5] = (120, 120, 120)
        zonas[self._render(self._sigma, (0.0, 0.0))[trozo] > 2.5 * s0] = (0, 140, 255)
        cen = cv2.resize(zonas, (lado, lado), interpolation=cv2.INTER_NEAREST)

        der = np.full((lado, lado, 3), 40, np.uint8)
        if frame is not None:
            g = _gris(frame)
            if g.shape != self._geom["shape"]:
                self._preparar(g.shape)
            gl = g / self._luz
            t = self._fase(gl, self._celda)
            r = gl - self._render(self._celda, t)
            r = cv2.GaussianBlur(r - float(np.median(r)), (0, 0), 1.5)
            v = (np.clip(0.5 + 8.0 * r, 0, 1) * 255).astype(np.uint8)
            s = min(v.shape)
            x = (v.shape[1] - s) // 2
            der = cv2.cvtColor(cv2.resize(v[:s, x:x + s], (lado, lado),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_GRAY2BGR)
        tira = np.hstack([izq, cen, der])
        inf = self.informe
        pie = np.full((64, tira.shape[1], 3), 30, np.uint8)
        l1 = (f"celda {math.hypot(*inf.a1):.1f} x {math.hypot(*inf.a2):.1f} px   "
              f"enganche {inf.enganche:.3f}   residuo {inf.residuo_pct:.2f} %")
        l2 = (f"tinta {inf.zona_tinta:.0%}  tela {inf.zona_tela:.0%}  "
              f"orilla {inf.zona_borde:.0%}   estado: {self.estado}")
        cv2.putText(pie, l1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(pie, l2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (235, 235, 235), 1, cv2.LINE_AA)
        return np.vstack([tira, pie])

    # ── guardar ───────────────────────────────────────────────────────────
    def guardar(self, ruta) -> None:
        """Se guarda con su aprobación, y la huella la amarra al contenido."""
        if self._celda is None:
            raise ValueError("no hay máscara")
        np.savez_compressed(
            Path(ruta), celda=self._celda, peso=self._peso, sigma=self._sigma,
            tinta=self._tinta, luz=self._luz, base=self._base,
            meta=json.dumps(dict(
                piso=list(self._piso), enganche_ref=self._enganche_ref,
                estado=self.estado, por=self.aprobada_por,
                en=self.aprobada_en, huella=self._huella_aprobada)))

    @classmethod
    def cargar(cls, ruta) -> "MascaraAprendida":
        d = np.load(Path(ruta), allow_pickle=False)
        m = cls()
        m._celda, m._peso, m._sigma = d["celda"], d["peso"], d["sigma"]
        m._tinta, m._luz, m._base = d["tinta"], d["luz"], d["base"]
        m._var = (m._sigma ** 2).astype(np.float32)
        meta = json.loads(str(d["meta"]))
        m._piso = tuple(float(x) for x in meta["piso"])
        m._enganche_ref = float(meta["enganche_ref"])
        m.estado = meta["estado"]
        m.aprobada_por = meta["por"]
        m.aprobada_en = meta["en"]
        m._huella_aprobada = meta["huella"]
        # Si alguien cambió la celda en el archivo, la aprobación no le toca.
        if m.estado == APROBADA and m._huella_aprobada != m.huella():
            m.estado = POR_APROBAR
            m._huella_aprobada = ""
        return m


def fusionar(mascara: MascaraAprendida, patron, named=None) -> list:
    """
    Detecciones de los dos motores para un solo tracker.

    Lo que la máscara ya marcó, Patrón no lo repite: es la misma regla que
    Patrón sigue con YOLO, y por el mismo motivo (contar una sola vez).
    """
    propias = mascara.as_detections()
    cajas = list(named or []) + list(mascara.regiones)
    return propias + patron.as_detections(named=cajas)
