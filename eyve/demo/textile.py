"""
Tela sintética: patrón de impresión que viaja infinitamente, con una capa
de defectos que viaja CON la tela.

Por qué la capa de defectos va en coordenadas de tela y no de pantalla:
un defecto pintado en la pantalla se quedaría quieto mientras la tela pasa
por debajo, y el tracker lo vería como un objeto estático. Pintado sobre la
tela, el defecto entra por un lado del encuadre, cruza y sale — que es
exactamente lo que el módulo de conteo necesita para contarlo una vez.

La ventana que ve "la cámara" se saca con np.take(..., mode="wrap"), así
que el viaje es infinito y sin costura, sin copiar la tela entera cada
frame. Después se aplica una rotación pequeña y un viñeteado para que el
frame se parezca a una toma real y no a un render perfecto.

El generador de dataset (eyve.demo.dataset) usa ESTA MISMA clase y estas
mismas primitivas de pintura, para que lo que dibuje el visitante en la
expo caiga dentro de la distribución con la que se entrenó el modelo.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

#: clases del proyecto demo, en el orden que espera el modelo (índice = class_id)
DEFECT_CLASSES: tuple[str, ...] = (
    "rayon",            # trazo: hilo roto, raspon, arrastre
    "mancha",           # suciedad, aceite, salpicadura
    "falta_impresion",  # llego poca tinta o ninguna
    "fantasma",         # eco palido del motivo, desplazado
    "offset",           # impresion movida: motivo duplicado y corrido
)

#: fallos que TRANSFORMAN el estampado en vez de pintar encima
_PRINT_FAULTS = ("falta_impresion", "fantasma", "offset")

#: Clases que se ENTRENAN en YOLO: lo puntual y nombrable, que es lo que
#: se cuenta y lo que se reporta por tipo.
YOLO_CLASSES: tuple[str, ...] = ("rayon", "mancha")

#: Fallos que NO son clases: variantes sin fin de "esto no se parece a lo
#: bueno".  Los cubre el modulo Patron, sin entrenar nada.  Se generan en
#: los frames de entrenamiento igual, pero sin etiqueta, para ensenarle a
#: YOLO que son fondo y no la clase mas parecida.
PRINT_FAULTS: tuple[str, ...] = _PRINT_FAULTS

#: paleta de la tela — azul sobre blanco, como pidió el guion de la demo
_INK = (168, 86, 28)        # azul de impresión (BGR)
_INK_SOFT = (205, 150, 95)  # azul claro para el relleno secundario
_CLOTH = (247, 247, 243)    # blanco de la tela impresa, no blanco puro
#: tela cruda que se ve donde no cayo tinta — mas calida y mate que la
#: impresa. La diferencia es de ~10 niveles: suficiente para detectarla,
#: no tanto como para que sea trivial.
_GREIGE = (233, 238, 241)

#: colores con los que se pinta cada defecto
_DEFECT_INK = {
    "rayon": (58, 42, 38),            # trazo oscuro
    "mancha": (72, 92, 150),          # mancha café-rojiza
    "falta_impresion": _GREIGE,       # patrón borrado = tela cruda
}

MOTIFS = ("diamantes", "flores", "rayas", "puntos")

#: ligamentos. "ninguno" deja la tela lisa (util para aislar en pruebas).
WEAVES = ("sarga", "tafetan", "sarga_fina", "canasta", "ninguno")


@dataclass
class Defect:
    """
    Un defecto en coordenadas de TELA.

    bbox crece a medida que se pinta: un trazo continuo del visitante es UN
    defecto, no uno por cada punto del arrastre.
    """
    cls: str
    x1: int
    y1: int
    x2: int
    y2: int

    def grow(self, x: int, y: int, r: int) -> None:
        self.x1 = min(self.x1, x - r)
        self.y1 = min(self.y1, y - r)
        self.x2 = max(self.x2, x + r)
        self.y2 = max(self.y2, y + r)

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


class TextilePattern:
    """
    Tela de `fabric_len` px de largo sobre el eje de viaje, que se repite.

    axis="x"  la tela viaja horizontalmente (banda transportadora)
    axis="y"  la tela viaja verticalmente (línea de impresión textil)
    """

    def __init__(self, width: int = 960, height: int = 540,
                 axis: str = "x", fabric_len: int = 3840,
                 motif: str = "diamantes", speed: float = 110.0,
                 tilt_deg: float = 1.8, seed: Optional[int] = None,
                 weave: str = "sarga", weave_amp: float = 0.07) -> None:
        self.width = int(width)
        self.height = int(height)
        self.axis = "y" if axis == "y" else "x"
        self.motif = motif if motif in MOTIFS else "diamantes"
        self.weave = weave if weave in WEAVES else "sarga"
        #: amplitud del ligamento. Bajo a proposito: es el relieve del hilo,
        #: no un estampado. Pasado de 0.15 compite con el motivo y el modelo
        #: aprende el tejido en vez del defecto.
        self.weave_amp = float(weave_amp)
        #: px por segundo sobre el eje de viaje
        self.speed = float(speed)
        self.tilt_deg = float(tilt_deg)
        self._rng = random.Random(seed)

        # La tela debe ser más larga que la ventana para que el viaje se
        # note, y su largo debe ser múltiplo del periodo del motivo para que
        # el wraparound no deje una costura visible.
        # Periodo del motivo. Mas chico = impresion mas densa, que es lo
        # que hace visible un faltante: sobre una tela casi vacia, borrar
        # el patron no deja nada que ver.
        self.period = 96
        span = max(int(fabric_len), (self.width if self.axis == "x" else self.height) * 2)
        self.fabric_len = int(round(span / self.period) * self.period)

        if self.axis == "x":
            self.fw, self.fh = self.fabric_len, self.height
        else:
            self.fw, self.fh = self.width, self.fabric_len

        self._base = self._render_fabric()
        self._weave = self._render_weave()
        # capa de defectos: color + alfa, en coordenadas de tela
        self._layer = np.zeros((self.fh, self.fw, 3), dtype=np.uint8)
        self._alpha = np.zeros((self.fh, self.fw), dtype=np.uint8)
        self.defects: list[Defect] = []
        self._stroke: Optional[Defect] = None

        self.offset = 0.0          # posición de la tela sobre su eje
        self._vignette = self._make_vignette()

    # ── la tela ───────────────────────────────────────────────────────────
    def _render_fabric(self) -> np.ndarray:
        """Dibuja el motivo repetido una sola vez, al construir."""
        img = np.full((self.fh, self.fw, 3), _CLOTH, dtype=np.uint8)
        p = self.period
        draw = {
            "diamantes": self._motif_diamantes,
            "flores":    self._motif_flores,
            "rayas":     self._motif_rayas,
            "puntos":    self._motif_puntos,
        }[self.motif]
        # El eje de viaje da la vuelta, asi que un motivo pegado al limite
        # tiene que dibujarse TAMBIEN del otro lado: si no, el diamante
        # queda cortado en la costura y se ve el salto cada vuelta.
        span = self.fw if self.axis == "x" else self.fh
        for gy in range(-p, self.fh + p, p):
            for gx in range(-p, self.fw + p, p):
                # filas alternas desfasadas medio periodo (tresbolillo), que
                # es como se imprime de verdad
                off = (p // 2) if ((gy // p) % 2) else 0
                cx, cy = gx + off, gy
                draw(img, cx, cy, p)
                here = cx if self.axis == "x" else cy
                if here < p or here > span - p:
                    shift = -span if here > span // 2 else span
                    if self.axis == "x":
                        draw(img, cx + shift, cy, p)
                    else:
                        draw(img, cx, cy + shift, p)
        # textura de hilo: ruido tenue, para que el detector no aprenda que
        # "liso perfecto" es lo normal
        noise = np.random.default_rng(7).integers(-5, 6, (self.fh, self.fw, 1),
                                                  dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def _motif_diamantes(img, cx, cy, p) -> None:
        r = p // 3
        pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]],
                       dtype=np.int32)
        cv2.fillPoly(img, [pts], _INK)
        r2 = r // 2
        pts2 = np.array([[cx, cy - r2], [cx + r2, cy], [cx, cy + r2], [cx - r2, cy]],
                        dtype=np.int32)
        cv2.fillPoly(img, [pts2], _CLOTH)

    @staticmethod
    def _motif_flores(img, cx, cy, p) -> None:
        r = p // 4
        for k in range(6):
            a = k * math.pi / 3
            px = int(cx + math.cos(a) * r)
            py = int(cy + math.sin(a) * r)
            cv2.circle(img, (px, py), max(3, r // 2), _INK_SOFT, -1)
        cv2.circle(img, (cx, cy), max(3, r // 2), _INK, -1)

    @staticmethod
    def _motif_rayas(img, cx, cy, p) -> None:
        t = max(3, p // 12)
        cv2.line(img, (cx - p // 2, cy - p // 2), (cx + p // 2, cy + p // 2),
                 _INK, t, cv2.LINE_AA)
        cv2.line(img, (cx - p // 2, cy), (cx + p // 2, cy + p),
                 _INK_SOFT, t, cv2.LINE_AA)

    @staticmethod
    def _motif_puntos(img, cx, cy, p) -> None:
        r = max(4, p // 7)
        cv2.circle(img, (cx, cy), r, _INK, -1)
        q = p // 2
        for ox, oy in ((q, q), (-q, q), (q, -q)):
            cv2.circle(img, (cx + ox, cy + oy), max(3, p // 10), _INK_SOFT, -1)

    def _render_weave(self) -> np.ndarray:
        """
        Multiplicador de luz (fh, fw, 1) con el ligamento del tejido.

        Se construye una vez y viaja con la tela.  Los periodos son primos
        entre si respecto del motivo para que no aparezcan bandas de moire
        donde el tejido y el estampado se alinean.
        """
        if self.weave == "ninguno" or self.weave_amp <= 0:
            return np.ones((self.fh, self.fw, 1), dtype=np.float32)

        yy, xx = np.meshgrid(np.arange(self.fh, dtype=np.float32),
                             np.arange(self.fw, dtype=np.float32),
                             indexing="ij")
        if self.weave == "sarga":
            # diagonal marcada: el hilo pasa sobre 2 y bajo 1, desplazado
            # una pasada por fila — es lo que da la linea a 45 grados
            t = np.sin((xx + yy) * (2 * math.pi / 7.0))
            t += 0.35 * np.sin(yy * (2 * math.pi / 3.0))
        elif self.weave == "sarga_fina":
            t = np.sin((xx + yy) * (2 * math.pi / 4.0))
            t += 0.25 * np.sin((xx - yy) * (2 * math.pi / 11.0))
        elif self.weave == "tafetan":
            # trama y urdimbre cruzando una a una
            t = (np.sin(xx * (2 * math.pi / 5.0)) *
                 np.sin(yy * (2 * math.pi / 5.0)))
            t += 0.4 * np.sin((xx + yy) * (2 * math.pi / 5.0))
        else:   # canasta: grupos de 2x2 hilos
            t = (np.sin(xx * (2 * math.pi / 9.0)) +
                 np.sin(yy * (2 * math.pi / 9.0)))
            t += 0.5 * np.sin(xx * (2 * math.pi / 4.5)) * \
                 np.sin(yy * (2 * math.pi / 4.5))

        t = t / (np.abs(t).max() or 1.0)
        # irregularidad del hilo: ninguna tela real es perfectamente regular
        rng = np.random.default_rng(13)
        t = t + rng.normal(0.0, 0.18, t.shape).astype(np.float32)
        w = (1.0 + self.weave_amp * t).astype(np.float32)
        return w[:, :, None]

    def _make_vignette(self) -> np.ndarray:
        """Caída de luz en las esquinas, como una cámara real con lente."""
        ys = np.linspace(-1.0, 1.0, self.height)[:, None]
        xs = np.linspace(-1.0, 1.0, self.width)[None, :]
        r = np.sqrt(xs ** 2 + ys ** 2) / math.sqrt(2.0)
        v = (1.0 - 0.28 * r ** 2).astype(np.float32)
        return v[:, :, None]

    # ── pintura de defectos (coordenadas de TELA) ─────────────────────────
    def begin_stroke(self, cls: str) -> None:
        """Abre un trazo nuevo: todo lo que se pinte hasta end_stroke() es
        UN defecto (una instancia para el tracker)."""
        if cls not in DEFECT_CLASSES:
            raise ValueError(f"clase de defecto desconocida: {cls}")
        self._stroke = None
        self._stroke_cls = cls

    def paint(self, fx: int, fy: int, radius: int = 7,
              cls: Optional[str] = None) -> None:
        """
        Pinta un punto del trazo en curso en (fx, fy) de la tela.

        "falta_impresion" pinta el blanco de la tela, que es justo cómo se
        ve un faltante de impresión: el motivo desaparece.
        """
        cls = cls or getattr(self, "_stroke_cls", "rayon")
        if cls not in DEFECT_CLASSES:
            return
        fx, fy = self._wrap_clamp(fx, fy)
        radius = max(2, int(radius))
        color = _DEFECT_INK[cls]
        # El color va SIN antialias y un pixel mas grande: asi el fleco
        # suave del alfa siempre tiene color solido debajo y nunca mezcla
        # con el negro con el que nace la capa.
        cv2.circle(self._layer, (fx, fy), radius + 1, color, -1, cv2.LINE_8)
        cv2.circle(self._alpha, (fx, fy), radius, 255, -1, cv2.LINE_AA)
        if self._stroke is None:
            self._stroke = Defect(cls, fx - radius, fy - radius,
                                  fx + radius, fy + radius)
            self.defects.append(self._stroke)
        else:
            self._stroke.grow(fx, fy, radius)

    def paint_segment(self, p0, p1, radius: int = 7,
                      cls: Optional[str] = None) -> None:
        """
        Pinta el segmento p0→p1 del trazo (el arrastre del mouse salta
        píxeles; sin interpolar, el trazo sale punteado).
        """
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) // max(1, radius // 2)))
        for i in range(n + 1):
            tt = i / n
            self.paint(int(x0 + (x1 - x0) * tt), int(y0 + (y1 - y0) * tt),
                       radius, cls)

    def end_stroke(self) -> Optional[Defect]:
        d, self._stroke = self._stroke, None
        return d

    def blob(self, fx: int, fy: int, size: int = 34,
             cls: str = "mancha") -> Defect:
        """
        Mancha irregular de un golpe (no un círculo: una mancha redonda
        perfecta no existe y el modelo aprendería la forma, no el defecto).
        """
        self.begin_stroke(cls)
        n = self._rng.randint(5, 9)
        for k in range(n):
            a = 2 * math.pi * k / n + self._rng.uniform(-0.3, 0.3)
            rr = size * self._rng.uniform(0.25, 0.55)
            self.paint(int(fx + math.cos(a) * rr * 0.6),
                       int(fy + math.sin(a) * rr * 0.6),
                       int(size * self._rng.uniform(0.3, 0.5)), cls)
        d = self.end_stroke()
        return d

    def streak(self, fx: int, fy: int, length: int = 110,
               angle: Optional[float] = None, thickness: int = 5,
               cls: str = "rayon") -> Defect:
        """Rayón: trazo largo y delgado, con un poco de temblor."""
        self.begin_stroke(cls)
        a = self._rng.uniform(0, math.pi) if angle is None else angle
        steps = max(4, length // 10)
        prev = None
        for i in range(steps + 1):
            tt = i / steps
            jx = self._rng.uniform(-3, 3)
            jy = self._rng.uniform(-3, 3)
            cur = (int(fx + math.cos(a) * length * (tt - 0.5) + jx),
                   int(fy + math.sin(a) * length * (tt - 0.5) + jy))
            # unir los puntos: un rayon es un trazo continuo, no una fila
            # de puntos sueltos
            if prev is None:
                self.paint(cur[0], cur[1], thickness, cls)
            else:
                self.paint_segment(prev, cur, thickness, cls)
            prev = cur
        return self.end_stroke()

    # ── fallos de impresion: transforman el estampado de debajo ───────────
    def _region(self, fx: int, fy: int, w: int, h: int):
        """
        Indices (ys, xs) de una region de tela, con vuelta en el eje de
        viaje.  Devuelve tambien la caja SIN envolver, para la etiqueta.
        """
        x0, y0 = int(fx - w // 2), int(fy - h // 2)
        x1, y1 = x0 + w, y0 + h
        if self.axis == "x":
            y0 = max(0, min(y0, self.fh - 1))
            y1 = max(y0 + 2, min(y1, self.fh))
        else:
            x0 = max(0, min(x0, self.fw - 1))
            x1 = max(x0 + 2, min(x1, self.fw))
        ys = np.arange(y0, y1) % self.fh
        xs = np.arange(x0, x1) % self.fw
        return ys, xs, (x0, y0, x1, y1)

    def _mottle(self, shape, rng_seed: int, escala: int = 14) -> np.ndarray:
        """
        Mancha suave 0..1 para que el fallo no sea un rectangulo perfecto.

        Un rectangulo exacto seria un atajo: el modelo aprenderia la forma
        del parche en vez del fallo de impresion.
        """
        h, w = shape
        rng = np.random.default_rng(rng_seed)
        chico = rng.random((max(2, h // escala), max(2, w // escala)))
        suave = cv2.resize(chico.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_CUBIC)
        suave = np.clip(suave, 0.0, 1.0)
        # bordes desvanecidos: el fallo se degrada hacia afuera
        fy_ = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        fx_ = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        caida = np.clip(1.25 - (fx_ ** 2 + fy_ ** 2), 0.0, 1.0)
        return suave * caida

    def _stamp(self, cls: str, ys, xs, caja, nuevo: np.ndarray,
               alfa: np.ndarray) -> Defect:
        """Escribe el resultado en la capa de defectos y registra la caja."""
        grid = np.ix_(ys, xs)
        a = np.clip(alfa, 0.0, 1.0)[:, :, None]
        viejo = self._layer[grid].astype(np.float32)
        va = (self._alpha[grid].astype(np.float32) / 255.0)[:, :, None]
        base = self._base[grid].astype(np.float32)
        # si ya habia algo pintado ahi, se respeta como fondo
        fondo = base * (1 - va) + viejo * va
        self._layer[grid] = np.clip(fondo * (1 - a) + nuevo * a,
                                    0, 255).astype(np.uint8)
        self._alpha[grid] = np.maximum(
            self._alpha[grid],
            (np.clip(alfa, 0.0, 1.0) * 255).astype(np.uint8))
        d = Defect(cls, caja[0], caja[1], caja[2], caja[3])
        self.defects.append(d)
        return d

    def ghost(self, fx: int, fy: int, size: int = 120,
              strength: float = 0.45) -> Defect:
        """
        Fantasma: un eco palido del motivo, desplazado unos milimetros.

        Se toma el propio estampado de al lado y se superpone muy suave.
        """
        w = int(size * self._rng.uniform(0.9, 1.6))
        h = int(size * self._rng.uniform(0.7, 1.3))
        ys, xs, caja = self._region(fx, fy, w, h)
        dx = int(self.period * self._rng.uniform(0.18, 0.45)) * \
            self._rng.choice((-1, 1))
        dy = int(self.period * self._rng.uniform(0.10, 0.35)) * \
            self._rng.choice((-1, 1))
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        eco = self._base[np.ix_((ys + dy) % self.fh,
                                (xs + dx) % self.fw)].astype(np.float32)
        # el eco solo OSCURECE donde el motivo pasa: la tinta no aclara
        nuevo = np.minimum(base, eco * strength + base * (1 - strength))
        alfa = self._mottle(nuevo.shape[:2], self._rng.randrange(1 << 30))
        return self._stamp("fantasma", ys, xs, caja, nuevo, alfa * 0.9)

    def ink_starved(self, fx: int, fy: int, size: int = 120,
                    severity: Optional[float] = None) -> Defect:
        """
        Falta de tinta: el motivo se desvanece hacia la tela cruda.

        severity 0.4 = impresion debil y a parches;  1.0 = tela desnuda.
        El rango entero es el mismo fallo visto con mas o menos tinta.
        """
        if severity is None:
            severity = self._rng.uniform(0.45, 1.0)
        w = int(size * self._rng.uniform(0.8, 1.5))
        h = int(size * self._rng.uniform(0.6, 1.2))
        ys, xs, caja = self._region(fx, fy, w, h)
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        cruda = np.array(_GREIGE, dtype=np.float32)[None, None, :]
        m = self._mottle(base.shape[:2], self._rng.randrange(1 << 30))
        # a mas severidad, mas uniforme el faltante
        k = np.clip(m * (0.5 + severity) + severity - 0.45, 0.0, 1.0)
        nuevo = base * (1 - k[:, :, None]) + cruda * k[:, :, None]
        return self._stamp("falta_impresion", ys, xs, caja, nuevo,
                           np.clip(k * 1.6, 0.0, 1.0))

    def misregister(self, fx: int, fy: int, size: int = 140,
                    shift: Optional[int] = None) -> Defect:
        """
        Offset / movido: el estampado sale duplicado y corrido.

        A diferencia del fantasma, las dos impresiones estan a buena
        intensidad: es un fallo de registro, no un rebote.
        """
        w = int(size * self._rng.uniform(0.9, 1.7))
        h = int(size * self._rng.uniform(0.7, 1.4))
        ys, xs, caja = self._region(fx, fy, w, h)
        if shift is None:
            shift = int(self.period * self._rng.uniform(0.25, 0.6))
        ang = self._rng.uniform(0, 2 * math.pi)
        dx = int(math.cos(ang) * shift)
        dy = int(math.sin(ang) * shift)
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        corrido = self._base[np.ix_((ys + dy) % self.fh,
                                    (xs + dx) % self.fw)].astype(np.float32)
        # las dos impresiones se ven: se queda la mas oscura de cada pixel
        nuevo = np.minimum(base, corrido)
        alfa = self._mottle(nuevo.shape[:2], self._rng.randrange(1 << 30))
        return self._stamp("offset", ys, xs, caja, nuevo,
                           np.clip(alfa * 1.8, 0.0, 1.0))

    def missing_print(self, fx: int, fy: int, size: int = 60) -> Defect:
        """
        Faltante de impresión severo.  Se conserva el nombre porque es la
        llamada que ya usaban la demo y las pruebas; ahora es el extremo
        del rango de ink_starved().
        """
        return self.ink_starved(fx, fy, size, severity=0.95)

    def clear_defects(self) -> None:
        self._layer[:] = 0
        self._alpha[:] = 0
        self.defects.clear()
        self._stroke = None

    # ── coordenadas: pantalla ↔ tela ──────────────────────────────────────
    def screen_to_fabric(self, sx: int, sy: int) -> Tuple[int, int]:
        """
        Punto de la ventana (lo que ve la cámara) → punto de la tela.

        Deshace la rotación alrededor del centro y suma el desplazamiento
        del viaje. Es la inversa exacta de lo que hace frame(), para que el
        defecto aparezca bajo el cursor y no corrido.
        """
        # Inversa exacta de la rotacion que aplica cv2.warpAffine en
        # frame(). Con Y hacia abajo, la matriz de cv2 para el angulo t es
        # [[cos, sin], [-sin, cos]]; su inversa es la de -t.
        cxx, cyy = self.width / 2.0, self.height / 2.0
        a = math.radians(self.tilt_deg)
        dx, dy = sx - cxx, sy - cyy
        rx = dx * math.cos(a) - dy * math.sin(a) + cxx
        ry = dx * math.sin(a) + dy * math.cos(a) + cyy
        off = int(self.offset)
        if self.axis == "x":
            return self._wrap_clamp(rx + off, ry)
        return self._wrap_clamp(rx, ry + off)

    def _wrap_clamp(self, fx: float, fy: float) -> Tuple[int, int]:
        """
        Lleva un punto a coordenadas validas de tela.

        Solo el eje de VIAJE da la vuelta: la tela es infinita en ese eje y
        finita en el otro, igual que un rollo real. Aplicar modulo al eje
        corto manda un punto que se salio por arriba al borde de abajo.
        """
        if self.axis == "x":
            return (int(fx) % self.fw,
                    int(min(max(fy, 0), self.fh - 1)))
        return (int(min(max(fx, 0), self.fw - 1)),
                int(fy) % self.fh)

    @staticmethod
    def _nearest(v: int, span: int, win: int) -> int:
        """
        Representante de v modulo span mas cercano al centro de la ventana.

        La tela da la vuelta, asi que una posicion de tela tiene infinitos
        equivalentes (v, v+span, v-span, ...). Para proyectar una caja hay
        que elegir el que cae junto a la ventana: con el representante
        ingenuo en [0, span), un defecto que esta 5 px ANTES del encuadre
        sale reportado a span-5 px, al otro extremo de la tela, y la caja
        queda absurda.
        """
        u = v % span
        if u - win / 2.0 > span / 2.0:
            u -= span
        return int(u)

    def fabric_to_screen(self, fx: int, fy: int) -> Tuple[int, int]:
        """Inversa de screen_to_fabric (para proyectar etiquetas)."""
        off = int(self.offset)
        if self.axis == "x":
            ux = self._nearest(fx - off, self.fw, self.width)
            uy = float(fy)
        else:
            ux = float(fx)
            uy = self._nearest(fy - off, self.fh, self.height)
        # La MISMA rotacion que aplica cv2.warpAffine en frame(), no su
        # inversa: la etiqueta tiene que caer donde quedo la tinta.
        cxx, cyy = self.width / 2.0, self.height / 2.0
        a = math.radians(-self.tilt_deg)
        dx, dy = ux - cxx, uy - cyy
        sx = dx * math.cos(a) - dy * math.sin(a) + cxx
        sy = dx * math.sin(a) + dy * math.cos(a) + cyy
        return int(round(sx)), int(round(sy))

    # ── el frame ──────────────────────────────────────────────────────────
    def advance(self, dt: float) -> None:
        """Avanza la tela dt segundos."""
        span = self.fw if self.axis == "x" else self.fh
        self.offset = (self.offset + self.speed * dt) % span

    def frame(self) -> np.ndarray:
        """
        La ventana que ve la cámara: tela + defectos, con el viaje aplicado,
        rotación pequeña y viñeteado.
        """
        off = int(self.offset)
        if self.axis == "x":
            idx = (np.arange(self.width) + off) % self.fw
            base = np.take(self._base, idx, axis=1)
            lay = np.take(self._layer, idx, axis=1)
            alp = np.take(self._alpha, idx, axis=1)
            wv = np.take(self._weave, idx, axis=1)
        else:
            idx = (np.arange(self.height) + off) % self.fh
            base = np.take(self._base, idx, axis=0)
            lay = np.take(self._layer, idx, axis=0)
            alp = np.take(self._alpha, idx, axis=0)
            wv = np.take(self._weave, idx, axis=0)

        out = base
        if alp.any():
            a = (alp.astype(np.float32) / 255.0)[:, :, None]
            out = (base.astype(np.float32) * (1.0 - a) +
                   lay.astype(np.float32) * a).astype(np.uint8)
        else:
            out = base.copy()

        # El tejido se aplica sobre la tinta tambien: es el relieve del
        # hilo y se ve igual en la zona impresa y en la cruda. Si solo
        # modulara el fondo, los defectos saldrian sospechosamente lisos y
        # el modelo aprenderia "lo liso es defecto".
        if self.weave != "ninguno" and self.weave_amp > 0:
            out = np.clip(out.astype(np.float32) * wv, 0, 255).astype(np.uint8)

        if abs(self.tilt_deg) > 0.01:
            M = cv2.getRotationMatrix2D((self.width / 2, self.height / 2),
                                        self.tilt_deg, 1.0)
            # Borde CONSTANTE con el color de la tela, no REPLICATE:
            # replicar embarra la ultima fila de pixeles hacia la esquina,
            # y si justo ahi hay un defecto, su tinta aparece estirada
            # fuera de su caja. Las cunas de tela lisa en las esquinas se
            # leen como la orilla del rollo.
            out = cv2.warpAffine(out, M, (self.width, self.height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=_CLOTH)
        out = np.clip(out.astype(np.float32) * self._vignette,
                      0, 255).astype(np.uint8)
        return out

    # ── etiquetas para el dataset ─────────────────────────────────────────
    def visible_labels(self, min_side: int = 10,
                       min_visible: float = 0.55) -> list[tuple[str, int, int, int, int]]:
        """
        Defectos visibles en la ventana actual, como
        (clase, x1, y1, x2, y2) en coordenadas de FRAME.

        Un defecto que está medio fuera del encuadre se descarta: etiquetar
        la mitad de una mancha le enseña al modelo que esa mitad es la
        mancha completa.
        """
        out = []
        span_w = self.width if self.axis == "x" else self.fw
        span_h = self.height if self.axis == "y" else self.fh
        for d in self.defects:
            if d.w < 2 or d.h < 2:
                continue
            # Un trazo pintado sobre la costura de la tela se envuelve: sus
            # puntos caen en los dos extremos y la caja abarca la tela
            # entera. No se puede etiquetar bien, y entrenar con esa caja
            # ensena una barbaridad — se descarta.
            if d.w > span_w or d.h > span_h:
                continue
            corners = [self.fabric_to_screen(x, y)
                       for x, y in ((d.x1, d.y1), (d.x2, d.y1),
                                    (d.x2, d.y2), (d.x1, d.y2))]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            full = max(1, (x2 - x1) * (y2 - y1))
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(self.width, x2), min(self.height, y2)
            if cx2 - cx1 < min_side or cy2 - cy1 < min_side:
                continue
            if ((cx2 - cx1) * (cy2 - cy1)) / full < min_visible:
                continue
            out.append((d.cls, cx1, cy1, cx2, cy2))
        return out
