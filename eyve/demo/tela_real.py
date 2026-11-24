"""
Tela realista: el fork mejorado de la demo textil.

La demo de siempre (`TextilePattern`) es una tela perfecta: el motivo se
repite píxel a píxel, el hilo es un seno, la luz no cambia y la cámara no
tiene ruido.  Sirve para enseñar la idea, pero una máscara aprendida sobre
eso no dice nada de una línea real: con 12 fotos de tela real (23-nov) se vio
que lo que rompe la inspección es justo lo que la demo no tenía.

Esta clase hereda de la demo, así que la pantalla Demo, el dataset y las
pruebas la pueden usar igual, y agrega:

  La tela, que viaja con el rollo (se dibuja una vez):
    - dos tintas con su propio registro, y un registro que deriva despacio a lo
      largo del rollo;
    - tinta que se corre en la orilla del motivo y densidad desigual (la malla
      no deja pasar lo mismo en todos lados);
    - la tela deformada: arco de trama (la trama se curva a lo ancho), sesgo y
      un estiramiento suave que cambia a lo largo del rollo;
    - hilo por hilo: cada hilo con su grosor, su brillo y algún nudo, en vez de
      una onda perfecta.

  La cámara, que cambia cuadro a cuadro:
    - velocidad con tirones y vaivén a lo ancho (la tela no corre por un riel);
    - luz desigual: viñeta, caída hacia un lado, un brillo de la lámpara y un
      parpadeo pequeño;
    - foco imperfecto, barrido por el movimiento, ruido de sensor y, si se
      pide, compresión JPEG como la de una cámara USB.

  Fallas de estampado como las de una línea (todas con su caja de verdad):
    raya_racla, franja_sin_tinta, doble_impresion, salpicadura, aceite,
    hilo_jalado, agujero, hilacha y arruga.  Cada una guarda `tipo` (el nombre
    real) y `cls` (la clase de la demo de siempre más cercana, para que lo que
    ya consume `cls` no se rompa).

Lo que NO es: no simula la física de la impresión ni de la tela.  Es una tela
sintética más difícil y más honesta.  Lo que se mida aquí no sustituye a una
tela real.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from eyve.demo.textile import (MOTIFS, WEAVES, Defect, TextilePattern,
                               _CLOTH, _GREIGE)

#: fallas reales -> clase de la demo de siempre (lo que ya consume `cls`)
FALLAS_REALES: dict = {
    "raya_racla": "rayon",
    "franja_sin_tinta": "falta_impresion",
    "doble_impresion": "fantasma",
    "salpicadura": "mancha",
    "aceite": "mancha",
    "hilo_jalado": "rayon",
    "agujero": "mancha",
    "hilacha": "rayon",
    "arruga": "offset",
}

#: Paletas de dos tintas (BGR): la principal y la de apoyo.  Una tela real casi
#: nunca es de una sola tinta, y el fallo de registro solo existe con dos.
PALETAS = {
    "azul": ((150, 78, 30), (60, 150, 225)),       # azul + mostaza
    "rojo": ((40, 40, 170), (110, 120, 60)),       # rojo + verde oliva
    "negro": ((45, 42, 40), (160, 120, 205)),      # negro + rosa
}

_FONDO_BANDA = (34, 32, 30)     # lo que se ve por un agujero: la banda


class TelaRealista(TextilePattern):
    """
    Misma interfaz que `TextilePattern`, con tela y cámara realistas.

    rollo: cambia el hilo, la deformación y la densidad de la tinta sin
        cambiar el estampado (dos rollos del mismo diseño).
    realismo: 0..1, escala todo lo que la hace difícil (1 = lo medido con
        tela real; 0 queda cerca de la demo de siempre).
    """

    def __init__(self, width: int = 960, height: int = 540, axis: str = "x",
                 fabric_len: int = 3840, motif: str = "diamantes",
                 speed: float = 110.0, tilt_deg: float = 1.2,
                 seed: Optional[int] = None, weave: str = "sarga",
                 weave_amp: float = 0.10, paleta: str = "azul",
                 rollo: int = 0, realismo: float = 1.0,
                 jpeg: bool = False) -> None:
        self.paleta = paleta if paleta in PALETAS else "azul"
        self.rollo = int(rollo)
        self.realismo = float(np.clip(realismo, 0.0, 1.0))
        self.jpeg = bool(jpeg)
        self._rr = np.random.default_rng(1009 + 7919 * self.rollo)
        self._t = 0.0
        super().__init__(width, height, axis, fabric_len, motif, speed,
                         tilt_deg, seed, weave, weave_amp)
        # Ruido de sensor: un banco de cuadros fijos que se rota.  Generarlo
        # cada cuadro costaba 9 ms; el banco no se nota (el ruido no se repite
        # en el mismo lugar porque la tela se mueve debajo).
        rng = np.random.default_rng(31 + self.rollo)
        self._ruido = [rng.normal(0.0, 1.0, (self.height, self.width, 1))
                       .astype(np.float32) for _ in range(6)]
        self._iruido = 0
        self._vaiven = 0.0

    # ── la tela: estampado con dos tintas, tinta corrida y tela deformada ──
    def _render_fabric(self) -> np.ndarray:
        r = getattr(self, "realismo", 1.0)
        rr = getattr(self, "_rr", np.random.default_rng(1009))
        h, w = self.fh, self.fw
        tinta1, tinta2 = PALETAS[getattr(self, "paleta", "azul")]

        # Las dos capas de tinta se dibujan como máscaras (0..1), al doble de
        # resolución, y luego se bajan: orillas suaves como las de verdad.
        m1, m2 = self._mascaras_tinta(2)
        m1 = cv2.resize(m1, (w, h), interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(m2, (w, h), interpolation=cv2.INTER_AREA)

        # Tinta corrida: la orilla se abre un poco y se suaviza.
        if r > 0:
            m1 = np.clip(cv2.GaussianBlur(m1, (0, 0), 0.6 + 0.5 * r) * 1.15, 0, 1)
            m2 = np.clip(cv2.GaussianBlur(m2, (0, 0), 0.6 + 0.5 * r) * 1.15, 0, 1)

        # Registro: la segunda tinta cae corrida respecto de la primera, y ese
        # corrimiento deriva despacio a lo largo del rollo (periódico en el
        # eje de viaje, para que el rollo no tenga costura).
        eje = np.arange(w if self.axis == "x" else h, dtype=np.float32)
        largo = float(len(eje))
        fase = rr.uniform(0, 2 * math.pi)
        deriva = (1.2 * r) * np.sin(2 * math.pi * eje / largo + fase)
        reg = (rr.uniform(-1.5, 1.5) * r, rr.uniform(-1.5, 1.5) * r)
        m2 = self._desplazar(m2, reg[0] + (deriva if self.axis == "x" else 0),
                             reg[1] + (deriva[:, None] if self.axis == "y" else 0))

        # Densidad desigual de la tinta: la malla no deja pasar lo mismo en
        # todos lados.  Baja frecuencia, periódica en el eje de viaje.
        dens1 = 1.0 - 0.12 * r * self._ruido_suave(h, w, 70, rr)
        dens2 = 1.0 - 0.12 * r * self._ruido_suave(h, w, 55, rr)
        a1 = (m1 * dens1)[..., None]
        a2 = (m2 * dens2)[..., None]

        # Tinta sobre tela: se multiplica (la tinta filtra la luz), no se pega.
        tela = np.array(_CLOTH, np.float32)[None, None, :]
        t1 = np.array(tinta1, np.float32)[None, None, :] / 255.0
        t2 = np.array(tinta2, np.float32)[None, None, :] / 255.0
        img = tela * (1 - a1 + a1 * t1)
        img = img * (1 - a2 + a2 * t2)

        # La tela deformada: arco de trama, sesgo y estiramiento suave.  Todo
        # viaja con el rollo, así que se hornea una vez aquí.
        img = self._deformar(img, r, rr)
        self._tinta = self._deformar(np.maximum(m1, m2), r, rr, gemelo=True)
        # fibra: ruido fino de la tela (no del sensor), viaja con ella
        fib = rr.normal(0.0, 3.0 * (0.5 + 0.5 * r), (h, w, 1)).astype(np.float32)
        return np.clip(img + fib, 0, 255).astype(np.uint8)

    def _mascaras_tinta(self, s: int):
        """Las dos tintas del motivo como máscaras 0..1, a escala s."""
        h, w, p = self.fh * s, self.fw * s, self.period * s
        m1 = np.zeros((h, w), np.float32)
        m2 = np.zeros((h, w), np.float32)
        span = w if self.axis == "x" else h
        for gy in range(-p, h + p, p):
            for gx in range(-p, w + p, p):
                off = (p // 2) if ((gy // p) % 2) else 0
                for dx, dy in ((0, 0), (span, 0), (-span, 0)) if self.axis == "x" \
                        else ((0, 0), (0, span), (0, -span)):
                    self._motivo(m1, m2, gx + off + dx, gy + dy, p, s)
        return m1, m2

    def _motivo(self, m1, m2, cx, cy, p, s) -> None:
        aa = cv2.LINE_AA
        if self.motif == "diamantes":
            r = p // 3
            pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r],
                            [cx - r, cy]], np.int32)
            cv2.fillPoly(m1, [pts], 1.0, aa)
            r2 = r // 2
            pts2 = np.array([[cx, cy - r2], [cx + r2, cy], [cx, cy + r2],
                             [cx - r2, cy]], np.int32)
            cv2.fillPoly(m1, [pts2], 0.0, aa)
            cv2.circle(m2, (cx, cy), max(2, r2 // 2), 1.0, -1, aa)
        elif self.motif == "flores":
            r = p // 4
            for k in range(6):
                a = k * math.pi / 3
                cv2.circle(m2, (int(cx + math.cos(a) * r), int(cy + math.sin(a) * r)),
                           max(3, r // 2), 1.0, -1, aa)
            cv2.circle(m1, (cx, cy), max(3, r // 2), 1.0, -1, aa)
        elif self.motif == "rayas":
            t = max(3, p // 12)
            cv2.line(m1, (cx - p // 2, cy - p // 2), (cx + p // 2, cy + p // 2),
                     1.0, t, aa)
            cv2.line(m2, (cx - p // 2, cy), (cx + p // 2, cy + p), 1.0, t, aa)
        else:   # puntos
            cv2.circle(m1, (cx, cy), max(4, p // 7), 1.0, -1, aa)
            q = p // 2
            for ox, oy in ((q, q), (-q, q), (q, -q)):
                cv2.circle(m2, (cx + ox, cy + oy), max(3, p // 10), 1.0, -1, aa)

    def _ruido_suave(self, h, w, escala, rng) -> np.ndarray:
        """Ruido 0..1 de baja frecuencia, periódico en el eje de viaje."""
        if self.axis == "x":
            gh, gw = max(2, h // escala), max(2, w // escala)
            g = rng.random((gh, gw)).astype(np.float32)
            g = np.concatenate([g, g[:, :1]], axis=1)          # cierra la vuelta
            big = cv2.resize(g, (w + w // gw, h), interpolation=cv2.INTER_CUBIC)
            return np.clip(big[:, :w], 0, 1)
        gh, gw = max(2, h // escala), max(2, w // escala)
        g = rng.random((gh, gw)).astype(np.float32)
        g = np.concatenate([g, g[:1]], axis=0)
        big = cv2.resize(g, (w, h + h // gh), interpolation=cv2.INTER_CUBIC)
        return np.clip(big[:h], 0, 1)

    def _desplazar(self, img, dx, dy):
        h, w = img.shape[:2]
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        mx = (xs - np.asarray(dx, np.float32))
        my = (ys - np.asarray(dy, np.float32))
        if self.axis == "x":
            mx = np.mod(mx, w)
        else:
            my = np.mod(my, h)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    def _deformar(self, img, r, rng, gemelo=False):
        """
        Arco de trama, sesgo y estiramiento suave.

        El arco curva la trama a lo ancho (es el defecto de geometría más
        común en tela tejida); el sesgo la inclina.  Los dos se dan en
        píxeles en la orilla.  `gemelo` reusa el campo del último llamado,
        para deformar la máscara de tinta exactamente igual que la tela.
        """
        if r <= 0:
            return img
        h, w = img.shape[:2]
        if not gemelo or not hasattr(self, "_campo"):
            corto = h if self.axis == "x" else w
            u = np.linspace(-1, 1, corto, dtype=np.float32)
            arco = rng.uniform(2.0, 5.0) * r * (u ** 2 - 1 / 3.0)
            sesgo = rng.uniform(-3.0, 3.0) * r * u
            est = (3.0 * r) * (self._ruido_suave(h, w, 160, rng) - 0.5)
            if self.axis == "x":
                d_viaje = (arco + sesgo)[:, None] + est
                d_ancho = (1.5 * r) * (self._ruido_suave(h, w, 200, rng) - 0.5)
            else:
                d_viaje = (arco + sesgo)[None, :] + est
                d_ancho = (1.5 * r) * (self._ruido_suave(h, w, 200, rng) - 0.5)
            self._campo = (d_viaje.astype(np.float32), d_ancho.astype(np.float32))
        d_viaje, d_ancho = self._campo
        if self.axis == "x":
            return self._desplazar(img, d_viaje, d_ancho)
        return self._desplazar(img, d_ancho, d_viaje)

    # ── el hilo, hilo por hilo ─────────────────────────────────────────────
    def _render_weave(self) -> np.ndarray:
        r = getattr(self, "realismo", 1.0)
        if self.weave == "ninguno" or self.weave_amp <= 0:
            return np.ones((self.fh, self.fw, 3), np.float32)
        rr = np.random.default_rng(4241 + 7919 * getattr(self, "rollo", 0))
        h, w = self.fh, self.fw
        paso = {"sarga": 4.2, "tafetan": 3.6, "sarga_fina": 3.0,
                "canasta": 4.5}.get(self.weave, 4.0)

        def hilos(n_px, periodico):
            # posiciones de cada hilo con su paso propio; si el eje da la
            # vuelta, el total se ajusta para cerrar sin costura
            k = int(round(n_px / paso))
            pasos = paso * (1 + rr.normal(0, 0.07 * r, k))
            if periodico:
                pasos *= n_px / pasos.sum()
            pos = np.concatenate([[0.0], np.cumsum(pasos)])
            idx = np.clip(np.searchsorted(pos, np.arange(n_px) + 0.5) - 1,
                          0, k - 1)
            frac = (np.arange(n_px) + 0.5 - pos[idx]) / pasos[idx]
            brillo = rr.normal(0, 0.35 * r, k)       # cada hilo distinto
            return idx, frac.astype(np.float32), brillo.astype(np.float32), k

        ix, fx, bx, kx = hilos(w, self.axis == "x")
        iy, fy, by, ky = hilos(h, self.axis == "y")
        # perfil de un hilo: redondo, más claro al centro
        px = np.cos((fx - 0.5) * math.pi).astype(np.float32)
        py = np.cos((fy - 0.5) * math.pi).astype(np.float32)
        I, J = np.meshgrid(ix, iy)
        if self.weave in ("sarga", "sarga_fina"):
            arriba = ((I + J) % 3) < (2 if self.weave == "sarga" else 1)
        elif self.weave == "canasta":
            arriba = ((I // 2 + J // 2) % 2) == 0
        else:
            arriba = ((I + J) % 2) == 0
        PX, PY = np.meshgrid(px, py)
        BX, BY = np.meshgrid(bx[ix], by[iy])
        t = np.where(arriba, PX + 0.6 * BX, PY + 0.6 * BY) - 0.5
        # nudos: tramos cortos de un hilo más gruesos y claros
        nudos = np.zeros((h, w), np.float32)
        for _ in range(int(0.00004 * h * w * r)):
            x0, y0 = int(rr.integers(0, w)), int(rr.integers(0, h))
            largo = int(rr.integers(8, 30))
            if self.axis == "x":
                cv2.line(nudos, (x0, y0), (x0 + largo, y0), 1.0, 2, cv2.LINE_AA)
            else:
                cv2.line(nudos, (x0, y0), (x0, y0 + largo), 1.0, 2, cv2.LINE_AA)
        t = t + 0.8 * nudos
        t = t / (np.abs(t).max() or 1.0)
        mult = (1.0 + self.weave_amp * t).astype(np.float32)
        return np.repeat(mult[:, :, None], 3, axis=2)

    # ── la luz ──────────────────────────────────────────────────────────────
    def _make_vignette(self) -> np.ndarray:
        r = getattr(self, "realismo", 1.0)
        ys = np.linspace(-1.0, 1.0, self.height, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, self.width, dtype=np.float32)[None, :]
        rad = np.sqrt(xs ** 2 + ys ** 2) / math.sqrt(2.0)
        v = 1.0 - 0.30 * rad ** 2
        v = v * (1.0 - 0.06 * r * xs)                 # cae hacia un lado
        # el brillo de la lámpara: una mancha clara y ancha, fuera de centro
        v = v + 0.05 * r * np.exp(-((xs + 0.35) ** 2 / 0.08 + (ys + 0.25) ** 2 / 0.3))
        return np.repeat(v[:, :, None].astype(np.float32), 3, axis=2)

    # ── el movimiento y la cámara ──────────────────────────────────────────
    def advance(self, dt: float) -> None:
        r = self.realismo
        self._t += dt
        # tirones: la velocidad no es constante
        v = self.speed * (1.0 + 0.06 * r * math.sin(self._t * 1.7) +
                          0.03 * r * self._rng.uniform(-1, 1))
        span = self.fw if self.axis == "x" else self.fh
        self.offset = (self.offset + v * dt) % span
        # vaivén a lo ancho, en px
        self._vaiven = 2.5 * r * math.sin(self._t * 0.9) + 0.6 * r * math.sin(self._t * 3.1)

    def frame(self) -> np.ndarray:
        r = self.realismo
        if r <= 0:
            return super().frame()
        tilt, self.tilt_deg = self.tilt_deg, 0.0
        vig, self._vignette = self._vignette, np.ones_like(self._vignette)
        try:
            out = super().frame()          # tela + fallas + hilo, sin girar
        finally:
            self.tilt_deg, self._vignette = tilt, vig
        # giro + vaivén en una sola transformación
        M = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), tilt, 1.0)
        if self.axis == "x":
            M[1, 2] += self._vaiven
        else:
            M[0, 2] += self._vaiven
        out = cv2.warpAffine(out, M, (self.width, self.height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=_CLOTH)
        # foco y barrido: la tela se mueve mientras el obturador está abierto
        barrido = max(1, int(round(self.speed * r / 110.0 * 2)))
        if barrido > 1:
            k = np.ones((1, barrido), np.float32) / barrido
            out = cv2.filter2D(out, -1, k if self.axis == "x" else k.T)
        out = cv2.GaussianBlur(out, (0, 0), 0.5 + 0.3 * r)
        # luz: viñeta, caída, brillo de lámpara y un parpadeo pequeño
        f = out.astype(np.float32) * self._vignette
        f *= 1.0 + 0.012 * r * math.sin(self._t * 12.0)
        # ruido de sensor: más en lo claro (disparo) más un piso (lectura)
        n = self._ruido[self._iruido % len(self._ruido)]
        self._iruido += 1
        f += n * (1.2 + 0.12 * np.sqrt(np.maximum(f, 0))) * r
        out = np.clip(f, 0, 255).astype(np.uint8)
        if self.jpeg:
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return out

    def fabric_to_screen(self, fx: int, fy: int):
        sx, sy = super().fabric_to_screen(fx, fy)
        if self.axis == "x":
            return sx, int(round(sy + self._vaiven))
        return int(round(sx + self._vaiven)), sy

    def screen_to_fabric(self, sx: int, sy: int):
        if self.axis == "x":
            return super().screen_to_fabric(sx, sy - self._vaiven)
        return super().screen_to_fabric(sx - self._vaiven, sy)

    # ── fallas reales ───────────────────────────────────────────────────────
    def _marcar(self, tipo: str, ys, xs, caja, nuevo, alfa) -> Defect:
        d = self._stamp(FALLAS_REALES[tipo], ys, xs, caja, nuevo, alfa)
        d.tipo = tipo
        return d

    def _region_viaje(self, fx, fy, largo, ancho):
        """Región alargada en el eje de viaje (las fallas de la máquina corren
        con la tela)."""
        if self.axis == "x":
            return self._region(fx, fy, largo, ancho)
        return self._region(fx, fy, ancho, largo)

    def _rampa_viaje(self, shape, entra=0.15):
        """0..1 a lo largo del viaje: la falla entra y sale de a poco."""
        h, w = shape
        n = w if self.axis == "x" else h
        u = np.linspace(0, 1, n, dtype=np.float32)
        r = np.clip(np.minimum(u, 1 - u) / entra, 0, 1)
        return r[None, :] if self.axis == "x" else r[:, None]

    def raya_racla(self, fx, fy, largo=900, ancho=3, fuerza=None) -> Defect:
        """Raya de racla: la cuchilla deja pasar tinta en una línea que corre
        con la tela.  Delgada, larga, del color de la tinta principal."""
        fuerza = self._rng.uniform(0.35, 0.8) if fuerza is None else fuerza
        ys, xs, caja = self._region_viaje(fx, fy, largo, ancho * 4 + 6)
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        h, w = base.shape[:2]
        m = np.zeros((h, w), np.float32)
        if self.axis == "x":
            y = h / 2.0 + np.cumsum(self._rng_np().normal(0, 0.05, w))
            for x in range(w):
                cv2.circle(m, (x, int(y[x])), max(1, ancho // 2), 1.0, -1, cv2.LINE_AA)
        else:
            x_ = w / 2.0 + np.cumsum(self._rng_np().normal(0, 0.05, h))
            for y in range(h):
                cv2.circle(m, (int(x_[y]), y), max(1, ancho // 2), 1.0, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 0.7) * self._rampa_viaje((h, w), 0.05)
        tinta = np.array(PALETAS[self.paleta][0], np.float32) / 255.0
        nuevo = base * (1 - fuerza + fuerza * tinta[None, None, :])
        return self._marcar("raya_racla", ys, xs, caja, nuevo, m)

    def franja_sin_tinta(self, fx, fy, largo=1200, ancho=24, severidad=None) -> Defect:
        """Malla tapada: una franja a lo largo del viaje donde la tinta casi no
        llega.  Solo aclara donde hay tinta: la tela de fondo no cambia."""
        severidad = self._rng.uniform(0.55, 0.95) if severidad is None else severidad
        ys, xs, caja = self._region_viaje(fx, fy, largo, ancho)
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        tinta = self._tinta[np.ix_(ys, xs)][..., None]
        cruda = np.array(_GREIGE, np.float32)[None, None, :]
        h, w = base.shape[:2]
        perfil = np.cos(np.linspace(-math.pi / 2, math.pi / 2,
                                    h if self.axis == "x" else w)) ** 0.6
        perfil = perfil[:, None] if self.axis == "x" else perfil[None, :]
        k = severidad * perfil * self._rampa_viaje((h, w), 0.2)
        k = k * (0.7 + 0.3 * self._mottle((h, w), self._rng.randrange(1 << 30), 20))
        nuevo = base * (1 - k[..., None] * tinta) + cruda * (k[..., None] * tinta)
        return self._marcar("franja_sin_tinta", ys, xs, caja, nuevo,
                            np.clip(k * 1.5, 0, 1))

    def doble_impresion(self, fx, fy, largo=700, ancho=220, fuerza=None) -> Defect:
        """Doble impresión: la malla toca dos veces y deja un eco corrido unos
        milímetros, a lo largo de un tramo del rollo."""
        fuerza = self._rng.uniform(0.25, 0.5) if fuerza is None else fuerza
        ys, xs, caja = self._region_viaje(fx, fy, largo, ancho)
        dx = int(self.period * self._rng.uniform(0.08, 0.25)) * self._rng.choice((-1, 1))
        dy = int(self.period * self._rng.uniform(0.05, 0.2)) * self._rng.choice((-1, 1))
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        eco = self._base[np.ix_((ys + dy) % self.fh, (xs + dx) % self.fw)].astype(np.float32)
        nuevo = np.minimum(base, eco * fuerza + base * (1 - fuerza))
        h, w = base.shape[:2]
        alfa = self._rampa_viaje((h, w), 0.25) * \
            (0.6 + 0.4 * self._mottle((h, w), self._rng.randrange(1 << 30), 30))
        return self._marcar("doble_impresion", ys, xs, caja, nuevo, alfa)

    def salpicadura(self, fx, fy, size=40) -> Defect:
        """Gota de tinta con gotitas alrededor y la orilla corrida en el hilo."""
        lado = int(size * 2.4)
        ys, xs, caja = self._region(fx, fy, lado, lado)
        h, w = len(ys), len(xs)
        m = np.zeros((h, w), np.float32)
        g = self._rng_np()
        cv2.circle(m, (w // 2, h // 2), int(size * 0.35), 1.0, -1, cv2.LINE_AA)
        for _ in range(int(g.integers(4, 10))):
            a = g.uniform(0, 2 * math.pi)
            d = g.uniform(0.4, 1.0) * size
            cv2.circle(m, (int(w / 2 + math.cos(a) * d), int(h / 2 + math.sin(a) * d)),
                       int(g.uniform(2, size * 0.15)), 1.0, -1, cv2.LINE_AA)
        # orilla irregular: el umbral sobre ruido la rompe
        rugo = self._mottle((h, w), self._rng.randrange(1 << 30), 6)
        m = np.clip(cv2.GaussianBlur(m, (0, 0), 1.5) * (0.75 + 0.5 * rugo), 0, 1)
        m = (m > 0.45).astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), 0.9)
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        tinta = np.array(PALETAS[self.paleta][self._rng.randrange(2)], np.float32) / 255.0
        nuevo = base * (0.15 + 0.85 * tinta[None, None, :])
        return self._marcar("salpicadura", ys, xs, caja, nuevo, m * 0.95)

    def aceite(self, fx, fy, size=90) -> Defect:
        """Grasa de la máquina: oscurece y amarillea, sin orilla clara."""
        lado = int(size * 1.6)
        ys, xs, caja = self._region(fx, fy, lado, int(lado * 0.8))
        h, w = len(ys), len(xs)
        m = self._mottle((h, w), self._rng.randrange(1 << 30), 18)
        m = np.clip((m - 0.25) * 2.2, 0, 1)
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        nuevo = base * np.array((0.72, 0.84, 0.92), np.float32)[None, None, :]
        return self._marcar("aceite", ys, xs, caja, nuevo, m * 0.9)

    def hilo_jalado(self, fx, fy, largo=None) -> Defect:
        """Un hilo de trama jalado: una línea a lo ancho donde el tejido se
        tensa y el estampado se corre un par de píxeles."""
        corto = self.fh if self.axis == "x" else self.fw
        largo = int(corto * self._rng.uniform(0.4, 0.9)) if largo is None else largo
        if self.axis == "x":
            ys, xs, caja = self._region(fx, fy, 14, largo)
        else:
            ys, xs, caja = self._region(fx, fy, largo, 14)
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        h, w = base.shape[:2]
        if self.axis == "x":
            u = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
            corr = 4.0 * np.exp(-u ** 2 / 0.15) * np.sign(u)
            mx, my = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            mov = cv2.remap(base, mx - corr, my, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
            linea = np.exp(-u ** 2 / 0.02)
        else:
            u = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
            corr = 4.0 * np.exp(-u ** 2 / 0.15) * np.sign(u)
            mx, my = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            mov = cv2.remap(base, mx, my - corr, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
            linea = np.exp(-u ** 2 / 0.02)
        nuevo = mov * (1 + 0.25 * linea[..., None])
        alfa = np.clip(0.3 + np.exp(-u ** 2 / 0.3), 0, 1) * np.ones((h, w), np.float32)
        return self._marcar("hilo_jalado", ys, xs, caja, nuevo, alfa)

    def agujero(self, fx, fy, size=10) -> Defect:
        """Agujero: se ve la banda de abajo, con la orilla deshilachada."""
        lado = size * 4
        ys, xs, caja = self._region(fx, fy, lado, lado)
        h, w = len(ys), len(xs)
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (w // 2, h // 2), (size // 2 + 1, int(size * 0.4) + 1),
                    self._rng.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
        rugo = self._mottle((h, w), self._rng.randrange(1 << 30), 3)
        m = np.clip(cv2.GaussianBlur(m, (0, 0), 1.2) * (0.6 + 0.8 * rugo), 0, 1)
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        hueco = np.array(_FONDO_BANDA, np.float32)[None, None, :]
        nuevo = hueco * np.ones_like(base)
        # fibras sueltas en la orilla
        g = self._rng_np()
        for _ in range(8):
            a = g.uniform(0, 2 * math.pi)
            cv2.line(m, (w // 2, h // 2), (int(w / 2 + math.cos(a) * size),
                                           int(h / 2 + math.sin(a) * size)),
                     0.5, 1, cv2.LINE_AA)
        return self._marcar("agujero", ys, xs, caja, nuevo, np.clip(m, 0, 1))

    def hilacha(self, fx, fy, largo=70) -> Defect:
        """Una fibra suelta encima de la tela: curva, delgada y clara u oscura."""
        lado = largo + 20
        ys, xs, caja = self._region(fx, fy, lado, lado)
        h, w = len(ys), len(xs)
        g = self._rng_np()
        p = np.array([[g.uniform(0, w), g.uniform(0, h)] for _ in range(4)])
        t = np.linspace(0, 1, 120)[:, None]
        curva = ((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] +
                 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]).astype(np.int32)
        m = np.zeros((h, w), np.float32)
        cv2.polylines(m, [curva.reshape(-1, 1, 2)], False, 1.0, 1, cv2.LINE_AA)
        clara = self._rng.random() < 0.5
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        nuevo = np.full_like(base, 250.0 if clara else 70.0)
        return self._marcar("hilacha", ys, xs, caja, nuevo, m * 0.8)

    def arruga(self, fx, fy, largo=260, ancho=40) -> Defect:
        """Arruga o pliegue: una banda de luz y sombra a lo ancho."""
        if self.axis == "x":
            ys, xs, caja = self._region(fx, fy, ancho, largo)
        else:
            ys, xs, caja = self._region(fx, fy, largo, ancho)
        base = self._base[np.ix_(ys, xs)].astype(np.float32)
        h, w = base.shape[:2]
        n = w if self.axis == "x" else h
        u = np.linspace(-1, 1, n, dtype=np.float32)
        perfil = 0.18 * np.sin(u * math.pi) * np.exp(-u ** 2 / 0.5)
        perfil = perfil[None, :] if self.axis == "x" else perfil[:, None]
        nuevo = base * (1 + perfil[..., None])
        # a lo largo del pliegue entra y sale de a poco
        m = h if self.axis == "x" else w
        v = np.linspace(0, 1, m, dtype=np.float32)
        rampa = np.clip(np.minimum(v, 1 - v) / 0.3, 0, 1)
        alfa = np.ones((h, w), np.float32) * (rampa[:, None] if self.axis == "x"
                                              else rampa[None, :])
        return self._marcar("arruga", ys, xs, caja, nuevo, alfa)

    def _rng_np(self) -> np.random.Generator:
        return np.random.default_rng(self._rng.randrange(1 << 30))

    def falla(self, tipo: str, fx: int, fy: int, **kw) -> Defect:
        """Una falla por su nombre real."""
        if tipo not in FALLAS_REALES:
            raise ValueError(f"falla desconocida: {tipo}")
        return getattr(self, tipo)(fx, fy, **kw)
