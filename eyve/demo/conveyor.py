"""
Escenas para explicar el módulo de conteo.

"Contar" suena a una sola cosa hasta que alguien pregunta *qué* cuenta, y
ahí se descubre que son cinco preguntas distintas.  Explicarlo hablando
cuesta; viéndolo es inmediato.  Por eso cada método tiene aquí su propia
escena, la que lo hace obvio:

    en pantalla    piezas entrando y saliendo — el número sube y baja
    cruce de meta  una banda y una línea — cada pieza suma al cruzarla
    zona           un área — suma al entrar, y muestra cuántas hay dentro
    al aparecer    piezas que se materializan donde sea
    al desaparecer piezas que alguien retira

De dónde salen las detecciones
──────────────────────────────
La escena sabe dónde está cada pieza, así que las entrega directamente en
vez de pasar por un modelo.  Es deliberado: lo que esta demo explica es la
LÓGICA DE CONTEO, y meter errores de detección en medio sólo enturbiaría la
explicación.  El seguimiento y el conteo sí son los de producción —el mismo
`InstanceTracker` y el mismo `CountingModule`—, así que lo que el visitante
ve contar es exactamente lo que cuenta en una planta.

La pantalla lo dice con todas sus letras; no se presenta como detección.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

#: cómo se llama la pieza que se cuenta (una sola clase: aquí no se
#: clasifica nada, sólo se cuenta)
PIECE_LABEL = "pieza"

#: escenas, en el orden en que el bucle las muestra
SCENES: tuple[str, ...] = ("screen", "line", "zone", "appear", "disappear")

_BG = (26, 22, 20)             # fondo industrial oscuro (BGR)
_BELT = (46, 41, 38)           # la banda
_BELT_EDGE = (70, 63, 58)
_PIECE = (120, 170, 235)       # la pieza: ámbar claro
_PIECE_DARK = (70, 110, 165)
_SHADOW = (14, 12, 11)


_uid_counter = 0


def _next_uid() -> int:
    global _uid_counter
    _uid_counter += 1
    return _uid_counter


@dataclass
class Piece:
    """
    Una pieza de la escena, en coordenadas de pantalla.

    `uid` existe para poder seguirla desde fuera.  No vale usar id() de
    Python: lo reutiliza en cuanto un objeto se libera, asi que dos piezas
    distintas pueden compartirlo y el seguimiento se corrompe en silencio.
    """
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    size: int = 54
    #: 0..1 — aparece y desaparece con un fundido, para que el visitante
    #: vea el momento en que entra o sale en vez de un parpadeo
    alpha: float = 1.0
    fading_in: bool = False
    fading_out: bool = False
    spin: float = 0.0
    uid: int = field(default_factory=_next_uid)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        h = self.size // 2
        return (int(self.x - h), int(self.y - h),
                int(self.x + h), int(self.y + h))


class CountingScene:
    """
    Genera la escena de un método de conteo.

    step(dt) devuelve (frame, detecciones).  Las detecciones son las piezas
    con alpha suficiente: una que se está desvaneciendo deja de reportarse
    cuando ya casi no se ve, igual que un detector dejaría de verla.
    """

    def __init__(self, method: str = "screen", width: int = 960,
                 height: int = 540, seed: Optional[int] = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self.method = method if method in SCENES else "screen"
        self._rng = random.Random(seed)
        self.pieces: list[Piece] = []
        self._t = 0.0
        self._next_spawn = 0.0
        self._belt_phase = 0.0
        self.reset()

    # ── control ───────────────────────────────────────────────────────────
    def set_method(self, method: str) -> None:
        if method in SCENES and method != self.method:
            self.method = method
            self.reset()

    def reset(self) -> None:
        self.pieces.clear()
        self._t = 0.0
        self._next_spawn = 0.0
        if self.method in ("screen", "line", "zone", "disappear"):
            # arrancar con la banda ya poblada: una escena que empieza vacía
            # tarda en explicar algo
            for i in range(4):
                self._spawn(x=self.width - 1 - i * 230)

    # ── geometría que la pantalla necesita dibujar ────────────────────────
    @property
    def belt_y(self) -> int:
        return self.height // 2

    @property
    def line_x(self) -> int:
        return self.width // 2

    @property
    def zone(self) -> tuple[int, int, int, int]:
        w, h = int(self.width * 0.26), int(self.height * 0.42)
        cx, cy = self.width // 2, self.belt_y
        return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)

    # ── piezas ────────────────────────────────────────────────────────────
    def _spawn(self, x: Optional[float] = None,
               y: Optional[float] = None) -> Piece:
        vel = -135.0            # la banda va de derecha a izquierda
        p = Piece(
            x=self.width + 40 if x is None else x,
            y=self.belt_y if y is None else y,
            vx=vel, vy=0.0,
            size=self._rng.randint(46, 62),
            spin=self._rng.uniform(-0.25, 0.25),
        )
        self.pieces.append(p)
        return p

    def _spawn_anywhere(self) -> Piece:
        m = 80
        p = Piece(
            x=self._rng.randint(m, self.width - m),
            y=self._rng.randint(m, self.height - m),
            vx=0.0, vy=0.0,
            size=self._rng.randint(46, 62),
            alpha=0.0, fading_in=True,
            spin=self._rng.uniform(-0.3, 0.3),
        )
        self.pieces.append(p)
        return p

    # ── un paso de simulación ─────────────────────────────────────────────
    def step(self, dt: float) -> tuple[np.ndarray, list]:
        dt = min(max(dt, 0.0), 0.2)      # un frame perdido no debe saltar
        self._t += dt
        self._belt_phase = (self._belt_phase + 135.0 * dt) % 48.0

        if self.method == "appear":
            self._step_appear(dt)
        elif self.method == "disappear":
            self._step_disappear(dt)
        else:
            self._step_belt(dt)

        for p in self.pieces:
            p.x += p.vx * dt
            p.y += p.vy * dt
            if p.fading_in:
                p.alpha = min(1.0, p.alpha + dt * 2.2)
                if p.alpha >= 1.0:
                    p.fading_in = False
            elif p.fading_out:
                p.alpha = max(0.0, p.alpha - dt * 2.2)

        # fuera las que ya no pintan nada
        self.pieces = [p for p in self.pieces
                       if p.x > -120 and not (p.fading_out and p.alpha <= 0.01)]

        return self._render(), self._detections()

    def _step_belt(self, dt: float) -> None:
        """Banda continua: sirve para 'en pantalla', 'cruce' y 'zona'."""
        if self._t >= self._next_spawn:
            self._next_spawn = self._t + self._rng.uniform(1.5, 2.6)
            self._spawn()

    def _step_appear(self, dt: float) -> None:
        """
        Piezas que se materializan donde sea y se quedan un rato.

        Es el caso de algo que aparece sin un lado fijo de llegada: piezas
        que caen, que se destapan, que se imprimen.
        """
        if self._t >= self._next_spawn:
            self._next_spawn = self._t + self._rng.uniform(1.3, 2.2)
            self._spawn_anywhere()
        # las más viejas se van, para que la escena no se sature
        if len(self.pieces) > 6:
            for p in self.pieces[:len(self.pieces) - 6]:
                p.fading_out = True

    def _step_disappear(self, dt: float) -> None:
        """
        Piezas que alguien retira: al llegar a la izquierda, se las llevan.

        Es el caso de piezas que salen del encuadre o que una mano recoge.
        """
        if self._t >= self._next_spawn:
            self._next_spawn = self._t + self._rng.uniform(1.6, 2.6)
            self._spawn()
        for p in self.pieces:
            if p.x < self.width * 0.22 and not p.fading_out:
                p.fading_out = True
                p.vx = -40.0

    # ── detecciones ───────────────────────────────────────────────────────
    def _detections(self) -> list:
        """
        Las piezas como detecciones, para el tracker de producción.

        Una pieza a medio desvanecer deja de reportarse: es lo que haría un
        detector real, y es lo que permite que "al desaparecer" cuente.
        """
        from eyve.inference.tracker import RawDetection
        out = []
        for p in self.pieces:
            if p.alpha < 0.45:
                continue
            x1, y1, x2, y2 = p.bbox
            if x2 < 0 or x1 > self.width:
                continue
            out.append(RawDetection(PIECE_LABEL, 0.95,
                                    max(0, x1), max(0, y1),
                                    min(self.width, x2), min(self.height, y2)))
        return out

    # ── dibujo ────────────────────────────────────────────────────────────
    def _render(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), _BG, dtype=np.uint8)
        if self.method != "appear":
            self._draw_belt(img)
        for p in self.pieces:
            self._draw_piece(img, p)
        return img

    def _draw_belt(self, img: np.ndarray) -> None:
        """Banda transportadora, con las traviesas moviéndose."""
        # ancha a proposito: con un tercio del alto quedaban franjas
        # negras arriba y abajo y la escena se veia a medio llenar
        h = int(self.height * 0.56)
        y1, y2 = self.belt_y - h // 2, self.belt_y + h // 2
        cv2.rectangle(img, (0, y1), (self.width, y2), _BELT, -1)
        cv2.line(img, (0, y1), (self.width, y1), _BELT_EDGE, 2)
        cv2.line(img, (0, y2), (self.width, y2), _BELT_EDGE, 2)
        # traviesas: dan la sensación de movimiento aunque no haya piezas
        for x in range(-48, self.width + 48, 48):
            xx = int(x - self._belt_phase)
            cv2.line(img, (xx, y1 + 3), (xx, y2 - 3), _BELT_EDGE, 1)

    def _draw_piece(self, img: np.ndarray, p: Piece) -> None:
        """
        La pieza: un cuadrado redondeado con una muesca, para que se note
        que gira y para que no parezca un punto abstracto.
        """
        a = max(0.0, min(1.0, p.alpha))
        if a <= 0.01:
            return
        capa = img.copy()
        s = p.size // 2
        ang = p.spin * self._t * 60.0
        M = cv2.getRotationMatrix2D((0, 0), ang, 1.0)
        esquinas = np.array([[-s, -s], [s, -s], [s, s], [-s, s]],
                            dtype=np.float32)
        rot = (esquinas @ M[:, :2].T) + np.array([p.x, p.y], dtype=np.float32)
        pts = rot.astype(np.int32)

        # pegada: separada parecian dos piezas superpuestas
        sombra = pts + np.array([2, 3], dtype=np.int32)
        cv2.fillConvexPoly(capa, sombra, _SHADOW, cv2.LINE_AA)
        cv2.fillConvexPoly(capa, pts, _PIECE, cv2.LINE_AA)
        cv2.polylines(capa, [pts], True, _PIECE_DARK, 2, cv2.LINE_AA)
        # muesca: un detalle que hace legible el giro
        cen = np.array([p.x, p.y], dtype=np.float32)
        muesca = ((esquinas[:2] * 0.42) @ M[:, :2].T) + cen
        cv2.line(capa, tuple(muesca[0].astype(int)),
                 tuple(muesca[1].astype(int)), _PIECE_DARK, 3, cv2.LINE_AA)

        if a >= 0.999:
            img[:] = capa
        else:
            cv2.addWeighted(capa, a, img, 1 - a, 0, img)


@dataclass
class SceneInfo:
    """Lo que la pantalla necesita contar sobre cada método."""
    key: str
    draws_line: bool = False
    draws_zone: bool = False
    expect: tuple[Optional[int], Optional[int]] = (None, None)


#: qué geometría y qué límites usa cada escena
SCENE_INFO: dict[str, SceneInfo] = {
    # "en pantalla" con límites esperados: es lo que lo convierte en un
    # criterio de inspección y no en un número suelto
    "screen":    SceneInfo("screen", expect=(2, 4)),
    "line":      SceneInfo("line", draws_line=True),
    "zone":      SceneInfo("zone", draws_zone=True),
    "appear":    SceneInfo("appear"),
    "disappear": SceneInfo("disappear"),
}
