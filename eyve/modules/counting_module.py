"""
Counting module — five ways to count, one tracker underneath.

Every method works ON TOP of InstanceTracker, which is what makes the count
immune to detection flicker: a part that blinks out for a few frames keeps
its ID (TrackerConfig.max_lost_frames) and cannot be counted twice.

Why five methods and not one: "contar" means a different event depending on
what is in front of the camera, and the line-crossing count that 2.1.0
shipped only covers the conveyor case.

    SCREEN      ¿cuántas hay AHORA en el encuadre?           (no acumula)
                tortillas en la charola, pines en el conector, cajas en la
                tarima.  Con límites esperados, veta el frame a NOT_OK.

    LINE        cruce de una meta, con sentido               (acumula)
                banda transportadora: todo lo que pasa la línea cuenta una
                vez.  Es el método de 2.1.0, ahora con conteo por sentido
                (entran / salen / neto).

    ZONE        entrada y/o salida de un área                (acumula)
                celda de trabajo, zona de carga: cuenta al entrar al área,
                al salir, o ambas, y reporta la ocupación actual.

    APPEAR      cada instancia nueva, una vez                (acumula)
                piezas que aparecen en cualquier parte del encuadre (caen,
                se destapan, se imprimen) sin un lado fijo de llegada.

    DISAPPEAR   cada instancia que se va                     (acumula)
                piezas que alguien retira, que salen del encuadre o que
                desaparecen bajo la herramienta.  Con filtro de borde,
                cuenta solo las que se fueron por el lado que te importa.

Geometry (line / zone rect) lives in FRAME coordinates; the production
screen converts the user's canvas drag.  `geometry_kind` tells the UI what
to ask the user to draw for the active method.
"""
from __future__ import annotations
from typing import Optional, Tuple

import cv2
import numpy as np

from eyve.modules.base import InspectionModule, ModuleVerdict

_LINE_COLOR = (0, 215, 255)      # amarillo (BGR)
_ZONE_COLOR = (255, 190, 0)      # cian (BGR)
_EDGE_COLOR = (120, 120, 255)    # rojo claro (BGR)
_NOK_COLOR  = (60, 60, 255)      # rojo: fuera del rango esperado (BGR)
_LINE_THICK = 2


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True if segment p1-p2 intersects segment p3-p4 (2D)."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _side_of_line(p, a, b) -> int:
    """Which side of the directed line a->b the point p falls on (-1/0/+1)."""
    v = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _point_in_rect(p, rect) -> bool:
    x1, y1, x2, y2 = rect
    return (min(x1, x2) <= p[0] <= max(x1, x2) and
            min(y1, y2) <= p[1] <= max(y1, y2))


class CountingModule(InspectionModule):
    name = "Conteo"

    #: internal method keys (the UI translates them via i18n: count_m_<key>)
    METHODS = ("screen", "line", "zone", "appear", "disappear")
    #: which direction of a line crossing feeds `total`
    DIRECTIONS = ("both", "fwd", "rev")
    #: which zone transition counts
    ZONE_TRIGGERS = ("enter", "exit", "both")
    #: which frame edge a disappearance must happen near to count
    EDGES = ("any", "left", "right", "top", "bottom")

    #: how close to the edge (px) a track's last position must be, for the
    #: disappearance method with an edge filter
    EDGE_MARGIN = 28

    def __init__(self) -> None:
        super().__init__()
        self.method: str = "line"
        #: finish line in frame coords: (x1, y1, x2, y2) — None until drawn
        self.line: Optional[Tuple[int, int, int, int]] = None
        #: zone rect in frame coords: (x1, y1, x2, y2) — None until drawn
        self.zone: Optional[Tuple[int, int, int, int]] = None
        self.direction: str = "both"
        self.zone_trigger: str = "enter"
        self.edge: str = "any"
        #: expected range for the SCREEN / ZONE-occupancy methods.
        #: None = no expectation, the module never vetoes.
        self.expect_min: Optional[int] = None
        self.expect_max: Optional[int] = None

        self.counts: dict[str, int] = {}          # label -> acumulado
        self.on_screen: dict[str, int] = {}       # label -> visibles ahora
        self.dir_counts: dict[str, int] = {"fwd": 0, "rev": 0}
        self._counted_ids: set[int] = set()
        self._prev_centers: dict[int, Tuple[int, int]] = {}
        self._prev_inside: dict[int, bool] = {}
        self._seen_ids: set[int] = set()          # para APPEAR
        self._verdict = ModuleVerdict()
        #: frame size of the last update — el filtro de borde lo necesita
        self._frame_wh: Optional[Tuple[int, int]] = None

    # ── configuration ─────────────────────────────────────────────────────
    def set_method(self, method: str) -> None:
        """
        Switching method clears the accumulated count: showing a tally that
        was built by counting a different event would be a lie.
        """
        if method not in self.METHODS or method == self.method:
            return
        self.method = method
        self.reset()

    @property
    def geometry_kind(self) -> Optional[str]:
        """
        What the active method needs the user to draw: 'line', 'rect' or
        None.  The production screen builds its draw button from this.
        """
        if self.method == "line":
            return "line"
        if self.method == "zone":
            return "rect"
        return None

    @property
    def needs_geometry(self) -> bool:
        """True when the method cannot run yet because nothing is drawn."""
        if self.method == "line":
            return self.line is None
        if self.method == "zone":
            return self.zone is None
        return False

    def clear_geometry(self) -> None:
        self.line = None
        self.zone = None
        self.reset()

    # kept for the 2.1.0 call sites
    def clear_line(self) -> None:
        self.clear_geometry()

    def reset(self) -> None:
        self.counts.clear()
        self.on_screen.clear()
        self.dir_counts = {"fwd": 0, "rev": 0}
        self._counted_ids.clear()
        self._prev_centers.clear()
        self._prev_inside.clear()
        self._seen_ids.clear()
        self._verdict = ModuleVerdict()

    # ── readouts ──────────────────────────────────────────────────────────
    @property
    def total(self) -> int:
        """
        The number the operator reads.  For SCREEN it is what is visible
        right now; for the accumulating methods, the running tally
        (restricted to one direction when the line is set to a single sense).
        """
        if self.method == "screen":
            return sum(self.on_screen.values())
        if self.method == "line" and self.direction in ("fwd", "rev"):
            return self.dir_counts[self.direction]
        return sum(self.counts.values())

    @property
    def occupancy(self) -> int:
        """Instances currently inside the zone (ZONE method)."""
        return sum(1 for v in self._prev_inside.values() if v)

    def summary(self) -> str:
        """One short line for the side panel."""
        if self.method == "screen":
            if not self.on_screen:
                return "0"
            if len(self.on_screen) == 1:
                return str(self.total)
            return "  ".join(f"{k}: {v}" for k, v in sorted(self.on_screen.items()))
        if self.method == "line" and self.direction == "both":
            fwd = self.dir_counts["fwd"]
            rev = self.dir_counts["rev"]
            return f"{self.total}   ^{fwd} v{rev}  neto {fwd - rev}"
        if self.method == "zone":
            return f"{self.total}   en zona: {self.occupancy}"
        if not self.counts:
            return "0"
        if len(self.counts) == 1:
            return str(self.total)
        return "  ".join(f"{k}: {v}" for k, v in sorted(self.counts.items()))

    # ── per-frame update (tracker-based, not detection-based) ─────────────
    def update_tracks(self, tracks: list, expired: Optional[list] = None,
                      frame_wh: Optional[Tuple[int, int]] = None) -> ModuleVerdict:
        """
        Feed this frame's active TrackedInstances.

        tracks    active instances from InstanceTracker.update()
        expired   instances purged this frame (tracker.last_expired) — the
                  DISAPPEAR method counts these and nothing else
        frame_wh  (width, height) of the frame, for the edge filter

        target_class None counts every class.  Returns the module verdict;
        ok=False only when an expected range is set and the live count falls
        outside it.
        """
        if not self.enabled:
            self._verdict = ModuleVerdict()
            return self._verdict
        if frame_wh:
            self._frame_wh = frame_wh

        mine = [tr for tr in tracks
                if not self.target_class or tr.label == self.target_class]
        confirmed = [tr for tr in mine if tr.confirmed]

        # visibles ahora — lo usa SCREEN, el resto lo muestra como contexto
        self.on_screen = {}
        for tr in confirmed:
            self.on_screen[tr.label] = self.on_screen.get(tr.label, 0) + 1

        if self.method == "screen":
            self._verdict = self._verdict_for(self.total)
            return self._verdict

        if self.method == "line":
            self._update_line(confirmed)
        elif self.method == "zone":
            self._update_zone(confirmed)
        elif self.method == "appear":
            self._update_appear(confirmed)
        elif self.method == "disappear":
            self._update_disappear(expired or [])

        self._forget_dead(tracks)
        if self.method == "zone":
            self._verdict = self._verdict_for(self.occupancy)
        else:
            self._verdict = ModuleVerdict(ok=True, label=self.summary())
        return self._verdict

    def _verdict_for(self, value: int) -> ModuleVerdict:
        """Veto the frame when a live count falls outside the expected range."""
        lo, hi = self.expect_min, self.expect_max
        if lo is None and hi is None:
            return ModuleVerdict(ok=True, label=self.summary())
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            if lo is not None and hi is not None:
                want = f"{lo}-{hi}"
            elif hi is None:
                want = f">={lo}"
            else:
                want = f"<={hi}"
            return ModuleVerdict(
                ok=False,
                label=f"{value} (esperado {want})",
                triggered_by=f"conteo {self.target_class or 'total'}",
                analyzed=value,
            )
        return ModuleVerdict(ok=True, label=self.summary(), analyzed=value)

    # ── method implementations ────────────────────────────────────────────
    def _update_line(self, tracks: list) -> None:
        if self.line is None:
            return
        a = (self.line[0], self.line[1])
        b = (self.line[2], self.line[3])
        for tr in tracks:
            cur = (tr.cx, tr.cy)
            prev = self._prev_centers.get(tr.id)
            self._prev_centers[tr.id] = cur
            if prev is None or tr.id in self._counted_ids:
                continue
            if not _segments_intersect(prev, cur, a, b):
                continue
            # sentido del cruce: a qué lado de la meta quedó la instancia
            sense = "fwd" if _side_of_line(cur, a, b) > 0 else "rev"
            self._counted_ids.add(tr.id)
            self.dir_counts[sense] += 1
            if self.direction != "both" and sense != self.direction:
                continue   # el cruce queda registrado en dir_counts, no en total
            self.counts[tr.label] = self.counts.get(tr.label, 0) + 1

    def _update_zone(self, tracks: list) -> None:
        if self.zone is None:
            return
        for tr in tracks:
            inside = _point_in_rect((tr.cx, tr.cy), self.zone)
            was = self._prev_inside.get(tr.id)
            self._prev_inside[tr.id] = inside
            if was is None or was == inside:
                continue
            entered = inside and not was
            if ((entered and self.zone_trigger in ("enter", "both")) or
                    (not entered and self.zone_trigger in ("exit", "both"))):
                self.counts[tr.label] = self.counts.get(tr.label, 0) + 1

    def _update_appear(self, tracks: list) -> None:
        for tr in tracks:
            if tr.id in self._seen_ids:
                continue
            self._seen_ids.add(tr.id)
            self.counts[tr.label] = self.counts.get(tr.label, 0) + 1

    def _update_disappear(self, expired: list) -> None:
        for tr in expired:
            if self.target_class and tr.label != self.target_class:
                continue
            if not tr.confirmed:
                continue        # parpadeo del detector, no una pieza que se fue
            if tr.id in self._counted_ids:
                continue
            if not self._left_by_edge(tr):
                continue
            self._counted_ids.add(tr.id)
            self.counts[tr.label] = self.counts.get(tr.label, 0) + 1

    def _left_by_edge(self, tr) -> bool:
        """
        True if the track's last known position justifies counting it as
        "left through the selected edge".  'any' accepts every exit.
        """
        if self.edge == "any" or not self._frame_wh:
            return True
        fw, fh = self._frame_wh
        x1, y1, x2, y2 = tr.bbox
        m = self.EDGE_MARGIN
        if self.edge == "left":
            return x1 <= m
        if self.edge == "right":
            return x2 >= fw - m
        if self.edge == "top":
            return y1 <= m
        if self.edge == "bottom":
            return y2 >= fh - m
        return True

    def _forget_dead(self, tracks: list) -> None:
        """
        Drop per-track scratch state for IDs the tracker no longer has, so the
        dicts don't grow for a whole shift.  _counted_ids and _seen_ids stay:
        IDs are never reused within a session, and forgetting them is exactly
        how a part gets counted twice.
        """
        alive = {tr.id for tr in tracks}
        for store in (self._prev_centers, self._prev_inside):
            for tid in list(store):
                if tid not in alive:
                    store.pop(tid, None)

    # ── overlay ───────────────────────────────────────────────────────────
    def draw(self, annotated: np.ndarray) -> None:
        """Draw the active method's geometry + running readout."""
        if not self.enabled:
            return
        h, w = annotated.shape[:2]
        anchor = (12, 30)

        if self.method == "line" and self.line is not None:
            x1, y1, x2, y2 = self.line
            cv2.line(annotated, (x1, y1), (x2, y2), _LINE_COLOR, _LINE_THICK)
            cv2.circle(annotated, (x1, y1), 4, _LINE_COLOR, -1)
            cv2.circle(annotated, (x2, y2), 4, _LINE_COLOR, -1)
            self._draw_sense_arrow(annotated, (x1, y1), (x2, y2))
            anchor = (min(x1, x2), max(20, min(y1, y2) - 8))

        elif self.method == "zone" and self.zone is not None:
            x1, y1, x2, y2 = self.zone
            cv2.rectangle(annotated, (min(x1, x2), min(y1, y2)),
                          (max(x1, x2), max(y1, y2)), _ZONE_COLOR, _LINE_THICK)
            anchor = (min(x1, x2), max(20, min(y1, y2) - 8))

        elif self.method == "disappear" and self.edge != "any":
            m = self.EDGE_MARGIN
            band = {"left":   ((0, 0), (m, h)),
                    "right":  ((w - m, 0), (w, h)),
                    "top":    ((0, 0), (w, m)),
                    "bottom": ((0, h - m), (w, h))}.get(self.edge)
            if band:
                cv2.rectangle(annotated, band[0], band[1], _EDGE_COLOR, 2)

        color = _ZONE_COLOR if self.method == "zone" else _LINE_COLOR
        if not self._verdict.ok:
            color = _NOK_COLOR
        cv2.putText(annotated, str(self.total), anchor,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    def _draw_sense_arrow(self, annotated: np.ndarray, a, b) -> None:
        """Arrow at the line's midpoint showing which sense feeds the total."""
        if self.direction == "both":
            return
        mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        # normal a la línea; fwd = lado positivo de _side_of_line
        nx, ny = -dy / n, dx / n
        if self.direction == "rev":
            nx, ny = -nx, -ny
        cv2.arrowedLine(annotated, (mx, my),
                        (int(mx + nx * 26), int(my + ny * 26)),
                        _LINE_COLOR, 2, tipLength=0.4)

    # InspectionModule contract — the real work happens in update_tracks()
    def process(self, frame, detections, annotated) -> ModuleVerdict:
        return self._verdict
