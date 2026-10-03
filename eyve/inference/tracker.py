"""
Persistent_Instance.py — Seguimiento persistente de instancias detectadas
══════════════════════════════════════════════════════════════════════════════
Asigna un ID único a cada IC / Pocket detectado y lo mantiene a través
de múltiples frames aunque se mueva por el encuadre.

Uso desde main_prod.py:

    from Persistent_Instance import InstanceTracker, RawDetection

    tracker = InstanceTracker(modules=["polarity"])

    # Por cada frame con resultados YOLO:
    tracks = tracker.update(detections, frame)

    for t in tracks:
        if t.needs_processing("polarity"):
            roi = t.get_roi()
            t.set_queued("polarity")
            # → enviar a PolarityWorker
            result = analyzer.analyze(roi)
            t.set_result("polarity", result)

Estados de una instancia:
    "pending"    → registrada, esperando ser procesada
    "queued"     → en cola de procesamiento
    "processing" → siendo procesada ahora mismo
    "done"       → módulo terminó, resultado guardado
    "skipped"    → omitida (confianza baja, tamaño insuficiente, etc.)
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ─────────────────────────────────────────────────────────────────────────────
#  Configuración
# ─────────────────────────────────────────────────────────────────────────────
IOU_MATCH_THRESHOLD = 0.25   # IoU mínima para considerar "misma instancia"
MAX_LOST_FRAMES     = 12     # frames sin ver antes de purgar
MIN_CONFIRM_FRAMES  = 2      # frames mínimos para considerar instancia "real"
PROCESS_CONF_MIN    = 0.40   # confianza mínima para enviar a procesamiento


# ─────────────────────────────────────────────────────────────────────────────
#  Configuración por instancia
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrackerConfig:
    """
    Parámetros de persistencia de instancia, ajustables en caliente.

    Hasta 2.1.0 estos cuatro valores eran constantes de módulo, así que la
    única forma de afinar "qué tanto seguimos reconociendo la misma
    instancia" era editar el código.  El módulo de conteo los necesita a la
    mano: una banda rápida con oclusión pide tolerancia alta
    (max_lost_frames), y piezas pegadas pide iou_match bajo.

    iou_match          IoU mínima para considerar que dos cajas son la misma
                       instancia entre frames.  Bajo = más permisivo (une
                       piezas que se mueven rápido); alto = más estricto
                       (evita robar el ID del vecino).
    max_lost_frames    Frames que una instancia puede desaparecer sin perder
                       su ID.  Es la tolerancia al parpadeo del detector.
    min_confirm_frames Frames vistos antes de considerar la instancia "real".
                       Filtra detecciones de un solo frame (falsos positivos).
    process_conf_min   Confianza mínima para mandar la instancia a un módulo.
    """
    iou_match:          float = IOU_MATCH_THRESHOLD
    max_lost_frames:    int   = MAX_LOST_FRAMES
    min_confirm_frames: int   = MIN_CONFIRM_FRAMES
    process_conf_min:   float = PROCESS_CONF_MIN

    def clamped(self) -> "TrackerConfig":
        """Valores dentro de rango usable (la UI no puede romper el tracker)."""
        return TrackerConfig(
            iou_match          = min(max(self.iou_match, 0.01), 0.95),
            max_lost_frames    = max(int(self.max_lost_frames), 0),
            min_confirm_frames = max(int(self.min_confirm_frames), 1),
            process_conf_min   = min(max(self.process_conf_min, 0.0), 0.99),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Detección de entrada
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RawDetection:
    label:      str
    confidence: float
    x1: int; y1: int; x2: int; y2: int

    @property
    def bbox(self) -> Tuple[int,int,int,int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def cx(self): return (self.x1 + self.x2) // 2

    @property
    def cy(self): return (self.y1 + self.y2) // 2

    @property
    def w(self): return abs(self.x2 - self.x1)

    @property
    def h(self): return abs(self.y2 - self.y1)

    @property
    def area(self): return self.w * self.h


# ─────────────────────────────────────────────────────────────────────────────
#  Instancia rastreada
# ─────────────────────────────────────────────────────────────────────────────
class TrackedInstance:
    _id_counter = 0

    def __init__(self, det: RawDetection, cfg: Optional[TrackerConfig] = None):
        TrackedInstance._id_counter += 1
        self.id           = TrackedInstance._id_counter
        self.label        = det.label
        self.confidence   = det.confidence
        self.bbox         = det.bbox
        self.bbox_history: List[Tuple] = [det.bbox]
        self._cfg         = cfg or TrackerConfig()

        self.frames_seen  = 1
        self.frames_lost  = 0
        self.first_seen   = time.time()
        self.last_seen    = time.time()
        self.confirmed    = (self._cfg.min_confirm_frames <= 1)

        self._proc_state: Dict[str, str] = {}
        self._results:    Dict[str, Any] = {}
        self._last_roi:   Optional[np.ndarray] = None

    # ── Propiedades ───────────────────────────────────────────────────────

    @property
    def cx(self): return (self.bbox[0] + self.bbox[2]) // 2

    @property
    def cy(self): return (self.bbox[1] + self.bbox[3]) // 2

    @property
    def w(self):  return abs(self.bbox[2] - self.bbox[0])

    @property
    def h(self):  return abs(self.bbox[3] - self.bbox[1])

    @property
    def is_active(self) -> bool:
        return self.frames_lost == 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.first_seen

    # ── ROI ───────────────────────────────────────────────────────────────

    def update_roi(self, frame: np.ndarray):
        x1, y1, x2, y2 = self.bbox
        pad = max(4, min(self.w, self.h) // 10)
        fh, fw = frame.shape[:2]
        crop = frame[max(0,y1-pad):min(fh,y2+pad),
                     max(0,x1-pad):min(fw,x2+pad)]
        if crop.size > 0:
            self._last_roi = crop.copy()

    def get_roi(self, frame: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if frame is not None:
            self.update_roi(frame)
        return self._last_roi

    # ── Estado de procesamiento ───────────────────────────────────────────

    def register_module(self, module: str):
        if module not in self._proc_state:
            self._proc_state[module] = "pending"

    def needs_processing(self, module: str) -> bool:
        if not self.confirmed:
            return False
        if self.confidence < self._cfg.process_conf_min:
            return False
        return self._proc_state.get(module, "pending") == "pending"

    def set_queued(self, module: str):
        self._proc_state[module] = "queued"

    def set_processing(self, module: str):
        self._proc_state[module] = "processing"

    def set_result(self, module: str, result: Any):
        self._proc_state[module] = "done"
        self._results[module]    = result

    def skip_module(self, module: str):
        self._proc_state[module] = "skipped"

    def get_result(self, module: str) -> Optional[Any]:
        return self._results.get(module)

    def module_state(self, module: str) -> str:
        return self._proc_state.get(module, "pending")

    # ── Actualización de posición ─────────────────────────────────────────

    def update(self, det: RawDetection):
        alpha = 0.65
        ox1, oy1, ox2, oy2 = self.bbox
        self.bbox = (
            int(ox1*(1-alpha) + det.x1*alpha),
            int(oy1*(1-alpha) + det.y1*alpha),
            int(ox2*(1-alpha) + det.x2*alpha),
            int(oy2*(1-alpha) + det.y2*alpha),
        )
        self.bbox_history.append(self.bbox)
        if len(self.bbox_history) > 30:
            self.bbox_history.pop(0)

        self.confidence  = det.confidence
        self.frames_seen += 1
        self.frames_lost  = 0
        self.last_seen    = time.time()

        if self.frames_seen >= self._cfg.min_confirm_frames:
            self.confirmed = True

    def mark_lost(self):
        self.frames_lost += 1

    def __repr__(self):
        return (f"Track#{self.id:03d}[{self.label}] "
                f"conf={self.confidence:.0%} seen={self.frames_seen} "
                f"lost={self.frames_lost}")


# ─────────────────────────────────────────────────────────────────────────────
#  Tracker principal
# ─────────────────────────────────────────────────────────────────────────────
class InstanceTracker:
    """
    Asigna y mantiene IDs persistentes a detecciones YOLO entre frames.
    Utiliza IoU para asociar detecciones a instancias existentes.
    """

    def __init__(self, modules: Optional[List[str]] = None,
                 config: Optional[TrackerConfig] = None):
        self._tracks: Dict[int, TrackedInstance] = {}
        self._modules: List[str] = modules or []
        self._frame_count = 0
        self.config: TrackerConfig = (config or TrackerConfig()).clamped()
        #: instancias purgadas en el último update() — el método de conteo
        #: "desaparición" cuenta justo estas (ya no hay forma de verlas luego).
        self.last_expired: List[TrackedInstance] = []

    def configure(self, **kwargs) -> None:
        """
        Ajusta la configuración en caliente (desde la UI, sin reiniciar).

        Los tracks vivos conservan la config con la que nacieron para su
        umbral de confirmación; los nuevos usan la nueva.  La tolerancia de
        purga y la IoU de asociación aplican de inmediato a todos.
        """
        cur = {
            "iou_match":          self.config.iou_match,
            "max_lost_frames":    self.config.max_lost_frames,
            "min_confirm_frames": self.config.min_confirm_frames,
            "process_conf_min":   self.config.process_conf_min,
        }
        cur.update({k: v for k, v in kwargs.items() if k in cur and v is not None})
        self.config = TrackerConfig(**cur).clamped()

    def update(self, detections: List[RawDetection],
               frame: Optional[np.ndarray] = None) -> List[TrackedInstance]:
        """
        Procesa las detecciones de un frame.
        Retorna lista de instancias activas en este frame.
        """
        self._frame_count += 1
        active  = list(self._tracks.values())
        matched_track_ids = set()
        matched_det_idxs  = set()

        # Matching por IoU
        if active and detections:
            iou_matrix = self._iou_matrix(active, detections)
            rows, cols = iou_matrix.shape

            while True:
                if iou_matrix.size == 0:
                    break
                max_val = float(iou_matrix.max())
                if max_val < self.config.iou_match:
                    break
                ti_arr, di_arr = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                ti, di = int(ti_arr), int(di_arr)

                track = active[ti]
                det   = detections[di]

                if self._compatible(track.label, det.label):
                    track.update(det)
                    if frame is not None:
                        track.update_roi(frame)
                    matched_track_ids.add(track.id)
                    matched_det_idxs.add(di)

                iou_matrix[ti, :] = -1.0
                iou_matrix[:, di] = -1.0

        # Marcar perdidos
        for t in active:
            if t.id not in matched_track_ids:
                t.mark_lost()

        # Crear nuevas instancias
        for i, det in enumerate(detections):
            if i not in matched_det_idxs:
                t = TrackedInstance(det, self.config)
                for m in self._modules:
                    t.register_module(m)
                if frame is not None:
                    t.update_roi(frame)
                self._tracks[t.id] = t

        # Purgar expirados.  Se guardan en last_expired ANTES de borrarlos:
        # es la única oportunidad de contarlos (conteo por desaparición).
        purge = [tid for tid, t in self._tracks.items()
                 if t.frames_lost > self.config.max_lost_frames]
        self.last_expired = [self._tracks[tid] for tid in purge]
        for tid in purge:
            del self._tracks[tid]

        return [t for t in self._tracks.values() if t.frames_lost == 0]

    def get_all(self) -> List[TrackedInstance]:
        return list(self._tracks.values())

    def get_by_id(self, track_id: int) -> Optional[TrackedInstance]:
        return self._tracks.get(track_id)

    def pending_for(self, module: str) -> List[TrackedInstance]:
        return [t for t in self._tracks.values()
                if t.needs_processing(module)]

    def reset(self):
        self._tracks.clear()
        self.last_expired = []
        TrackedInstance._id_counter = 0
        self._frame_count = 0

    def stats(self) -> Dict:
        tracks = list(self._tracks.values())
        return {
            "active":     len([t for t in tracks if t.frames_lost == 0]),
            "total":      len(tracks),
            "confirmed":  len([t for t in tracks if t.confirmed]),
            "frames":     self._frame_count,
        }

    # ── Internos ──────────────────────────────────────────────────────────

    @staticmethod
    def _iou(a: Tuple, b: Tuple) -> float:
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
        ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0: return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / (union + 1e-6)

    def _iou_matrix(self, tracks: List[TrackedInstance],
                    dets: List[RawDetection]) -> np.ndarray:
        m = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for ti, t in enumerate(tracks):
            for di, d in enumerate(dets):
                m[ti, di] = self._iou(t.bbox, d.bbox)
        return m

    @staticmethod
    def _compatible(a: str, b: str) -> bool:
        """
        Misma clase = misma instancia.  (La versión 2.0 traía grupos
        semánticos hardcodeados del proyecto de polaridad — en 2.1 las
        clases son definidas por el usuario, así que match exacto.)
        """
        return a == b
