"""
Jobs: los parámetros de producción de un SKU, guardados.

Un proyecto tiene UN modelo y va creciendo (se entrenan más SKUs contra el
mismo modelo). Lo que cambia entre un SKU y otro no es el modelo: es cómo
se inspecciona — qué módulos van encendidos, con qué método, dónde está la
línea de conteo, con qué sensibilidad mira el Patrón. Eso es un job.

    <proyecto>/
        jobs/
            SKU-1234.yaml
            SKU-1234.ref.npy      (sólo si el Patrón aprendió una referencia)

Hasta 2.1.1 nada de esto se guardaba: al abrir Producción había que ponerlo
todo de cero, y los cuatro deslizadores de persistencia del tracker —lo
único que sí persistía— vivían en el config GLOBAL de la app, así que
ajustarlos para un SKU se los cambiaba en silencio a todos los demás. Aquí
se mudan al job, que es donde pertenecen.

Lo que el job NO guarda, a propósito:

  * el modelo y las clases, que son del proyecto;
  * los resultados de producción, que son de la sesión (`ProductionSession`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from eyve.core.logger import log

#: Caracteres que se permiten en el NOMBRE DE ARCHIVO de un job. El nombre
#: que ve el usuario puede traer espacios, puntos o barras —un SKU real los
#: trae— así que el archivo se llama con una versión limpia y el nombre de
#: verdad va dentro del YAML.
_SLUG_MALO = re.compile(r"[^A-Za-z0-9_\-]+")

#: Claves de la persistencia del tracker, con sus valores por defecto. Son
#: las mismas que `TrackerConfig`, y las que hasta ahora vivían sueltas en
#: el config global como track_iou / track_lost / track_confirm /
#: track_conf_min.
TRACKER_DEFAULTS = {
    "iou_match": 0.25,
    "max_lost_frames": 12,
    "min_confirm_frames": 2,
    "process_conf_min": 0.40,
}

#: Fuente y umbral. `video_path` vacío = cámara.
SOURCE_DEFAULTS = {
    "camera_source": 0,
    "video_path": "",
    "conf": 0.50,
}


def slug(nombre: str) -> str:
    """Nombre de archivo seguro para un SKU cualquiera."""
    s = _SLUG_MALO.sub("_", (nombre or "").strip()).strip("_")
    return s or "job"


@dataclass
class Job:
    """Los parámetros de producción de un SKU."""

    name: str
    notes: str = ""
    #: Para encontrar el job entre cientos: compañía y proyecto (en la
    #: interfaz se llama «Proyecto»; aquí `product` para no confundirlo con
    #: el proyecto de Eyve, que es el modelo). Vacío = sin agrupar; los jobs
    #: de antes de estos campos caen ahí y siguen abriendo igual.
    company: str = ""
    product: str = ""
    source: dict = field(default_factory=lambda: dict(SOURCE_DEFAULTS))
    tracker: dict = field(default_factory=lambda: dict(TRACKER_DEFAULTS))
    #: key del módulo -> su config(). Ver InspectionModule.config().
    modules: dict = field(default_factory=dict)
    #: cuándo se calibró el Patrón, en ISO. La calibración depende del
    #: material y de la luz: sin fecha, un día se inspecciona con una
    #: referencia de hace semanas y nadie se entera.
    calibrated_at: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    #: la referencia aprendida del Patrón, si la hay. No va al YAML: se
    #: guarda al lado como .npy.
    reference_image: Optional[np.ndarray] = None
    #: nombre del archivo en disco, sin extensión. No se serializa: lo fija
    #: save() o load(), y existe para que el .npy acompañe al .yaml aunque
    #: haya habido que desempatar el nombre.
    _stem: str = ""

    # ── serialización ────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "notes": self.notes,
            "company": self.company,
            "product": self.product,
            "source": dict(self.source),
            "tracker": dict(self.tracker),
            "modules": {k: dict(v) for k, v in self.modules.items()},
            "calibrated_at": self.calibrated_at,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        job = cls(name=str(d.get("name") or "job"))
        job.notes = str(d.get("notes") or "")
        job.company = str(d.get("company") or "").strip()
        job.product = str(d.get("product") or "").strip()
        job.source = {**SOURCE_DEFAULTS, **(d.get("source") or {})}
        job.tracker = {**TRACKER_DEFAULTS, **(d.get("tracker") or {})}
        job.modules = {k: dict(v) for k, v in (d.get("modules") or {}).items()}
        job.calibrated_at = str(d.get("calibrated_at") or "")
        job.created_at = str(d.get("created_at") or datetime.now().isoformat())
        job.last_modified = str(d.get("last_modified") or job.created_at)
        return job

    # ── disco ────────────────────────────────────────────────────────────
    def _resolve_stem(self, jobs_dir: Path) -> str:
        """
        Nombre de archivo libre para este job.

        `slug()` tiene que colapsar caracteres —un SKU real trae puntos,
        espacios o barras— y eso hace que "SKU 1" y "SKU/1" caigan en el
        mismo archivo. Dos SKU distintos pisándose en silencio es de los
        fallos que no se notan hasta que una línea inspecciona con los
        parámetros de otra pieza. Si el archivo ya es de OTRO nombre, se
        desempata con un sufijo corto y estable.
        """
        base = slug(self.name)
        p = Path(jobs_dir) / f"{base}.yaml"
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    de_quien = (yaml.safe_load(f) or {}).get("name")
            except Exception:
                de_quien = None
            if de_quien is not None and de_quien != self.name:
                import hashlib
                h = hashlib.sha256(self.name.encode()).hexdigest()[:4]
                return f"{base}_{h}"
        return base

    def yaml_path(self, jobs_dir: Path) -> Path:
        stem = self._stem or slug(self.name)
        return Path(jobs_dir) / f"{stem}.yaml"

    def ref_path(self, jobs_dir: Path) -> Path:
        stem = self._stem or slug(self.name)
        return Path(jobs_dir) / f"{stem}.ref.npy"

    def save(self, jobs_dir: Path) -> Path:
        jobs_dir = Path(jobs_dir)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        self.last_modified = datetime.now().isoformat()
        if not self._stem:
            self._stem = self._resolve_stem(jobs_dir)
        p = self.yaml_path(jobs_dir)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True,
                           sort_keys=False)
        ref = self.ref_path(jobs_dir)
        if self.reference_image is not None:
            np.save(ref, self.reference_image)
        elif ref.exists():
            # Sin referencia en memoria, el .npy de antes miente: cargarlo
            # devolvería una referencia que este job ya no usa.
            ref.unlink()
        log.info(f"Job guardado: {p}")
        return p

    @classmethod
    def load(cls, path: Path, ref: bool = True) -> "Job":
        """*ref*=False no lee la referencia (.npy): para listar no hace falta."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        job = cls.from_dict(d)
        job._stem = path.stem
        npy = path.parent / f"{job._stem}.ref.npy"
        if ref and npy.exists():
            try:
                job.reference_image = np.load(npy)
            except Exception:
                log.warning(f"Job {job.name}: no se pudo leer {npy.name}")
        return job

    def delete(self, jobs_dir: Path) -> None:
        for p in (self.yaml_path(jobs_dir), self.ref_path(jobs_dir)):
            if p.exists():
                p.unlink()

    # ── licencia ─────────────────────────────────────────────────────────
    def pro_modules(self, modulos) -> list[str]:
        """
        Nombres de los módulos Pro que este job pide ENCENDIDOS.

        Se guardan igual aunque la licencia no los permita: perder los
        parámetros por abrir el job en una máquina sin Pro sería peor que
        no poder encenderlo.
        """
        fuera = []
        for m in modulos:
            cfg = self.modules.get(m.key) or {}
            if m.pro and cfg.get("enabled"):
                fuera.append(m.name)
        return fuera


# ── pasar de los módulos al job y al revés ───────────────────────────────
def capture(modulos, tracker=None, source: Optional[dict] = None,
            name: str = "", notes: str = "",
            base: Optional[Job] = None,
            company: Optional[str] = None,
            product: Optional[str] = None) -> Job:
    """Construye un Job con el estado ACTUAL de los módulos y el tracker."""
    job = Job(name=name or (base.name if base else "job"))
    if base is not None:
        job.created_at = base.created_at
        job.notes = base.notes
        job.company = base.company
        job.product = base.product
    if notes:
        job.notes = notes
    if company is not None:
        job.company = company.strip()
    if product is not None:
        job.product = product.strip()
    if source:
        job.source = {**SOURCE_DEFAULTS, **source}
    if tracker is not None:
        cfg = getattr(tracker, "config", None)
        for k in TRACKER_DEFAULTS:
            v = getattr(cfg, k, None)
            if v is not None:
                job.tracker[k] = v
    for m in modulos:
        job.modules[m.key] = m.config()
        img = getattr(m, "reference_image", None)
        if img is not None:
            job.reference_image = img
        if getattr(m, "calibrated", False):
            job.calibrated_at = datetime.now().isoformat()
    return job


def apply(job: Job, modulos, tracker=None) -> None:
    """
    Deja los módulos y el tracker como los describe *job*.

    Nunca lanza: un job con una clave rara o de una versión más nueva deja
    el resto aplicado en vez de tirar la pantalla de Producción abajo.
    """
    for m in modulos:
        cfg = job.modules.get(m.key)
        if not cfg:
            continue
        try:
            m.apply_config(cfg)
        except Exception:
            log.debug(f"job: {m.key} no aceptó su config", exc_info=True)
        if job.reference_image is not None and hasattr(m, "load_reference_image"):
            try:
                m.load_reference_image(job.reference_image)
            except Exception:
                log.debug("job: referencia no aplicada", exc_info=True)
    if tracker is not None:
        try:
            tracker.configure(**{k: job.tracker[k]
                                 for k in TRACKER_DEFAULTS if k in job.tracker})
        except Exception:
            log.debug("job: tracker no aceptó su config", exc_info=True)


# ── listado ──────────────────────────────────────────────────────────────
def list_jobs(jobs_dir: Path, refs: bool = True) -> list[Job]:
    """
    Los jobs de un proyecto, por nombre.

    *refs*=False no lee las referencias del Patrón: un selector con cientos
    de SKU no debe cargar cientos de imágenes para enseñar nombres. Esos
    jobs NO se aplican ni se guardan; para eso, `find()`.
    """
    jobs_dir = Path(jobs_dir)
    if not jobs_dir.exists():
        return []
    out = []
    for p in sorted(jobs_dir.glob("*.yaml")):
        try:
            out.append(Job.load(p, ref=refs))
        except Exception:
            log.warning(f"Job ilegible: {p.name}")
    return sorted(out, key=lambda j: j.name.lower())


def find(jobs_dir: Path, name: str) -> Optional[Job]:
    """
    El job que se llama *name*, o None.

    Primero el archivo que le toca por su slug, pero solo si de verdad es
    suyo: "SKU 1" y "SKU/1" comparten slug, y el segundo en guardarse queda
    con sufijo (`_resolve_stem`). Devolver el del slug sin mirar el nombre
    cargaba los parámetros del OTRO SKU.
    """
    p = Path(jobs_dir) / f"{slug(name)}.yaml"
    if p.exists():
        try:
            job = Job.load(p)
            if job.name == name:
                return job
        except Exception:
            pass
    for job in list_jobs(jobs_dir, refs=False):
        if job.name == name:
            return Job.load(Path(jobs_dir) / f"{job._stem}.yaml")
    return None


# ── agrupar: compañía → proyecto → SKU ───────────────────────────────────
# Con cientos de SKU un solo menú no sirve: el operador va filtrando. None
# en un filtro = todos; "" = los que no tienen ese campo.

def companies(jobs) -> list[str]:
    """Las compañías que hay, en orden; "" al final si hay jobs sin compañía."""
    hay = {j.company for j in jobs}
    return sorted((c for c in hay if c), key=str.lower) + ([""] if "" in hay else [])


def products(jobs, company: Optional[str] = None) -> list[str]:
    """Los proyectos de *company* (None = de todas), con "" al final si hay sin proyecto."""
    hay = {j.product for j in jobs if company is None or j.company == company}
    return sorted((p for p in hay if p), key=str.lower) + ([""] if "" in hay else [])


def filter_jobs(jobs, company: Optional[str] = None,
                product: Optional[str] = None, text: str = "") -> list:
    """
    Los jobs que pasan los filtros. *text* busca en SKU, compañía y
    proyecto, sin mayúsculas, y con varias palabras exige todas.
    """
    palabras = [w for w in (text or "").lower().split() if w]
    out = []
    for j in jobs:
        if company is not None and j.company != company:
            continue
        if product is not None and j.product != product:
            continue
        donde = f"{j.name} {j.company} {j.product}".lower()
        if all(w in donde for w in palabras):
            out.append(j)
    return out
