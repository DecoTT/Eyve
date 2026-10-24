"""
Paquete de job (.eyvejob): un job con todo lo que necesita para correr en
otra PC.

Para pasarle a otra estación —o a un cliente en su semana de prueba— lo que
ya funciona aquí: las clases, el modelo entrenado y los parámetros de
producción del SKU (módulos, referencia aprendida, tolerancias, tracker).
No lleva imágenes ni datasets: es para producir, no para reentrenar.

Es un ZIP con:

    manifest.json        formato, versión de Eyve, proyecto, job, sha256 del modelo
    project.yaml         con el modelo en ruta relativa
    classes.yaml
    models/best.pt
    jobs/<job>.yaml      (+ jobs/<job>.ref.npy si el Patrón está calibrado)

Al importar se crea un proyecto nuevo, nunca se escribe encima de uno que ya
exista, y la fuente del job pasa a la cámara 0: el video o el stream de la
PC de origen no existen en la de destino.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from eyve import __version__
from eyve.core import job as J
from eyve.core.logger import log
from eyve.core.project_manager import Project, load_project

EXT = ".eyvejob"
FORMATO = "eyve-job"
VERSION_FORMATO = 1


class PaqueteInvalido(Exception):
    """El archivo no es un paquete de job que esta versión sepa abrir."""


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def exportar(proj: Project, job_name: str, destino: Path) -> Path:
    """Escribe el paquete del job *job_name* en *destino* (se le pone .eyvejob)."""
    job = J.find(proj.paths.jobs, job_name)
    if job is None:
        raise ValueError(f"No existe el job «{job_name}»")
    modelo = Path(proj.active_model) if proj.active_model else proj.paths.best_model
    if not modelo.exists():
        raise ValueError("El proyecto no tiene modelo entrenado")
    destino = Path(destino)
    if destino.suffix.lower() != EXT:
        destino = destino.with_name(destino.name + EXT)

    stem = job.yaml_path(proj.paths.jobs).stem
    proyecto = proj.to_project_dict()
    proyecto.update(active_model="models/best.pt", last_job=job.name)
    manifest = {
        "formato": FORMATO,
        "version_formato": VERSION_FORMATO,
        "eyve": __version__,
        "creado": datetime.now().isoformat(timespec="seconds"),
        "proyecto": proj.name,
        "job": job.name,
        "job_archivo": stem,
        "clases": [c.name for c in proj.classes],
        "modelo_sha256": _sha256(modelo),
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("project.yaml", yaml.safe_dump(proyecto, allow_unicode=True))
        zf.writestr("classes.yaml", yaml.safe_dump(proj.to_classes_dict(), allow_unicode=True))
        zf.write(modelo, "models/best.pt")
        zf.write(job.yaml_path(proj.paths.jobs), f"jobs/{stem}.yaml")
        ref = job.ref_path(proj.paths.jobs)
        if ref.exists():
            zf.write(ref, f"jobs/{stem}.ref.npy")
    tmp.replace(destino)          # sin medio paquete si algo falla a la mitad
    log.info(f"Paquete de job: {destino}")
    return destino


def leer_manifest(paquete: Path) -> dict:
    """El manifest del paquete, validado. Lanza PaqueteInvalido."""
    try:
        with zipfile.ZipFile(paquete) as zf:
            m = json.loads(zf.read("manifest.json").decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as e:
        raise PaqueteInvalido(f"No es un paquete de job de Eyve ({e})") from e
    if m.get("formato") != FORMATO:
        raise PaqueteInvalido("No es un paquete de job de Eyve")
    if int(m.get("version_formato") or 0) > VERSION_FORMATO:
        raise PaqueteInvalido(f"El paquete es de una versión más nueva de Eyve ({m.get('eyve')}); "
                              "actualiza Eyve para abrirlo")
    return m


def _miembros(m: dict) -> dict[str, str]:
    """Lo único que se extrae: nombre en el ZIP → ruta dentro del proyecto.
    Lo demás se ignora, y así ningún nombre del ZIP decide dónde se escribe."""
    stem = J.slug(str(m.get("job_archivo") or m.get("job") or "job"))
    return {"project.yaml": "project.yaml", "classes.yaml": "classes.yaml",
            "models/best.pt": "models/best.pt",
            f"jobs/{stem}.yaml": f"jobs/{stem}.yaml",
            f"jobs/{stem}.ref.npy": f"jobs/{stem}.ref.npy"}


def _carpeta_libre(padre: Path, nombre: str) -> Path:
    base = J.slug(nombre) or "proyecto"
    p, n = padre / base, 2
    while p.exists():
        p, n = padre / f"{base}_{n}", n + 1
    return p


def importar(paquete: Path, padre: Path) -> Project:
    """Crea un proyecto nuevo en *padre* con el job del paquete y lo devuelve."""
    m = leer_manifest(paquete)
    miembros = _miembros(m)
    padre = Path(padre)
    padre.mkdir(parents=True, exist_ok=True)
    root = _carpeta_libre(padre, str(m.get("proyecto") or "proyecto"))

    # todo a una carpeta temporal y, si está completo, se mueve a su lugar
    stage = Path(tempfile.mkdtemp(prefix="eyve_job_", dir=padre))
    try:
        with zipfile.ZipFile(paquete) as zf:
            nombres = set(zf.namelist())
            for req in ("project.yaml", "classes.yaml", "models/best.pt"):
                if req not in nombres:
                    raise PaqueteInvalido(f"Al paquete le falta {req}")
            for en_zip, rel in miembros.items():
                if en_zip in nombres:
                    out = stage / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(en_zip))
        if m.get("modelo_sha256") and _sha256(stage / "models/best.pt") != m["modelo_sha256"]:
            raise PaqueteInvalido("El modelo del paquete está dañado (no coincide su sha256)")
        stage.rename(root)
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise

    proj = load_project(root)
    proj.name = root.name
    proj.active_model = str(proj.paths.best_model)
    proj.paths.create_all()
    job = J.find(proj.paths.jobs, str(m.get("job") or ""))
    if job is not None:
        # la fuente de la PC de origen (un video, un stream) aquí no existe
        job.source = {**J.SOURCE_DEFAULTS, "conf": job.source.get("conf", J.SOURCE_DEFAULTS["conf"])}
        job.save(proj.paths.jobs)
        proj.last_job = job.name
    proj.save()
    log.info(f"Paquete importado: {paquete} → {root}")
    return proj
