"""
Actualización en sitio desde GitHub Releases.

El problema que resuelve: hasta ahora actualizar Eyve era "borra la
carpeta, descomprime la nueva, vuelve a cargar tus proyectos".  Eso pierde
proyectos por accidente y hace que la gente simplemente no actualice.

Cómo funciona
─────────────
1.  `check()` pregunta a la API pública de GitHub cuál es el último release
    y compara su tag con `eyve.__version__`.  No necesita llave de licencia
    ni token: el repo es público y la API anónima basta de sobra para una
    consulta por sesión.
2.  `download()` baja el ZIP del release y **verifica el SHA256** contra el
    `SHA256SUMS.txt` del mismo release.  Si no coincide, se aborta ahí: un
    ZIP a medias que se descomprime encima de la instalación la deja rota.
3.  `apply()` respalda lo que va a reemplazar, reemplaza, y si algo falla
    restaura el respaldo.

Lo que NUNCA se toca
────────────────────
    projects/        los proyectos del usuario
    .venv/           el entorno (se reinstala solo si requirements cambió)
    ~/.eyve/         config, licencia, modelos descargados
    cualquier archivo o carpeta que no venga dentro del ZIP

Se reemplaza solo `eyve/` (el paquete) más los archivos sueltos de la raíz
que trae el ZIP (requirements.txt, setup.bat, run.bat, LICENSE…).

Sobre el ZIP como medio y no `git pull`: el usuario final no tiene git, y
el pipeline de release ya produce ZIP + checksum.  Un `git clone` obligaría
a instalar git en cada planta donde corre Eyve.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import ssl
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from eyve import __version__
from eyve.core import config as _cfg
from eyve.core.logger import log

#: repo público y nombre fijo del asset (ver RELEASE.md §0)
GITHUB_REPO = "DecoTT/Eyve"
ASSET_NAME = "Eyve-2.1-setup.zip"
SUMS_NAME = "SHA256SUMS.txt"
_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

#: carpetas y archivos que jamás se reemplazan, vengan o no en el ZIP
PROTECTED = {"projects", ".venv", "venv", ".git", "sessions", "runs", "models"}

_TIMEOUT = 20
_UA = f"Eyve/{__version__} (+https://github.com/{GITHUB_REPO})"

#: Accept para preguntarle a la API de GitHub.
#:
#: NO vale "application/octet-stream": la API contesta 415 Unsupported
#: Media Type.  Es lo que mandaba Eyve hasta 2.1.0 para TODO, asi que la
#: consulta fallaba siempre — y como check() se traga los errores a
#: proposito (abrir Eyve sin internet no debe dar un dialogo), el
#: actualizador contestaba "no hay actualizacion" para siempre, sin
#: ruido.  Comprobado contra la API real: octet-stream da 415 y
#: vnd.github+json da 200.
_ACCEPT_API = "application/vnd.github+json"

#: Accept para bajar un asset, que ahi si es un binario.
_ACCEPT_BIN = "application/octet-stream"


# ── versiones ────────────────────────────────────────────────────────────────
def parse_version(text: str) -> tuple[int, ...]:
    """
    'v2.1.3' → (2, 1, 3).  Lo que no sea numérico se ignora.

    Comparar cadenas no sirve: '2.10.0' < '2.9.0' como texto, y esa es
    exactamente la versión en la que el updater dejaría de ofrecer
    actualizaciones sin que nadie se diera cuenta.
    """
    nums = re.findall(r"\d+", text or "")
    return tuple(int(n) for n in nums[:4]) or (0,)


def is_newer(remote: str, local: str = __version__) -> bool:
    a, b = parse_version(remote), parse_version(local)
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    return a > b


# ── consulta ─────────────────────────────────────────────────────────────────
@dataclass
class UpdateInfo:
    available: bool = False
    version: str = ""
    current: str = __version__
    url: str = ""
    sums_url: str = ""
    size: int = 0
    notes: str = ""
    error: str = ""


def _get(url: str, timeout: int = _TIMEOUT,
         accept: str = _ACCEPT_BIN) -> bytes:
    """
    Descarga *url*.  `accept` por defecto es el de un binario; la consulta
    a la API tiene que pasar `_ACCEPT_API` o GitHub responde 415.
    """
    req = Request(url, headers={"User-Agent": _UA, "Accept": accept})
    ctx = ssl.create_default_context()
    with urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()


def check(timeout: int = _TIMEOUT) -> UpdateInfo:
    """
    Pregunta a GitHub por el último release.  Nunca lanza: sin red, la
    respuesta es "no hay actualización" con el motivo en `error`, porque un
    diálogo de error por abrir Eyve sin internet sería peor que la falta de
    la función.
    """
    info = UpdateInfo()
    try:
        raw = _get(_API, timeout, accept=_ACCEPT_API)
        data = json.loads(raw.decode("utf-8"))
    except (HTTPError, URLError, OSError, ValueError) as e:
        info.error = str(e)
        log.debug(f"Update check failed: {e}")
        return info

    tag = str(data.get("tag_name") or "")
    info.version = tag.lstrip("vV")
    info.notes = (data.get("body") or "").strip()

    for asset in data.get("assets") or []:
        name = asset.get("name") or ""
        if name == ASSET_NAME:
            info.url = asset.get("browser_download_url") or ""
            info.size = int(asset.get("size") or 0)
        elif name == SUMS_NAME:
            info.sums_url = asset.get("browser_download_url") or ""

    if not info.url:
        info.error = f"el release {tag} no trae {ASSET_NAME}"
        return info

    info.available = is_newer(info.version)
    _cfg.set("update_last_check", time.time())
    _cfg.set("update_last_seen", info.version)
    return info


def should_check(every_hours: int = 24) -> bool:
    """
    True si toca consultar.  Evita pegarle a la API en cada arranque: una
    vez al día es más que suficiente para un release cada varias semanas.
    """
    if _cfg.get("update_check_disabled", False):
        return False
    last = float(_cfg.get("update_last_check", 0) or 0)
    return (time.time() - last) > every_hours * 3600


# ── descarga ─────────────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_sha(sums_text: str, asset: str = ASSET_NAME) -> Optional[str]:
    """Saca el hash de *asset* de un SHA256SUMS.txt ('<hash>  <archivo>')."""
    for line in (sums_text or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].strip() == asset:
            h = parts[0].strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", h):
                return h
    return None


def download(info: UpdateInfo, dest_dir: Optional[Path] = None,
             progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """
    Baja el ZIP y verifica su SHA256.  Devuelve la ruta del ZIP verificado.

    Lanza RuntimeError si el checksum no cuadra: descomprimir un ZIP que no
    es el que se publicó encima de la instalación es justo el fallo que
    deja a un cliente sin Eyve en medio de un turno.
    """
    if not info.url:
        raise RuntimeError("no hay URL de descarga")
    dest_dir = Path(dest_dir or tempfile.mkdtemp(prefix="eyve_update_"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / ASSET_NAME

    req = Request(info.url, headers={"User-Agent": _UA})
    ctx = ssl.create_default_context()
    with urlopen(req, timeout=_TIMEOUT, context=ctx) as r, open(zip_path, "wb") as f:
        total = int(r.headers.get("Content-Length") or info.size or 0)
        done = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)

    if not info.sums_url:
        raise RuntimeError(f"el release no trae {SUMS_NAME}; no se puede verificar")
    try:
        sums = _get(info.sums_url).decode("utf-8", "replace")
    except (HTTPError, URLError, OSError) as e:
        raise RuntimeError(f"no se pudo bajar {SUMS_NAME}: {e}") from e

    want = expected_sha(sums)
    if not want:
        raise RuntimeError(f"{SUMS_NAME} no menciona {ASSET_NAME}")
    got = _sha256(zip_path)
    if got != want:
        raise RuntimeError(
            f"el archivo descargado no coincide con el publicado\n"
            f"  esperado: {want}\n  obtenido: {got}")
    log.info(f"Update {info.version} downloaded and verified ({zip_path})")
    return zip_path


# ── aplicación ───────────────────────────────────────────────────────────────
def install_root() -> Path:
    """
    Carpeta donde vive la instalación (la que contiene `eyve/`).

    No se usa el cwd: Eyve se abre con un acceso directo y el cwd puede ser
    cualquier cosa.
    """
    return Path(__file__).resolve().parents[2]


@dataclass
class ApplyResult:
    ok: bool = False
    replaced: list[str] = None
    backup: Optional[Path] = None
    requirements_changed: bool = False
    error: str = ""

    def __post_init__(self):
        if self.replaced is None:
            self.replaced = []


def _zip_root(zf: zipfile.ZipFile) -> str:
    """
    Prefijo común del ZIP.  El build actual produce estructura plana, pero
    un ZIP generado a mano suele traer todo bajo 'Eyve_2.1.0/'; detectarlo
    evita instalar una carpeta dentro de otra.
    """
    names = [n for n in zf.namelist() if not n.endswith("/")]
    if not names:
        return ""
    first = names[0].split("/")[0]
    if all(n.split("/")[0] == first for n in names) and "/" in names[0]:
        return first + "/"
    return ""


def _safe_members(zf: zipfile.ZipFile, root: str) -> list[tuple[str, str]]:
    """
    (nombre en el zip, ruta relativa de destino) de los miembros instalables.

    Descarta rutas absolutas y cualquier '..': un ZIP manipulado con
    '../../windows/system32/...' escribiría fuera de la instalación.  No es
    paranoia gratuita — es el fallo clásico de descomprimir sin mirar, y
    aquí el ZIP viene de la red.
    """
    out = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        rel = name[len(root):] if root and name.startswith(root) else name
        rel = rel.replace("\\", "/").lstrip("/")
        if not rel:
            continue
        parts = Path(rel).parts
        if any(p == ".." for p in parts) or Path(rel).is_absolute():
            log.warning(f"Update: entrada de ZIP descartada por insegura: {name}")
            continue
        if parts[0] in PROTECTED:
            continue
        out.append((name, rel))
    return out


def apply(zip_path: Path, root: Optional[Path] = None,
          backup_dir: Optional[Path] = None) -> ApplyResult:
    """
    Instala el ZIP verificado sobre *root*.

    Orden a propósito: primero se extrae TODO a una carpeta temporal, luego
    se respalda lo que se va a reemplazar, y solo entonces se mueve.  Así
    un error a mitad de la extracción no deja la instalación con medio
    paquete nuevo y medio viejo.
    """
    root = Path(root or install_root())
    res = ApplyResult()
    stage = Path(tempfile.mkdtemp(prefix="eyve_stage_"))
    backup = Path(backup_dir or (root / ".eyve_backup"))

    try:
        with zipfile.ZipFile(zip_path) as zf:
            base = _zip_root(zf)
            members = _safe_members(zf, base)
            if not any(rel.startswith("eyve/") for _, rel in members):
                res.error = "el ZIP no contiene el paquete eyve/; no se instala"
                return res
            for name, rel in members:
                target = stage / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        # ¿cambió requirements.txt?  Si sí, el llamador tiene que correr pip.
        new_req = stage / "requirements.txt"
        old_req = root / "requirements.txt"
        if new_req.exists():
            res.requirements_changed = (
                not old_req.exists()
                or new_req.read_bytes().strip() != old_req.read_bytes().strip())

        # respaldo de lo que se va a reemplazar
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        backup.mkdir(parents=True, exist_ok=True)
        top = sorted({Path(rel).parts[0] for _, rel in members})
        for item in top:
            cur = root / item
            if cur.exists():
                if cur.is_dir():
                    shutil.copytree(cur, backup / item, dirs_exist_ok=True)
                else:
                    shutil.copy2(cur, backup / item)

        # reemplazo
        try:
            for item in top:
                src = stage / item
                dst = root / item
                if src.is_dir():
                    # El paquete viejo se BORRA antes de poner el nuevo: si
                    # solo se copiara encima, un modulo eliminado en la
                    # version nueva seguiria ahi y se seguiria importando.
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                res.replaced.append(item)
        except Exception as e:
            log.error(f"Update failed mid-apply, restoring backup: {e}")
            _restore(backup, root, top)
            res.error = f"fallo al instalar, se restauro el respaldo: {e}"
            return res

        res.backup = backup
        res.ok = True
        log.info(f"Update applied: {', '.join(res.replaced)}")
        return res

    except Exception as e:
        res.error = str(e)
        log.error(f"Update apply error: {e}")
        return res
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _restore(backup: Path, root: Path, items: list[str]) -> None:
    for item in items:
        src = backup / item
        dst = root / item
        if not src.exists():
            continue
        try:
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        except Exception as e:
            log.error(f"Restore failed for {item}: {e}")


def rollback(root: Optional[Path] = None,
             backup_dir: Optional[Path] = None) -> bool:
    """Vuelve a la versión anterior desde el respaldo de la última update."""
    root = Path(root or install_root())
    backup = Path(backup_dir or (root / ".eyve_backup"))
    if not backup.exists():
        return False
    items = [p.name for p in backup.iterdir()]
    _restore(backup, root, items)
    log.info(f"Rolled back: {', '.join(items)}")
    return True


# ── orquestación ─────────────────────────────────────────────────────────────
def update_now(info: UpdateInfo,
               progress: Optional[Callable[[int, int], None]] = None,
               root: Optional[Path] = None) -> ApplyResult:
    """Bajar + verificar + instalar, en un paso, para la UI."""
    tmp = Path(tempfile.mkdtemp(prefix="eyve_update_"))
    try:
        zip_path = download(info, tmp, progress)
        return apply(zip_path, root)
    except Exception as e:
        return ApplyResult(ok=False, error=str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_async(callback: Callable[[UpdateInfo], None]) -> None:
    """
    Consulta en segundo plano.  El arranque de Eyve no espera a la red:
    una planta con el proxy caído no puede quedarse mirando un splash.
    """
    def work():
        try:
            callback(check())
        except Exception as e:      # el updater nunca tumba la app
            log.debug(f"check_async error: {e}")
    threading.Thread(target=work, daemon=True).start()


def restart_command() -> list[str]:
    """
    Cómo relanzar Eyve después de actualizar.

    run.bat es el camino bueno: activa el venv.  Lanzar sys.executable
    directo funciona solo si ya se corría desde el venv.
    """
    root = install_root()
    bat = root / "run.bat"
    if os.name == "nt" and bat.exists():
        return ["cmd", "/c", "start", "", str(bat)]
    return [sys.executable, "-m", "eyve.main"]
