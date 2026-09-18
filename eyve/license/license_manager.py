"""
License manager — Eyve 2.1 · contrato con sbcsuite (docs/licencias-eyve.md).

La llave es un JWT firmado con EdDSA (Ed25519) que emite la tienda
(sbcsuite.com.mx). Lleva dentro todo lo que Eyve necesita para decidir SIN RED:
nivel, titular, emisión y vencimiento. Se verifica aquí con la clave pública
embebida; el servidor es *aditivo*: registra el equipo, cuenta hasta tres y
entrega renovaciones. Si no hay red, Eyve corre igual.

Regla de la casa: POR HONOR. NADA BLOQUEA.
  - Sin llave              -> Eyve Free.
  - Llave falsa            -> aviso claro, sigue en Free.
  - Llave vencida          -> sigue con el nivel que trae y avisa en cada arranque.
  - Sin cupo (4.º equipo)  -> aviso con la lista de equipos, sigue funcionando.
  - Sin red                -> se guarda y se reintenta al siguiente arranque.

Archivos (junto al config de la app, ~/.eyve/):
  licencia.jwt   la llave tal cual se pegó (una sola; pegar otra la reemplaza)
  licencia.json  estado local: activación pendiente, último check-in, lo que
                 contestó el servidor la última vez (equipos, estado real)

AGPL note: this is part of the official build experience. Users who build
from source may modify or remove this — that is permitted under AGPL-3.0.
"""
from __future__ import annotations

import hashlib
import json
import platform
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import jwt
from jwt.algorithms import OKPAlgorithm

from eyve import __version__ as EYVE_VERSION

# ── contrato ──────────────────────────────────────────────────────────────────

ISSUER = "sbcgroup.com.mx/eyve"
# Clave PÚBLICA de firma (supabase/functions/_shared/eyve-public.jwk.json).
# La privada sólo existe en el servidor.
PUBLIC_JWK = {
    "kty": "OKP", "crv": "Ed25519", "alg": "EdDSA", "kid": "eyve-lic-2026-09",
    "x": "aCG1V9fOCcbprd2YnSKGqT6nT1mPbDvN7yv6-qrsO50",
}
SUPABASE_URL = "https://dmwcaohtaftbvmixbplr.supabase.co"
# Anon key del proyecto: pública por diseño (la misma que usa la tienda).
ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRtd2Nhb2h0YWZ0YnZtaXhicGxyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYzMjM0NzUsImV4cCI6MjA3MTg5OTQ3NX0."
    "7y2BoM5laL9gRdgaH8YyiXiSDnVKjKQKYCN_5SEjfj0"
)
URL_CUENTA_LICENCIAS = "https://sbcsuite.com.mx/store/cuenta/licencias"
URL_COMPRAR = "https://sbcgroup.com.mx/prueba-eyve/"

NIVELES = ("free", "estudiante", "normal", "pro")
NIVEL_LABEL = {"free": "Free", "estudiante": "Estudiante", "normal": "Normal", "pro": "Pro"}
MAX_EQUIPOS_DEFAULT = 3
CHECKIN_CADA = timedelta(days=7)
_HTTP_TIMEOUT = 8

_APP_DIR = Path.home() / ".eyve"
_KEY_FILE = _APP_DIR / "licencia.jwt"
_STATE_FILE = _APP_DIR / "licencia.json"

_PUB = OKPAlgorithm.from_jwk(json.dumps(PUBLIC_JWK))


class LlaveInvalida(Exception):
    """Firma falsa, emisor equivocado o formato roto. NO es 'vencida'."""


# ── la llave ──────────────────────────────────────────────────────────────────

@dataclass
class Licencia:
    llave: str
    nivel: str
    sub: str                 # correo del titular
    jti: str                 # id en la base de sbcsuite
    exp: datetime
    iat: Optional[datetime]
    nombre: str = ""
    maxeq: int = MAX_EQUIPOS_DEFAULT

    @property
    def vencida(self) -> bool:
        return datetime.now(timezone.utc) > self.exp

    @property
    def vence_str(self) -> str:
        return self.exp.astimezone().strftime("%Y-%m-%d")

    @property
    def titular(self) -> str:
        return self.nombre or self.sub


def leer_llave(llave: str) -> Licencia:
    """
    Verifica firma + emisor SIN RED y devuelve los claims.
    Lanza LlaveInvalida si la firma es falsa. Una llave vencida se devuelve
    igual (con .vencida = True): por honor, no se degrada.
    """
    llave = (llave or "").strip()
    if not llave:
        raise LlaveInvalida("empty")
    try:
        claims = jwt.decode(
            llave, key=_PUB, algorithms=["EdDSA"], issuer=ISSUER,
            options={"verify_exp": False, "require": ["exp", "sub", "jti", "iss"]},
        )
    except jwt.PyJWTError as e:
        raise LlaveInvalida(str(e)) from e
    nivel = str(claims.get("nivel", "")).lower()
    if nivel not in NIVELES:
        raise LlaveInvalida(f"nivel desconocido: {nivel!r}")
    iat = claims.get("iat")
    return Licencia(
        llave=llave, nivel=nivel, sub=str(claims["sub"]), jti=str(claims["jti"]),
        exp=datetime.fromtimestamp(int(claims["exp"]), tz=timezone.utc),
        iat=datetime.fromtimestamp(int(iat), tz=timezone.utc) if iat else None,
        nombre=str(claims.get("nombre") or ""),
        maxeq=int(claims.get("maxeq") or MAX_EQUIPOS_DEFAULT),
    )


# ── el equipo ─────────────────────────────────────────────────────────────────

def _machine_id() -> str:
    """Id de máquina que no cambia entre arranques. MachineGuid en Windows."""
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SOFTWARE\Microsoft\Cryptography",
                                0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
                return str(winreg.QueryValueEx(k, "MachineGuid")[0])
        except OSError:
            pass
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            return Path(p).read_text().strip()
        except OSError:
            pass
    return str(uuid.getnode())


def hash_equipo() -> str:
    """sha256 hex (64 chars) del id de máquina. Es lo que identifica al equipo
    en sbcsuite para liberarlo después: no debe cambiar entre arranques."""
    return hashlib.sha256(f"{_machine_id()}|eyve".encode()).hexdigest()


def nombre_equipo() -> str:
    return (platform.node() or "equipo")[:120]


# ── estado local ──────────────────────────────────────────────────────────────

@dataclass
class EstadoServidor:
    """Lo último que contestó sbcsuite. Informativo: nunca decide si Eyve corre."""
    estado: str = ""                 # vigente · vencida · revocada · pendiente
    equipos_activos: int = 0
    max_equipos: int = MAX_EQUIPOS_DEFAULT
    motivo: str = ""                 # sin_cupo · llave_invalida · no_registrada · sin_red · ...
    mensaje: str = ""
    equipos: list = field(default_factory=list)
    consultado_en: str = ""          # ISO


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class LicenseManager:
    """
    Una sola llave activa. Lectura barata: se puede instanciar donde haga falta.
    Las llamadas de red van en `activar_online` / `checkin`, pensadas para
    correr en hilo; nunca hay que esperarlas para arrancar.
    """

    def __init__(self) -> None:
        self._state: dict = _load_json(_STATE_FILE)
        self.licencia: Optional[Licencia] = None
        self.error_llave: str = ""       # llave guardada pero inválida
        self._cargar_llave()

    # ── carga / guardado ─────────────────────────────────────────────────────

    def _cargar_llave(self) -> None:
        self.licencia, self.error_llave = None, ""
        if not _KEY_FILE.exists():
            return
        try:
            self.licencia = leer_llave(_KEY_FILE.read_text(encoding="utf-8"))
        except LlaveInvalida as e:
            self.error_llave = str(e)

    def _save_state(self) -> None:
        _save_json(_STATE_FILE, self._state)

    # ── lo que Eyve consulta ─────────────────────────────────────────────────

    @property
    def nivel(self) -> str:
        """Nivel efectivo. Sin llave o llave falsa → free. Vencida → el que trae."""
        return self.licencia.nivel if self.licencia else "free"

    @property
    def nivel_label(self) -> str:
        return NIVEL_LABEL[self.nivel]

    def status(self) -> str:
        """'sin_llave' | 'invalida' | 'vigente' | 'vencida'"""
        if self.licencia is None:
            return "invalida" if self.error_llave else "sin_llave"
        return "vencida" if self.licencia.vencida else "vigente"

    def es_pro(self) -> bool:
        return self.nivel == "pro"

    def permite_checks_avanzados(self) -> bool:
        """Módulos de check de SBC (polaridad, flujo, serigrafía, patrones): Pro.
        Free: detección, conteo y log. Estudiante/Normal: plataforma abierta."""
        return self.nivel == "pro"

    @property
    def servidor(self) -> EstadoServidor:
        d = self._state.get("servidor") or {}
        return EstadoServidor(**{k: d[k] for k in EstadoServidor.__dataclass_fields__ if k in d})

    @property
    def activacion_pendiente(self) -> bool:
        return bool(self.licencia) and bool(self._state.get("pendiente_activar"))

    @property
    def ultimo_checkin(self) -> Optional[datetime]:
        raw = self._state.get("ultimo_checkin")
        try:
            return datetime.fromisoformat(raw) if raw else None
        except ValueError:
            return None

    def resumen(self) -> str:
        """Una línea para la barra de estado."""
        lic = self.licencia
        if lic is None:
            return "Eyve Free"
        s = f"Eyve {NIVEL_LABEL[lic.nivel]} · {lic.titular}"
        if lic.vencida:
            s += f" · vencida {lic.vence_str}"
        return s

    # ── pegar / quitar llave ─────────────────────────────────────────────────

    def guardar_llave(self, llave: str) -> Licencia:
        """
        Verifica sin red y guarda. Reemplaza la anterior. Lanza LlaveInvalida.
        Deja la activación pendiente: llama a `activar_online` (en hilo) después.
        """
        lic = leer_llave(llave)
        _APP_DIR.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(lic.llave, encoding="utf-8")
        # Llave nueva: el estado del servidor de la anterior ya no aplica.
        self._state = {"pendiente_activar": True, "guardada_en": _now_iso()}
        self._save_state()
        self._cargar_llave()
        return lic

    def quitar_llave(self) -> None:
        try:
            _KEY_FILE.unlink()
        except OSError:
            pass
        self._state = {}
        self._save_state()
        self._cargar_llave()

    # ── red ──────────────────────────────────────────────────────────────────

    def _post(self, fn: str, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            f"{SUPABASE_URL}/functions/v1/{fn}",
            data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "apikey": ANON_KEY,
                     "Authorization": f"Bearer {ANON_KEY}",
                     "User-Agent": f"Eyve/{EYVE_VERSION}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
                return r.status, json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8") or "{}")
            except ValueError:
                return e.code, {}
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise ConnectionError(str(e)) from e

    def _guardar_respuesta(self, code: int, j: dict, *, checkin: bool) -> EstadoServidor:
        srv = EstadoServidor(
            estado=str(j.get("estado", "")),
            equipos_activos=int(j.get("equipos_activos") or 0),
            max_equipos=int(j.get("max_equipos") or (self.licencia.maxeq if self.licencia else MAX_EQUIPOS_DEFAULT)),
            motivo="" if j.get("ok") else str(j.get("motivo") or f"http_{code}"),
            mensaje=str(j.get("mensaje") or ""),
            equipos=list(j.get("equipos") or []),
            consultado_en=_now_iso(),
        )
        self._state["servidor"] = srv.__dict__
        if j.get("ok"):
            self._state["pendiente_activar"] = False
            self._state["ultimo_checkin"] = _now_iso()
        elif srv.motivo in ("sin_cupo", "llave_invalida", "no_registrada"):
            # Respuesta definitiva: no tiene caso reintentar en cada arranque.
            self._state["pendiente_activar"] = False
        self._save_state()
        return srv

    def activar_online(self) -> EstadoServidor:
        """
        POST licencia-activar. Registra este equipo (o cuenta como check-in si
        ya estaba). Lanza ConnectionError si no hay red: el llamador decide
        (normalmente: nada, se reintenta al siguiente arranque).
        """
        if not self.licencia:
            raise RuntimeError("sin llave")
        code, j = self._post("licencia-activar", {
            "llave": self.licencia.llave, "hash_equipo": hash_equipo(),
            "nombre_equipo": nombre_equipo(), "version_eyve": EYVE_VERSION,
        })
        return self._guardar_respuesta(code, j, checkin=False)

    def checkin(self) -> EstadoServidor:
        """
        POST licencia-estado. Actualiza último check-in, trae el estado real y,
        si hay `llave_nueva` (renovación), la guarda en lugar de la actual.
        """
        if not self.licencia:
            raise RuntimeError("sin llave")
        code, j = self._post("licencia-estado", {
            "llave": self.licencia.llave, "hash_equipo": hash_equipo(),
            "version_eyve": EYVE_VERSION,
        })
        srv = self._guardar_respuesta(code, j, checkin=True)
        nueva = j.get("llave_nueva")
        if j.get("ok") and nueva:
            try:
                lic = leer_llave(nueva)
                _KEY_FILE.write_text(lic.llave, encoding="utf-8")
                self._state["renovada_en"] = _now_iso()
                # El equipo ya está registrado bajo la anterior; la nueva se
                # activa en el siguiente arranque con red.
                self._state["pendiente_activar"] = True
                self._save_state()
                self._cargar_llave()
            except LlaveInvalida:
                pass
        return srv

    def comprobar(self) -> EstadoServidor:
        """Lo que hace el botón 'Comprobar ahora' y el arranque con red."""
        return self.activar_online() if self.activacion_pendiente else self.checkin()

    def toca_checkin(self) -> bool:
        if not self.licencia:
            return False
        if self.activacion_pendiente:
            return True
        last = self.ultimo_checkin
        return last is None or datetime.now(timezone.utc) - last >= CHECKIN_CADA

    def comprobar_en_hilo(self, on_done: Optional[Callable[[Optional[EstadoServidor], Optional[str]], None]] = None,
                          solo_si_toca: bool = False) -> None:
        """
        Corre `comprobar()` en un hilo daemon. Nunca se espera. `on_done(srv, error)`
        se llama desde el hilo: si toca UI, el llamador lo manda a `after()`.
        """
        if not self.licencia or (solo_si_toca and not self.toca_checkin()):
            return

        def run() -> None:
            try:
                srv = self.comprobar()
                if on_done:
                    on_done(srv, None)
            except ConnectionError as e:
                if on_done:
                    on_done(None, f"sin_red: {e}")
            except Exception as e:  # noqa: BLE001 — nunca tumbar la app por esto
                if on_done:
                    on_done(None, str(e))

        threading.Thread(target=run, name="eyve-licencia", daemon=True).start()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
