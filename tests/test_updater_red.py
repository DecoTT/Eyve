# -*- coding: utf-8 -*-
"""
Lo que el actualizador le manda de verdad a GitHub.

`test_updater.py` cubre toda la logica —versiones, respaldo, archivos
protegidos, instalacion— sustituyendo `_get` y `urlopen` por falsos que
aceptan cualquier cosa.  Por eso no cazo el fallo real: hasta 2.1.0 Eyve
pedia la API con `Accept: application/octet-stream`, GitHub contestaba
**415 Unsupported Media Type**, y como `check()` se traga los errores a
proposito, el actualizador decia "no hay actualizacion" para siempre sin
hacer ruido.  Cien comprobaciones en verde y la funcion muerta.

Aqui se mira justo lo que aquellas no miraban:

  * sin red: QUE cabecera se manda, con el caso malo (la descarga de un
    asset si tiene que pedir octet-stream);
  * con red: que la API de verdad contesta 200 con esa cabecera y 415 con
    la vieja.  Es la unica que habria cazado el fallo original.

Sin internet las de red se saltan y se dice; las de cabecera corren
siempre, que son las que impiden la regresion.

    .venv\\Scripts\\python.exe tests\\test_updater_red.py
"""
from __future__ import annotations

import io
import json
import ssl
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eyve.core import updater as U

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


# ╔══ sin red: que cabecera se manda ═════════════════════════════════════╗
print("\n[1] La consulta a la API pide JSON, no un binario")

capturadas: list[Request] = []


class RespuestaFalsa:
    """Devuelve un release plausible para que check() siga su curso."""

    def __init__(self):
        self._d = io.BytesIO(json.dumps({
            "tag_name": "v9.9.9",
            "body": "",
            "assets": [
                {"name": U.ASSET_NAME, "size": 1,
                 "browser_download_url": "https://ejemplo/x.zip"},
                {"name": U.SUMS_NAME, "size": 1,
                 "browser_download_url": "https://ejemplo/s.txt"},
            ],
        }).encode())

    def read(self, n=-1):
        return self._d.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def urlopen_espia(req, timeout=20, context=None):
    capturadas.append(req)
    return RespuestaFalsa()


_real_urlopen = U.urlopen
_real_set = U._cfg.set
U.urlopen = urlopen_espia
U._cfg.set = lambda k, v: None          # que no escriba en el config real
try:
    U.check()
finally:
    U.urlopen = _real_urlopen
    U._cfg.set = _real_set

check("la consulta sale", len(capturadas) == 1, f"{len(capturadas)} peticiones")
if capturadas:
    acc = capturadas[0].get_header("Accept")
    check("va a la API de releases", capturadas[0].full_url == U._API,
          capturadas[0].full_url)
    check("el Accept NO es octet-stream (eso da 415)",
          acc != "application/octet-stream", repr(acc))
    check("el Accept es el JSON de GitHub",
          acc == "application/vnd.github+json", repr(acc))
    check("manda User-Agent", bool(capturadas[0].get_header("User-agent")))

print("\n[2] El caso malo: bajar un asset SI pide un binario")
capturadas.clear()
U.urlopen = urlopen_espia
try:
    U._get("https://ejemplo/x.zip")
finally:
    U.urlopen = _real_urlopen
if capturadas:
    acc = capturadas[0].get_header("Accept")
    check("la descarga pide octet-stream",
          acc == "application/octet-stream", repr(acc))
    check("y NO el JSON de la API",
          acc != "application/vnd.github+json", repr(acc))
else:
    check("la descarga sale", False)

# ╔══ con red: contra la API de verdad ═══════════════════════════════════╗
print("\n[3] Contra la API de GitHub de verdad")


def pedir(accept: str):
    req = Request(U._API, headers={"User-Agent": U._UA, "Accept": accept})
    try:
        with urlopen(req, timeout=20,
                     context=ssl.create_default_context()) as r:
            return r.status, json.loads(r.read().decode())
    except HTTPError as e:
        return e.code, None


hay_red = True
try:
    estado, datos = pedir(U._ACCEPT_API)
except (URLError, OSError) as e:
    hay_red = False
    print(f"  --    sin red ({e}); las de red se saltan")

if hay_red:
    check("con el Accept de Eyve la API contesta 200", estado == 200,
          str(estado))
    check("y trae un tag", bool(datos and datos.get("tag_name")),
          str(datos and datos.get("tag_name")))
    # el control: la cabecera vieja TIENE que seguir fallando, o esta
    # prueba no esta midiendo lo que cree
    estado_viejo, _ = pedir("application/octet-stream")
    check("y con la cabecera vieja sigue dando 415", estado_viejo == 415,
          str(estado_viejo))

    print("\n[4] check() completo contra la API real")
    info = U.check()
    check("sin error", not info.error, info.error)
    check("trae version", bool(info.version), info.version)
    check("trae el asset con el nombre fijo",
          info.url.endswith(U.ASSET_NAME), info.url)
    check("trae el SHA256SUMS", info.sums_url.endswith(U.SUMS_NAME),
          info.sums_url)
    # Coherencia: available tiene que concordar con la comparacion de
    # versiones, no ser un booleano suelto.
    check("'disponible' concuerda con la comparacion de versiones",
          info.available == U.is_newer(info.version, U.__version__),
          f"available={info.available} remota={info.version} "
          f"local={U.__version__}")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ACTUALIZADOR: LE PIDE A GITHUB LO QUE GITHUB ACEPTA")
