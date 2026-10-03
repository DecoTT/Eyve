# -*- coding: utf-8 -*-
"""
Prueba del updater con instalaciones y ZIPs de verdad en disco.

Lo que de verdad importa no es que una actualizacion buena funcione, sino
que una MALA no destruya la instalacion. Asi que cada caso bueno va con su
caso malo: checksum que no cuadra, ZIP con '..', ZIP sin el paquete eyve/,
y fallo a mitad de la instalacion.
"""
import sys
import hashlib
import io
import shutil
import zipfile
import tempfile
from pathlib import Path

# El repo es el padre de tests/: nada de rutas absolutas, para que las
# pruebas corran en cualquier maquina y desde cualquier carpeta.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Salidas de las pruebas (hojas de contacto, proyectos temporales).
SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

from eyve.core import updater as U

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FALLA'}  {name}: got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


def check_true(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


def fake_install(root: Path):
    """Una instalacion de Eyve como la que tendria un cliente."""
    if root.exists():
        shutil.rmtree(root)
    (root / "eyve" / "ui").mkdir(parents=True)
    (root / "eyve" / "__init__.py").write_text('__version__ = "2.1.0"\n')
    (root / "eyve" / "viejo_modulo.py").write_text("# se va en la 2.2\n")
    (root / "eyve" / "ui" / "app.py").write_text("# v1\n")
    (root / "requirements.txt").write_text("ultralytics>=8.3,<9\n")
    (root / "run.bat").write_text("@echo off\n")
    # lo del usuario, que NO se debe tocar
    (root / "projects" / "MiProyecto" / "models").mkdir(parents=True)
    (root / "projects" / "MiProyecto" / "project.yaml").write_text("name: MiProyecto\n")
    (root / "projects" / "MiProyecto" / "models" / "best.pt").write_bytes(b"PESOS")
    (root / ".venv" / "Scripts").mkdir(parents=True)
    (root / ".venv" / "Scripts" / "python.exe").write_bytes(b"PYTHON")
    (root / "notas_del_cliente.txt").write_text("no me borres\n")
    return root


def make_zip(path: Path, entries: dict, prefix: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, content in entries.items():
            zf.writestr(prefix + rel, content)
    return path


GOOD = {
    "eyve/__init__.py": '__version__ = "2.2.0"\n',
    "eyve/ui/app.py": "# v2\n",
    "eyve/nuevo_modulo.py": "# nuevo en la 2.2\n",
    "requirements.txt": "ultralytics>=8.3,<9\nPyJWT[crypto]\n",
    "run.bat": "@echo off\nrem v2\n",
    "LICENSE": "AGPL-3.0\n",
}

if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)

print("\n[1] Comparacion de versiones")
check("v2.2.0 > 2.1.0", U.is_newer("v2.2.0", "2.1.0"), True)
check("2.1.1 > 2.1.0", U.is_newer("2.1.1", "2.1.0"), True)
check("2.1.0 no > 2.1.0", U.is_newer("2.1.0", "2.1.0"), False)
check("2.0.9 no > 2.1.0", U.is_newer("2.0.9", "2.1.0"), False)
# el caso que rompe la comparacion por texto
check("2.10.0 > 2.9.0 (no es comparacion de texto)",
      U.is_newer("2.10.0", "2.9.0"), True)
check("2.9.0 no > 2.10.0", U.is_newer("2.9.0", "2.10.0"), False)
check("2.1 vs 2.1.0 (longitudes distintas)", U.is_newer("2.1", "2.1.0"), False)
check("2.1.0.1 > 2.1.0", U.is_newer("2.1.0.1", "2.1.0"), True)
check("basura no se considera mas nueva", U.is_newer("", "2.1.0"), False)

print("\n[2] Lectura de SHA256SUMS.txt")
sums = ("8060b7050bd9c24edac98669802eced58e14295ce9ae1639abeebc24867d7758  "
        "Eyve-2.1-setup.zip\n"
        "aaaa0000000000000000000000000000000000000000000000000000000000ff  otro.zip\n")
check("saca el hash correcto", U.expected_sha(sums),
      "8060b7050bd9c24edac98669802eced58e14295ce9ae1639abeebc24867d7758")
check("otro asset", U.expected_sha(sums, "otro.zip"),
      "aaaa0000000000000000000000000000000000000000000000000000000000ff")
check("asset ausente -> None", U.expected_sha(sums, "noexiste.zip"), None)
check("texto vacio -> None", U.expected_sha(""), None)
check("hash mal formado -> None",
      U.expected_sha("nohexadecimal  Eyve-2.1-setup.zip"), None)
check("nombre parecido no cuela",
      U.expected_sha("ab" * 32 + "  Eyve-2.1-setup.zip.bak"), None)

print("\n[3] Instalacion buena")
root = fake_install(SCRATCH / "inst1")
z = make_zip(SCRATCH / "good.zip", GOOD)
res = U.apply(z, root)
check("ok", res.ok, True)
check("sin error", res.error, "")
check("version nueva en disco",
      (root / "eyve" / "__init__.py").read_text().strip(), '__version__ = "2.2.0"')
check("archivo actualizado", (root / "eyve" / "ui" / "app.py").read_text().strip(), "# v2")
check_true("archivo nuevo instalado", (root / "eyve" / "nuevo_modulo.py").exists())
check_true("modulo ELIMINADO en la version nueva ya no esta",
           not (root / "eyve" / "viejo_modulo.py").exists())
check("requirements actualizado",
      "PyJWT" in (root / "requirements.txt").read_text(), True)
check("detecta que requirements cambio", res.requirements_changed, True)
check_true("LICENSE nuevo instalado", (root / "LICENSE").exists())

print("\n[4] Lo del usuario queda INTACTO")
check_true("projects/ sigue ahi",
           (root / "projects" / "MiProyecto" / "project.yaml").exists())
check("los pesos del usuario intactos",
      (root / "projects" / "MiProyecto" / "models" / "best.pt").read_bytes(), b"PESOS")
check_true(".venv intacto", (root / ".venv" / "Scripts" / "python.exe").exists())
check("archivo suelto del usuario intacto",
      (root / "notas_del_cliente.txt").read_text().strip(), "no me borres")

print("\n[5] Respaldo y vuelta atras")
check_true("hay respaldo", res.backup is not None and res.backup.exists())
check_true("el respaldo tiene el eyve/ viejo",
           (res.backup / "eyve" / "viejo_modulo.py").exists())
check("vuelta atras", U.rollback(root), True)
check("version vieja restaurada",
      (root / "eyve" / "__init__.py").read_text().strip(), '__version__ = "2.1.0"')
check_true("el modulo viejo volvio", (root / "eyve" / "viejo_modulo.py").exists())
check_true("el modulo nuevo se fue", not (root / "eyve" / "nuevo_modulo.py").exists())
check_true("projects/ sigue intacto tras la vuelta atras",
           (root / "projects" / "MiProyecto" / "project.yaml").exists())

print("\n[6] ZIP con una carpeta raiz (Eyve_2.2.0/...)")
root = fake_install(SCRATCH / "inst2")
z = make_zip(SCRATCH / "nested.zip", GOOD, prefix="Eyve_2.2.0/")
res = U.apply(z, root)
check("ok", res.ok, True)
check_true("instalado en la raiz, no anidado",
           (root / "eyve" / "nuevo_modulo.py").exists())
check_true("no creo una carpeta Eyve_2.2.0",
           not (root / "Eyve_2.2.0").exists())

print("\n[7] ZIP MALICIOSO con '..' no escribe fuera")
root = fake_install(SCRATCH / "inst3")
fuera = SCRATCH / "VICTIMA.txt"
if fuera.exists():
    fuera.unlink()
mal = dict(GOOD)
mal["../VICTIMA.txt"] = "te hackee\n"
mal["../../VICTIMA2.txt"] = "te hackee otra vez\n"
z = make_zip(SCRATCH / "evil.zip", mal)
res = U.apply(z, root)
check("instala lo legitimo", res.ok, True)
check_true("NO escribio fuera de la instalacion", not fuera.exists())
check_true("tampoco dos niveles arriba",
           not (SCRATCH.parent / "VICTIMA2.txt").exists())
check_true("el paquete si se actualizo", (root / "eyve" / "nuevo_modulo.py").exists())

print("\n[8] ZIP que intenta pisar projects/ o .venv/")
root = fake_install(SCRATCH / "inst4")
mal = dict(GOOD)
mal["projects/MiProyecto/project.yaml"] = "name: SECUESTRADO\n"
mal["projects/MiProyecto/models/best.pt"] = "BASURA"
mal[".venv/Scripts/python.exe"] = "TROYANO"
z = make_zip(SCRATCH / "greedy.zip", mal)
res = U.apply(z, root)
check("instala lo legitimo", res.ok, True)
check("project.yaml del usuario NO fue pisado",
      (root / "projects" / "MiProyecto" / "project.yaml").read_text().strip(),
      "name: MiProyecto")
check("los pesos del usuario NO fueron pisados",
      (root / "projects" / "MiProyecto" / "models" / "best.pt").read_bytes(), b"PESOS")
check("el venv NO fue pisado",
      (root / ".venv" / "Scripts" / "python.exe").read_bytes(), b"PYTHON")

print("\n[9] ZIP sin el paquete eyve/ se rechaza")
root = fake_install(SCRATCH / "inst5")
z = make_zip(SCRATCH / "nopkg.zip", {"README.md": "hola\n", "run.bat": "@echo off\n"})
res = U.apply(z, root)
check("se rechaza", res.ok, False)
check_true("dice por que", "eyve/" in res.error, res.error)
check("la instalacion quedo como estaba",
      (root / "eyve" / "__init__.py").read_text().strip(), '__version__ = "2.1.0"')
check_true("run.bat NO fue reemplazado",
           (root / "run.bat").read_text().strip() == "@echo off")

print("\n[10] requirements IGUAL no se reporta como cambiado")
root = fake_install(SCRATCH / "inst6")
same = dict(GOOD)
same["requirements.txt"] = "ultralytics>=8.3,<9\n"
z = make_zip(SCRATCH / "same_req.zip", same)
res = U.apply(z, root)
check("instalado", res.ok, True)
check("requirements sin cambios", res.requirements_changed, False)

print("\n[11] Fallo a mitad de la instalacion -> restaura")
root = fake_install(SCRATCH / "inst7")
z = make_zip(SCRATCH / "good2.zip", GOOD)
orig_copytree = shutil.copytree
estado = {"n": 0}


def copytree_que_falla(src, dst, *a, **kw):
    # Falla solo al copiar HACIA la instalacion. No sirve distinguir por
    # dirs_exist_ok: la recursion interna de shutil.copytree lo pasa
    # POSICIONALMENTE, asi que kw.get("dirs_exist_ok") es None en las
    # llamadas anidadas del respaldo.
    if ".eyve_backup" in str(dst):
        return orig_copytree(src, dst, *a, **kw)
    estado["n"] += 1
    if estado["n"] == 1:
        raise OSError("disco lleno (simulado)")
    return orig_copytree(src, dst, *a, **kw)


shutil.copytree = copytree_que_falla
try:
    res = U.apply(z, root)
finally:
    shutil.copytree = orig_copytree
check("reporta el fallo", res.ok, False)
check_true("dice que restauro", "restaur" in res.error.lower(), res.error)
check("la version vieja sigue ahi",
      (root / "eyve" / "__init__.py").read_text().strip(), '__version__ = "2.1.0"')
check_true("el modulo viejo sigue ahi", (root / "eyve" / "viejo_modulo.py").exists())
check_true("projects/ intacto tras el fallo",
           (root / "projects" / "MiProyecto" / "project.yaml").exists())

print("\n[11b] Fallo durante el RESPALDO: no se toca nada")
root = fake_install(SCRATCH / "inst8")


def copytree_falla_en_respaldo(src, dst, *a, **kw):
    if ".eyve_backup" in str(dst):
        raise OSError("sin permisos para respaldar (simulado)")
    return orig_copytree(src, dst, *a, **kw)


shutil.copytree = copytree_falla_en_respaldo
try:
    res = U.apply(z, root)
finally:
    shutil.copytree = orig_copytree
check("reporta el fallo", res.ok, False)
check_true("no dice que instalo nada", res.replaced == [], str(res.replaced))
check("la instalacion quedo intacta",
      (root / "eyve" / "__init__.py").read_text().strip(), '__version__ = "2.1.0"')
check_true("sin archivos de la version nueva",
           not (root / "eyve" / "nuevo_modulo.py").exists())
check_true("projects/ intacto",
           (root / "projects" / "MiProyecto" / "project.yaml").exists())

print("\n[12] download() rechaza un checksum que no cuadra")
import eyve.core.updater as UU

z = make_zip(SCRATCH / "dl.zip", GOOD)
real = hashlib.sha256(z.read_bytes()).hexdigest()
descargas = {"file://good": z.read_bytes()}


def fake_get(url, timeout=20):
    if url.endswith("SHA256SUMS_ok"):
        return (real + "  " + U.ASSET_NAME + "\n").encode()
    if url.endswith("SHA256SUMS_mal"):
        return ("00" * 32 + "  " + U.ASSET_NAME + "\n").encode()
    raise AssertionError("url inesperada " + url)


class FakeResp:
    def __init__(self, data): self._d = io.BytesIO(data); self.headers = {}
    def read(self, n=-1): return self._d.read(n)
    def __enter__(self): return self
    def __exit__(self, *a): return False


def fake_urlopen(req, timeout=20, context=None):
    return FakeResp(z.read_bytes())


UU._get = fake_get
UU.urlopen = fake_urlopen
try:
    info_ok = U.UpdateInfo(available=True, version="2.2.0", url="http://x/a.zip",
                           sums_url="http://x/SHA256SUMS_ok")
    p = U.download(info_ok, SCRATCH / "dl_ok")
    check_true("descarga con checksum bueno", p.exists())

    info_mal = U.UpdateInfo(available=True, version="2.2.0", url="http://x/a.zip",
                            sums_url="http://x/SHA256SUMS_mal")
    try:
        U.download(info_mal, SCRATCH / "dl_mal")
        fails.append("acepto un checksum que no cuadra")
        print("  FALLA  acepto un checksum que no cuadra")
    except RuntimeError as e:
        check_true("rechaza el checksum malo", "no coincide" in str(e), str(e)[:60])

    info_sin = U.UpdateInfo(available=True, version="2.2.0", url="http://x/a.zip",
                            sums_url="")
    try:
        U.download(info_sin, SCRATCH / "dl_sin")
        fails.append("acepto un release sin SHA256SUMS")
        print("  FALLA  acepto un release sin SHA256SUMS")
    except RuntimeError as e:
        check_true("exige SHA256SUMS", "verificar" in str(e), str(e)[:60])
finally:
    import importlib
    importlib.reload(UU)

print("\n[13] check() sin red no revienta")
import eyve.core.updater as U2


def urlopen_muerto(*a, **kw):
    raise OSError("sin red")


U2.urlopen = urlopen_muerto
info = U2.check()
check("no hay actualizacion", info.available, False)
check_true("reporta el motivo", bool(info.error), info.error)
check("conserva la version local", info.current, U2.__version__)
importlib = __import__("importlib")
importlib.reload(U2)

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("UPDATER: TODAS LAS PRUEBAS PASARON")
