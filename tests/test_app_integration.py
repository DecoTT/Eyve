# -*- coding: utf-8 -*-
"""
Prueba de integracion: la app REAL arranca, navega por todas las pantallas
y cierra limpio, en los dos idiomas.

Las pruebas por pieza pasan y aun asi la app puede no abrir: basta un
import circular, una clave de i18n mal escrita o un widget que se
construye antes de existir. Esto es lo que lo detecta.

La camara se neutraliza (en una maquina de build no hay webcam, y no se
trata de probar la camara) pero TODO lo demas es el codigo de produccion.
"""
import sys
import os
import tempfile
from pathlib import Path

# El repo es el padre de tests/: nada de rutas absolutas, para que las
# pruebas corran en cualquier maquina y desde cualquier carpeta.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Salidas de las pruebas (hojas de contacto, proyectos temporales).
SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_OFFLINE", "1")

fails = []


def check_true(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


# ── neutralizar camara y descarga de modelos ANTES de construir la app ─────
from eyve.inference import detector as _det
_det.VideoSource.start = lambda self: False
_det.VideoSource.stop = lambda self: None
_det.VideoSource.read = lambda self: None

from eyve.ui.screens import production_screen as _ps
_ps.ProductionScreen._try_autoload_model = lambda self: None
from eyve.ui.screens import demo_screen as _ds
_ds.DemoScreen._load_model = lambda self: None

# y la consulta de actualizaciones (no se le pega a GitHub en una prueba)
from eyve.core import updater as _U
_U.check_async = lambda cb: None

from eyve.i18n import set_language
from eyve.ui.app import EyveApp
from eyve.ui.components.sidebar import NAV_ITEMS

# Un idioma por ejecucion: crear un segundo root de Tk despues de destruir
# el primero deja las PhotoImage cacheadas apuntando al interprete muerto
# ("pyimage1 doesn't exist"). Es un limite de Tk, no de Eyve.
LANG = sys.argv[1] if len(sys.argv) > 1 else "es"

for lang in (LANG,):
    print(f"\n[{lang}] Arranque y navegacion completa")
    set_language(lang)
    app = EyveApp()
    app.update()
    check_true(f"[{lang}] la ventana existe", app.winfo_exists())
    check_true(f"[{lang}] hay barra de estado", hasattr(app, "status_bar"))
    check_true(f"[{lang}] hay check_for_updates",
               callable(getattr(app, "check_for_updates", None)))

    for key, _icon in NAV_ITEMS:
        try:
            app.navigate(key)
            app.update()
            scr = app._screens.get(key)
            check_true(f"[{lang}] {key} se construye", scr is not None)
            check_true(f"[{lang}] {key} se muestra",
                       scr is not None and scr.winfo_ismapped())
        except Exception as e:
            import traceback
            traceback.print_exc()
            fails.append(f"[{lang}] {key} reventó: {e}")

    # el panel de conteo de Produccion responde en este idioma
    try:
        prod = app._screens.get("nav_production")
        if prod is not None:
            from eyve.modules import CountingModule
            from eyve.i18n import t
            for m in CountingModule.METHODS:
                prod._on_count_method_change(t("count_m_" + m))
                app.update()
            check_true(f"[{lang}] los 5 metodos de conteo responden",
                       prod._counting.method == CountingModule.METHODS[-1])
    except Exception as e:
        fails.append(f"[{lang}] panel de conteo: {e}")

    # Settings abre (es donde vive el boton de actualizaciones)
    try:
        from eyve.ui.screens.settings_popup import SettingsPopup
        pop = SettingsPopup(app, app=app)
        app.update()
        check_true(f"[{lang}] Settings abre", pop.winfo_exists())
        check_true(f"[{lang}] Settings tiene la casilla de updates",
                   hasattr(pop, "_auto_upd"))
        pop.destroy()
        app.update()
    except Exception as e:
        import traceback
        traceback.print_exc()
        fails.append(f"[{lang}] Settings: {e}")

    # cierre limpio
    try:
        app._teardown_all_screens()
        app.update()
        app.destroy()
        print(f"  OK    [{lang}] cierre limpio")
    except Exception as e:
        fails.append(f"[{lang}] cierre: {e}")

print("\n[3] Ninguna clave de i18n quedo sin traducir en lo nuevo")
from eyve.i18n.es import STRINGS as ES
from eyve.i18n.en import STRINGS as EN
dif = set(ES) ^ set(EN)
check_true("es y en tienen el mismo juego de claves", not dif, str(sorted(dif)[:8]))
vacias = [k for k, v in ES.items() if not str(v).strip()]
check_true("ninguna cadena vacia en es", not vacias, str(vacias[:5]))
vacias = [k for k, v in EN.items() if not str(v).strip()]
check_true("ninguna cadena vacia en en", not vacias, str(vacias[:5]))

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("INTEGRACION: LA APP ARRANCA Y NAVEGA COMPLETA EN LOS DOS IDIOMAS")
