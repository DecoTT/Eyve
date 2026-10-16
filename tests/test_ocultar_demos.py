# -*- coding: utf-8 -*-
"""
Esconder las pantallas de demo desde Settings.

Se hicieron para el stand. En una máquina de producción sobran en el menú,
y la textil además deja un hilo generando tela mientras siga instanciada.
Por eso esconderlas no es sólo quitarlas de la lista: tiene que
DESMONTARLAS, o no se libera nada y el ajuste engaña.

Dos cosas que, mal hechas, no se notan:

  * esconder la pantalla en la que estás te deja en un sitio sin entrada en
    el menú — justo lo que el modo kiosco vino a evitar;
  * esconderlas sin desmontarlas deja el hilo de la tela corriendo para
    siempre, y el ajuste parece funcionar.

Cada caso bueno va con el malo: antes de exigir que desaparezcan se exige
que estuvieran, y antes de exigir que el hilo muera se exige que estuviera
vivo.
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("YOLO_OFFLINE", "1")

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


# ── neutralizar cámara, modelo y red ──────────────────────────────────────
from eyve.inference import detector as _det
_det.VideoSource.start = lambda self: False
_det.VideoSource.stop = lambda self: None
_det.VideoSource.read = lambda self: None

from eyve.ui.screens import production_screen as _ps
_ps.ProductionScreen._try_autoload_model = lambda self: None
_ps.ProductionScreen._auto_start_preview = lambda self: None
from eyve.ui.screens import demo_screen as _ds
_ds.DemoScreen._load_model = lambda self: None
_ds.DemoScreen._paint = lambda self, canvas, frame, attr: (960, 540, 960, 540, 0, 0)
from eyve.ui.screens import counting_demo_screen as _cds
_cds.CountingDemoScreen._pintar = lambda self, frame: None

from eyve.core import updater as _U
_U.check_async = lambda cb: None

from eyve.core import config
from eyve.ui.app import EyveApp
from eyve.ui.components.sidebar import DEMO_ITEMS, NAV_ITEMS, visible_nav_items

# El valor de FABRICA, antes de sustituir nada. Se mira en _DEFAULTS y no
# con config.get: el config del usuario puede traer la clave puesta, y lo
# que importa es con que sale Eyve de serie. Sin esto, todo lo de abajo
# pasaria igual aunque viniera escondido por defecto.
check("de fabrica las demos NO vienen escondidas",
      config._DEFAULTS.get("hide_demos") is False,
      repr(config._DEFAULTS.get("hide_demos")))

# El ajuste se simula tocando config.get: config.set escribe en el config
# de verdad del usuario, y una prueba no debe dejar rastro.
_real_get = config.get
_escondidas = {"v": False}
config.get = lambda k, d=None: (_escondidas["v"] if k == "hide_demos"
                                else _real_get(k, d))


def settle(app, n=4):
    for _ in range(n):
        app.update()
        app.update_idletasks()


print("\n[1] Por defecto las demos están")
check("las dos están en NAV_ITEMS",
      set(DEMO_ITEMS) <= {k for k, _ in NAV_ITEMS}, str(DEMO_ITEMS))
check("y visible_nav_items las enseña",
      set(DEMO_ITEMS) <= {k for k, _ in visible_nav_items()})
check("son 8 entradas", len(visible_nav_items()) == 8,
      str(len(visible_nav_items())))

print("\n[2] Con el ajuste puesto, desaparecen de la lista")
_escondidas["v"] = True
check("ya no están",
      not (set(DEMO_ITEMS) & {k for k, _ in visible_nav_items()}),
      str([k for k, _ in visible_nav_items()]))
check("quedan 6", len(visible_nav_items()) == 6, str(len(visible_nav_items())))
check("y no se llevó por delante ninguna otra",
      {k for k, _ in visible_nav_items()}
      == {k for k, _ in NAV_ITEMS} - set(DEMO_ITEMS))
_escondidas["v"] = False

print("\n[3] En la app: el menú, la pantalla visible y el hilo de la tela")
app = EyveApp()
app.geometry("1200x800+60+60")
app.navigate("nav_demo")
settle(app)

demo = app._screens.get("nav_demo")
check("PREVIO: la demo textil está montada", demo is not None)
check("PREVIO: es la pantalla visible", app._current == "nav_demo",
      str(app._current))
check("PREVIO: hay botón de Demo en el menú",
      "nav_demo" in app.sidebar._buttons)
hilo = getattr(demo._source, "_thread", None)
check("PREVIO: el hilo de la tela está vivo",
      hilo is not None and hilo.is_alive())

_escondidas["v"] = True
app.refresh_nav()
settle(app)

check("el botón desaparece del menú",
      "nav_demo" not in app.sidebar._buttons and
      "nav_cdemo" not in app.sidebar._buttons,
      str(sorted(app.sidebar._buttons)))
check("salimos de la demo", app._current == "nav_home", str(app._current))
check("la pantalla se desmonta", "nav_demo" not in app._screens,
      str(sorted(app._screens)))
# ── lo que de verdad libera recursos ────────────────────────────────────
for _ in range(20):
    if not hilo.is_alive():
        break
    time.sleep(0.05)
    settle(app, 1)
check("y el hilo de la tela se para",
      not hilo.is_alive(),
      "si siguiera vivo, esconderlas no liberaría nada")

print("\n[4] Volver a enseñarlas las devuelve")
_escondidas["v"] = False
app.refresh_nav()
settle(app)
check("los botones vuelven",
      "nav_demo" in app.sidebar._buttons and "nav_cdemo" in app.sidebar._buttons)
check("y se puede volver a entrar", app.navigate("nav_demo") is None
      and app._current == "nav_demo", str(app._current))
settle(app)

print("\n[5] Escondidas estando en kiosco: también sale del kiosco")
app.enter_kiosk()
settle(app)
check("PREVIO: está en kiosco", app.kiosk)
_escondidas["v"] = True
app.refresh_nav()
settle(app)
check("sale del kiosco", not app.kiosk)
check("y la barra lateral vuelve a verse", app.sidebar.winfo_ismapped())
check("en Inicio", app._current == "nav_home", str(app._current))

print("\n[6] Estando en otra pantalla, esconderlas no te mueve")
_escondidas["v"] = False
app.refresh_nav()
app.navigate("nav_home")
settle(app)
app.navigate("nav_classes")
settle(app)
check("PREVIO: estamos en Categorías", app._current == "nav_classes",
      str(app._current))
_escondidas["v"] = True
app.refresh_nav()
settle(app)
check("seguimos en Categorías", app._current == "nav_classes",
      str(app._current))
check("y sigue marcada en el menú",
      app.sidebar._active == "nav_classes", str(app.sidebar._active))

print("\n[7] El interruptor de Settings escribe y rehace el menú")
# Que la casilla exista no basta: tiene que escribir la clave QUE LEE el
# menú y pedir que se rehaga. Son los dos sitios donde se corta la cadena.
guardado = {}
llamado = {"refresh": 0}
_real_set = config.set
config.set = lambda k, v: guardado.__setitem__(k, v)
_real_refresh = app.refresh_nav
app.refresh_nav = lambda: llamado.__setitem__("refresh", llamado["refresh"] + 1)
try:
    from eyve.ui.screens.settings_popup import SettingsPopup
    pop = SettingsPopup(app, app=app)
    app.update()
    check("Settings tiene el interruptor", hasattr(pop, "_hide_demos_var"))
    pop._hide_demos_var.set(True)
    pop._toggle_hide_demos()
    check("encenderlo guarda hide_demos=True",
          guardado.get("hide_demos") is True, str(guardado.get("hide_demos")))
    check("y pide rehacer el menú", llamado["refresh"] == 1,
          str(llamado["refresh"]))
    pop._hide_demos_var.set(False)
    pop._toggle_hide_demos()
    check("apagarlo guarda hide_demos=False",
          guardado.get("hide_demos") is False, str(guardado.get("hide_demos")))
    pop.destroy()
    app.update()
except Exception as e:
    import traceback
    traceback.print_exc()
    fails.append(f"Settings: {e}")
finally:
    config.set = _real_set
    app.refresh_nav = _real_refresh

# ── cierre ────────────────────────────────────────────────────────────────
config.get = _real_get
try:
    app._teardown_all_screens()
    app.update()
    app.destroy()
    print("\n  OK    cierre limpio")
except Exception as e:
    fails.append(f"cierre: {e}")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ESCONDER LAS DEMOS: SALEN DEL MENU, SE DESMONTAN Y SUELTAN EL HILO")
