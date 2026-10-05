# -*- coding: utf-8 -*-
"""
Modo kiosco: pantalla completa sin barra lateral ni barra de estado.

Cada comprobacion va con su contraria, porque una prueba que pasaria igual
con el codigo mal no prueba nada.  Los dos pares que mas importan:

  * "al salir la ventana vuelve a donde estaba" pasaria sola si la pantalla
    completa no hubiera ocurrido nunca.  Por eso antes se exige que la
    geometria en kiosco SEA DISTINTA de la de partida.

  * "el contenido llena la ventana" pasaria sola si el contenido llenase la
    ventana tambien fuera del kiosco.  Por eso antes se exige que fuera del
    kiosco sea ESTRICTAMENTE MAS PEQUENO — la barra lateral y la de estado
    ocupan su sitio.
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("YOLO_OFFLINE", "1")

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


# ── neutralizar camara, modelo y red antes de construir la app ────────────
from eyve.inference import detector as _det
_det.VideoSource.start = lambda self: False
_det.VideoSource.stop = lambda self: None
_det.VideoSource.read = lambda self: None

from eyve.ui.screens import production_screen as _ps
_ps.ProductionScreen._try_autoload_model = lambda self: None
from eyve.ui.screens import demo_screen as _ds
_ds.DemoScreen._load_model = lambda self: None

from eyve.core import updater as _U
_U.check_async = lambda cb: None

# ── y el repintado de los lienzos ──────────────────────────────────────────
# Medido en tests/perf_kiosco.py: a pantalla completa en un monitor de
# 3440x1440 repintar un lienzo cuesta 18.9 ms en vez de 2.8 ms, y la demo
# repinta DOS por frame — 37.7 ms contra los 33 ms del bucle.  Con el
# repintado puesto, update() nunca termina de drenar la cola (siempre hay
# un temporizador vencido) y la prueba se queda girando para siempre.
# Aqui se mide el LAYOUT del kiosco, que no depende de los pixeles; el
# coste del repintado tiene su propia medicion y se comprueba abriendo la
# app.
_ds.DemoScreen._paint = lambda self, canvas, frame, attr: (960, 540, 960, 540, 0, 0)
from eyve.ui.screens import counting_demo_screen as _cds
_cds.CountingDemoScreen._pintar = lambda self, frame: None

from eyve.core import config
from eyve.i18n import t
from eyve.ui import theme as T
from eyve.ui.app import EyveApp, KIOSK_SCREENS


def settle(app, veces=6):
    """Tk aplica los cambios de ventana en diferido; hay que dejarle vueltas."""
    for _ in range(veces):
        app.update()
        app.update_idletasks()


app = EyveApp()
# Geometria conocida: el config del usuario puede traer una ventana casi tan
# grande como el monitor, y entonces "pantalla completa cambia algo" seria
# una diferencia de dos pixeles.
app.geometry("1200x800+80+40")
settle(app)

print("\n[1] El kiosco solo existe en las dos pantallas de demo")
check("KIOSK_SCREENS son nav_demo y nav_cdemo",
      tuple(KIOSK_SCREENS) == ("nav_demo", "nav_cdemo"), str(KIOSK_SCREENS))

# ── caso malo: en Inicio no debe entrar ───────────────────────────────────
app.navigate("nav_home")
settle(app)
check("en Inicio kiosk_available() es False", not app.kiosk_available())
app.enter_kiosk()
settle(app)
check("en Inicio enter_kiosk() no hace nada", not app.kiosk)
check("en Inicio la barra lateral sigue visible", app.sidebar.winfo_ismapped())
app.event_generate("<F11>")
settle(app)
check("en Inicio F11 no entra a kiosco", not app.kiosk)
# Esc fuera de kiosco no debe reventar ni cambiar nada
app.event_generate("<Escape>")
settle(app)
check("en Inicio Esc no hace nada", not app.kiosk and app.winfo_exists())

print("\n[2] Geometria antes, durante y despues")
app.navigate("nav_demo")
settle(app)
check("en la demo kiosk_available() es True", app.kiosk_available())

geom_antes = app.geometry()
# tamanos del contenido FUERA del kiosco: el caso malo de "no deja huecos"
w_root_antes, h_root_antes = app.winfo_width(), app.winfo_height()
w_cont_antes = app.content.winfo_width()
h_cont_antes = app.content.winfo_height()
check("fuera del kiosco el contenido es mas estrecho que la ventana",
      w_cont_antes < w_root_antes, f"{w_cont_antes} < {w_root_antes}")
check("fuera del kiosco el contenido es mas bajo que la ventana",
      h_cont_antes < h_root_antes, f"{h_cont_antes} < {h_root_antes}")

app.enter_kiosk()
settle(app)
check("entra a kiosco", app.kiosk)
check("el atributo -fullscreen esta puesto",
      bool(app.attributes("-fullscreen")))
geom_kiosco = app.geometry()
# ── la guarda: sin esto, lo de abajo pasaria aunque nada hubiera pasado ──
check("la geometria en kiosco es DISTINTA de la de partida",
      geom_kiosco != geom_antes, f"{geom_antes} -> {geom_kiosco}")
check("en kiosco la ventana ocupa el ancho del monitor",
      app.winfo_width() >= app.winfo_screenwidth(),
      f"{app.winfo_width()} >= {app.winfo_screenwidth()}")

print("\n[3] La barra lateral y la de estado se quitan sin dejar huecos")
check("la barra lateral esta oculta", not app.sidebar.winfo_ismapped())
check("la barra de estado esta oculta", not app.status_bar.winfo_ismapped())
check("el contenido llena el ancho de la ventana",
      app.content.winfo_width() == app.winfo_width(),
      f"{app.content.winfo_width()} == {app.winfo_width()}")
check("el contenido llena el alto de la ventana",
      app.content.winfo_height() == app.winfo_height(),
      f"{app.content.winfo_height()} == {app.winfo_height()}")
check("la pantalla de demo sigue mostrandose",
      app._screens["nav_demo"].winfo_ismapped())

print("\n[4] La pista de como salir esta en pantalla")
check("hay aviso de salida", app._kiosk_toast is not None
      and app._kiosk_toast.winfo_exists())
check("el aviso va con place(), no con grid()",
      app._kiosk_toast in app.place_slaves()
      and app._kiosk_toast not in app.grid_slaves())
check("el texto del aviso nombra F11 y Esc",
      "F11" in t("kiosk_exit_hint") and "Esc" in t("kiosk_exit_hint"),
      t("kiosk_exit_hint"))
check("la pista de entrada nombra F11", "F11" in t("kiosk_enter_hint"),
      t("kiosk_enter_hint"))

print("\n[4b] La pista de la esquina dice lo que toca, y se lee")
# Salio de abrir la app: el rotulo estaba en TEXT_DIM y a un metro de la
# pantalla no se leia, y el aviso grande se desvanece a los 6 s — quien
# llega al stand mas tarde se queda sin ninguna pista de como salir.
demo = app._screens["nav_demo"]
check("los dos textos de la pista son distintos",
      t("kiosk_enter_hint") != t("kiosk_exit_hint"))
check("en kiosco la pista dice como SALIR",
      demo._kiosk_lbl.cget("text") == t("kiosk_exit_hint"),
      demo._kiosk_lbl.cget("text"))
check("la pista no esta en el gris mas oscuro",
      demo._kiosk_lbl.cget("text_color") == T.TEXT_SEC
      and T.TEXT_SEC != T.TEXT_DIM,
      f"{demo._kiosk_lbl.cget('text_color')} (DIM es {T.TEXT_DIM})")

print("\n[5] En kiosco el teclado no saca de la demo")
app.navigate("nav_home")
settle(app)
check("navigate('nav_home') en kiosco no cambia de pantalla",
      app._current == "nav_demo")
check("la demo sigue visible", app._screens["nav_demo"].winfo_ismapped())
check("Inicio NO se muestra",
      "nav_home" not in app._screens
      or not app._screens["nav_home"].winfo_ismapped())
check("sigue en kiosco", app.kiosk)
# force=True si debe pasar: es la puerta del cambio de idioma y de tema
app._navigate("nav_demo", force=True)
settle(app)
check("force=True si navega", app._current == "nav_demo")

print("\n[6] Al cerrar en kiosco no se guarda el tamano del monitor")
guardado = {}
_real_set, _real_destroy = config.set, app.destroy
config.set = lambda k, v: guardado.__setitem__(k, v)
app.destroy = lambda: None
try:
    w_kiosco = app.winfo_width()
    w_esperado = int(app._kiosk_geom.split("+")[0].split("x")[0])
    # la guarda otra vez: si el monitor midiera lo mismo que la ventana
    # previa, esta comprobacion no distinguiria nada
    check("el ancho en kiosco difiere del ancho previo",
          w_kiosco != w_esperado, f"{w_kiosco} != {w_esperado}")
    app._on_close()
    check("se guarda el ancho de ANTES del kiosco",
          guardado.get("window_width") == w_esperado,
          f"{guardado.get('window_width')} == {w_esperado}")
    check("no se guarda el ancho del monitor",
          guardado.get("window_width") != w_kiosco)
finally:
    config.set = _real_set

print("\n[7] Al salir, la ventana vuelve a donde estaba")
app.exit_kiosk()
settle(app)
check("sale del kiosco", not app.kiosk)
check("el atributo -fullscreen esta quitado",
      not bool(app.attributes("-fullscreen")))
check("la geometria es la de partida", app.geometry() == geom_antes,
      f"{geom_antes} -> {app.geometry()}")
check("la barra lateral vuelve", app.sidebar.winfo_ismapped())
check("la barra de estado vuelve", app.status_bar.winfo_ismapped())
# grid_info() devuelve {} si el widget no esta en el grid: con .get() la
# prueba informa una falla limpia en vez de reventar con KeyError.
info_s = app.sidebar.grid_info()
info_b = app.status_bar.grid_info()
check("la barra lateral vuelve a su celda",
      (info_s.get("row"), info_s.get("column")) == (0, 0), str(info_s))
check("la barra de estado vuelve a su celda",
      (info_b.get("row"), info_b.get("column"), info_b.get("columnspan"))
      == (1, 0, 2), str(info_b))
check("el contenido vuelve a dejarles sitio",
      app.content.winfo_width() == w_cont_antes
      and app.content.winfo_height() == h_cont_antes,
      f"{app.content.winfo_width()}x{app.content.winfo_height()} "
      f"== {w_cont_antes}x{h_cont_antes}")
check("el aviso de salida desaparecio", app._kiosk_toast is None)
check("fuera del kiosco la pista vuelve a decir como ENTRAR",
      demo._kiosk_lbl.cget("text") == t("kiosk_enter_hint"),
      demo._kiosk_lbl.cget("text"))

print("\n[7b] La geometria se devuelve a mano, no por cortesia de Tk")
# Medido: al quitar -fullscreen, Tk ya devuelve la ventana a su sitio, asi
# que la comprobacion de [7] pasaria igual sin restaurar nada.  Lo que SI
# distingue es que algo haya tocado la geometria durante el kiosco: ahi Tk
# sale a la geometria nueva (900x600) y solo la restauracion explicita
# devuelve la de partida.
app.navigate("nav_demo")
settle(app)
app.enter_kiosk()
settle(app)
app.geometry("900x600+10+10")
settle(app)
check("en kiosco la ventana sigue a pantalla completa",
      app.winfo_width() >= app.winfo_screenwidth(), app.geometry())
app.exit_kiosk()
settle(app)
check("vuelve a la geometria de partida, no a la que se colo",
      app.geometry() == geom_antes,
      f"esperado {geom_antes}, salio {app.geometry()}")

print("\n[8] Fuera del kiosco todo vuelve a funcionar (el caso malo del [5])")
app.navigate("nav_home")
settle(app)
check("fuera del kiosco si se navega a Inicio", app._current == "nav_home")
check("Inicio se muestra", app._screens["nav_home"].winfo_ismapped())

print("\n[9] F11 y Esc por evento de teclado")
# Hasta donde llega esta prueba, dicho de frente: event_generate entrega el
# evento directamente a los bindings de la raiz, asi que comprueba que el
# binding ESTA y que hace lo suyo — no que una pulsacion de verdad llegue.
# F11 si se comprobo a mano en la app abierta, entrando y saliendo.  El Esc
# real quedo sin comprobar: la automatizacion de escritorio no entrega
# Escape a NINGUNA ventana (probado con una ventana Tk minima: F11 y la
# letra 'a' llegan, Escape no llega ni al cazatodo <Key>).  Hay que
# pulsarlo con el teclado de verdad.
app.navigate("nav_cdemo")
settle(app)
app.event_generate("<F11>")
settle(app)
check("F11 entra a kiosco en la demo de conteo", app.kiosk)
app.event_generate("<F11>")
settle(app)
check("F11 otra vez sale", not app.kiosk)
app.event_generate("<F11>")
settle(app)
check("F11 vuelve a entrar", app.kiosk)
app.event_generate("<Escape>")
settle(app)
check("Esc sale", not app.kiosk)
check("la geometria vuelve tambien por Esc", app.geometry() == geom_antes,
      f"{geom_antes} -> {app.geometry()}")

print("\n[10] Cambiar de idioma apaga el kiosco (si no, ventana sin salida)")
app.navigate("nav_demo")
settle(app)
app.enter_kiosk()
settle(app)
check("en kiosco antes de cambiar idioma", app.kiosk)
config.set = lambda k, v: guardado.__setitem__(k, v)
try:
    from eyve.i18n import get_language
    app.switch_language(get_language())
    settle(app)
finally:
    config.set = _real_set
check("cambiar idioma saca del kiosco", not app.kiosk)
check("la barra lateral esta de vuelta", app.sidebar.winfo_ismapped())
check("y aterriza en Inicio", app._current == "nav_home")

# ── cierre ────────────────────────────────────────────────────────────────
app.destroy = _real_destroy
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
print("KIOSCO: ENTRA, SALE, NO DEJA HUECOS Y NO SE ESCAPA DE LA DEMO")
