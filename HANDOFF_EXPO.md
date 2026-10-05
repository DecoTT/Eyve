# Traspaso — Eyve 2.1 para stand de expo

Documento de arranque para una sesión nueva. Describe **dónde está el proyecto
hoy** y **las cuatro tareas acordadas**, con lo necesario para hacerlas sin
tener que redescubrir el contexto.

Fecha de corte: 4-oct-2026. Rama de trabajo: `2.1-dev`. Último commit:
`842e802`.

---

## 1. Qué se construyó en la tanda del 2-4 de octubre

Diecisiete commits desde `d1c6c57`. Lo que importa saber:

**Módulo de conteo — 5 métodos** (`eyve/modules/counting_module.py`)
En pantalla · cruce de meta con sentido · zona con ocupación · al aparecer ·
al desaparecer. Rango esperado min/max que veta el frame a NOT OK.
`TrackerConfig` (`eyve/inference/tracker.py`) hace configurables desde la UI
los cuatro parámetros de persistencia de instancia, antes constantes.

**Módulo Patrón — inspección sin clases** (`eyve/modules/pattern_module.py`)
No se entrena ni se etiqueta. Tres métodos: `periodo` (el material que se
repite es su propia referencia), `layout` (distingue tinta de más de tinta
que falta) y `referencia` (aprende de material bueno). **Requiere
calibrarse** con material bueno: no existe umbral fijo que sirva porque el
ruido del material sano cambia con el material.

**Demo textil** (`eyve/demo/`, `eyve/ui/screens/demo_screen.py`)
Tela sintética con 4 ligamentos y 4 estampados que viaja infinitamente. El
visitante dibuja rayones y manchas o coloca fallos de impresión. Los dos
motores corren a la vez y se ve la diferencia: YOLO nombra lo entrenado,
Patrón descubre lo que nunca vio.

**Demo del conteo** (`eyve/demo/conveyor.py`,
`eyve/ui/screens/counting_demo_screen.py`)
Pantalla que recorre los 5 métodos en bucle, cada uno con la escena que lo
explica. Avanza solo cada 14 s.

**Actualizador** (`eyve/core/updater.py`, `eyve/ui/screens/update_dialog.py`)
GitHub Releases + verificación SHA256 + respaldo. Nunca toca `projects/`,
`.venv/` ni `~/.eyve/`.

### Reparto de clases (decisión de arquitectura)

YOLO entrena **sólo** `rayon` y `mancha` — lo puntual, lo nombrable, lo que
se cuenta. Los fallos de impresión (fantasma, offset, falta de tinta) los
cubre el módulo Patrón sin entrenar nada, pero **sí se generan en los frames
de entrenamiento, sin etiqueta**, para que YOLO aprenda que son fondo.
Verificado: YOLO no reclama ninguno de los tres.

### Números medidos (no estimados)

| | |
|---|---|
| Modelo demo | mAP50 **0.981** (900 frames, 40 épocas, 2 clases) |
| Tejido vs YOLO | recall **98.9 %** idéntico en los 5 ligamentos — no le cuesta |
| Patrón, detección | rayón 4/4 · offset 4/4 · mancha 4/4 · falta de tinta 3/4 · fantasma 3/4 |
| Patrón, falsas alarmas | **0** en 64 frames de material bueno, tras calibrar |
| Escenas de conteo | cruce 16/16 · zona 16/16 · desaparecer 16/16 · aparecer 19→17 |

### Límites conocidos, dichos de frente

- **Acumular conteos de anomalías no es exacto.** Un fallo que cruza el
  encuadre se cuenta entre 1 y 4 veces: su región se fragmenta al salir.
  Contar las *visibles ahora* sí es fiable, y la demo arranca ahí. El camino
  no agotado: estabilizar las regiones temporalmente siguiendo el movimiento
  del material, en vez de parchear el conteo aguas abajo.
- **El fantasma se detecta en 2-3 de los 4 estampados.** Para enseñarlo, usar
  flores o puntos.
- El módulo Patrón se apoya en que la mayor parte del encuadre sea material
  sano. Pasado un tercio de área anómala la referencia se degrada.

---

## 2. Las cuatro tareas

### T1 — Modo kiosco (pantalla completa) — HECHO

**Por qué:** en un stand, que un visitante se salga a "Etiquetado" y no sepa
volver es el fallo más probable. Además el dueño comentó que la pantalla
completa "empieza a ser familiar", así que se usará también fuera del stand.

**Qué:** pantalla completa real, sin barra lateral ni barra de estado, para
las dos pantallas de demo (`nav_demo` y `nav_cdemo`).

Criterios:
- Se entra y se sale con una tecla visible y recordable (sugerencia: `F11`
  para entrar/salir, `Esc` para salir). Debe haber una pista en pantalla la
  primera vez, porque quien atiende el stand no leerá documentación.
- Al salir, la ventana vuelve al tamaño y posición que tenía.
- La barra lateral y la de estado se ocultan y se restauran sin dejar huecos
  en el `grid` (ver `eyve/ui/app.py:_build_layout`).
- No debe poderse navegar a otra pantalla con el teclado estando en kiosco.
- Verificarlo **abriendo la app**, no sólo con pruebas.

**Cómo quedó.** `KIOSK_SCREENS` en `eyve/ui/app.py`; `F11` entra y sale,
`Esc` sólo sale, y los dos se enlazan en la raíz con `add=True` para no
pisar los atajos de las pantallas. Al entrar sale un aviso centrado que se
desvanece a los 6 s, y el rótulo de la esquina de las dos demos cambia de
"F11 pantalla completa" a "pulsa F11 o Esc para salir" (hook `on_kiosk`).

41 comprobaciones en `tests/test_kiosco.py` y 12 mutaciones deliberadas,
todas cazadas. Dos cosas que conviene no repetir:

- La comprobación de "al salir vuelve a su geometría" **pasaba igual con
  la restauración desactivada**: Tk ya devuelve la ventana solo al quitar
  `-fullscreen`. Sólo distingue si algo toca la geometría *durante* el
  kiosco; ahí Tk sale a la geometría nueva y hace falta restaurar a mano.
- `Esc` quedó **sin verificar a mano, y no porque falle**: la
  automatización de escritorio no entrega Escape a ninguna ventana
  (probado con una ventana Tk mínima: `F11` y la letra `a` llegan, Escape
  no llega ni al cazatodo `<Key>`). `F11` sí se comprobó a ojo. **Hay que
  pulsar Esc con el teclado de verdad.**

Pendiente relacionado, pre-existente: `_on_close` guarda `winfo_width()`
estando maximizada, así que el config acaba con `window_width: 3440` y la
ventana se sale del monitor por la derecha, escondiendo el rótulo de la
esquina. El kiosco esquiva el problema (guarda lo de antes de entrar) pero
el caso normal sigue ahí.

### T2 — Vuelta sola a modo automático — HECHO

**Por qué:** si el último visitante deja la tela llena de garabatos, el
siguiente se encuentra un desastre y la demo pierde fuerza.

**Qué:** tras N minutos sin tocar (sugerencia: 3), la pantalla de demo
textil vuelve al modo automático y limpia la tela.

Criterios:
- El temporizador se reinicia con cualquier interacción del visitante
  (dibujar, botones de fallo, limpiar, pausar, cambiar material).
- No debe reiniciarse solo por el paso del tiempo ni por el modo automático
  colocando defectos: sólo cuenta lo que hace una persona.
- La vuelta debe ser visible (un aviso breve), no un salto brusco.
- La demo de conteo (`nav_cdemo`) ya vuelve sola por su propio bucle; revisar
  si necesita algo equivalente cuando alguien la deja detenida en un método.

**Cómo quedó.** `_INACTIVIDAD_S = 180.0` en las dos pantallas de demo. El
reloj vive en `_last_touch` y **sólo lo mueve `_touch()`**, que se llama
desde los manejadores de la interfaz y de ningún otro sitio — en
particular nunca desde `_auto_tick`. Los dos sitios donde eso se podía
colar llevan envoltorio: el botón "Limpiar tela" llama a
`_on_clear_click()` (porque `reset_demo()` la usa también la vuelta
automática) y, en la de conteo, los botones ‹ › y Auto llaman a
`_on_saltar()` y `_on_auto_click()` (porque `_saltar()` la usa el bucle al
avanzar solo). La vuelta limpia la tela, pone los contadores a cero,
enciende el automático y lo avisa 6 s en la cabecera.

**Sí hacía falta en la de conteo:** su bucle avanza solo, pero el botón de
Auto lo para y entonces se queda en ese método para siempre.

No recalibra el módulo Patrón, y es a propósito: `calibrate()` sólo se
llama explícitamente, así que los garabatos del visitante no contaminan la
calibración. El material no cambia, la referencia sigue valiendo. (T4 sí
tiene que recalibrar, porque allí se puede haber cambiado de tela.)

36 comprobaciones en `tests/test_inactividad.py` y 13 mutaciones, todas
cazadas. La que encontró un agujero real: la prueba llamaba a los
*métodos*, no a los *botones*, así que recablear un botón al callback
equivocado pasaba desapercibido. Ahora los pulsa de verdad con
`.invoke()`.

### T3 — Publicar 2.1.1

**Por qué:** hay 20 commits sin publicar y `__version__` sigue en `2.1.0`,
así que el actualizador **no tiene a qué actualizar** — hoy diría "ya tienes
la última versión". Publicar lo vuelve demostrable, que es argumento de
venta.

**Qué:** seguir `RELEASE.md`, que ya documenta el procedimiento y los nombres
fijos. Resumen:

1. `eyve/__init__.py` → `__version__ = "2.1.1"`.
2. Actualizar `RELEASE.md` con la sección de la versión nueva.
3. `cd Release && .\build_release.ps1` (valida los `.bat` y la estructura).
4. Tag `v2.1.1` y `gh release create --latest` con
   `Eyve-2.1-setup.zip` + `SHA256SUMS.txt`. **El nombre del asset no cambia**
   o se rompe la URL fija de la tienda.
5. Probar el actualizador de verdad: instalar 2.1.0 en una carpeta limpia,
   abrirlo, y comprobar que detecta 2.1.1, la instala y conserva `projects/`.

**Confirmar con el dueño antes de publicar.** Es una acción pública e
irreversible; el repo es `DecoTT/Eyve` y la tienda apunta a
`releases/latest/download/`.

### T4 — Botón grande de "empezar de cero" — HECHO

**Por qué:** quien atiende el stand necesita resetear sin buscar entre
controles pequeños.

**Qué:** un botón prominente en la demo textil que deje todo como recién
abierto: tela limpia, contadores en cero, tracker reiniciado, modo
automático encendido y el módulo Patrón recalibrado.

Criterios:
- Recalibrar es imprescindible: el piso de ruido depende del material, y sin
  recalibrar tras cambiar de tela el módulo o grita o se queda callado.
- Debe funcionar estando en cualquier estado (pausado, dibujando, con el
  automático apagado).

**Cómo quedó.** `empezar_de_cero()` en `DemoScreen`, con el botón en su
propia fila, a todo el ancho del panel derecho y de 54 px de alto frente a
los 36 de los demás controles. Hace, por este orden: olvida el trazo a
medias y el último punto, quita la pausa, limpia tela / tracker /
contadores, devuelve el método de conteo al inicial, enciende el
automático y **recalibra** el módulo Patrón — recalibrar va al final,
porque se calibra con material bueno y para eso la tela ya tiene que estar
limpia.

Lo que no era obvio y por eso está comentado en el código:

- Si se pulsa **a mitad de un trazo**, hay que olvidar `_last_pt`, o el
  siguiente movimiento del ratón pinta una raya desde donde estaba la mano
  antes del reinicio. Y hay que olvidar `_pausa_previa`, o al soltar el
  ratón la tela se vuelve a pausar sola.
- El método de conteo vuelve al inicial (`_METODO_INICIAL = "screen"`).
  **Esto no estaba en la lista de criterios**: se añadió porque "como
  recién abierto" incluye el panel de conteo. El material (estampado,
  tejido) y la velocidad **no** se tocan, porque cambiarlos obliga a una
  recalibración más larga y son una elección deliberada de quien atiende.
- `_MARGEN_TRAS_REINICIO = 6.0`. Salió de abrir la app: con los 2 s que
  usa el botón de Modo auto, el primer defecto automático caía antes de
  que diera tiempo a enseñar la tela limpia, y el "de cero" no se veía.

36 comprobaciones en `tests/test_empezar_de_cero.py`, todas con su
condición previa afirmada (si la tela no estuviera sucia antes, "queda
limpia" no probaría nada), y 12 mutaciones, todas cazadas. Dos cosas que
salieron de las mutaciones:

- La llamada a `end_stroke()` que había puesto era **código muerto**:
  `reset_demo()` → `clear_defects()` ya deja el trazo en `None`. Quitarla
  no cambiaba ninguna comprobación, así que se quitó.
- La comprobación de "olvida que la tela estaba pausada" **pasaba sola**,
  porque en la prueba se empezaba a dibujar con la tela corriendo y
  entonces `_pausa_previa` ya valía `False`. Ahora la prueba pausa la tela
  antes de dibujar, que es el caso que de verdad distingue.

Una prueba que se cuelga tampoco informa: la mutación "el recalibrado
nunca termina" dejaba colgado el bucle que espera a la calibración, así
que ese bucle lleva tope y ahora falla limpio.

---

## 3. Cómo trabajar en este repo

### Pruebas

```bash
cd "D:\Desarrollo\Claude Code\Eyve\2.1-dev"
.venv\Scripts\python.exe tests\run_all.py            # 17 suites, ~12 min
.venv\Scripts\python.exe tests\run_all.py --rapido   # salta la que carga YOLO
```

Cada suite corre en su **propio proceso** a propósito: las que abren Tk no
pueden compartir intérprete (crear un segundo root tras destruir el primero
deja las `PhotoImage` apuntando al intérprete muerto).

### El criterio de las pruebas en este proyecto

**Una comprobación que pasaría igual con los datos mal no verifica nada.**
Por cada caso bueno hay que escribir el caso malo. Esto no es teoría: en esta
tanda, una prueba de ida y vuelta de coordenadas pasaba perfecta mientras las
etiquetas salían giradas el doble del ángulo, porque componer `+a` y `−a` se
cancela con cualquier signo.

Relacionado: **no confiar en una sola corrida**. Varias conclusiones de esta
tanda resultaron falsas por medir con una semilla. Donde haya aleatoriedad,
medir con 8-12 semillas y reportar la distribución.

### Verificar la interfaz

Las pruebas no bastan: en esta tanda, **abrir la aplicación encontró tres
bugs que 400 comprobaciones automáticas no vieron**. Para abrirla y mirarla:

```powershell
# lanzar
cd "D:\Desarrollo\Claude Code\Eyve\2.1-dev"; Start-Process run.bat

# maximizar desde PowerShell (el botón de maximizar es fácil de errar)
$p = Get-Process python* | Where-Object { $_.MainWindowTitle -eq 'Eyve 2.1' }
# ShowWindow($p.MainWindowHandle, 3) vía Add-Type
```

Para capturar y hacer clic hace falta `request_access` de computer-use con
**`"Python 3.12 (64-bit)"`** — no resuelve por "Eyve" ni por "Python".

### Trampas de la herramienta

- Los **heredocs de bash se truncan** alrededor de las 130 líneas y rompen el
  comando. Para scripts de parche largos: escribir el `.py` con la
  herramienta Write y ejecutarlo.
- Los scripts de parche deben **afirmar** (`assert s.count(old) == 1`) antes
  de sustituir, y escribir con `newline="\n"`.
- `PYTHONIOENCODING=utf-8` en PowerShell o los acentos revientan la salida.

### El proyecto demo

`projects/Demo_Textil/` está en `.gitignore` (274 MB). Para regenerarlo:

```bash
.venv\Scripts\python.exe -m eyve.demo.train --out "projects\Demo_Textil" --frames 900 --epochs 40 --imgsz 512
```

~45 min en CPU. El venv trae **torch CPU a propósito** (es lo que recibe el
usuario final). La máquina tiene una RTX 3080 usable con torch CUDA instalado
aparte. Para llevar la demo a otra máquina basta copiar
`projects/Demo_Textil/models/best.pt`.

### Idiomas

Todo texto visible va en `eyve/i18n/es.py` y `en.py`, **siempre en los dos**.
`tests/test_app_integration.py` falla si los juegos de claves difieren.

---

## 4. Maratón de stand: resultado

Se simularon **30 minutos de stand desatendido** con el modo automático
poniendo defectos. `tests/maraton_stand.py` quedó en el repo, así que se
puede volver a correr con más minutos cuando se quiera.

**No hay fuga ni degradación:**

| | inicio | final |
|---|---|---|
| memoria | 80.7 MB | 75.5 MB |
| ms/frame | 42.5 | 41.9 |
| tracks vivos | 0 | 2 |
| defectos en la tela | 1 | 4 |

Los tracks y los defectos quedan acotados, y la limpieza automática de la
tela a los 4 defectos funciona. Esto deja de ser un riesgo.

### T5 — La demo iba a 24 fps, y no era por la inspección — HECHO

42 ms por frame, **sin YOLO**. Perfilado:

| parte | ms |
|---|---|
| **generar el frame de tela** | **40.4** |
| Patrón `analyze()` completo | 14.6 |
| tracker + conteo | ~0 |

Y dentro de generar el frame, los 40 ms son **tres pasadas float32 sobre el
frame entero**:

| paso | ms |
|---|---|
| compuesto alfa de la capa de defectos | 17.8 |
| multiplicar por el tejido | 9.1 |
| multiplicar por la viñeta | 9.1 |
| slices, rotación y el resto | 4.5 |

La tela se genera en su propio hilo (`SyntheticSource`), así que no bloquea
la interfaz, pero compite por el GIL con la UI y con la inferencia. En la
máquina del stand la demo se verá a ~20 fps en vez de 30. No es fatal, pero
una demo que se arrastra se nota.

Tres arreglos, de mayor a menor beneficio y ninguno arriesgado:

1. **Componer sólo donde hay defectos** (ahorra la mayor parte de los
   17.8 ms). `_alpha` es casi todo ceros: basta recortar al rectángulo que
   contiene los píxeles no nulos y componer ahí. Ya existe el atajo
   `if alp.any()` para el caso sin defectos; falta el caso con pocos.
2. **Fundir tejido y viñeta en un solo multiplicador** (ahorra ~9 ms). Hoy
   son dos pasadas porque el tejido va antes de rotar y la viñeta después;
   con una inclinación de 1.8° aplicar la viñeta antes de rotar es
   visualmente idéntico y permite una sola multiplicación.
3. **Usar `cv2.multiply` en vez de float32 de numpy** para lo que quede:
   está vectorizado con SIMD.

Medir antes y después con el mismo perfilado — y comprobar a ojo que la tela
sigue viéndose igual, porque el punto 2 cambia el orden de dos efectos.

### T5b — Y en kiosco costaba el doble (medido al hacer T1) — HECHO

El repintado de los lienzos no era sospechoso porque en ventana no se
nota. A pantalla completa sí: `tests/perf_kiosco.py`, 40 repeticiones.

| lienzo | repintar uno | los dos por frame |
|---|---|---|
| ventana 1200x800 | 2.8 ms | 5.6 ms |
| kiosco 3440x1440 | **18.9 ms** | **37.7 ms** |

Sumado a los 42 ms de tela e inspección: **~21 fps en ventana, ~12.5 fps
en kiosco**. Y como el stand va a correr en kiosco, el número que importa
es el de abajo.

Dos avisos para quien lo optimice:

- La primera medición dio 1.5 ms planos para los dos tamaños y era falsa:
  la ventana de prueba era de 200x100 y recortaba el lienzo, así que Tk no
  dibujaba lo que no se veía. El script lleva ahora un `assert` que lo
  impide.
- El frame nativo es de 960x540 y en kiosco se amplía a ~1700x950. Ampliar
  no añade detalle, sólo coste; pero recortar el tamaño mostrado encoge la
  demo en la pantalla grande, que es justo lo que el stand no quiere. Es
  un compromiso, no un arreglo obvio.

---

## 5. Lo que se hizo con T5 y T5b

**Decisión del dueño:** el kiosco se queda —la pantalla completa ayuda a
que el visitante se familiarice con Eyve— y lo que se optimiza es el
coste.

### Lo que se cambió

**La tela, 4× más rápida** (`eyve/demo/textile.py`). Dos cosas:

| | |
|---|---|
| componer sólo donde hay defectos | 31.3 → 20.3 ms, imagen **idéntica** |
| `cv2.multiply` en vez de float32 de numpy | 20.3 → **3.5 ms**, diferencia ≤ 2 |

`_alpha` es casi todo ceros, así que componer el encuadre entero era pagar
el 100 % por el 1 %. Y `cv2.multiply` hace conversión, producto y
saturación en una pasada SIMD donde numpy hacía cuatro pasadas y dos
reservas grandes por modulador. El tejido y la viñeta pasan a 3 canales
porque OpenCV no difunde canales: cuesta 17 MB más y ahorra 17 ms.

**El repintado** (`_paint` y `_pintar`). El PhotoImage y el item del canvas
se crean una vez y después se hace `paste()`; sólo se rehacen al cambiar
de tamaño. De los 18.2 ms que costaba repintar un lienzo en kiosco, 14.4
se iban en tirar y rehacer lo que podía reutilizarse.

### Segunda ronda, porque el stand corre en una NUC

Con la primera ronda el kiosco subió a 19.9 fps, todavía por debajo de los
30 que pide el bucle. Dos recortes más:

- **Desplazar la tela por rebanadas** en vez de `np.take`. El viaje es un
  desplazamiento circular, o sea una rebanada: mientras la ventana cabe
  sin dar la vuelta (960 px de un rollo de 3840) son vistas y salen
  gratis. `frame()`: 10.7 → 6.2 ms.
- **No ampliar el frame más allá de su tamaño nativo**
  (`T.MAX_ESCALA_DEMO = 1.0`). Repintar un lienzo en kiosco: 13.3 → 5.4 ms.

### El resultado, medido (`tests/perf_kiosco.py`)

| | tela | Patrón | 2 lienzos | TOTAL | fps |
|---|---|---|---|---|---|
| ventana | 6.2 ms | 15.2 ms | 3.6 ms | 25.0 ms | **40.0** |
| kiosco 3440×1440 | 6.2 ms | 15.2 ms | 10.7 ms | 32.1 ms | **31.1** |

De donde se partía: ~21 fps en ventana y **~12.5 en kiosco**. El kiosco
cabe ahora dentro de los 33 ms que pide el bucle, que es el umbral que
importa: por encima, `update()` no termina de drenar la cola de eventos y
la interfaz se arrastra.

**Lo que cuesta el tope de ampliación:** en un monitor muy grande la tela
se ve más pequeña, con margen alrededor. En **1920×1080 no se nota** —
comprobado abriendo la app: el panel mide ~940 px y la tela de 960 lo
llena de lado a lado. El tope sólo recorta donde antes se estiraba.

**Dos avisos para el stand:**

- Estos números son de una máquina de desarrollo (20 núcleos). **En la NUC
  serán peores.** `tests/perf_kiosco.py` corre solo y sin dependencias de
  la interfaz: conviene ejecutarlo en la NUC con la pantalla del stand
  conectada antes de dar la demo por lista.
- El módulo Patrón pasa a ser el mayor coste: 15.2 de los 32.1 ms, casi la
  mitad. Si en la NUC no llega, ahí es donde habría que mirar — y eso ya
  es tocar el producto, no la demo.

Efecto secundario que lo confirma: las suites que generan tela bajaron
solas — `test_conteo_anomalias_fiabilidad` de 311 a 129 s y `test_textile`
de 166 a 125 s.

### Lo que se descartó, y por qué

El tercer arreglo que proponía T5 —fundir tejido y viñeta en un solo
multiplicador aplicando la viñeta antes de rotar— **se midió y no sirve**:
sale más lento (4.4 ms contra 3.5, porque hay que combinar los dos
moduladores cada frame y eso cuesta lo mismo que aplicarlos) y además
cambia la imagen hasta 70 niveles, porque rotar después arrastra las
esquinas oscuras.

También estaba mal el orden de prioridad: T5 ponía el recorte del
compuesto como la mayor ganancia y `cv2.multiply` como el remate. Es al
revés — el recorte ahorra 11 ms y `cv2.multiply` otros 17.

### La diferencia de imagen, caracterizada

`tests/test_frame_rapido.py` carga el `textile.py` anterior desde git y
compara píxel a píxel con 4 ligamentos, 4 estampados, 8 semillas de
defectos y una tela saturada. La diferencia es **siempre +0, +1 o +2,
nunca negativa**, y vale 0.87 niveles de media sobre 255 (0.34 % de
brillo, uniforme): es el cambio de truncar a redondear al más cercano, que
es aritmética más correcta. Como es uniforme, el contraste local no cambia
y el módulo Patrón no lo nota.

Exigir que la diferencia **nunca sea negativa** es más severo que una
tolerancia simétrica: un cambio de verdad en la tela movería píxeles en
los dos sentidos.

### Lo que queda sobre la mesa

En kiosco el repintado sigue siendo 26.7 de los 50.4 ms, y la única
palanca que queda es el número de píxeles: el frame nativo es 960×540 y se
amplía a 1696×954, un 3× de píxeles que no añade ni un detalle. Bajar el
tamaño mostrado a ~1.4× daría unos 25 fps, a cambio de una imagen un 21 %
más pequeña con márgenes laterales. **Es una decisión de cómo se ve la
demo, no un arreglo técnico**, y por eso no se tomó sola.

El otro camino sería generar la tela a más resolución nativa (ahora se
puede, cuesta 9.9 ms), pero eso toca el modelo entrenado, la geometría del
conteo y el coste del módulo Patrón, que crece con los píxeles.
