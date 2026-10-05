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

### T1 — Modo kiosco (pantalla completa)

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

### T2 — Vuelta sola a modo automático

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

### T4 — Botón grande de "empezar de cero"

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

## 4. Pendiente de confirmar

Al cerrar este documento quedaba corriendo una **maratón de 8 horas
simuladas** de stand (`scratchpad/maraton.py`) midiendo memoria, tiempo por
frame, tracks vivos y defectos acumulados. Es el riesgo que no se había
tocado: una fuga no se nota en 20 minutos de pruebas y sí a media tarde de
expo. **Revisar su resultado antes de dar la demo por lista**, y si muestra
degradación, eso pasa a ser T0.
