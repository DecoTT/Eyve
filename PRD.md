# Eyve 2.1 — Product Requirements Document

**Estado:** En desarrollo para release público
**Versión objetivo:** Eyve 2.1
**Fecha de revisión:** 2026-08-10
**Owner:** SBC Group México
**Licencia:** AGPL-3.0

> Este documento sustituye al PRD interno previo. Es la referencia única de alcance,
> prioridades y criterios de liberación de Eyve 2.1.

---

## 1. Objetivo de Eyve 2.1

Eyve 2.1 es el **release público de Eyve**.

El objetivo de esta versión es que cualquier usuario interesado en inspección visual con IA pueda:

1. Descargar Eyve.
2. Descomprimirlo en una PC con Windows.
3. Ejecutar el instalador/configurador.
4. Conectar una cámara o utilizar un video.
5. Crear un proyecto.
6. Definir qué quiere detectar.
7. Capturar ejemplos.
8. Etiquetarlos.
9. Entrenar un modelo localmente.
10. Utilizar ese modelo inmediatamente en modo Producción.

El criterio principal de Eyve 2.1 no es competir por tener el modelo más avanzado ni cubrir todavía todas las necesidades de una instalación industrial.

El objetivo es mucho más concreto:

> **Que Eyve pueda descargarse, instalarse y utilizarse de principio a fin por una persona externa a SBC Group, sin requerir conocimientos de machine learning ni intervención del desarrollador.**

Eyve 2.1 debe convertir el flujo:

```text
Captura → Etiquetado → Entrenamiento → Producción
```

en una experiencia funcional, entendible y repetible.

---

## 2. Visión del producto

Eyve es una aplicación local de inspección visual con inteligencia artificial orientada a manufactura.

Todo el ciclo ocurre en la computadora del usuario:

- captura de imágenes;
- etiquetado;
- construcción del dataset;
- entrenamiento;
- validación;
- inferencia;
- inspección en producción;
- registro de resultados.

No es necesario enviar imágenes a servicios externos para utilizar la aplicación.

La propuesta de Eyve no es desarrollar una nueva arquitectura de inteligencia artificial.

Eyve funciona como una **capa de aplicación o distribución** que integra tecnologías de visión artificial existentes en una herramienta que pueda utilizar un ingeniero de proceso, técnico, estudiante, integrador o persona interesada en experimentar con inspección visual.

---

## 3. Objetivo específico del release

Eyve 2.1 debe ser suficientemente estable para que una persona que nunca ha utilizado Eyve pueda completar por sí sola su **primera inspección entrenada**.

La experiencia ideal es:

```text
Descargar
   ↓
Descomprimir
   ↓
Ejecutar setup.bat
   ↓
Ejecutar run.bat
   ↓
Crear proyecto
   ↓
Crear clases
   ↓
Capturar imágenes / grabar video
   ↓
Etiquetar / Tag Live
   ↓
Entrenar
   ↓
Abrir Producción
   ↓
Ver su propio modelo funcionando
```

El éxito de Eyve 2.1 se medirá principalmente por qué tan sencillo y confiable sea completar este flujo.

---

## 4. Usuarios objetivo de Eyve 2.1

Eyve 2.1 no debe diseñarse únicamente para especialistas en IA.

| Usuario | Qué quiere hacer |
|---|---|
| Ingeniero de proceso | Probar si IA puede resolver una inspección específica |
| Ingeniero de calidad | Evaluar detección de condiciones OK/NOK |
| Integrador | Experimentar con aplicaciones de visión |
| Técnico | Crear una inspección sencilla con cámara |
| Estudiante | Aprender el flujo completo de visión artificial |
| Desarrollador | Probar, modificar o extender Eyve |
| PyME manufacturera | Evaluar una aplicación antes de invertir en una solución industrial |

La interfaz debe asumir que el usuario **no conoce YOLO, epochs, datasets, bounding boxes, ONNX ni pipelines de inferencia**.

Cuando sea posible, Eyve debe traducir esos conceptos a acciones comprensibles.

---

## 5. Gestión de proyectos

El usuario debe poder:

- crear un proyecto;
- darle un nombre;
- abrir proyectos existentes;
- visualizar proyectos recientes;
- guardar automáticamente la configuración;
- cerrar Eyve y continuar posteriormente;
- eliminar proyectos.

Cada proyecto conserva de forma independiente: clases, imágenes, videos, etiquetas,
dataset, modelos, configuración de cámara, configuración de producción, sesiones,
screenshots y logs.

### 5.1 Ubicación de los proyectos

**Decisión:** los proyectos viven **fuera de la carpeta de la aplicación**.

Default propuesto: `Documentos/Eyve Projects/`

Razón: permite actualizar Eyve borrando y reemplazando la carpeta de la app sin tocar
el trabajo del usuario (ver §17). Hoy `home_screen.py:252` abre `askdirectory()` sin
`initialdir`, así que el proyecto cae donde Windows decida — hay que fijar el default.

**Criterio de aceptación:** crear un proyecto sin escribir una ruta manualmente deja el
proyecto fuera de la carpeta de instalación.

---

## 6. Clases de inspección

El usuario debe poder definir las condiciones que quiere reconocer.

Cada clase incluye: **nombre**, **tipo** y **color**.

```text
OK
NOK
IGNORE
```

Ejemplos:

```text
componente_correcto → OK
polaridad_invertida → NOK
rayon               → NOK
pocket_vacio        → NOK
fondo               → IGNORE
```

Las clases deben poder crearse, editarse, eliminarse y mantenerse asociadas
correctamente con las etiquetas y modelos del proyecto.

---

## 7. Captura

La pantalla de Captura genera el material de entrenamiento.

Fuentes soportadas en Eyve 2.1: **cámara USB** y **archivo de video**.

El usuario debe poder:

- seleccionar una cámara;
- visualizar preview;
- capturar imágenes individuales;
- **grabar video** (ver §8.2 — el video es material de entrenamiento, no solo evidencia);
- cambiar resolución cuando sea compatible;
- visualizar FPS de cámara y FPS de render por separado.

La cámara inicia automáticamente cuando es razonable.

Capturar y grabar **no requieren** tener una categoría seleccionada: Captura es captura,
no etiquetado. Si no hay clase seleccionada el material se guarda en la raíz de
`raw/images/`.

### 7.1 Requisitos mínimos de cámara

| Métrica | Objetivo |
|---|---:|
| Apertura de cámara | < 1.5 s |
| Preview 720p | ≥ 25 FPS |
| CPU con preview | razonable para una PC estándar |
| CPU en idle (pantalla Home) | < 10 % |
| Cambio entre pantallas | sin dejar procesos de cámara activos |

En Windows se mantiene **DSHOW** como backend preferente por el comportamiento validado
durante el desarrollo de 2.1 (ver Anexo A).

**Criterio de aceptación:** navegar Captura → Producción → Etiquetado → Captura tres
veces seguidas no deja cámaras ocupadas ni degrada los FPS.

---

## 8. Etiquetado

El etiquetado es una función central del producto.

El usuario debe poder:

- visualizar las imágenes capturadas;
- seleccionar una clase;
- dibujar bounding boxes;
- eliminar etiquetas incorrectas;
- avanzar y retroceder entre imágenes;
- visualizar cuántas imágenes han sido etiquetadas;
- guardar automáticamente las anotaciones.

### 8.1 Tag Live

Eyve 2.1 permite generar ejemplos directamente desde cámara o video, reduciendo pasos
entre:

```text
ver una condición → capturarla → etiquetarla → agregarla al dataset
```

El usuario captura material desde la fuente activa y continúa inmediatamente con el
etiquetado. **Tag Live es una de las características principales de Eyve 2.1.**

### 8.2 El video como material de entrenamiento

> Esta sección existe para evitar un malentendido recurrente: el video grabado **sí es
> material de entrenamiento**, no solo evidencia.

Flujo previsto:

```text
Captura graba video          →  raw/images/<clase>/vid_*.mp4
Etiquetado abre ese video    →  fuente = Video
Usuario navega y congela     →  frame de interés
Tag Live guarda el frame     →  raw/images/<clase>/tag_*.jpg
Usuario dibuja las cajas     →  tagged/labels/*.txt
```

Que el dataset final se construya de `*.jpg` **no significa que el video no se usó**:
el jpg es el producto de haber revisado el video cuadro por cuadro. Grabar video es
frecuentemente la forma más rápida de capturar una condición que ocurre de manera
intermitente en la línea.

**Mejora identificada (P2):** hoy Etiquetado abre el video con un `askopenfilename`
genérico. Debería listar directamente los videos ya grabados dentro del proyecto para
que el usuario no tenga que navegar el sistema de archivos.

---

## 9. Dataset

Eyve construye automáticamente el dataset de entrenamiento a partir de las imágenes
etiquetadas.

El usuario **no debe** necesitar mover archivos, crear carpetas train/val, generar
`data.yaml` ni entender la estructura interna del dataset.

Eyve valida antes del entrenamiento que:

- existen imágenes;
- existen etiquetas;
- **las etiquetas corresponden con las imágenes correctas** (ver BUG-01);
- existen clases válidas;
- existe material suficiente para intentar un entrenamiento.

Si falta algo, el mensaje debe explicar **qué necesita hacer el usuario**, no solamente
mostrar un error técnico:

> No hay suficientes imágenes etiquetadas para entrenar.
> Etiqueta al menos algunas imágenes de cada clase y vuelve a intentarlo.

---

## 10. Entrenamiento local

El usuario debe poder: iniciar entrenamiento, observar progreso, visualizar epoch actual
y estado, cancelar, visualizar resultado, identificar cuál modelo fue generado y utilizar
automáticamente el modelo entrenado en Producción.

El entrenamiento se ejecuta fuera del thread principal para evitar congelamientos.

### 10.1 CPU y GPU

```text
CPU            → soportado
GPU NVIDIA/CUDA → opcional
```

La ausencia de GPU no impide utilizar Eyve. El usuario debe recibir una explicación clara
de que el entrenamiento será más lento en CPU, con un estimado antes de arrancar.

### 10.2 Versionado de modelos

Hoy cada entrenamiento sobrescribe `models/best.pt` y reutiliza `runs/train`
(`exist_ok=True`). No hay forma de volver al modelo anterior si el nuevo salió peor.

**Requisito mínimo para 2.1:** conservar el modelo previo al reentrenar
(ej. `best_20260810_1432.pt`) y que `best.pt` sea el activo. Comparación formal de
modelos queda en P2.

---

## 11. Producción

Producción es la etapa donde el usuario utiliza el modelo que entrenó.

Debe permitir: seleccionar cámara o video, cargar el modelo activo del proyecto, ejecutar
inferencia en tiempo real, visualizar bounding boxes / clase / confianza, mostrar
resultado general y mantener contadores.

Estados:

```text
OK   NOT_OK   NO_DETECT   REVIEW   ERROR
```

La lógica considera el tipo de cada clase (`OK` / `NOK` / `IGNORE`).

### 11.1 Modelo genérico de arranque

**Decisión:** cuando el proyecto no tiene modelo entrenado, Producción **carga un modelo
genérico** (`yolov8n.pt`, clases COCO) como base de trabajo. Esto se mantiene: permite
que la pantalla sea funcional desde el minuto uno y que el usuario vea la cámara y el
pipeline de inferencia trabajando antes de haber entrenado nada.

**Requisito:** el modelo genérico debe estar **claramente identificado como tal** en la
interfaz. No debe confundirse con un modelo del proyecto.

```text
Modelo: yolov8n (genérico — demo)
Este proyecto todavía no tiene un modelo entrenado.
[ Ir a Entrenamiento ]
```

Lo que se evita no es usar el genérico, sino usarlo **silenciosamente**.

---

## 12. Sesiones y evidencia

Eyve registra las sesiones de producción localmente. Como mínimo: fecha, hora, proyecto,
modelo, duración, resultados y conteos.

Cuando se detecta una condición NOK, Eyve guarda evidencia visual.

La información permanece dentro de la computadora del usuario.

---

## 13. Offline y privacidad

Eyve opera localmente. **Después de la instalación y de la primera descarga de pesos**:

- la cámara funciona sin internet;
- el etiquetado funciona sin internet;
- el entrenamiento funciona sin internet;
- Producción funciona sin internet.

> **Matiz importante:** la primera vez que se entrena o se abre Producción sin modelo,
> Eyve descarga los pesos base (`yolov8n.pt`, ~6 MB) desde el hub de ultralytics.
> Existe `offline_mode` y un `models_dir` cacheado en `~/.eyve/models`, pero el modelo
> base debe estar precacheado para operar 100 % sin red. Esto debe quedar explícito en
> la documentación del usuario para no prometer de más.

Eyve no requiere subir imágenes del proceso a servicios externos para completar el flujo
principal.

---

## 14. Licencia y modelo de distribución

**Decisión (18-sep-2026): licencias de sbcsuite, por honor. Nada bloquea.**

Eyve 2.1 se publica bajo **AGPL-3.0**. La llave la emite la tienda
(`sbcsuite.com.mx`) y es un **JWT firmado con EdDSA** (contrato en
`qr-lead-connect/docs/licencias-eyve.md`). Módulo: `eyve/license/license_manager.py`.

- La llave lleva dentro nivel, titular, emisión y vencimiento; Eyve la verifica
  **sin red** con la clave pública embebida (`PyJWT[crypto]`).
- Sin llave → **Free** (detección, conteo y log). Estudiante / Normal → plataforma
  abierta. **Pro** → módulos de check de SBC (polaridad, flujo, serigrafía, patrones).
  En producción se requiere Pro; los demás corren igual y solo ven el aviso.
- Llave falsa → aviso, sigue en Free. Llave vencida → sigue con su nivel y avisa
  en cada arranque. Sin cupo (4.º equipo) → aviso con la lista, sigue funcionando.
- Con red: `licencia-activar` al pegar (registra el equipo, máx. 3) y
  `licencia-estado` como mucho una vez por semana, en hilo, sin esperar; si el
  servidor manda `llave_nueva` (renovación) se guarda sola.
- Archivos: `~/.eyve/licencia.jwt` (la llave) y `~/.eyve/licencia.json` (estado).
- `hash_equipo` = sha256(MachineGuid); es lo que el usuario libera desde
  `https://sbcsuite.com.mx/store/cuenta/licencias`.
- UI: diálogo "Licencia" (Settings → Licencia…, o clic en la barra de estado).

El esquema anterior (trial de 30 días + llaves `EYVE-XXXX` con HMAC) se retiró;
`tools/gen_license.py` ya no existe. Las llaves las emite el servidor (automático
al pagar / pedido Free, o manual desde el admin de la tienda).

---

## 15. Distribución pública

La prioridad de desarrollo actual es que Eyve 2.1 pueda ser **descargado y ejecutado por
usuarios externos**.

### 15.1 Formato inicial

```text
Eyve_2.1.zip
```

Descomprimible con WinRAR, 7-Zip o el descompresor integrado de Windows.

Después de extraer:

```text
setup.bat    →  instala
run.bat      →  ejecuta
```

### 15.2 setup.bat

1. Verificar Python.
2. **Si no existe, instalarlo automáticamente** (ver §15.3).
3. Crear el entorno virtual.
4. Instalar dependencias.
5. Informar claramente el progreso.
6. Mostrar cualquier error de instalación.
7. Indicar cuando Eyve está listo.

### 15.3 Instalación automática de Python

**Este es el punto de fuga #1 para usuarios no desarrolladores.** Hoy `setup.bat` detecta
que falta Python y le pide al usuario que lo instale — ahí se acaba el viaje para buena
parte de la audiencia objetivo.

**Requisito 2.1:** si Python no está presente, `setup.bat` lo instala sin intervención:

```bat
winget install -e --id Python.Python.3.12 --scope machine
```

Con fallback documentado a descarga manual si `winget` no está disponible
(Windows 10 antiguo). Después de instalar debe refrescar el `PATH` de la sesión o
indicar al usuario que vuelva a ejecutar `setup.bat`.

### 15.4 SmartScreen y antivirus

Un `.bat` que crea un entorno virtual y descarga desde internet **será marcado** por
Windows SmartScreen o Defender en algunas máquinas. Es uno de los tres principales
motivos de soporte en distribución pública por ZIP.

Mitigaciones para 2.1:

- documentar en el README qué aviso aparece y cómo continuar ("Más información →
  Ejecutar de todas formas");
- publicar el hash SHA-256 del ZIP junto a la descarga;
- evaluar firma de código para el launcher (ver §26.2).

---

## 16. Requisitos del sistema

```text
Windows 10 / Windows 11
Python 3.10+                (instalado automáticamente por setup.bat)
8 GB RAM mínimo             (16 GB recomendado para entrenar)
10 GB de espacio libre      (venv con torch ≈ 4-6 GB + modelos + dataset)
Conexión a internet         REQUERIDA para la instalación
Cámara USB                  (opcional si se trabaja con video)
GPU NVIDIA                  (opcional — acelera el entrenamiento)
```

Dos perfiles de uso:

| Perfil | RAM | Uso |
|---|---|---|
| Inspección | 8 GB | Producción con un modelo ya entrenado |
| Completo | 16 GB | Captura + Etiquetado + **Entrenamiento** + Producción |

La compatibilidad real debe verificarse antes del release en máquinas limpias.

---

## 17. Actualización y versionado

### 17.1 Estrategia 2.1 (manual, simple)

Instalación **lado a lado** por versión:

```text
Eyve/
  2.1/     ← versión actual
  2.1.1/   ← versión nueva
```

Y los proyectos **fuera** de ambas (§5.1), de modo que actualizar sea:

1. descomprimir la versión nueva junto a la anterior;
2. ejecutar su `setup.bat`;
3. abrir los mismos proyectos desde la versión nueva;
4. borrar la carpeta anterior cuando se confirme que todo funciona.

Los modelos entrenados viven dentro del proyecto (`<proyecto>/models/`), así que
sobreviven automáticamente.

**Criterio de aceptación:** un usuario puede pasar de 2.1 a 2.1.1 sin perder proyectos,
sin mover archivos a mano y sin volver a etiquetar.

### 17.2 Futuro (P2)

Actualizador integrado que detecte versión nueva, descargue y migre. No es alcance de 2.1.

---

## 18. Experiencia de primer uso

El usuario que abre Eyve por primera vez debe entender qué hacer. La pantalla inicial
orienta hacia:

```text
1. Crear proyecto
2. Definir clases
3. Capturar ejemplos
4. Etiquetar
5. Entrenar
6. Probar en Producción
```

No debe ser necesario leer documentación técnica para descubrir el flujo.

Cuando una acción todavía no sea posible, Eyve debe indicar el motivo:

```text
Entrenamiento no disponible
Necesitas etiquetar imágenes antes de entrenar.
```

En lugar de simplemente deshabilitar el botón sin explicación.

---

## 19. Proyecto demo

El release debe incluir al menos un ejemplo sencillo que permita explorar Eyve sin
construir inmediatamente un dataset propio. El demo debe estar claramente separado de los
proyectos reales.

Complementa —no sustituye— al modelo genérico de §11.1: el genérico deja ver el pipeline
funcionando; el demo deja ver un proyecto **completo y coherente** (clases, imágenes,
etiquetas, modelo entrenado).

---

## 20. Idiomas

**Eyve 2.1 es bilingüe: Español + Inglés.** Ambos idiomas son de primera clase.

### 20.1 Estado actual — deuda a corregir (P1)

Verificado en código: **41 strings de interfaz están hardcoded fuera del sistema i18n**,
mezclando ambos idiomas dentro de la misma pantalla.

| Pantalla | Strings fuera de `t()` |
|---|---:|
| `capture_screen.py` | 15 |
| `tagging_screen.py` | 17 |
| `production_screen.py` | 9 |

Ejemplos convivientes en la misma UI:

```text
"Fuente"        "⏺  Record"            "● En vivo"
"Resolución"    "⏹  Stop & Save"       "Abriendo cámara…"
"Cámara"        "No frame yet…"        "Opening camera…"
"Sin límite"    "Could not create video file"
```

Consecuencia: un usuario en inglés ve *"Fuente"* y *"Capturar y Etiquetar"*; uno en
español ve *"Record"* y *"No frame yet — wait for live feed"*.

**Requisito:** cero strings de interfaz fuera de `t()` antes del release. Incluye revisar
la estrategia de empaquetado para que agregar una pantalla nueva no vuelva a permitirlo
(ej. check en CI o revisión de PR).

---

## 21. Manejo de errores

Una aplicación pública no puede depender de errores visibles únicamente en consola.

Eyve debe:

- registrar excepciones en log con `exc_info`;
- **evitar `except Exception: pass`** en operaciones críticas;
- mostrar errores comprensibles;
- impedir múltiples diálogos modales simultáneos;
- evitar cierres inesperados;
- permitir reportar bugs con información suficiente.

Log principal: `eyve_run.log`

---

## 22. P0 — Bugs bloqueantes

### 🔴 BUG-01 — Asociación imagen/label rota

**Síntoma:** después de etiquetar N imágenes, Entrenamiento reporta "sin datos". El
contador de Etiquetado sí muestra las imágenes etiquetadas — la inconsistencia confunde.

**Causa:** dos convenciones de nombre distintas para el mismo archivo.

`tagging_screen.py:640` **escribe** con stem único derivado de la ruta relativa:

```python
rel = img_path.relative_to(proj.paths.raw_images)
unique_stem = str(rel.with_suffix("")).replace("/", "_").replace("\\", "_")
# raw/images/rayon/img_001.jpg → tagged/labels/rayon_img_001.txt
```

`dataset_builder.py:46` y `:106` **leen** con el stem pelón:

```python
label = p.tagged_labels / f"{img.stem}.txt"
# busca tagged/labels/img_001.txt → NUNCA existe
```

**Impacto:** el pipeline de entrenamiento está muerto para cualquier imagen dentro de una
subcarpeta de clase, que es el flujo normal.

**Fix:** una sola función canónica (ej. `ProjectPaths.label_stem(img_path)`) usada por los
tres sitios. Incluir migración para proyectos con etiquetas del nombre viejo.

---

### 🔴 BUG-02 — `BaseCallback` no existe en Ultralytics 8.4.x

**Síntoma:** al dar Start en Entrenamiento, la barra pasa a "error" de inmediato.

**Causa:** `train_manager.py:104`

```python
from ultralytics.utils.callbacks.base import BaseCallback
```

Verificado contra la versión instalada:

```text
ultralytics 8.4.49
BaseCallback import: FAILS → ImportError
```

La `ImportError` cae en un `except Exception` genérico y se reporta como error de
entrenamiento, sin pista de que fue un problema de import.

**Fix:** migrar al sistema actual de callbacks (función suelta registrada con
`model.add_callback(...)`) y pinear `ultralytics>=8.3,<9`.

Nota: el log ya usa `exc_info=True` (`train_manager.py:204`); el problema es que la UI
solo muestra `str(e)`, que para una `ImportError` no dice nada útil al usuario.

---

### 🔴 Validación End-to-End

Debe existir una prueba real completa y **repetible**:

```text
crear proyecto → crear clases → capturar ≥30 imágenes → etiquetar →
construir dataset → entrenar → cargar best.pt → abrir Producción →
detectar con el modelo nuevo → cerrar Eyve → reabrir → continuar
```

Hasta que esta prueba funcione de forma repetible, Eyve 2.1 no está listo para release.

---

## 23. P1 — Antes del release público

**Funcionalidad**
- [ ] Instalación automática de Python en `setup.bat` (§15.3)
- [ ] Etiquetar el modelo genérico como tal en Producción (§11.1)
- [ ] Fijar ubicación default de proyectos fuera de la app (§5.1)
- [ ] Conservar modelo previo al reentrenar (§10.2)
- [ ] Crear proyecto demo (§19)
- [ ] Listar videos del proyecto en Etiquetado (§8.2)

**Calidad**
- [ ] Corregir los 41 strings fuera de i18n (§20.1)
- [ ] Resolver diálogos modales simultáneos
- [ ] Eliminar errores silenciosos en operaciones críticas
- [ ] Mejorar validación del dataset con mensajes accionables
- [ ] Pinear versiones de dependencias

**Verificación**
- [ ] Instalación limpia en Windows 10
- [ ] Instalación limpia en Windows 11
- [ ] Operación solo CPU
- [ ] Al menos una GPU NVIDIA
- [ ] Persistencia de proyectos tras reinicio
- [ ] Cambiar de pantalla libera cámara y threads
- [ ] Producción utiliza el modelo correcto
- [ ] Actualización 2.1 → 2.1.1 sin pérdida (§17.1)

**Documentación**
- [ ] README de instalación (incluye aviso de SmartScreen y hash SHA-256)
- [ ] Guía rápida de primer proyecto
- [ ] Proceso sencillo para reportar bugs
- [ ] Declarar política de licencia (por honor, niveles) desde el README (§14)

---

## 24. P2 — Deseables si no retrasan el release

- [ ] editar/mover bounding boxes existentes;
- [ ] captura automática cada N segundos;
- [ ] extracción automática de frames cada N cuadros desde video;
- [ ] exportar CSV de sesiones;
- [ ] ROI configurable;
- [ ] augmentation configurable;
- [ ] comparación entre modelos;
- [ ] exportar/importar proyecto;
- [ ] instalador completamente offline;
- [ ] instalador `.exe` / launcher compilado (§26.2);
- [ ] actualizador integrado (§17.2);
- [ ] detección automática y configuración simplificada de CUDA.

---

## 25. Fuera del alcance de Eyve 2.1

Todo lo relacionado con **despliegue industrial multiestación**: integración PLC, paro de
línea, buses industriales, MES/ERP, servidores centrales, múltiples cámaras simultáneas,
analítica centralizada, alta disponibilidad, funciones safety y validaciones industriales
formales.

Esas capacidades pertenecen a **Eyve Enterprise** y pueden desarrollarse alrededor de
Eyve, pero no deben distraer el desarrollo actual.

---

## 26. Arquitectura hacia adelante

### 26.1 Una sola costura que importa

Eyve 2.1 usa un backend de detección (YOLO/Ultralytics). A futuro pueden existir motores
para anomaly detection, segmentación, pre-etiquetado o distintos runtimes.

No hay que implementarlos en 2.1. Pero sí hay **un requisito concreto** que evita quedar
amarrados:

> **Producción debe hablar con una interfaz `Detector`, no con `YOLOWorker` directamente.**

Hoy `production_screen.py` importa `YOLOWorker` de forma directa. Definir una interfaz
mínima (`load()` / `push_frame()` / `get_result()` / `class_names`) permite sustituir el
motor después sin tocar la UI. Es trabajo de una tarde y ahorra un refactor mayor.

### 26.2 Compilación y empaquetado (P2 / futuro)

Camino previsto conforme el producto madure:

1. **Hoy:** `.bat` + Python interpretado.
2. **Siguiente:** launcher compilado (`.exe`) que sustituya `run.bat` — mejora la
   experiencia, permite firma de código y reduce falsos positivos de SmartScreen (§15.4).
3. **Después:** módulos críticos de rendimiento o de propiedad intelectual compilados
   (C++/Cython/Nuitka) mientras la capa de aplicación sigue en Python.

Compatible con AGPL-3.0 siempre que el código correspondiente siga disponible. No es
alcance de 2.1, pero las decisiones de estructura de hoy no deben cerrar esta puerta.

---

## 27. Métricas de éxito del release

### Métrica principal — Time to First Tag

Tiempo desde que el usuario **inicia la descarga** hasta que **etiqueta su primera
imagen** en Eyve.

```text
Objetivo: ≤ 30 minutos
```

**Por qué termina en el primer tag y no en la primera inspección:** a partir de ahí el
tiempo depende de cuántas muestras tenga el usuario, qué tan bien las etiquete y qué
hardware use para entrenar. Eso no lo controlamos. **El deploy sí** — y es exactamente lo
que esta métrica mide.

Supuesto declarado: conexión ≥ 20 Mbps (la descarga de dependencias es ~1-2 GB).

### Métricas adicionales

| Métrica | Meta |
|---|---:|
| Instalación sin intervención de SBC | ≥ 90 % |
| Completar flujo completo sin ayuda | ≥ 80 % |
| App sin cierre inesperado durante 30 min | 100 % |
| Apertura de cámara | < 1.5 s |
| Preview 720p | ≥ 25 FPS |
| CPU idle en Home | < 10 % |
| Proyecto conserva información tras reiniciar | 100 % |
| Entrenamiento genera modelo utilizable | 100 % en prueba controlada |
| Strings de UI fuera de i18n | 0 |

Durante las primeras versiones públicas, **los reportes de error son parte del objetivo
del release**.

---

## 28. Definition of Done — Eyve 2.1

### 28.1 Flujo principal

Una PC limpia debe completar:

```text
1.  Descargar Eyve_2.1.zip
2.  Descomprimir
3.  Ejecutar setup.bat  (instala Python si falta)
4.  Ejecutar run.bat
5.  Crear proyecto
6.  Crear clases OK/NOK
7.  Abrir cámara
8.  Capturar imágenes y/o grabar video
9.  Etiquetar (incluye tagear frames desde el video grabado)
10. Entrenar
11. Generar modelo
12. Abrir Producción
13. Detectar utilizando ese modelo
14. Registrar una sesión
15. Cerrar Eyve
16. Volver a abrir Eyve
17. Abrir el mismo proyecto
18. Continuar funcionando correctamente
```

Sin editar código, usar terminal manualmente, mover archivos, crear datasets a mano,
modificar YAML, instalar herramientas fuera del flujo indicado ni depender del
desarrollador.

### 28.2 Pruebas negativas (obligatorias)

El happy path no es suficiente. Debe verificarse también:

- [ ] cerrar Eyve **durante** el entrenamiento → el proyecto queda consistente
- [ ] desconectar la cámara con el preview en vivo → mensaje claro, sin crash
- [ ] entrar a Producción sin modelo entrenado → genérico bien identificado
- [ ] entrar a Entrenamiento sin etiquetas → mensaje accionable, no error técnico
- [ ] disco lleno durante grabación o entrenamiento → error comprensible
- [ ] abrir un proyecto creado por una versión anterior de Eyve
- [ ] dos instancias de Eyve abiertas a la vez → la segunda no roba la cámara en silencio

---

## 29. Prioridad de desarrollo actual

```text
1. Pipeline end-to-end
2. Estabilidad
3. Instalación
4. Experiencia de primer uso
5. Manejo de errores
6. Documentación
7. Release
```

No es prioridad: Enterprise, PLC, multiestación, nuevos modelos, nuevos protocolos,
nuevas integraciones, arquitecturas complejas, features que no ayuden a completar la
primera inspección.

La pregunta para cada cambio durante esta etapa:

> **¿Esto ayuda a que una persona externa pueda descargar Eyve 2.1 y completar su primera
> inspección?**

Si la respuesta es no, probablemente puede esperar hasta después del release.

---

## 30. Principio rector

> **Descargar. Enseñar. Entrenar. Inspeccionar.**

Eyve 2.1 debe demostrar que una herramienta de visión artificial para manufactura puede
usarse como software descargable y local, sin que el usuario tenga que convertirse
primero en especialista en machine learning.

El release no necesita resolver todavía todos los escenarios industriales. Necesita
resolver muy bien el primero:

> **Que alguien pueda instalar Eyve y hacerlo funcionar con su propia cámara, sus propias
> piezas y sus propios ejemplos.**

---
---

# Anexo A — Notas técnicas aprendidas

Lecciones del desarrollo de 2.1. Documentadas para no repetir los mismos errores.

**Cámara (Windows)**

- **DSHOW vs MSMF:** DSHOW abre en ~200 ms sin renegociar; MSMF tarda 5-6 s y renegocia el
  pipeline completo en *cada* `cap.set()`. Siempre DSHOW primero en Windows.
- **MJPEG es obligatorio para HD:** sin `FOURCC=MJPG` el driver manda YUV sin comprimir y
  satura USB 2.0 → 5 fps a 1080p. Con MJPEG la cámara comprime on-chip → 30 fps.
- **Un solo `cap.set()` de FOURCC:** ponerlo dos veces (open + apply_resolution) mete una
  segunda renegociación de +1.4 s y deja la cámara en mal estado.
- **DSHOW libera de forma asíncrona:** `cap.release()` regresa de inmediato pero el filter
  graph sigue desmontándose 200-500 ms. Requiere reintentos al reabrir entre pantallas.

**Interfaz (Tkinter / CustomTkinter)**

- **`grid_remove()` NO detiene los `after()`:** las pantallas ocultas seguían corriendo sus
  loops en background (CPU 48 % en idle). Requiere `on_hide()` / `on_show()` explícito.
- **`create_image()` acumula:** hay que crear el item una vez y actualizarlo con
  `itemconfig()`. Miles de items apilados degradan el render progresivamente.
- **`cv2.resize` antes de PIL:** `Image.thumbnail()` sobre 1080p tarda 20-40 ms;
  `cv2.resize` (C++ SIMD) tarda 1-3 ms.
- **Tkinter no acepta `#RRGGBBAA`:** hay que mezclar el alpha manualmente a 6 caracteres.
- **Encoding y captura no van en el mismo thread:** `VideoWriter.write()` dentro del loop de
  cámara hace que un códec o disco lento frene el preview. Cola acotada + thread dedicado.
- **Defaults duplicados se pisan:** un default en `config._DEFAULTS` gana sobre el fallback
  de `config.get(key, default)`, porque `_load()` siempre mergea los defaults. Mantener un
  solo lugar de verdad.

---

# Anexo B — Estado por módulo

Leyenda: ✅ funciona · ⚠️ funciona con problemas · ❌ roto

| Módulo | Estado | Nota |
|---|---|---|
| Home | ✅ | Crear/abrir/recientes. Falta default de ubicación (§5.1) |
| Categorías | ✅ | CRUD, kind ok/nok/ignore, color |
| Captura | ✅ | Cámara + video, auto-start, grabación en thread aparte |
| Etiquetado | ✅ | BUG-01 cerrado; playbar de video (pausa/seek), congelar sin huérfanas |
| Entrenamiento | ✅ | BUG-02 cerrado; E2E validado 2026-08-10: mAP50 0.865 con 40 imgs reales |
| Producción | ✅ | Auto-start, genérico de arranque, OK/NOK, contadores |
| Enumeración de cámaras | ✅ | MFEnumDeviceSources — orden correcto, 0 ms |
| Licencias | ✅ | JWT de sbcsuite, verificación offline, activar/check-in, puerta por nivel (§14) |
| i18n | ⚠️ | Sistema completo, pero 41 strings lo evaden (§20.1) |
| Tema dark/light | ✅ | |
| Settings | ✅ | Idioma, tema, FPS cap, offline mode, licencia |
| Release / instalador | ⚠️ | ZIP funciona; falta auto-install de Python (§15.3) |

---

# Anexo C — Estructura de un proyecto en disco

```text
<proyecto>/
  project.yaml            nombre, target, cámara, modelo activo
  classes.yaml            clases (name, kind: ok|nok|ignore, color)
  raw/images/<clase>/     imágenes capturadas y videos grabados
  tagged/images/          copia de imágenes etiquetadas
  tagged/labels/          .txt en formato YOLO
  datasets/yolo_dataset/  split train/val + data.yaml
  models/                 best.pt, last.pt, training_metadata.yaml
  runs/                   salida cruda de ultralytics
  production/sessions/    logs de sesión
  production/screenshots/ capturas de NOK
  packages/               paquetes de entrenamiento exportados
  logs/
```
