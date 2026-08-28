# Eyve 2.1 Beta

**Inspección visual con IA — 100 % local, en tu PC.**

Eyve te permite entrenar tu propio modelo de visión artificial con tus propias
piezas, sin enviar una sola imagen a internet y sin saber nada de machine learning.

```
Captura  →  Etiquetado  →  Entrenamiento  →  Producción
```

---

## Instalación

### 1. Descomprime el ZIP

En cualquier carpeta donde tengas permisos de escritura, por ejemplo
`C:\Eyve\` o tu escritorio. **No lo dejes dentro del ZIP** — descomprímelo de verdad.

### 2. Doble clic en `setup.bat`

Se encarga de todo:

- Detecta si tienes Python. **Si no lo tienes, lo instala automáticamente** (te pide confirmación).
- Crea un entorno aislado para Eyve (no toca tu Python ni otros programas).
- Descarga las dependencias (~1-2 GB, solo la primera vez).

Tarda entre 10 y 30 minutos según tu conexión. **No cierres la ventana.**

### 3. Doble clic en `run.bat`

Listo. Eyve abre.

---

## ⚠ Si Windows muestra una advertencia

Es normal: los archivos `.bat` descargados de internet no están firmados digitalmente.

**Windows protegió tu PC** → clic en **Más información** → **Ejecutar de todas formas**.

Si tu antivirus bloquea la descarga de dependencias, agrégale una excepción a la
carpeta de Eyve. Puedes verificar que el ZIP es el original comparando su hash
SHA-256 con el publicado junto a la descarga:

```powershell
Get-FileHash Eyve_2.1_Beta.zip -Algorithm SHA256
```

---

## Requisitos

| | Mínimo | Recomendado |
|---|---|---|
| Sistema | Windows 10 | Windows 11 |
| RAM | 8 GB | 16 GB (para entrenar) |
| Disco libre | 10 GB | 15 GB |
| Internet | **Requerido para instalar** | — |
| Cámara | USB (opcional si usas video) | Webcam HD |
| GPU | No necesaria | NVIDIA (acelera el entrenamiento) |

Con 8 GB de RAM puedes usar Producción con un modelo ya entrenado. Para entrenar
cómodamente, 16 GB.

Después de instalar, Eyve funciona sin internet (salvo la primera descarga de los
pesos base del modelo, ~6 MB).

---

## Tu primer proyecto en 6 pasos

1. **Inicio → Crear proyecto.** Ponle nombre. Se guarda en `Documentos\Eyve Projects`.
2. **Categorías.** Define qué quieres detectar. Cada categoría es `OK`, `NOK` o `IGNORE`.
   Ejemplo: `componente_correcto` → OK, `polaridad_invertida` → NOK.
3. **Captura.** Toma fotos de tus piezas con la cámara, o graba video.
   Junta al menos 30-40 imágenes con variedad (ángulos, iluminación, posiciones).
4. **Etiquetado.** Dibuja un recuadro alrededor de cada pieza y elige su categoría.
   - Fuente **Imágenes**: recorre las fotos que capturaste.
   - Fuente **Video**: reproduce, pausa donde te interese (`⏸`), dibuja, Guardar `[S]`,
     y usa la barra para saltar a otro momento.
   - Fuente **Cámara**: congela con `[G]`, dibuja y guarda.
5. **Entrenamiento → Iniciar.** Con CPU tarda unos minutos; con GPU, menos.
   Al terminar, tu modelo queda activo automáticamente.
6. **Producción.** Apunta la cámara (o carga un video) y verás tu modelo trabajando:
   OK / NOT OK en tiempo real, con conteo de piezas y evidencia de los defectos.

### Atajos útiles

| Tecla | Acción |
|---|---|
| `1`–`9` | Elegir categoría (Etiquetado) |
| `S` | Guardar etiquetas |
| `N` / `P` | Siguiente / anterior imagen |
| `G` | Congelar frame de cámara/video |
| `Supr` | Borrar última caja |
| `Esc` | Cancelar |
| `Espacio` | Pausar/reanudar (Producción) |

---

## Módulos (vista previa de Eyve Pro)

En Producción encontrarás la sección **Módulos**, con inspecciones que van más allá
de solo detectar:

- **Polaridad** — verifica que los componentes estén orientados correctamente
  (banda de cátodo, muescas, puntos). Aprende la orientación correcta sola, o se la
  fijas tú. La vista previa en la esquina te muestra exactamente qué está analizando.
- **Conteo** — dibuja una línea sobre el video y Eyve cuenta las piezas que la cruzan.
  Cada pieza cuenta una sola vez.

Son experimentales y están en evolución. Tu feedback sobre ellos es especialmente valioso.

---

## Licencia

Eyve es software libre bajo **AGPL-3.0**. Puedes usarlo, estudiarlo, modificarlo y
distribuirlo.

Después de 30 días verás un recordatorio de licencia. **Eyve sigue funcionando igual,
sin limitaciones** — el recordatorio solo desaparece con una licencia comercial, que
además incluye soporte. Estilo WinRAR: nada se bloquea nunca.

---

## Reportar un problema

Esta es una beta: **los reportes de errores son parte del objetivo**. Si algo falla:

1. Anota qué estabas haciendo cuando pasó.
2. Adjunta el archivo `eyve_run.log` (está en la carpeta de Eyve).
3. Si hay algo visible en pantalla, una captura ayuda mucho.
4. Menciona tu Windows (10 u 11) y si tienes GPU NVIDIA.

Envíalo al equipo de Eyve o abre un issue en el repositorio.

---

## Preguntas frecuentes

**¿Mis imágenes salen de mi computadora?**
No. Todo el ciclo ocurre localmente. Eyve solo usa internet para instalarse y para
descargar los pesos base del modelo la primera vez.

**¿Necesito GPU?**
No. Sin GPU el entrenamiento es más lento pero funciona bien para datasets pequeños
(40 imágenes ≈ 5 minutos en un CPU moderno).

**¿Puedo mover o copiar mis proyectos?**
Sí. Cada proyecto es una carpeta autocontenida en `Documentos\Eyve Projects` con sus
imágenes, etiquetas y modelos. Cópiala a otra PC y ábrela ahí.

**¿Cómo actualizo a una versión nueva?**
Descomprime la nueva junto a la anterior (`Eyve\2.1\`, `Eyve\2.1.1\`) y corre su
`setup.bat`. Tus proyectos no se tocan porque viven fuera de la carpeta de Eyve.
Cuando confirmes que la nueva funciona, borra la anterior.

**El entrenamiento dice "sin imágenes etiquetadas" pero ya etiqueté.**
Verifica que dibujaste al menos una caja en cada imagen y presionaste Guardar `[S]`.
Recorrer imágenes sin dibujar nada no cuenta como etiquetada.
