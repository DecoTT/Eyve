# Sistema de Inspección Visual Automatizado

Este proyecto implementa un sistema de inspección visual automatizado para entornos industriales, permitiendo la detección de defectos, medición de dimensiones y control de calidad a través de procesamiento de imágenes y visión por computadora.

## Estructura del Proyecto

```
proyecto_inspeccion_visual/
├── config/                  # Configuraciones generales y específicas por SKU
├── core/                    # Funcionalidades centrales del sistema
├── database/                # Módulos de conexión y operaciones con BD
├── gui/                     # Interfaces gráficas
├── inspection_modules/      # Módulos específicos de inspección
├── models/                  # Modelos de IA preentrenados
├── utils/                   # Utilidades generales
└── main.py                  # Punto de entrada principal
```

## Requisitos

- Python 3.8+
- OpenCV 4.5+
- NumPy
- Pillow (PIL)
- tkinter
- SQLite (o el motor de base de datos elegido)

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Componentes Principales

### 1. Sistema de Gestión Central (SystemManager)

El administrador central del sistema coordina todos los módulos y componentes:

- Gestión de usuarios y autenticación
- Carga de configuraciones de trabajo
- Conexión con cámaras y dispositivos
- Registro de actividades y resultados

### 2. Interfaz de Usuario

#### Pantalla de Login
- Autenticación de usuarios
- Selección de trabajos/SKUs
- Registro de fecha/hora de inicio

#### Pantalla Principal
- Visualización en tiempo real
- Controles de cámara
- Visualización de resultados de inspección
- Estadísticas y métricas

### 3. Módulo de Cámara

- Conexión y configuración de dispositivos de captura
- Calibración de parámetros
- Streaming de video en tiempo real
- Captura de imágenes para inspección

### 4. Módulos de Inspección

El sistema incluye varios módulos de inspección especializada:

- **ColorDetectionModule**: Detecta colores fuera de rango
- **DefectDetectionModule**: Identifica defectos visuales mediante comparación con referencia
- **DimensionMeasurementModule**: Mide dimensiones de productos y verifica tolerancias

## Flujo de Trabajo

1. El usuario inicia sesión en el sistema
2. Selecciona el trabajo/SKU a inspeccionar
3. El sistema carga la configuración específica para ese producto
4. Se establece conexión con las cámaras configuradas
5. La interfaz muestra la visualización en tiempo real
6. El sistema captura imágenes automáticamente o bajo demanda
7. Los módulos de inspección analizan las imágenes
8. Se muestran los resultados y se registran en la base de datos

## Configuración

El sistema utiliza archivos de configuración en formato JSON para:

- **Parámetros generales**: Rutas, logs, conexiones, etc.
- **Configuración de SKUs**: Especificaciones por producto
- **Módulos de inspección**: Umbrales, rangos, dimensiones, etc.

Ejemplo de configuración de SKU:

```json
{
  "sku_id": "SKU123",
  "name": "Pieza Metálica A",
  "inspection_modules": {
    "color": {
      "enabled": true,
      "color_ranges": {
        "red": {"lower": [0, 100, 100], "upper": [10, 255, 255]}
      }
    },
    "dimensions": {
      "enabled": true,
      "pixels_per_mm": 10.5,
      "target_dimensions": {
        "width": {"min": 95, "max": 105},
        "height": {"min": 45, "max": 55}
      }
    }
  }
}
```

## Base de Datos

El sistema utiliza una base de datos para almacenar:

- Usuarios y permisos
- Configuraciones de SKU
- Resultados de inspección
- Registro de actividad

## Desarrollo Futuro

Áreas para expandir el sistema:

- **Módulos adicionales de inspección**: OCR, análisis de textura, etc.
- **Integración con IA**: Redes neuronales para detección avanzada
- **Dashboard web**: Visualización remota y estadísticas
- **Notificaciones**: Alertas por correo o SMS
- **Control de calidad estadístico**: SPC/SQC con gráficos de control
- **Integración con sistemas ERP/MES**: Intercambio de datos con sistemas empresariales

## Cómo Contribuir

1. Haz un fork del repositorio
2. Crea una rama para tu función: `git checkout -b feature/nueva-funcionalidad`
3. Haz commit de tus cambios: `git commit -m 'Añadir nueva funcionalidad'`
4. Envía los cambios a tu rama: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

## Licencia

[MIT License](LICENSE)

## Contacto

Para preguntas o sugerencias, por favor abre un issue en este repositorio.
