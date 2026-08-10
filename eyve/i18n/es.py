STRINGS = {
    # ── App ──────────────────────────────────────────────────────────────────
    "app_title": "Eyve 2.1 Beta",
    "app_subtitle": "Plataforma de Inspección Visual",

    # ── Nav / sidebar ────────────────────────────────────────────────────────
    "nav_home": "Inicio",
    "nav_classes": "Categorías",
    "nav_capture": "Captura",
    "nav_tagging": "Etiquetado",
    "nav_training": "Entrenamiento",
    "nav_production": "Producción",

    # ── Home screen ──────────────────────────────────────────────────────────
    "home_welcome": "Bienvenido a Eyve",
    "home_tagline": "Crea, entrena y ejecuta proyectos de inspección visual localmente.",
    "home_new_project": "Nuevo Proyecto",
    "home_open_project": "Abrir Proyecto",
    "home_recent": "Proyectos Recientes",
    "home_no_recent": "Sin proyectos recientes.",
    "home_test_camera": "Probar Cámara",
    "home_help": "Ayuda / Documentación",

    # ── Project creation ─────────────────────────────────────────────────────
    "proj_create_title": "Crear Nuevo Proyecto",
    "proj_name": "Nombre del Proyecto",
    "proj_name_hint": "ej. papitas_inspeccion",
    "proj_folder": "Ubicación",
    "proj_folder_browse": "Explorar…",
    "proj_camera": "Fuente de Cámara",
    "proj_target": "Objeto a Inspeccionar",
    "proj_target_hint": "¿Qué vas a inspeccionar? ej. Papitas, PCB, Piezas",
    "proj_create_btn": "Crear Proyecto",
    "proj_cancel": "Cancelar",
    "proj_name_required": "El nombre del proyecto es obligatorio.",
    "proj_folder_required": "La ubicación es obligatoria.",
    "proj_name_invalid": "El nombre solo puede contener letras, números y guiones bajos.",
    "proj_exists": "Ya existe un proyecto con ese nombre en esa carpeta.",
    "proj_created_ok": "Proyecto creado exitosamente.",

    # ── Project dashboard ────────────────────────────────────────────────────
    "dash_project": "Proyecto",
    "dash_classes": "Categorías",
    "dash_images": "Imágenes",
    "dash_labeled": "Etiquetadas",
    "dash_model": "Modelo",
    "dash_no_model": "Sin modelo cargado",
    "dash_status": "Estado",

    # ── Classes screen ───────────────────────────────────────────────────────
    "cls_title": "Definir Categorías",
    "cls_subtitle": (
        "Las categorías definen qué detectará tu modelo. "
        "Una vez iniciado el entrenamiento, no se pueden cambiar sin crear un nuevo proyecto."
    ),
    "cls_warning_permanent": (
        "Importante: El modelo entrenado queda ligado permanentemente a estas categorías.\n"
        "Si necesitas detectar una nueva categoría más adelante, debes crear un nuevo proyecto."
    ),
    "cls_add": "Agregar Categoría",
    "cls_name_hint": "Nombre de categoría…",
    "cls_type_ok": "OK",
    "cls_type_nok": "NO OK",
    "cls_type_ignore": "Ignorar",
    "cls_delete": "Eliminar",
    "cls_rename": "Renombrar",
    "cls_save": "Guardar Categorías",
    "cls_empty": "Sin categorías definidas. Agrega al menos 2.",
    "cls_need_ok": "Al menos una categoría debe marcarse como OK.",
    "cls_need_nok": "Al menos una categoría debe marcarse como NO OK.",
    "cls_name_dup": "Ya existe una categoría con ese nombre.",
    "cls_name_empty": "El nombre no puede estar vacío.",
    "cls_count": "{n} categoría(s) definida(s)",
    "cls_in_use": "No se puede eliminar '{name}' — tiene {n} imagen(es) etiquetada(s).",
    "cls_saved_ok": "Categorías guardadas.",

    # ── Capture screen ───────────────────────────────────────────────────────
    "cap_title": "Capturar Imágenes",
    "cap_source_camera": "Cámara",
    "cap_source_video": "Archivo de Video",
    "cap_select_video": "Seleccionar Video…",
    "cap_camera_id": "Índice de Cámara",
    "cap_start": "Iniciar Vista Previa",
    "cap_stop": "Detener",
    "cap_capture": "Capturar Frame  [C]",
    "cap_session_class": "Asignar a Categoría",
    "cap_no_class": "— selecciona categoría —",
    "cap_captured": "Capturadas: {n}",
    "cap_no_camera": "Cámara no encontrada. Verifica el índice.",
    "cap_no_classes": "Define categorías antes de capturar.",
    "cap_no_class_selected": "Selecciona una categoría antes de capturar.",
    "cap_saved": "Frame guardado en el proyecto.",

    # ── Tagging screen ───────────────────────────────────────────────────────
    "tag_title": "Etiquetado",
    "tag_image": "Imagen {current} / {total}",
    "tag_class": "Categoría",
    "tag_prev": "← Ant  [P]",
    "tag_next": "Sig →  [N]",
    "tag_save": "Guardar  [S]",
    "tag_delete_box": "Eliminar Caja  [Supr]",
    "tag_clear": "Limpiar Todas",
    "tag_skip": "Saltar Imagen",
    "tag_no_images": "No hay imágenes capturadas. Ve a Captura primero.",
    "tag_shortcuts": "Dibuja caja: arrastra el mouse | Teclas: 1-9 = categoría · S = guardar · N/P = sig/ant · Supr = eliminar",
    "tag_saved": "Etiqueta guardada.",
    "tag_class_stats": "Etiquetadas por categoría:",
    "tag_progress": "{labeled} / {total} imágenes etiquetadas",

    # ── Training screen ──────────────────────────────────────────────────────
    "train_title": "Entrenamiento",
    "train_local": "Entrenar Localmente",
    "train_package": "Empaquetar para Entrenamiento Externo",
    "train_model_base": "Modelo Base",
    "train_epochs": "Épocas",
    "train_imgsz": "Tamaño de Imagen",
    "train_batch": "Tamaño de Lote",
    "train_device": "Dispositivo",
    "train_device_auto": "Auto (GPU si está disponible)",
    "train_device_cpu": "Solo CPU",
    "train_device_gpu": "GPU (CUDA)",
    "train_conf_default": "Confianza Predeterminada",
    "train_start": "Iniciar Entrenamiento",
    "train_stop": "Detener Entrenamiento",
    "train_status_idle": "Listo para entrenar.",
    "train_status_preparing": "Preparando dataset…",
    "train_status_training": "Entrenando… época {epoch}/{total}",
    "train_status_done": "Entrenamiento completo. Modelo guardado.",
    "train_status_error": "Error en entrenamiento: {error}",
    "train_eta": "ETA: {time}",
    "train_elapsed": "Transcurrido: {time}",
    "train_best_model": "Mejor modelo: {path}",
    "train_hardware_note": (
        "El tiempo de entrenamiento depende de tu hardware, tamaño del dataset, modelo seleccionado y épocas."
    ),
    "train_no_data": "Sin imágenes etiquetadas. Etiqueta imágenes antes de entrenar.",
    "train_few_samples": "La categoría '{cls}' solo tiene {n} muestra(s). Los resultados pueden ser débiles.",
    "train_split": "División Train / Val: {train} / {val} imágenes",
    "train_dataset_ok": "Dataset listo: {n} imágenes, {c} categorías.",

    # ── Package export ───────────────────────────────────────────────────────
    "pkg_title": "Empaquetar para Entrenamiento Externo",
    "pkg_description": (
        "Empaqueta tu dataset en un zip para transferir por USB, Mega, Dropbox,\n"
        "o envíalo al equipo Eyve para que lo entrenen por ti."
    ),
    "pkg_size_small": "Pequeño  (≤ 500 imágenes, CPU rápido)",
    "pkg_size_medium": "Mediano  (500–2000 imágenes, GPU recomendado)",
    "pkg_size_large": "Grande  (2000–5000 imágenes, GPU requerido)",
    "pkg_size_complex": "Complejo  (5000+ imágenes — requiere cotización)",
    "pkg_detected_size": "Tamaño detectado: {label}",
    "pkg_create": "Crear Paquete",
    "pkg_open_folder": "Abrir Carpeta",
    "pkg_done": "Paquete creado: {path}",
    "pkg_note": "Tras empaquetar, envía el zip y el equipo te devolverá el modelo entrenado.",

    # ── Production screen ────────────────────────────────────────────────────
    "prod_title": "Producción / Vista en Vivo",
    "prod_load_model": "Cargar Modelo",
    "prod_model_loaded": "Modelo: {name}",
    "prod_no_model": "Sin modelo cargado. Carga un modelo para iniciar.",
    "prod_start": "Iniciar  [Espacio]",
    "prod_stop": "Detener  [Espacio]",
    "prod_status_ok": "OK",
    "prod_status_nok": "NO OK",
    "prod_status_none": "SIN DETECCIÓN",
    "prod_status_review": "REVISAR",
    "prod_status_error": "ERROR",
    "prod_fps": "FPS: {fps}",
    "prod_conf": "Conf: {conf}%",
    "prod_session": "Sesión: {id}",
    "prod_elapsed": "Transcurrido: {t}",
    "prod_save_nok": "Guardar capturas NO OK",
    "prod_save_logs": "Guardar log de detecciones",
    "prod_class_mismatch": (
        "Advertencia: las categorías del modelo no coinciden con el proyecto.\n"
        "Modelo espera: {model}\nProyecto tiene: {project}"
    ),
    "prod_pause":    "Pausar  [Espacio]",
    "prod_resume":   "Reanudar  [Espacio]",
    "prod_paused":   "⏸  PAUSADO",
    "prod_count_ok": "✓ OK: {n}",
    "prod_count_nok": "✗ NO OK: {n}",

    # ── Status bar ───────────────────────────────────────────────────────────
    "status_no_project": "Sin proyecto abierto.",
    "status_project": "Proyecto: {name}",
    "status_ready": "Listo.",
    "status_camera_ok": "Cámara OK",
    "status_camera_err": "Error de cámara",

    # ── License / trial ──────────────────────────────────────────────────────
    "lic_trial_active": "Periodo de prueba activo — {days} día(s) restante(s).",
    "lic_trial_expired": "Periodo de prueba finalizado.",
    "lic_reminder_title": "Apoya Eyve",
    "lic_reminder_body": (
        "Tu periodo de prueba de 30 días ha terminado.\n\n"
        "Eyve Community sigue siendo completamente funcional.\n"
        "Si te resulta útil, considera apoyar el proyecto:\n\n"
        "Estándar: $45 USD  ·  Estudiante: $25 USD\n\n"
        "eyve.app/licencia"
    ),
    "lic_remind_later": "Recordar más tarde",
    "lic_enter_key": "Ingresar Clave de Licencia",
    "lic_key_valid": "Licencia activada. ¡Gracias!",
    "lic_key_invalid": "Clave de licencia inválida.",
    "lic_watermark": "Eyve Community — Build No Activado",

    # ── Settings popup ───────────────────────────────────────────────────────
    "settings_appearance":   "Apariencia",
    "settings_dark":         "Oscuro",
    "settings_light":        "Claro",
    "settings_license":      "Licencia",
    "settings_status_trial": "Prueba — {days} día(s) restante(s)",
    "settings_status_active":"✓ Activado",
    "settings_status_expired":"Expirado",
    "settings_device_id":    "ID de dispositivo",
    "settings_device_hint":  "Comparte este ID al solicitar tu clave de licencia.",
    "settings_activate":     "Activar licencia",
    "settings_key_hint":     "Pega tu clave aquí…",
    "settings_activate_btn": "Activar",
    "settings_activated_ok": "¡Licencia activada. Gracias!",
    "settings_key_invalid":  "Clave de licencia inválida.",
    "settings_about":        "Acerca de Eyve",

    # ── Dialogs / common ─────────────────────────────────────────────────────
    "ok": "Aceptar",
    "cancel": "Cancelar",
    "yes": "Sí",
    "no": "No",
    "close": "Cerrar",
    "error": "Error",
    "warning": "Advertencia",
    "info": "Información",
    "confirm_delete": "¿Seguro que deseas eliminar '{name}'?",
    "unsaved_changes": "Tienes cambios sin guardar. ¿Descartarlos?",
    "loading": "Cargando…",
    "saving": "Guardando…",
    "done": "Listo",
}
