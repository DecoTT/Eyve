STRINGS = {
    # ── App ──────────────────────────────────────────────────────────────────
    "app_title": "Eyve 2.1",
    "app_subtitle": "Plataforma de Inspección Visual",

    # ── Nav / sidebar ────────────────────────────────────────────────────────
    "nav_home": "Inicio",
    "nav_classes": "Categorías",
    "nav_capture": "Captura",
    "nav_tagging": "Etiquetado",
    "nav_training": "Entrenamiento",
    "nav_production": "Producción",

    "nav_demo": "Demo",

    # ── demo del modulo de conteo ────────────────────────────────────────────
    "nav_cdemo":   "Conteo",
    "cdemo_title": "Contar, de cinco maneras",
    "cdemo_sub":   "El mismo modulo, cinco preguntas distintas",
    "cdemo_paso":  "Metodo {n} de {total}",
    "cdemo_count": "Lo que lleva contado",
    "cdemo_donde": "Donde sirve",
    "cdemo_auto_on":  "Avanzando solo",
    "cdemo_auto_off": "Detenido en este",
    "cdemo_dir":      "{a} en un sentido, {b} en el otro",
    "cdemo_zona":     "{n} dentro del area ahora mismo",
    "cdemo_esperado": "Se esperan entre {lo} y {hi}",

    "cdemo_q_screen":    "¿Cuantas hay AHORA?",
    "cdemo_q_line":      "¿Cuantas han pasado por aqui?",
    "cdemo_q_zone":      "¿Cuantas entraron al area, y cuantas hay dentro?",
    "cdemo_q_appear":    "¿Cuantas piezas nuevas han salido?",
    "cdemo_q_disappear": "¿Cuantas se han ido?",

    "cdemo_como_screen":    "Mira el encuadre y dice cuantas hay en este momento. "
                            "El numero sube y baja; no acumula.",
    "cdemo_como_line":      "Una linea atraviesa la banda. Cada pieza suma una "
                            "sola vez al cruzarla, aunque despues siga ahi.",
    "cdemo_como_zone":      "Un area marcada. Suma al entrar, y aparte dice "
                            "cuantas hay dentro en este momento.",
    "cdemo_como_appear":    "Cada pieza nueva suma una vez, aparezca donde "
                            "aparezca. Las que ya conto no vuelven a contar.",
    "cdemo_como_disappear": "Suma cuando una pieza deja de estar. Se puede "
                            "limitar al lado por el que de verdad se van.",

    "cdemo_uso_screen":    "Cuando lo que importa es el estado de ahora mismo: "
                           "12 tortillas en la charola, 8 pines en el conector, "
                           "4 cajas en la tarima. Con un rango esperado, salirse "
                           "del rango es un defecto.",
    "cdemo_uso_line":      "La banda transportadora. Cada pieza suma una vez al "
                           "cruzar la meta, y el sentido separa lo que entra de "
                           "lo que sale: el neto es la produccion real.",
    "cdemo_uso_zone":      "Una celda de trabajo o una zona de carga. Dice "
                           "cuantas pasaron y cuantas hay dentro en este momento, "
                           "que es lo que avisa de un cuello de botella.",
    "cdemo_uso_appear":    "Piezas que llegan sin un lado fijo: caen, se "
                           "destapan, se imprimen. Cada una suma una sola vez, "
                           "aparezca donde aparezca.",
    "cdemo_uso_disappear": "Piezas que alguien retira o que salen del encuadre. "
                           "Con un borde elegido, cuenta solo las que salen por "
                           "el lado que importa.",

    # ── pantalla de demo (expo) ──────────────────────────────────────────────
    "demo_title":     "Demo en vivo",
    "demo_hint":      "Dibuja un defecto sobre la tela de la derecha. Eyve lo encuentra solo.",
    "demo_side_eyve": "Lo que ve Eyve",
    "demo_side_you":  "Dibuja aqui",
    "demo_tool_rayon":           "Rayon",
    "demo_tool_mancha":          "Mancha",
    "demo_clear":   "Limpiar tela",
    "demo_pause":   "Pausar tela",
    "demo_resume":  "Reanudar",
    "demo_speed":   "Velocidad",
    "demo_found":   "Defectos: {n}",
    "demo_clean":   "LIMPIO",
    "demo_defect":  "DEFECTO",
    "demo_loading": "Cargando el modelo de la demo...",
    "demo_no_model": "Falta el modelo de la demo. Genera uno con: python -m eyve.demo.train",
    "demo_model_error": "No se pudo cargar el modelo: {err}",

    # ── demo ampliada: patron, conteo, modo automatico, material ─────────────
    "demo_yolo_title":    "Lo que le ensenaste",
    "demo_yolo_sub":      "YOLO: nombra el defecto, pero solo los que entreno",
    "demo_pattern_title": "Lo que nunca vio",
    "demo_pattern_sub":   "Patron: sin entrenar; dice donde algo no cuadra",
    "demo_count_title":   "Conteo",
    "demo_count_reset":   "Reiniciar",
    "demo_nothing":       "nada",
    "demo_faults":        "Fallos de impresion (no son clases: los ve el modulo Patron)",
    "demo_fault_fantasma":        "Fantasma",
    "demo_fault_offset":          "Movido",
    "demo_fault_falta_impresion": "Falta tinta",
    "demo_auto":          "Modo auto",
    "demo_auto_stop":     "Tomar control",
    "demo_auto_on":       "Modo automatico - toca para tomar el control",
    "demo_auto_off":      "Tu tienes el control",
    "demo_material":      "Material",
    "demo_calibrating":   "aprendiendo el material...",

    "motif_diamantes": "Diamantes",
    "motif_flores":    "Flores",
    "motif_rayas":     "Rayas",
    "motif_puntos":    "Puntos",

    "weave_sarga":      "Sarga",
    "weave_tafetan":    "Tafetan",
    "weave_sarga_fina": "Sarga fina",
    "weave_canasta":    "Canasta",
    "weave_ninguno":    "Liso",

    # ── modulo Patron en Produccion ──────────────────────────────────────────
    "pat_title":        "Patron (sin clases)",
    "pat_method":       "Metodo",
    "pat_m_periodo":    "Periodicidad",
    "pat_m_layout":     "Layout",
    "pat_m_referencia": "Referencia",
    "pat_help_periodo":    "El material que se repite es su propia referencia. "
                           "Compara cada repeticion con sus vecinas.",
    "pat_help_layout":     "Reconstruye donde deberia ir la tinta y distingue "
                           "tinta de mas de tinta que falta.",
    "pat_help_referencia": "Aprende de material bueno. Para piezas que no se "
                           "repiten.",
    "pat_sens":         "Sensibilidad",
    "pat_calibrate":    "Calibrar con material bueno",
    "pat_calibrating":  "Aprendiendo... {n}",
    "pat_calibrated":   "Calibrado ({n} frames)",
    "pat_uncalibrated": "Sin calibrar - ensenale material bueno primero",

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
    "tag_no_frame": "No hay imagen para etiquetar — inicia la cámara o el video.",
    "tag_draw_box_first": "Dibuja al menos una caja antes de guardar.",
    "tag_vid_paused": "⏸ Pausado — dibuja cajas y presiona Guardar",
    "tag_frozen_hint": "❚❚ Congelado — dibuja cajas y presiona Guardar [S]",
    "tag_grab_btn": "📷 Congelar y Etiquetar  [G]",
    "prod_select_video": "Seleccionar Video…",
    "prod_video_missing": "Selecciona un archivo de video primero.",
    "prod_modules": "Módulos (Pro)",
    "prod_mod_class": "Clase",
    "prod_mod_method": "Método",
    "prod_mod_ref": "Referencia",
    "prod_mod_need_class": "Selecciona la clase a inspeccionar primero.",
    "prod_mod_all": "(todas)",
    "prod_draw_line": "✏ Dibujar meta",
    "prod_line_hint": "Arrastra sobre el video para dibujar la meta",
    "prod_arc": "Arco",
    # ── counting module (5 metodos) ──────────────────────────────────────────
    "count_m_screen":     "En pantalla",
    "count_m_line":       "Cruce de meta",
    "count_m_zone":       "Zona (area)",
    "count_m_appear":     "Al aparecer",
    "count_m_disappear":  "Al desaparecer",

    "count_help_screen":    "Cuenta lo que hay AHORA en el encuadre. No acumula. "
                            "Con un rango esperado, un conteo fuera de rango marca NOT OK.",
    "count_help_line":      "Cada instancia cuenta una vez al cruzar la meta. "
                            "Dibuja la linea sobre el video.",
    "count_help_zone":      "Cuenta al entrar al area, al salir, o ambas, y reporta "
                            "cuantas hay dentro. Dibuja el area sobre el video.",
    "count_help_appear":    "Cuenta cada instancia nueva una sola vez, aparezca donde "
                            "aparezca. Para piezas sin un lado fijo de llegada.",
    "count_help_disappear": "Cuenta cada instancia que se va del encuadre. Con un borde "
                            "elegido, solo las que salen por ese lado.",

    "count_sense":        "Sentido",
    "count_counts":       "Cuenta",
    "count_exit_by":      "Sale por",
    "count_expected":     "Esperado",

    "count_d_both":       "Ambos",
    "count_d_fwd":        "Un sentido",
    "count_d_rev":        "Sentido inverso",

    "count_z_enter":      "Al entrar",
    "count_z_exit":       "Al salir",
    "count_z_both":       "Entrar y salir",

    "count_e_any":        "Cualquier lado",
    "count_e_left":       "Izquierda",
    "count_e_right":      "Derecha",
    "count_e_top":        "Arriba",
    "count_e_bottom":     "Abajo",

    "count_persistence":  "Persistencia de instancia",
    "count_persistence_hint": "Tolerancia = frames que una pieza puede desaparecer "
                              "sin perder su ID (sube si el detector parpadea). "
                              "IoU = que tan parecidas deben ser dos cajas para ser "
                              "la misma pieza (baja si se mueven rapido).",
    "count_tolerance":    "Tolerancia",
    "count_confirm":      "Confirmar",
    "count_iou":          "IoU minima",
    "count_conf_min":     "Conf. minima",

    "prod_draw_zone":     "\u270f Dibujar zona",
    "prod_zone_hint":     "Arrastra sobre el video para dibujar el area",


    # ── i18n pass (antes hardcoded) ──────────────────────────────────────────
    "cam_loading": "⟳  Cargando cámaras…",
    "cam_refreshing": "⟳  Actualizando…",
    "cam_none": "Sin cámaras detectadas",
    "cap_refresh_cams": "⟳ Actualizar cámaras",
    "cap_resolution": "Resolución",
    "cap_record": "⏺  Grabar",
    "cap_stop_save": "⏹  Detener y Guardar",
    "cap_open_folder": "📂 Abrir carpeta",
    "cap_opening_cam": "Abriendo cámara…",
    "cap_no_frame_yet": "Aún no hay imagen — espera el video en vivo",
    "cap_rec_file_error": "No se pudo crear el archivo de video",
    "cap_recording": "Grabando…",
    "tag_source": "Fuente",
    "tag_source_images": "Imágenes",
    "tag_start": "▶ Iniciar",
    "tag_stop": "■ Parar",
    "tag_select_video_first": "Selecciona un video primero.",
    "tag_video_open_error": "No se pudo abrir el video.",
    "tag_live": "● En vivo",
    "tag_open_project_first": "Abre un proyecto primero.",
    "tag_define_classes_first": "Define categorías primero.",
    "prod_downloading_generic": "⟳  Descargando yolov8n.pt…",
    "prod_loading_model": "⟳  Cargando…",
    "prod_model_error": "Error cargando modelo: {err}",
    "prod_cam_retry": "⚠  Cámara no disponible — reintentando…",
    "prod_connecting_cam": "Conectando cámara…",
    "set_title": "⚙  Ajustes",
    "set_perf": "Rendimiento",
    "set_tagline": "Plataforma de inspección visual local",
    "set_models_title": "Modelos y Modo Offline",
    "set_offline": "Modo offline",
    "set_manage_models": "⬇  Gestionar modelos descargados",
    "set_no_models": "⚠  Sin modelos descargados. Abre el gestor para descargar uno.",
    "mm_title": "Descargar Modelos Base YOLO",
    "mm_stored_in": "Los modelos se guardan en  {path}",
    "mm_model": "Modelo:",
    "mm_download": "⬇  Descargar",
    "mm_close": "Cerrar",
    "mm_downloaded": "✓  Descargado",
    "mm_ready_offline": "✓  Ya descargado — listo para uso offline",
    "mm_not_downloaded": "✗  No descargado",
    "mm_downloading": "⏳ Descargando…",
    "mm_retry": "⬇  Reintentar",
    "prod_generic_model": "⚠  yolov8n (genérico — demo)\nEste proyecto aún no tiene un modelo entrenado.",
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
    "train_no_new_data": "No hay etiquetas nuevas desde el último entrenamiento. Etiqueta más imágenes o desmarca la casilla.",
    "train_model_project": "★ Modelo del proyecto (best.pt) — seguir mejorando",
    "train_model_browse": "Elegir archivo .pt…",
    "train_only_new_short": "Solo etiquetas nuevas desde el último entrenamiento",
    "train_only_new": "Solo etiquetas nuevas desde el último entrenamiento ({date})",
    "train_only_new_warn": "⚠ Entrenar solo con lo nuevo puede hacer que el modelo olvide lo anterior. Úsalo cuando el material previo ya no aplica (cambió la pieza, la cámara o la luz). Para mejorar el mismo modelo, lo normal es entrenar con todo.",
    "train_dataset_ok_new": "Dataset listo: {n} imágenes nuevas (de {total} etiquetadas), {c} categorías.",
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

    # ── License (sbcsuite · JWT · por honor) ─────────────────────────────────
    "lic_title": "Licencia de Eyve",
    "lic_honor": "Por honor: nada bloquea. Eyve corre sin internet hasta la fecha que trae la llave. El servidor solo registra tus equipos (hasta 3) y entrega renovaciones.",
    "lic_titular": "Titular",
    "lic_vence": "Vence el",
    "lic_equipos": "Equipos activos",
    "lic_ultimo_checkin": "Último check-in",
    "lic_never": "nunca",
    "lic_paste_label": "Pega tu llave (la recibiste en la página de gracias y por correo)",
    "lic_btn_save": "Guardar llave",
    "lic_btn_check": "Comprobar ahora",
    "lic_btn_remove": "Quitar",
    "lic_btn_account": "Liberar equipos en mi cuenta ↗",
    "lic_btn_buy": "Conseguir una licencia ↗",
    "lic_device": "ID de este equipo",
    "lic_no_key": "Sin llave: Eyve Free (detección, conteo y log). Pega una llave para subir de nivel.",
    "lic_invalid_stored": "La llave guardada no es válida (firma incorrecta). Eyve sigue en Free.",
    "lic_key_invalid": "Llave inválida: la firma no corresponde. Revisa que la copiaste completa.",
    "lic_key_saved": "Llave guardada: Eyve {level}. Registrando este equipo…",
    "lic_paste_first": "Pega la llave primero.",
    "lic_expired_tag": "vencida",
    "lic_expired_notice": "Tu licencia venció el {date}. Eyve sigue funcionando con este nivel; renueva cuando puedas.",
    "lic_pending": "Este equipo aún no se ha registrado en el servidor (sin red). Se reintenta al siguiente arranque; Eyve funciona igual.",
    "lic_no_slots": "Esta licencia ya está en 3 equipos. Eyve sigue funcionando; libera uno desde tu cuenta para registrar este.",
    "lic_devices_list": "Equipos",
    "lic_revoked": "El servidor reporta esta licencia como revocada. Escríbenos a contacto@sbcgroup.com.mx.",
    "lic_student_pending": "Licencia de Estudiante pendiente de verificación. Funciona igual; responde al correo con tu credencial.",
    "lic_not_registered": "La llave es auténtica pero el servidor no la encuentra. Escríbenos a contacto@sbcgroup.com.mx.",
    "lic_checking": "Comprobando…",
    "lic_offline": "Sin conexión. Eyve funciona igual; se reintenta después.",
    "lic_server_ok": "Equipo registrado. {n} de {max} equipos activos.",
    "lic_server_err": "El servidor no pudo registrar el equipo",
    "lic_removed": "Llave quitada. Eyve Free.",
    "lic_status_expired": "vencida",
    "lic_pro_required": "Los módulos de check de SBC requieren licencia Pro. Eyve Free/Normal: detección, conteo y log; tus propios checks en Python.",
    "lic_prod_notice": "En producción se requiere Pro. Puedes correr aquí con {level}; la licencia no lo ampara.",

    # ── Settings popup ───────────────────────────────────────────────────────
    "settings_appearance":   "Apariencia",
    "settings_dark":         "Oscuro",
    "settings_light":        "Claro",
    "settings_license":      "Licencia",
    "settings_license_btn":  "Licencia…",
    # ── actualizaciones ──────────────────────────────────────────────────────
    "settings_updates":    "Actualizaciones",
    "settings_version":    "Version instalada: {v}",
    "settings_update_btn": "Buscar...",
    "settings_update_auto": "Buscar actualizaciones al iniciar",

    "upd_title":     "Actualizar Eyve",
    "upd_current":   "Tienes la version {v}",
    "upd_from_to":   "De la {a} a la {b}",
    "upd_checking":  "Buscando actualizaciones...",
    "upd_offline":   "No se pudo consultar. Revisa la conexion e intenta de nuevo.",
    "upd_uptodate":  "Ya tienes la ultima version.",
    "upd_available": "Hay una version nueva: {v}",
    "upd_safe":      "Tus proyectos, tus modelos entrenados y tu licencia no se "
                     "tocan. Solo se reemplaza el programa, y se guarda una copia "
                     "de la version anterior por si algo sale mal.",
    "upd_install":   "Actualizar ahora",
    "upd_later":     "Ahora no",
    "upd_working":   "Actualizando...",
    "upd_downloading": "Descargando y verificando...",
    "upd_done":      "Listo: Eyve {v} instalado.",
    "upd_deps_changed": "Esta version necesita librerias nuevas. Cierra Eyve y "
                        "ejecuta setup.bat una vez antes de volver a abrirlo.",
    "upd_failed":    "No se pudo actualizar: {err}",
    "upd_retry":     "Reintentar",
    "upd_restart":   "Reiniciar Eyve",
    "upd_banner":    "Version {v} disponible",

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
