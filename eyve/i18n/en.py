STRINGS = {
    # ── App ──────────────────────────────────────────────────────────────────
    "app_title": "Eyve 2.1 Beta",
    "app_subtitle": "Visual Inspection Platform",

    # ── Nav / sidebar ────────────────────────────────────────────────────────
    "nav_home": "Home",
    "nav_classes": "Classes",
    "nav_capture": "Capture",
    "nav_tagging": "Tagging",
    "nav_training": "Training",
    "nav_production": "Production",

    # ── Home screen ──────────────────────────────────────────────────────────
    "home_welcome": "Welcome to Eyve",
    "home_tagline": "Create, train and run local visual inspection projects.",
    "home_new_project": "New Project",
    "home_open_project": "Open Project",
    "home_recent": "Recent Projects",
    "home_no_recent": "No recent projects.",
    "home_test_camera": "Test Camera",
    "home_help": "Help / Docs",

    # ── Project creation ─────────────────────────────────────────────────────
    "proj_create_title": "Create New Project",
    "proj_name": "Project Name",
    "proj_name_hint": "e.g. potato_chips",
    "proj_folder": "Save Location",
    "proj_folder_browse": "Browse…",
    "proj_camera": "Camera Source",
    "proj_target": "Inspection Target",
    "proj_target_hint": "What are you inspecting? e.g. Potato chips, PCB, Parts",
    "proj_create_btn": "Create Project",
    "proj_cancel": "Cancel",
    "proj_name_required": "Project name is required.",
    "proj_folder_required": "Save location is required.",
    "proj_name_invalid": "Project name may only contain letters, numbers and underscores.",
    "proj_exists": "A project with that name already exists in that folder.",
    "proj_created_ok": "Project created successfully.",

    # ── Project dashboard ────────────────────────────────────────────────────
    "dash_project": "Project",
    "dash_classes": "Classes",
    "dash_images": "Images",
    "dash_labeled": "Labeled",
    "dash_model": "Model",
    "dash_no_model": "No model loaded",
    "dash_status": "Status",

    # ── Classes screen ───────────────────────────────────────────────────────
    "cls_title": "Define Classes",
    "cls_subtitle": (
        "Classes define what your model will detect. "
        "Once training starts, classes cannot be changed without creating a new project."
    ),
    "cls_warning_permanent": (
        "Important: The trained model is permanently tied to these classes.\n"
        "If you need to detect a new category later, you must create a new project."
    ),
    "cls_add": "Add Class",
    "cls_name_hint": "Class name…",
    "cls_type_ok": "OK",
    "cls_type_nok": "NOT OK",
    "cls_type_ignore": "Ignore",
    "cls_delete": "Delete",
    "cls_rename": "Rename",
    "cls_save": "Save Classes",
    "cls_empty": "No classes defined yet. Add at least 2 classes.",
    "cls_need_ok": "At least one class must be marked as OK.",
    "cls_need_nok": "At least one class must be marked as NOT OK.",
    "cls_name_dup": "A class with that name already exists.",
    "cls_name_empty": "Class name cannot be empty.",
    "cls_count": "{n} class(es) defined",
    "cls_in_use": "Cannot delete class '{name}' — it has {n} tagged image(s).",
    "cls_saved_ok": "Classes saved.",

    # ── Capture screen ───────────────────────────────────────────────────────
    "cap_title": "Capture Images",
    "cap_source_camera": "Camera",
    "cap_source_video": "Video File",
    "cap_select_video": "Select Video…",
    "cap_camera_id": "Camera Index",
    "cap_start": "Start Preview",
    "cap_stop": "Stop",
    "cap_capture": "Capture Frame  [C]",
    "cap_session_class": "Assign to Class",
    "cap_no_class": "— select class —",
    "cap_captured": "Captured: {n}",
    "cap_no_camera": "Camera not found. Check the camera index.",
    "cap_no_classes": "Define classes before capturing.",
    "cap_no_class_selected": "Select a class before capturing.",
    "cap_saved": "Frame saved to project.",

    # ── Tagging screen ───────────────────────────────────────────────────────
    "tag_title": "Tagging",
    "tag_image": "Image {current} / {total}",
    "tag_class": "Class",
    "tag_prev": "← Prev  [P]",
    "tag_next": "Next →  [N]",
    "tag_save": "Save  [S]",
    "tag_delete_box": "Delete Box  [Del]",
    "tag_clear": "Clear All Boxes",
    "tag_skip": "Skip Image",
    "tag_no_images": "No captured images found. Go to Capture first.",
    "tag_shortcuts": "Draw box: drag mouse | Keys: 1-9 = select class · S = save · N/P = next/prev · Del = delete box",
    "tag_saved": "Label saved.",
    "tag_class_stats": "Tagged per class:",
    "tag_progress": "{labeled} / {total} images labeled",

    # ── Training screen ──────────────────────────────────────────────────────
    "train_title": "Training",
    "train_local": "Train Locally",
    "train_package": "Package for External Training",
    "train_model_base": "Base Model",
    "train_epochs": "Epochs",
    "train_imgsz": "Image Size",
    "train_batch": "Batch Size",
    "train_device": "Device",
    "train_device_auto": "Auto (GPU if available)",
    "train_device_cpu": "CPU only",
    "train_device_gpu": "GPU (CUDA)",
    "train_conf_default": "Default Confidence",
    "train_start": "Start Training",
    "train_stop": "Stop Training",
    "train_status_idle": "Ready to train.",
    "train_status_preparing": "Preparing dataset…",
    "train_status_training": "Training… epoch {epoch}/{total}",
    "train_status_done": "Training complete. Model saved.",
    "train_status_error": "Training failed: {error}",
    "train_eta": "ETA: {time}",
    "train_elapsed": "Elapsed: {time}",
    "train_best_model": "Best model: {path}",
    "train_hardware_note": (
        "Training time depends on your hardware, dataset size, selected model and epochs."
    ),
    "train_no_data": "No labeled images found. Tag images before training.",
    "train_few_samples": "Class '{cls}' has only {n} sample(s). Results may be weak.",
    "train_split": "Train / Val split: {train} / {val} images",
    "train_dataset_ok": "Dataset ready: {n} images, {c} classes.",

    # ── Package export ───────────────────────────────────────────────────────
    "pkg_title": "Package for External Training",
    "pkg_description": (
        "Pack your dataset into a zip file you can transfer via USB, Mega, Dropbox,\n"
        "or send to the Eyve team for training."
    ),
    "pkg_size_small": "Small  (≤ 500 images, fast CPU training)",
    "pkg_size_medium": "Medium  (500–2000 images, GPU recommended)",
    "pkg_size_large": "Large  (2000–5000 images, GPU required)",
    "pkg_size_complex": "Complex  (5000+ images — requires quote)",
    "pkg_detected_size": "Detected size: {label}",
    "pkg_create": "Create Package",
    "pkg_open_folder": "Open Output Folder",
    "pkg_done": "Package created: {path}",
    "pkg_note": "After packaging, send the zip file and the team will return your trained model.",

    # ── Production screen ────────────────────────────────────────────────────
    "prod_title": "Production / Live Feed",
    "prod_load_model": "Load Model",
    "prod_model_loaded": "Model: {name}",
    "prod_no_model": "No model loaded. Load a model to start.",
    "prod_start": "Start  [Space]",
    "prod_stop": "Stop  [Space]",
    "prod_status_ok": "OK",
    "prod_status_nok": "NOT OK",
    "prod_status_none": "NO DETECTION",
    "prod_status_review": "REVIEW",
    "prod_status_error": "ERROR",
    "prod_fps": "FPS: {fps}",
    "prod_conf": "Conf: {conf}%",
    "prod_session": "Session: {id}",
    "prod_elapsed": "Elapsed: {t}",
    "prod_save_nok": "Save NOK screenshots",
    "prod_save_logs": "Save detection log",
    "prod_class_mismatch": (
        "Warning: model classes do not match project classes.\n"
        "Model expects: {model}\nProject has: {project}"
    ),
    "prod_pause":    "Pause  [Space]",
    "prod_resume":   "Resume  [Space]",
    "prod_paused":   "⏸  PAUSED",
    "prod_count_ok": "✓ OK: {n}",
    "prod_count_nok": "✗ NOK: {n}",

    # ── Status bar ───────────────────────────────────────────────────────────
    "status_no_project": "No project open.",
    "status_project": "Project: {name}",
    "status_ready": "Ready.",
    "status_camera_ok": "Camera OK",
    "status_camera_err": "Camera error",

    # ── License / trial ──────────────────────────────────────────────────────
    "lic_trial_active": "Trial active — {days} day(s) remaining.",
    "lic_trial_expired": "Trial expired.",
    "lic_reminder_title": "Support Eyve",
    "lic_reminder_body": (
        "Your 30-day free trial has ended.\n\n"
        "Eyve Community remains fully functional.\n"
        "If you find it useful, consider supporting the project:\n\n"
        "Standard: $45 USD  ·  Student: $25 USD\n\n"
        "eyve.app/license"
    ),
    "lic_remind_later": "Remind me later",
    "lic_enter_key": "Enter License Key",
    "lic_key_valid": "License activated. Thank you!",
    "lic_key_invalid": "Invalid license key.",
    "lic_watermark": "Eyve Community — Unactivated Build",

    # ── Settings popup ───────────────────────────────────────────────────────
    "settings_appearance":   "Appearance",
    "settings_dark":         "Dark",
    "settings_light":        "Light",
    "settings_license":      "License",
    "settings_status_trial": "Trial — {days} day(s) remaining",
    "settings_status_active":"✓ Activated",
    "settings_status_expired":"Expired",
    "settings_device_id":    "Device ID",
    "settings_device_hint":  "Share this ID when requesting your license key.",
    "settings_activate":     "Activate License",
    "settings_key_hint":     "Paste your key here…",
    "settings_activate_btn": "Activate",
    "settings_activated_ok": "License activated. Thank you!",
    "settings_key_invalid":  "Invalid license key.",
    "settings_about":        "About Eyve",

    # ── Dialogs / common ─────────────────────────────────────────────────────
    "ok": "OK",
    "cancel": "Cancel",
    "yes": "Yes",
    "no": "No",
    "close": "Close",
    "error": "Error",
    "warning": "Warning",
    "info": "Information",
    "confirm_delete": "Are you sure you want to delete '{name}'?",
    "unsaved_changes": "You have unsaved changes. Discard them?",
    "loading": "Loading…",
    "saving": "Saving…",
    "done": "Done",
}
