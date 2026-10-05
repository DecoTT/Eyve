STRINGS = {
    # ── App ──────────────────────────────────────────────────────────────────
    "app_title": "Eyve 2.1",
    "app_subtitle": "Visual Inspection Platform",

    # ── Nav / sidebar ────────────────────────────────────────────────────────
    "nav_home": "Home",
    "nav_classes": "Classes",
    "nav_capture": "Capture",
    "nav_tagging": "Tagging",
    "nav_training": "Training",
    "nav_production": "Production",
    "nav_demo": "Demo",

    # ── counting module demo ─────────────────────────────────────────────────
    "nav_cdemo":   "Counting",
    "cdemo_title": "Counting, five ways",
    "cdemo_sub":   "One module, five different questions",
    "cdemo_idle_back": "Nobody touched it for a while: the tour resumes",
    "cdemo_paso":  "Method {n} of {total}",
    "cdemo_count": "Counted so far",
    "cdemo_donde": "Where it fits",
    "cdemo_auto_on":  "Advancing on its own",
    "cdemo_auto_off": "Holding on this one",
    "cdemo_dir":      "{a} one way, {b} the other",
    "cdemo_zona":     "{n} inside the area right now",
    "cdemo_esperado": "Expecting between {lo} and {hi}",

    "cdemo_q_screen":    "How many are there RIGHT NOW?",
    "cdemo_q_line":      "How many have gone past here?",
    "cdemo_q_zone":      "How many entered the area, and how many are inside?",
    "cdemo_q_appear":    "How many new parts have shown up?",
    "cdemo_q_disappear": "How many have left?",

    "cdemo_como_screen":    "Looks at the frame and says how many are there "
                            "right now. The number goes up and down; it does "
                            "not accumulate.",
    "cdemo_como_line":      "A line across the belt. Each part counts once as "
                            "it crosses, even if it stays in view afterwards.",
    "cdemo_como_zone":      "A marked area. It counts on entry, and separately "
                            "reports how many are inside right now.",
    "cdemo_como_appear":    "Each new part counts once, wherever it shows up. "
                            "The ones already counted never count again.",
    "cdemo_como_disappear": "Counts when a part stops being there. It can be "
                            "limited to the side they actually leave by.",

    "cdemo_uso_screen":    "When what matters is the state right now: 12 "
                           "tortillas on the tray, 8 pins in the connector, 4 "
                           "boxes on the pallet. With an expected range, falling "
                           "outside it is a defect.",
    "cdemo_uso_line":      "The conveyor. Each part counts once as it crosses "
                           "the finish line, and the sense separates what comes "
                           "in from what goes out: the net is real output.",
    "cdemo_uso_zone":      "A work cell or a loading area. It tells you how many "
                           "went through and how many are inside right now, which "
                           "is what warns you about a bottleneck.",
    "cdemo_uso_appear":    "Parts that arrive with no fixed side: they drop, get "
                           "uncovered, get printed. Each counts once, wherever it "
                           "shows up.",
    "cdemo_uso_disappear": "Parts someone removes, or that leave the frame. With "
                           "an edge selected, it counts only the ones leaving "
                           "through the side that matters.",

    # ── kiosk mode (full screen for the trade show booth) ────────────────────
    "kiosk_enter_hint": "F11  full screen",
    "kiosk_exit_hint":  "Full screen - press F11 or Esc to exit",

    # ── demo screen (trade show) ─────────────────────────────────────────────
    "demo_title":     "Live demo",
    "demo_hint":      "Draw a defect on the fabric at right. Eyve finds it on its own.",
    "demo_side_eyve": "What Eyve sees",
    "demo_side_you":  "Draw here",
    "demo_tool_rayon":           "Scratch",
    "demo_tool_mancha":          "Stain",
    "demo_clear":   "Clean fabric",
    "demo_restart":      "START OVER",
    "demo_restart_done": "Fresh start: clean fabric, counters at zero, recalibrating",
    "demo_pause":   "Pause fabric",
    "demo_resume":  "Resume",
    "demo_speed":   "Speed",
    "demo_found":   "Defects: {n}",
    "demo_clean":   "CLEAN",
    "demo_defect":  "DEFECT",
    "demo_loading": "Loading the demo model...",
    "demo_no_model": "Demo model missing. Build one with: python -m eyve.demo.train",
    "demo_model_error": "Could not load the model: {err}",


    # ── extended demo: pattern, counting, auto mode, material ────────────────
    "demo_yolo_title":    "What you taught it",
    "demo_yolo_sub":      "YOLO: names the defect, but only the ones it trained on",
    "demo_pattern_title": "What it never saw",
    "demo_pattern_sub":   "Pattern: untrained; says where something is off",
    "demo_count_title":   "Counting",
    "demo_count_reset":   "Reset",
    "demo_nothing":       "nothing",
    "demo_faults":        "Print faults (not classes: the Pattern module finds these)",
    "demo_fault_fantasma":        "Ghosting",
    "demo_fault_offset":          "Misregister",
    "demo_fault_falta_impresion": "Ink starved",
    "demo_idle_back":     "Nobody touched it for a while: clean fabric, auto mode",
    "demo_auto":          "Auto mode",
    "demo_auto_stop":     "Take control",
    "demo_auto_on":       "Auto mode - touch to take control",
    "demo_auto_off":      "You are in control",
    "demo_material":      "Material",
    "demo_calibrating":   "learning the material...",

    "motif_diamantes": "Diamonds",
    "motif_flores":    "Flowers",
    "motif_rayas":     "Stripes",
    "motif_puntos":    "Dots",

    "weave_sarga":      "Twill",
    "weave_tafetan":    "Plain",
    "weave_sarga_fina": "Fine twill",
    "weave_canasta":    "Basket",
    "weave_ninguno":    "Flat",

    # ── Pattern module in Production ─────────────────────────────────────────
    "pat_title":        "Pattern (no classes)",
    "pat_method":       "Method",
    "pat_m_periodo":    "Periodicity",
    "pat_m_layout":     "Layout",
    "pat_m_referencia": "Reference",
    "pat_help_periodo":    "Repeating material is its own reference. Compares "
                           "each repeat against its neighbours.",
    "pat_help_layout":     "Rebuilds where the ink should be and tells excess "
                           "ink from missing ink.",
    "pat_help_referencia": "Learns from good material. For parts that do not "
                           "repeat.",
    "pat_sens":         "Sensitivity",
    "pat_calibrate":    "Calibrate on good material",
    "pat_calibrating":  "Learning... {n}",
    "pat_calibrated":   "Calibrated ({n} frames)",
    "pat_uncalibrated": "Not calibrated - show it good material first",

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
    "tag_no_frame": "No frame to tag — start the camera or video first.",
    "tag_draw_box_first": "Draw at least one box before saving.",
    "tag_vid_paused": "⏸ Paused — draw boxes and press Save",
    "tag_frozen_hint": "❚❚ Frozen — draw boxes and press Save [S]",
    "tag_grab_btn": "📷 Freeze & Tag  [G]",
    "prod_select_video": "Select Video…",
    "prod_video_missing": "Select a video file first.",
    "prod_modules": "Modules (Pro)",
    "prod_mod_class": "Class",
    "prod_mod_method": "Method",
    "prod_mod_ref": "Reference",
    "prod_mod_need_class": "Select the class to inspect first.",
    "prod_mod_all": "(all)",
    "prod_draw_line": "✏ Draw finish line",
    "prod_line_hint": "Drag on the video to draw the finish line",
    "prod_arc": "Arc",
    # ── counting module (5 methods) ──────────────────────────────────────────
    "count_m_screen":     "On screen",
    "count_m_line":       "Line crossing",
    "count_m_zone":       "Zone (area)",
    "count_m_appear":     "On appear",
    "count_m_disappear":  "On disappear",

    "count_help_screen":    "Counts what is in frame RIGHT NOW. Does not accumulate. "
                            "With an expected range, a count outside it flags NOT OK.",
    "count_help_line":      "Each instance counts once as it crosses the finish line. "
                            "Draw the line over the video.",
    "count_help_zone":      "Counts on entering the area, on leaving, or both, and "
                            "reports how many are inside. Draw the area over the video.",
    "count_help_appear":    "Counts every new instance exactly once, wherever it shows "
                            "up. For parts with no fixed side of arrival.",
    "count_help_disappear": "Counts every instance that leaves the frame. With an edge "
                            "selected, only the ones leaving through that side.",

    "count_sense":        "Sense",
    "count_counts":       "Counts",
    "count_exit_by":      "Leaves by",
    "count_expected":     "Expected",

    "count_d_both":       "Both",
    "count_d_fwd":        "One way",
    "count_d_rev":        "Reverse",

    "count_z_enter":      "On enter",
    "count_z_exit":       "On exit",
    "count_z_both":       "Enter and exit",

    "count_e_any":        "Any side",
    "count_e_left":       "Left",
    "count_e_right":      "Right",
    "count_e_top":        "Top",
    "count_e_bottom":     "Bottom",

    "count_persistence":  "Instance persistence",
    "count_persistence_hint": "Tolerance = frames a part may vanish without losing its "
                              "ID (raise it if the detector flickers). IoU = how alike "
                              "two boxes must be to count as the same part (lower it "
                              "when they move fast).",
    "count_tolerance":    "Tolerance",
    "count_confirm":      "Confirm",
    "count_iou":          "Min IoU",
    "count_conf_min":     "Min conf.",

    "prod_draw_zone":     "\u270f Draw zone",
    "prod_zone_hint":     "Drag on the video to draw the area",


    # ── i18n pass (previously hardcoded) ─────────────────────────────────────
    "cam_loading": "⟳  Loading cameras…",
    "cam_refreshing": "⟳  Refreshing…",
    "cam_none": "No cameras found",
    "cap_refresh_cams": "⟳ Refresh cameras",
    "cap_resolution": "Resolution",
    "cap_record": "⏺  Record",
    "cap_stop_save": "⏹  Stop & Save",
    "cap_open_folder": "📂 Open folder",
    "cap_opening_cam": "Opening camera…",
    "cap_no_frame_yet": "No frame yet — wait for the live feed",
    "cap_rec_file_error": "Could not create the video file",
    "cap_recording": "Recording…",
    "tag_source": "Source",
    "tag_source_images": "Images",
    "tag_start": "▶ Start",
    "tag_stop": "■ Stop",
    "tag_select_video_first": "Select a video first.",
    "tag_video_open_error": "Could not open the video.",
    "tag_live": "● Live",
    "tag_open_project_first": "Open a project first.",
    "tag_define_classes_first": "Define classes first.",
    "prod_downloading_generic": "⟳  Downloading yolov8n.pt…",
    "prod_loading_model": "⟳  Loading…",
    "prod_model_error": "Model load error: {err}",
    "prod_cam_retry": "⚠  Camera unavailable — retrying…",
    "prod_connecting_cam": "Connecting camera…",
    "set_title": "⚙  Settings",
    "set_perf": "Performance",
    "set_tagline": "Local-first visual inspection platform",
    "set_models_title": "Models & Offline Mode",
    "set_offline": "Offline mode",
    "set_manage_models": "⬇  Manage downloaded models",
    "set_no_models": "⚠  No models downloaded yet. Open the manager to download one.",
    "mm_title": "Download YOLO Base Models",
    "mm_stored_in": "Models are stored in  {path}",
    "mm_model": "Model:",
    "mm_download": "⬇  Download",
    "mm_close": "Close",
    "mm_downloaded": "✓  Downloaded",
    "mm_ready_offline": "✓  Already downloaded — ready for offline use",
    "mm_not_downloaded": "✗  Not downloaded",
    "mm_downloading": "⏳ Downloading…",
    "mm_retry": "⬇  Retry",
    "prod_generic_model": "⚠  yolov8n (generic — demo)\nThis project has no trained model yet.",
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
    "train_no_new_data": "No new labels since the last training. Label more images or untick the box.",
    "train_model_project": "★ Project model (best.pt) — keep improving",
    "train_model_browse": "Choose .pt file…",
    "train_only_new_short": "Only labels new since last training",
    "train_only_new": "Only labels new since last training ({date})",
    "train_only_new_warn": "⚠ Training on new material only can make the model forget what it learned before. Use it when the old material no longer applies (part, camera or lighting changed). To improve the same model, training on everything is the norm.",
    "train_dataset_ok_new": "Dataset ready: {n} new images (of {total} labeled), {c} classes.",
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

    # ── License (sbcsuite · JWT · honor-based) ───────────────────────────────
    "lic_title": "Eyve License",
    "lic_honor": "Honor-based: nothing locks you out. Eyve runs offline until the date inside the key. The server only registers your machines (up to 3) and delivers renewals.",
    "lic_titular": "Licensed to",
    "lic_vence": "Expires on",
    "lic_equipos": "Active machines",
    "lic_ultimo_checkin": "Last check-in",
    "lic_never": "never",
    "lic_paste_label": "Paste your key (you got it on the thank-you page and by email)",
    "lic_btn_save": "Save key",
    "lic_btn_check": "Check now",
    "lic_btn_remove": "Remove",
    "lic_btn_account": "Release machines in my account ↗",
    "lic_btn_buy": "Get a license ↗",
    "lic_device": "This machine's ID",
    "lic_no_key": "No key: Eyve Free (detection, counting and logging). Paste a key to upgrade.",
    "lic_invalid_stored": "The stored key is not valid (bad signature). Eyve stays on Free.",
    "lic_key_invalid": "Invalid key: the signature does not match. Make sure you copied it whole.",
    "lic_key_saved": "Key saved: Eyve {level}. Registering this machine…",
    "lic_paste_first": "Paste the key first.",
    "lic_expired_tag": "expired",
    "lic_expired_notice": "Your license expired on {date}. Eyve keeps working at this tier; renew when you can.",
    "lic_pending": "This machine is not registered on the server yet (no network). It retries on next start; Eyve works the same.",
    "lic_no_slots": "This license is already on 3 machines. Eyve keeps working; release one from your account to register this one.",
    "lic_devices_list": "Machines",
    "lic_revoked": "The server reports this license as revoked. Write to contacto@sbcgroup.com.mx.",
    "lic_student_pending": "Student license pending verification. Works the same; reply to the email with your credential.",
    "lic_not_registered": "The key is authentic but the server cannot find it. Write to contacto@sbcgroup.com.mx.",
    "lic_checking": "Checking…",
    "lic_offline": "No connection. Eyve works the same; it will retry later.",
    "lic_server_ok": "Machine registered. {n} of {max} machines active.",
    "lic_server_err": "The server could not register the machine",
    "lic_removed": "Key removed. Eyve Free.",
    "lic_status_expired": "expired",
    "lic_pro_required": "SBC check modules require a Pro license. Eyve Free/Normal: detection, counting and logging; your own checks in Python.",
    "lic_prod_notice": "Production requires Pro. You can run here with {level}; the license does not cover it.",

    # ── Settings popup ───────────────────────────────────────────────────────
    "settings_appearance":   "Appearance",
    "settings_dark":         "Dark",
    "settings_light":        "Light",
    "settings_license":      "License",
    "settings_license_btn":  "License…",
    # ── updates ──────────────────────────────────────────────────────────────
    "settings_updates":    "Updates",
    "settings_version":    "Installed version: {v}",
    "settings_update_btn": "Check...",
    "settings_update_auto": "Check for updates on startup",

    "upd_title":     "Update Eyve",
    "upd_current":   "You have version {v}",
    "upd_from_to":   "From {a} to {b}",
    "upd_checking":  "Checking for updates...",
    "upd_offline":   "Could not check. Check your connection and try again.",
    "upd_uptodate":  "You already have the latest version.",
    "upd_available": "A new version is available: {v}",
    "upd_safe":      "Your projects, your trained models and your license are left "
                     "alone. Only the program is replaced, and a copy of the "
                     "previous version is kept in case anything goes wrong.",
    "upd_install":   "Update now",
    "upd_later":     "Not now",
    "upd_working":   "Updating...",
    "upd_downloading": "Downloading and verifying...",
    "upd_done":      "Done: Eyve {v} installed.",
    "upd_deps_changed": "This version needs new libraries. Close Eyve and run "
                        "setup.bat once before opening it again.",
    "upd_failed":    "Update failed: {err}",
    "upd_retry":     "Retry",
    "upd_restart":   "Restart Eyve",
    "upd_banner":    "Version {v} available",

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
