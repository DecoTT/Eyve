#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Punto de Entrada Principal
------------------------
Script principal que inicializa y ejecuta el sistema de inspección visual Eyve.
"""

import os
import sys
import time
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Importar PyQt5 para la interfaz gráfica
try:
    from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
    from PyQt5.QtGui import QPixmap, QFont
    from PyQt5.QtCore import Qt, QTimer
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Establecer el directorio base
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Añadir directorios al path
sys.path.insert(0, os.path.join(BASE_DIR, 'core'))
sys.path.insert(0, os.path.join(BASE_DIR, 'database'))
sys.path.insert(0, os.path.join(BASE_DIR, 'gui'))
sys.path.insert(0, os.path.join(BASE_DIR, 'inspection_modules'))
sys.path.insert(0, os.path.join(BASE_DIR, 'utils'))

# Importar componentes del sistema
try:
    from system_manager import get_system_manager
    from main_screen import MainScreen
    from login_screen import LoginDialog
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    sys.exit(1)


def setup_logging():
    """Configura el sistema de logging."""
    # Crear directorio de logs si no existe
    logs_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Nombre de archivo con timestamp
    log_file = os.path.join(logs_dir, f"eyve_{datetime.datetime.now().strftime('%Y%m%d')}.log")
    
    # Configurar formato
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configurar logger raíz
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Crear logger específico para el sistema
    system_logger = logging.getLogger('system_logger')
    system_logger.setLevel(logging.INFO)
    
    # Limitar logs de las bibliotecas externas
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('cv2').setLevel(logging.WARNING)
    
    return system_logger


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas.
    
    Returns:
        bool: True si todas las dependencias están instaladas
    """
    required_modules = [
        'PyQt5', 'numpy', 'cv2', 'sqlite3'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module.split('-')[0])  # Usar solo la primera parte para módulos con guion
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("Faltan las siguientes dependencias:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPor favor, instale las dependencias faltantes con pip:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
        
    return True


def create_splash_screen():
    """Crea y devuelve una pantalla de bienvenida.
    
    Returns:
        QSplashScreen: Pantalla de bienvenida
    """
    if not GUI_AVAILABLE:
        return None
        
    # Crear un pixmap para la pantalla de bienvenida
    # En una aplicación real, usaríamos una imagen
    # splash_pixmap = QPixmap("assets/splash.png")
    
    # Como alternativa, creamos un pixmap vacío
    splash_pixmap = QPixmap(500, 300)
    splash_pixmap.fill(Qt.white)
    
    # Crear la pantalla de bienvenida
    splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
    
    # Personalizar la pantalla
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.setEnabled(False)
    
    # Mostrar un mensaje de bienvenida
    splash.showMessage(
        "<html><body>"
        "<h2 style='color: #2c3e50; margin-top: 20px; text-align: center;'>Eyve Inspection</h2>"
        "<p style='color: #7f8c8d; text-align: center;'>Sistema de Inspección Visual</p>"
        "<p style='color: #7f8c8d; text-align: center; margin-top: 50px;'>Iniciando componentes...</p>"
        "</body></html>",
        Qt.AlignCenter, Qt.darkGray
    )
    
    return splash


def parse_arguments():
    """Analiza los argumentos de la línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(description='Sistema de Inspección Visual Eyve')
    
    parser.add_argument(
        '--no-gui', 
        dest='no_gui', 
        action='store_true',
        help='Ejecutar en modo sin interfaz gráfica'
    )
    
    parser.add_argument(
        '--config', 
        dest='config_file',
        type=str, 
        default=None,
        help='Ruta a un archivo de configuración alternativo'
    )
    
    parser.add_argument(
        '--debug', 
        dest='debug',
        action='store_true',
        help='Activar modo de depuración'
    )
    
    return parser.parse_args()


def load_config(config_file=None):
    """Carga la configuración del sistema.
    
    Args:
        config_file: Ruta a un archivo de configuración alternativo
        
    Returns:
        Dict[str, Any]: Configuración cargada
    """
    # Si no se especifica un archivo, usar el predeterminado
    if not config_file:
        config_file = os.path.join(BASE_DIR, 'config', 'system_config.json')
        
    # Verificar si el archivo existe
    if not os.path.exists(config_file):
        return {}
        
    # Cargar configuración
    try:
        import json
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error al cargar configuración: {e}")
        return {}


def run_gui_mode():
    """Ejecuta la aplicación en modo interfaz gráfica."""
    if not GUI_AVAILABLE:
        print("Error: PyQt5 no está disponible. No se puede iniciar la interfaz gráfica.")
        sys.exit(1)
        
    # Crear aplicación Qt
    app = QApplication(sys.argv)
    app.setApplicationName("Eyve Inspection")
    app.setStyle('Fusion')
    
    # Mostrar pantalla de bienvenida
    splash = create_splash_screen()
    if splash:
        splash.show()
        app.processEvents()
        
    # Inicializar el gestor del sistema
    system_manager = get_system_manager(BASE_DIR)
    
    # Mensaje de bienvenida
    if splash:
        splash.showMessage(
            "<html><body>"
            "<h2 style='color: #2c3e50; margin-top: 20px; text-align: center;'>Eyve Inspection</h2>"
            "<p style='color: #7f8c8d; text-align: center;'>Sistema de Inspección Visual</p>"
            "<p style='color: #7f8c8d; text-align: center; margin-top: 50px;'>Inicializando sistema...</p>"
            "</body></html>",
            Qt.AlignCenter, Qt.darkGray
        )
        app.processEvents()
        
    # Inicializar sistema
    if not system_manager.initialize():
        if splash:
            splash.close()
        QMessageBox.critical(None, "Error", "No se pudo inicializar el sistema.")
        return 1
        
    if not system_manager.start():
        if splash:
            splash.close()
        QMessageBox.critical(None, "Error", "No se pudo iniciar el sistema.")
        return 1
        
    # Iniciar sesión
    if splash:
        splash.showMessage(
            "<html><body>"
            "<h2 style='color: #2c3e50; margin-top: 20px; text-align: center;'>Eyve Inspection</h2>"
            "<p style='color: #7f8c8d; text-align: center;'>Sistema de Inspección Visual</p>"
            "<p style='color: #7f8c8d; text-align: center; margin-top: 50px;'>Iniciando sesión...</p>"
            "</body></html>",
            Qt.AlignCenter, Qt.darkGray
        )
        app.processEvents()
        
    # Cerrar splash y mostrar pantalla de login
    if splash:
        splash.close()
        
    # Mostrar diálogo de login
    login_dialog = LoginDialog()
    if login_dialog.exec_() != login_dialog.Accepted:
        # Si se cancela el login, salir
        system_manager.stop()
        return 0
        
    # Crear y mostrar ventana principal
    main_window = MainScreen()
    main_window.show()
    
    # Ejecutar bucle principal
    return app.exec_()


def run_cli_mode():
    """Ejecuta la aplicación en modo línea de comandos."""
    print("Iniciando Eyve Inspection en modo línea de comandos")
    
    # Inicializar el gestor del sistema
    system_manager = get_system_manager(BASE_DIR)
    
    # Inicializar sistema
    if not system_manager.initialize():
        print("Error: No se pudo inicializar el sistema.")
        return 1
        
    if not system_manager.start():
        print("Error: No se pudo iniciar el sistema.")
        return 1
        
    print("Sistema iniciado correctamente")
    
    try:
        # Bucle principal
        print("Presione Ctrl+C para salir")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDeteniendo sistema...")
    finally:
        # Detener sistema
        system_manager.stop()
        
    return 0


def main():
    """Función principal."""
    # Configurar logging
    logger = setup_logging()
    logger.info("Iniciando Eyve Inspection")
    
    # Verificar dependencias
    if not check_dependencies():
        logger.error("Faltan dependencias necesarias")
        return 1
        
    # Parsear argumentos
    args = parse_arguments()
    
    # Configurar nivel de log para modo debug
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Modo de depuración activado")
        
    # Cargar configuración
    config = load_config(args.config_file)
    logger.info(f"Configuración cargada: {config.get('version', 'N/A')}")
    
    # Ejecutar en modo CLI o GUI
    if args.no_gui:
        return run_cli_mode()
    else:
        return run_gui_mode()


if __name__ == "__main__":
    sys.exit(main())
