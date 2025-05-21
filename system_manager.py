#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Administrador del Sistema
-----------------------
Módulo central que coordina todos los componentes del sistema de inspección visual,
incluyendo la inicialización, gestión de ciclo de vida, y control de operaciones.
"""

import os
import sys
import time
import logging
import threading
import datetime
import traceback
from typing import Dict, List, Any, Optional, Union, Callable

# Importar módulos del sistema
# En una implementación real, estos serían importados desde sus respectivos paquetes
try:
    from database_module import DatabaseManager, UserManager, ProductManager, InspectionDataManager, SystemConfigManager
    from config_manager import ConfigManager
    from camera_module import CameraManager, create_usb_camera, create_ip_camera, create_file_camera
    from inspection_module import InspectionController, create_standard_inspection_controller
except ImportError:
    # Manejo alternativo para desarrollo/pruebas
    pass


class SystemManager:
    """Clase principal para gestionar el sistema de inspección visual."""
    
    def __init__(self, base_path: str = "."):
        """Inicializa el gestor del sistema.
        
        Args:
            base_path: Ruta base para archivos del sistema
        """
        self.base_path = os.path.abspath(base_path)
        self.config_path = os.path.join(self.base_path, "config")
        self.data_path = os.path.join(self.base_path, "data")
        self.logs_path = os.path.join(self.base_path, "logs")
        
        # Crear directorios necesarios
        for path in [self.base_path, self.config_path, self.data_path, self.logs_path]:
            os.makedirs(path, exist_ok=True)
            
        # Configurar sistema de logs
        self._setup_logging()
        self.logger = logging.getLogger('system_logger')
        
        # Inicializar componentes con valores None (se inicializarán bajo demanda)
        self.db_manager = None
        self.config_manager = None
        self.camera_manager = None
        self.inspection_controller = None
        self.user_manager = None
        self.product_manager = None
        self.inspection_data_manager = None
        self.system_config_manager = None
        
        # Estado del sistema
        self.system_status = {
            "initialized": False,
            "running": False,
            "inspection_active": False,
            "current_user": None,
            "current_product": None,
            "current_batch": None,
            "last_inspection_result": None,
            "inspection_count": 0,
            "start_time": None,
            "uptime_seconds": 0
        }
        
        # Mutex para acceso a estado
        self.status_lock = threading.RLock()
        
        # Eventos y temporizadores
        self.shutdown_event = threading.Event()
        self.background_thread = None
        
        self.logger.info("Gestor del sistema inicializado")
        
    def _setup_logging(self) -> None:
        """Configura el sistema de logging."""
        log_file = os.path.join(self.logs_path, "system.log")
        
        # Configurar formato de logs
        log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Configurar handler para archivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        # Configurar handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        # Configurar logger principal
        logger = logging.getLogger('system_logger')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    def initialize(self) -> bool:
        """Inicializa todos los componentes del sistema.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        with self.status_lock:
            if self.system_status["initialized"]:
                self.logger.warning("El sistema ya está inicializado")
                return True
                
            try:
                self.logger.info("Iniciando inicialización del sistema...")
                
                # Inicializar gestores
                self._init_database()
                self._init_config()
                self._init_cameras()
                self._init_inspection()
                
                # Actualizar estado
                self.system_status["initialized"] = True
                self.system_status["start_time"] = datetime.datetime.now()
                
                self.logger.info("Sistema inicializado correctamente")
                return True
            except Exception as e:
                self.logger.error(f"Error durante la inicialización del sistema: {str(e)}")
                self.logger.error(traceback.format_exc())
                return False
                
    def _init_database(self) -> None:
        """Inicializa el gestor de base de datos y servicios relacionados."""
        db_path = os.path.join(self.data_path, "inspection_system.db")
        self.logger.info(f"Inicializando base de datos: {db_path}")
        
        self.db_manager = DatabaseManager(db_path)
        
        # Inicializar servicios relacionados con la base de datos
        self.user_manager = UserManager(self.db_manager)
        self.product_manager = ProductManager(self.db_manager)
        self.inspection_data_manager = InspectionDataManager(self.db_manager)
        self.system_config_manager = SystemConfigManager(self.db_manager)
        
        # Verificar/crear usuario administrador por defecto
        self._ensure_default_admin()
        
    def _ensure_default_admin(self) -> None:
        """Asegura que exista un usuario administrador por defecto."""
        # Buscar administrador existente
        admin_query = "SELECT id FROM users WHERE role = 'admin' LIMIT 1"
        cursor = self.db_manager.execute_query(admin_query)
        
        if cursor and cursor.fetchone() is None:
            # Crear administrador por defecto
            self.logger.info("Creando usuario administrador por defecto")
            self.user_manager.create_user(
                username="admin",
                password="admin123",  # En un sistema real, se debería forzar el cambio al primer inicio
                full_name="Administrador",
                role="admin"
            )
        
    def _init_config(self) -> None:
        """Inicializa el gestor de configuración."""
        self.logger.info("Inicializando gestor de configuración")
        self.config_manager = ConfigManager(self.config_path)
        

    def _init_cameras(self) -> None:
        """Inicializa el gestor de cámaras."""
        self.logger.info("Inicializando gestor de cámaras")
        self.camera_manager = CameraManager()
        
        try:
            # Definimos explícitamente nuestras cámaras conocidas
            known_cameras = [
                {"index": 0, "name": "main_camera", "description": "Logitech Webcam"},
                {"index": 1, "name": "camera_1", "description": "Segunda cámara"},
                {"index": 2, "name": "camera_2", "description": "Tercera cámara"}
                # Omitimos cámara 3 intencionalmente (NVIDIA virtual camera)
            ]
            
            for camera_info in known_cameras:
                camera = create_usb_camera(
                    camera_id=camera_info["name"],
                    device_index=camera_info["index"],
                    resolution=(1280, 720),
                    fps=30
                )
                self.camera_manager.add_camera(camera)
                self.logger.info(f"Cámara registrada: {camera_info['name']} (índice: {camera_info['index']})")
            
            # Establecer cámara principal como predeterminada
            self.camera_manager.set_default_camera("main_camera")
            
        except Exception as e:
            self.logger.error(f"Error al configurar cámaras: {str(e)}")
            # Configuración de respaldo con cámara de archivos
            test_images_dir = os.path.join(self.base_path, "test_images")
            os.makedirs(test_images_dir, exist_ok=True)
            
            # Crear imagen de prueba si no hay ninguna
            if not os.listdir(test_images_dir):
                try:
                    # Crear una imagen de prueba
                    import numpy as np
                    import cv2
                    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(test_image, "Camara de prueba", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imwrite(os.path.join(test_images_dir, "test_image.jpg"), test_image)
                except:
                    pass
            
            file_camera = create_file_camera(
                camera_id="fallback_camera",
                file_path=test_images_dir,
                fps=1
            )
            self.camera_manager.add_camera(file_camera)
            self.logger.info("Usando cámara de respaldo basada en archivos")
               
        
    def _init_inspection(self) -> None:
        """Inicializa el controlador de inspección."""
        self.logger.info("Inicializando controlador de inspección")
        self.inspection_controller = create_standard_inspection_controller()
        
    def start(self) -> bool:
        """Inicia la ejecución del sistema.
        
        Returns:
            bool: True si se inició correctamente
        """
        with self.status_lock:
            if not self.system_status["initialized"]:
                if not self.initialize():
                    return False
                    
            if self.system_status["running"]:
                self.logger.warning("El sistema ya está en ejecución")
                return True
                
            # Iniciar hilo de monitoreo en segundo plano
            self.shutdown_event.clear()
            self.background_thread = threading.Thread(target=self._background_tasks)
            self.background_thread.daemon = True
            self.background_thread.start()
            
            self.system_status["running"] = True
            self.logger.info("Sistema iniciado correctamente")
            return True
            
    def stop(self) -> bool:
        """Detiene la ejecución del sistema.
        
        Returns:
            bool: True si se detuvo correctamente
        """
        with self.status_lock:
            if not self.system_status["running"]:
                self.logger.warning("El sistema no está en ejecución")
                return True
                
            # Detener inspección activa si la hay
            if self.system_status["inspection_active"]:
                self.stop_inspection()
                
            # Detener hilo de monitoreo
            self.shutdown_event.set()
            if self.background_thread:
                self.background_thread.join(timeout=5.0)
                
            # Desconectar cámaras
            if self.camera_manager:
                self.camera_manager.disconnect_all_cameras()
                
            # Cerrar conexión a base de datos
            if self.db_manager:
                self.db_manager.close()
                
            self.system_status["running"] = False
            self.logger.info("Sistema detenido correctamente")
            return True
            
    def _background_tasks(self) -> None:
        """Ejecuta tareas en segundo plano periódicamente."""
        self.logger.info("Hilo de tareas en segundo plano iniciado")
        
        while not self.shutdown_event.is_set():
            try:
                # Actualizar tiempo de actividad
                with self.status_lock:
                    if self.system_status["start_time"]:
                        delta = datetime.datetime.now() - self.system_status["start_time"]
                        self.system_status["uptime_seconds"] = delta.total_seconds()
                
                # Verificar conexión de cámaras
                if self.camera_manager:
                    for camera_id, camera in self.camera_manager.get_all_cameras().items():
                        if not camera.is_connected and not camera.is_capturing:
                            try:
                                camera.connect()
                            except Exception as e:
                                pass  # Ignorar errores de reconexión
                
                # Otras tareas periódicas
                # ...
                
            except Exception as e:
                self.logger.error(f"Error en tareas en segundo plano: {str(e)}")
                
            # Pausa antes de la siguiente iteración
            for _ in range(10):  # Dividir en intervalos más pequeños para responder más rápido al shutdown
                if self.shutdown_event.is_set():
                    break
                time.sleep(0.5)
                
        self.logger.info("Hilo de tareas en segundo plano detenido")
        
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Autentica un usuario en el sistema.
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            
        Returns:
            Optional[Dict[str, Any]]: Datos del usuario o None si falla la autenticación
        """
        if not self.system_status["initialized"]:
            if not self.initialize():
                return None
                
        user_data = self.user_manager.authenticate_user(username, password)
        
        if user_data:
            with self.status_lock:
                self.system_status["current_user"] = user_data
                
            self.logger.info(f"Usuario '{username}' autenticado correctamente")
            return user_data
            
        return None
        
    def logout_user(self) -> None:
        """Cierra la sesión del usuario actual."""
        with self.status_lock:
            if self.system_status["current_user"]:
                username = self.system_status["current_user"].get("username", "")
                self.system_status["current_user"] = None
                self.logger.info(f"Usuario '{username}' ha cerrado sesión")
                
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Obtiene los datos del usuario actual.
        
        Returns:
            Optional[Dict[str, Any]]: Datos del usuario o None si no hay sesión
        """
        with self.status_lock:
            return self.system_status["current_user"]
            
    def select_product(self, product_id: int) -> Optional[Dict[str, Any]]:
        """Selecciona un producto para inspección.
        
        Args:
            product_id: ID del producto
            
        Returns:
            Optional[Dict[str, Any]]: Datos del producto o None si no existe
        """
        product_data = self.product_manager.get_product(product_id=product_id)
        
        if product_data:
            with self.status_lock:
                self.system_status["current_product"] = product_data
                
            # Cargar configuración del producto
            if self.config_manager and product_data.get("sku"):
                product_config = self.config_manager.load_product_config(product_data["sku"])
                
                # Configurar módulos de inspección
                if product_config and self.inspection_controller:
                    if "modules" in product_config:
                        for module_name, module_config in product_config["modules"].items():
                            module = self.inspection_controller.get_module(module_name)
                            if module:
                                module.set_config(module_config)
                                
                                # Habilitar/deshabilitar módulos según configuración
                                if module_config.get("enabled", True):
                                    module.enable()
                                else:
                                    module.disable()
                
            self.logger.info(f"Producto seleccionado: {product_data.get('name', '')} (ID: {product_id})")
            return product_data
            
        return None
        
    def start_batch(self, batch_code: str, notes: str = "") -> Optional[int]:
        """Inicia un nuevo lote de producción.
        
        Args:
            batch_code: Código del lote
            notes: Notas adicionales
            
        Returns:
            Optional[int]: ID del lote creado o None si falla
        """
        with self.status_lock:
            user = self.system_status["current_user"]
            product = self.system_status["current_product"]
            
            if not user or not product:
                self.logger.error("No hay usuario o producto seleccionado para crear lote")
                return None
                
            batch_id = self.inspection_data_manager.create_batch(
                batch_code=batch_code,
                product_id=product["id"],
                operator_id=user["id"],
                notes=notes
            )
            
            if batch_id > 0:
                self.system_status["current_batch"] = batch_id
                self.logger.info(f"Lote iniciado: {batch_code} (ID: {batch_id})")
                return batch_id
                
        return None
        
    def close_batch(self, status: str = "completed") -> bool:
        """Cierra el lote actual.
        
        Args:
            status: Estado final del lote
            
        Returns:
            bool: True si se cerró correctamente
        """
        with self.status_lock:
            batch_id = self.system_status["current_batch"]
            
            if not batch_id:
                self.logger.warning("No hay lote activo para cerrar")
                return False
                
            result = self.inspection_data_manager.close_batch(batch_id, status)
            
            if result:
                self.system_status["current_batch"] = None
                self.logger.info(f"Lote ID {batch_id} cerrado con estado '{status}'")
                return True
                
        return False
        
    def get_batch_statistics(self) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas del lote actual.
        
        Returns:
            Optional[Dict[str, Any]]: Estadísticas del lote o None si no hay lote activo
        """
        with self.status_lock:
            batch_id = self.system_status["current_batch"]
            
            if not batch_id:
                return None
                
            return self.inspection_data_manager.get_batch_statistics(batch_id)
            
    def start_inspection(self, camera_id: str = None) -> bool:
        """Inicia el proceso de inspección continua.
        
        Args:
            camera_id: ID de la cámara a utilizar o None para la predeterminada
            
        Returns:
            bool: True si se inició correctamente
        """
        with self.status_lock:
            if self.system_status["inspection_active"]:
                self.logger.warning("La inspección ya está activa")
                return True
                
            if not self.camera_manager:
                self.logger.error("El gestor de cámaras no está inicializado")
                return False
                
            # Obtener cámara
            camera = self.camera_manager.get_camera(camera_id)
            if not camera:
                self.logger.error(f"Cámara no encontrada: {camera_id}")
                return False
                
            # Iniciar captura
            if not camera.start_capture():
                self.logger.error(f"No se pudo iniciar la captura en cámara {camera.camera_id}")
                return False
                
            self.system_status["inspection_active"] = True
            self.logger.info(f"Inspección iniciada con cámara {camera.camera_id}")
            return True
            
    def stop_inspection(self) -> bool:
        """Detiene el proceso de inspección continua.
        
        Returns:
            bool: True si se detuvo correctamente
        """
        with self.status_lock:
            if not self.system_status["inspection_active"]:
                self.logger.warning("No hay inspección activa para detener")
                return True
                
            # Detener captura en todas las cámaras
            if self.camera_manager:
                for camera_id, camera in self.camera_manager.get_all_cameras().items():
                    if camera.is_capturing:
                        camera.stop_capture()
                        
            self.system_status["inspection_active"] = False
            self.logger.info("Inspección detenida")
            return True
            
    def inspect_current_frame(self, camera_id: str = None, 
                             save_image: bool = True) -> Optional[Dict[str, Any]]:
        """Realiza una inspección con el fotograma actual.
        
        Args:
            camera_id: ID de la cámara a utilizar o None para la predeterminada
            save_image: Si se debe guardar la imagen inspeccionada
            
        Returns:
            Optional[Dict[str, Any]]: Resultados de la inspección o None si falla
        """
        if not self.system_status["initialized"]:
            self.logger.error("El sistema no está inicializado")
            return None
            
        # Obtener cámara
        if not self.camera_manager:
            self.logger.error("El gestor de cámaras no está inicializado")
            return None
            
        camera = self.camera_manager.get_camera(camera_id)
        if not camera:
            self.logger.error(f"Cámara no encontrada: {camera_id}")
            return None
            
        # Obtener fotograma
        frame = camera.get_last_frame()
        if frame is None:
            self.logger.warning(f"No hay fotograma disponible de cámara {camera.camera_id}")
            return None
            
        # Verificar controlador de inspección
        if not self.inspection_controller:
            self.logger.error("El controlador de inspección no está inicializado")
            return None
            
        # Realizar inspección
        results = self.inspection_controller.inspect_image(frame)
        if not results:
            self.logger.error("Error al realizar la inspección")
            return None
            
        # Guardar imagen si es necesario
        image_path = ""
        if save_image:
            image_dir = os.path.join(self.data_path, "images")
            os.makedirs(image_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = os.path.join(image_dir, f"inspection_{timestamp}.jpg")
            
            try:
                import cv2
                cv2.imwrite(image_path, frame)
            except Exception as e:
                self.logger.error(f"Error al guardar imagen: {str(e)}")
                image_path = ""
                
        # Registrar inspección en base de datos si hay lote activo
        with self.status_lock:
            batch_id = self.system_status["current_batch"]
            user = self.system_status["current_user"]
            
            if batch_id and user and self.inspection_data_manager:
                # Determinar resultado
                result_status = "pass" if results.get("status") == "pass" else "fail"
                
                # Registrar en base de datos
                inspection_id = self.inspection_data_manager.record_inspection(
                    batch_id=batch_id,
                    result=result_status,
                    image_path=image_path,
                    data=results,
                    operator_id=user["id"]
                )
                
                if inspection_id > 0:
                    results["inspection_id"] = inspection_id
                    
            # Actualizar estado
            self.system_status["last_inspection_result"] = results
            self.system_status["inspection_count"] += 1
            
        self.logger.info(f"Inspección realizada: {results.get('status', 'unknown')}")
        return results
        
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema.
        
        Returns:
            Dict[str, Any]: Estado del sistema
        """
        with self.status_lock:
            # Crear copia del estado para no exponer el objeto interno
            status_copy = dict(self.system_status)
            
            # Agregar información adicional
            if self.camera_manager:
                cameras = {}
                for camera_id, camera in self.camera_manager.get_all_cameras().items():
                    cameras[camera_id] = camera.get_camera_info()
                status_copy["cameras"] = cameras
                
            # Formatear uptime
            uptime = status_copy.get("uptime_seconds", 0)
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            status_copy["uptime_formatted"] = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            return status_copy
            
    def backup_database(self) -> str:
        """Realiza una copia de seguridad de la base de datos.
        
        Returns:
            str: Ruta al archivo de copia de seguridad o cadena vacía si falla
        """
        if not self.db_manager:
            self.logger.error("El gestor de base de datos no está inicializado")
            return ""
            
        try:
            backup_dir = os.path.join(self.data_path, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"db_backup_{timestamp}.db")
            
            import shutil
            shutil.copy2(self.db_manager.db_path, backup_file)
            
            self.logger.info(f"Copia de seguridad creada: {backup_file}")
            return backup_file
        except Exception as e:
            self.logger.error(f"Error al crear copia de seguridad: {str(e)}")
            return ""
            
    def backup_configuration(self) -> str:
        """Realiza una copia de seguridad de la configuración.
        
        Returns:
            str: Ruta al archivo de copia de seguridad o cadena vacía si falla
        """
        if not self.config_manager:
            self.logger.error("El gestor de configuración no está inicializado")
            return ""
            
        backup_path = self.config_manager.backup_all_configs()
        if backup_path:
            self.logger.info(f"Copia de seguridad de configuración creada: {backup_path}")
            
        return backup_path
        

# Instancia global para acceso desde la aplicación
_system_manager = None

def get_system_manager(base_path: str = ".") -> SystemManager:
    """Obtiene o crea la instancia global del gestor del sistema.
    
    Args:
        base_path: Ruta base para los archivos del sistema
        
    Returns:
        SystemManager: Instancia del gestor
    """
    global _system_manager
    
    if _system_manager is None:
        _system_manager = SystemManager(base_path)
        
    return _system_manager
