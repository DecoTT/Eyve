#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Cámara
---------------
Proporciona clases y funciones para capturar imágenes desde diferentes
fuentes (cámaras USB, IP, archivos, etc.) y gestionar la configuración
de cámaras para el sistema de inspección visual.
"""

import cv2
import os
import time
import logging
import threading
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable


class CameraDevice:
    """Clase base para todos los dispositivos de cámara."""
    
    def __init__(self, camera_id: str, config: Dict[str, Any] = None):
        """Inicializa el dispositivo de cámara base.
        
        Args:
            camera_id: Identificador único para la cámara
            config: Configuración de la cámara
        """
        self.logger = logging.getLogger('system_logger')
        self.camera_id = camera_id
        self.config = config or {}
        self.is_connected = False
        self.is_capturing = False
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        
    def connect(self) -> bool:
        """Conecta con el dispositivo de cámara.
        
        Debe ser implementado por las clases hijas.
        
        Returns:
            bool: True si la conexión fue exitosa
        """
        raise NotImplementedError("El método connect debe ser implementado por la clase hija")
        
    def disconnect(self) -> None:
        """Desconecta el dispositivo de cámara.
        
        Debe ser implementado por las clases hijas.
        """
        raise NotImplementedError("El método disconnect debe ser implementado por la clase hija")
        
    def capture_frame(self) -> Optional[np.ndarray]:
        """Captura un único fotograma.
        
        Debe ser implementado por las clases hijas.
        
        Returns:
            Optional[np.ndarray]: Fotograma capturado o None si falla
        """
        raise NotImplementedError("El método capture_frame debe ser implementado por la clase hija")
        
    def start_capture(self) -> bool:
        """Inicia la captura continua de fotogramas.
        
        Debe ser implementado por las clases hijas.
        
        Returns:
            bool: True si se inició correctamente
        """
        raise NotImplementedError("El método start_capture debe ser implementado por la clase hija")
        
    def stop_capture(self) -> None:
        """Detiene la captura continua de fotogramas.
        
        Debe ser implementado por las clases hijas.
        """
        raise NotImplementedError("El método stop_capture debe ser implementado por la clase hija")
        
    def get_last_frame(self) -> Optional[np.ndarray]:
        """Obtiene el último fotograma capturado.
        
        Returns:
            Optional[np.ndarray]: Último fotograma o None si no hay
        """
        return self.last_frame
        
    def get_camera_info(self) -> Dict[str, Any]:
        """Obtiene información sobre la cámara.
        
        Returns:
            Dict[str, Any]: Información de la cámara
        """
        return {
            "camera_id": self.camera_id,
            "is_connected": self.is_connected,
            "is_capturing": self.is_capturing,
            "frame_count": self.frame_count,
            "fps": self.calculate_fps(),
            "config": self.config
        }
        
    def calculate_fps(self) -> float:
        """Calcula los fotogramas por segundo actuales.
        
        Returns:
            float: Fotogramas por segundo
        """
        if self.last_frame_time == 0:
            return 0.0
            
        elapsed = time.time() - self.last_frame_time
        if elapsed > 0:
            return 1.0 / elapsed
        return 0.0
        
    def set_config(self, config: Dict[str, Any]) -> bool:
        """Actualiza la configuración de la cámara.
        
        Args:
            config: Nueva configuración
            
        Returns:
            bool: True si se aplicó correctamente
        """
        self.config.update(config)
        self.logger.info(f"Configuración actualizada para cámara {self.camera_id}")
        return True


class USBCamera(CameraDevice):
    """Clase para cámaras conectadas por USB usando OpenCV."""
    
    def __init__(self, camera_id: str, device_index: int = 0, config: Dict[str, Any] = None):
        """Inicializa una cámara USB.
        
        Args:
            camera_id: Identificador único para la cámara
            device_index: Índice del dispositivo en el sistema
            config: Configuración de la cámara
        """
        super().__init__(camera_id, config)
        self.device_index = device_index
        self.capture = None
        self.capture_thread = None
        self.stop_event = threading.Event()
        
    def connect(self) -> bool:
        """Conecta con la cámara USB utilizando DirectShow explícitamente.
        
        Returns:
            bool: True si la conexión fue exitosa
        """
        if self.is_connected:
            return True
            
        try:
            # Usar explícitamente DirectShow con timeout más corto
            start_time = time.time()
            self.capture = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            
            # Configurar resolución
            if "width" in self.config and "height" in self.config:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["width"])
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["height"])
                
            # Configurar FPS
            if "fps" in self.config:
                self.capture.set(cv2.CAP_PROP_FPS, self.config["fps"])
                
            # Verificar conexión con timeout
            connection_timeout = 2.0  # 2 segundos máximo
            while not self.capture.isOpened() and time.time() - start_time < connection_timeout:
                time.sleep(0.1)
                
            if not self.capture.isOpened():
                self.logger.error(f"No se pudo abrir la cámara USB {self.device_index}")
                return False
            
            # Leer un frame para verificar
            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.logger.error(f"No se pudo capturar un frame de la cámara {self.device_index}")
                self.capture.release()
                return False
                
            self.is_connected = True
            self.logger.info(f"Cámara USB {self.camera_id} (índice: {self.device_index}) conectada")
            return True
        except Exception as e:
            self.logger.error(f"Error al conectar cámara USB {self.device_index}: {str(e)}")
            if self.capture:
                self.capture.release()
            return False                
            
    def disconnect(self) -> None:
        """Desconecta la cámara USB."""
        if self.is_capturing:
            self.stop_capture()
            
        if self.capture:
            self.capture.release()
            
        self.is_connected = False
        self.logger.info(f"Cámara USB {self.camera_id} desconectada")
        
    def capture_frame(self) -> Optional[np.ndarray]:
        """Captura un único fotograma.
        
        Returns:
            Optional[np.ndarray]: Fotograma capturado o None si falla
        """
        if not self.is_connected:
            if not self.connect():
                return None
                
        # Intentar capturar fotograma con reintentos
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Capturar fotograma
                ret, frame = self.capture.read()
                
                if ret and frame is not None and frame.size > 0:
                    # Verificar que el frame es válido mirando la suma de sus valores
                    if np.sum(frame) == 0:
                        # Frame completamente negro, posible error
                        self.logger.warning(f"Frame negro capturado de cámara USB {self.camera_id}, reintentando...")
                        time.sleep(0.1)  # Pequeña pausa
                        continue
                        
                    self.last_frame = frame.copy()
                    self.last_frame_time = time.time()
                    self.frame_count += 1
                    return frame
                else:
                    # Pausa breve y reintento
                    self.logger.warning(f"Error en frame de cámara USB {self.camera_id}, reintento {attempt+1}/{max_attempts}")
                    time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error al capturar frame: {str(e)}")
                time.sleep(0.1)
                
        # Si llegamos aquí, es porque todos los intentos fallaron
        # Intentar reconectar la cámara
        self.logger.warning(f"Reconectando cámara USB {self.camera_id} después de fallos de captura")
        self.disconnect()
        if not self.connect():
            return None
            
        # Un último intento después de reconectar
        try:
            ret, frame = self.capture.read()
            if ret and frame is not None and frame.size > 0:
                self.last_frame = frame.copy()
                self.last_frame_time = time.time()
                self.frame_count += 1
                return frame
        except:
            pass
                
        self.logger.error(f"No se pudo capturar fotograma de cámara USB {self.camera_id}")
        return None
            
    def _capture_loop(self) -> None:
        """Bucle de captura continua ejecutado en un hilo separado."""
        self.logger.info(f"Iniciando bucle de captura para cámara USB {self.camera_id}")
        
        while not self.stop_event.is_set():
            frame = self.capture_frame()
            if frame is None:
                # Pausa breve si hay error
                time.sleep(0.1)
                
            # Pausa según FPS configurados
            if "fps" in self.config and self.config["fps"] > 0:
                time.sleep(1.0 / self.config["fps"])
                
        self.logger.info(f"Bucle de captura detenido para cámara USB {self.camera_id}")
        
    def start_capture(self) -> bool:
        """Inicia la captura continua de fotogramas en un hilo separado.
        
        Returns:
            bool: True si se inició correctamente
        """
        if self.is_capturing:
            return True
            
        if not self.is_connected:
            if not self.connect():
                return False
                
        # Iniciar hilo de captura
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.is_capturing = True
        self.logger.info(f"Captura continua iniciada para cámara USB {self.camera_id}")
        return True
        
    def stop_capture(self) -> None:
        """Detiene la captura continua de fotogramas."""
        if not self.is_capturing:
            return
            
        # Detener hilo de captura
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
            
        self.is_capturing = False
        self.logger.info(f"Captura continua detenida para cámara USB {self.camera_id}")
        
    def set_config(self, config: Dict[str, Any]) -> bool:
        """Actualiza la configuración de la cámara USB.
        
        Args:
            config: Nueva configuración
            
        Returns:
            bool: True si se aplicó correctamente
        """
        was_capturing = self.is_capturing
        if was_capturing:
            self.stop_capture()
            
        was_connected = self.is_connected
        if was_connected:
            self.disconnect()
            
        result = super().set_config(config)
        
        # Actualizar índice de dispositivo si se proporciona
        if "device_index" in config:
            self.device_index = config["device_index"]
            
        # Reconectar si estaba conectado
        if was_connected:
            self.connect()
            if was_capturing:
                self.start_capture()
                
        return result


class IPCamera(CameraDevice):
    """Clase para cámaras IP usando OpenCV."""
    
    def __init__(self, camera_id: str, url: str, config: Dict[str, Any] = None):
        """Inicializa una cámara IP.
        
        Args:
            camera_id: Identificador único para la cámara
            url: URL de la transmisión de la cámara IP
            config: Configuración de la cámara
        """
        super().__init__(camera_id, config)
        self.url = url
        self.capture = None
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.connection_timeout = config.get("connection_timeout", 10)
        self.reconnect_attempts = config.get("reconnect_attempts", 3)
        
    def connect(self) -> bool:
        """Conecta con la cámara IP.
        
        Returns:
            bool: True si la conexión fue exitosa
        """
        if self.is_connected:
            return True
            
        try:
            # Intentar establecer conexión con reintentos
            for attempt in range(self.reconnect_attempts):
                self.logger.info(f"Intentando conectar a cámara IP {self.camera_id} (intento {attempt+1})")
                
                self.capture = cv2.VideoCapture(self.url)
                
                # Esperar un poco para que se establezca la conexión
                start_time = time.time()
                while time.time() - start_time < self.connection_timeout:
                    if self.capture.isOpened():
                        break
                    time.sleep(0.5)
                    
                if self.capture.isOpened():
                    self.is_connected = True
                    self.logger.info(f"Cámara IP {self.camera_id} conectada: {self.url}")
                    return True
                else:
                    self.logger.warning(f"Intento {attempt+1} fallido para cámara IP {self.camera_id}")
                    if self.capture:
                        self.capture.release()
                        
                    # Esperar antes del siguiente intento
                    time.sleep(1.0)
                    
            self.logger.error(f"No se pudo conectar a la cámara IP {self.camera_id} después de {self.reconnect_attempts} intentos")
            return False
        except Exception as e:
            self.logger.error(f"Error al conectar cámara IP {self.camera_id}: {str(e)}")
            return False
            
    def disconnect(self) -> None:
        """Desconecta la cámara IP."""
        if self.is_capturing:
            self.stop_capture()
            
        if self.capture:
            self.capture.release()
            
        self.is_connected = False
        self.logger.info(f"Cámara IP {self.camera_id} desconectada")
        
    def capture_frame(self) -> Optional[np.ndarray]:
        """Captura un único fotograma.
        
        Returns:
            Optional[np.ndarray]: Fotograma capturado o None si falla
        """
        if not self.is_connected:
            if not self.connect():
                return None
                
        # Capturar fotograma con timeout
        start_time = time.time()
        timeout = self.config.get("frame_timeout", 5.0)
        
        while time.time() - start_time < timeout:
            ret, frame = self.capture.read()
            
            if ret:
                self.last_frame = frame
                self.last_frame_time = time.time()
                self.frame_count += 1
                return frame
            
            # Pequeña pausa para no saturar CPU
            time.sleep(0.01)
            
        # Si llega aquí, no pudo capturar después del timeout
        self.logger.warning(f"Timeout al capturar fotograma de cámara IP {self.camera_id}")
        return None
        
    def _capture_loop(self) -> None:
        """Bucle de captura continua ejecutado en un hilo separado."""
        self.logger.info(f"Iniciando bucle de captura para cámara IP {self.camera_id}")
        
        consecutive_failures = 0
        max_failures = self.config.get("max_consecutive_failures", 5)
        
        while not self.stop_event.is_set():
            frame = self.capture_frame()
            
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self.logger.warning(f"Demasiados fallos consecutivos, reconectando cámara IP {self.camera_id}")
                    self.disconnect()
                    if self.connect():
                        consecutive_failures = 0
                    else:
                        # Pausa más larga si la reconexión falla
                        time.sleep(5.0)
                else:
                    # Pausa breve si hay error
                    time.sleep(0.5)
            else:
                consecutive_failures = 0
                
            # Pausa según FPS configurados
            if "fps" in self.config and self.config["fps"] > 0:
                time.sleep(1.0 / self.config["fps"])
                
        self.logger.info(f"Bucle de captura detenido para cámara IP {self.camera_id}")
        
    def start_capture(self) -> bool:
        """Inicia la captura continua de fotogramas en un hilo separado.
        
        Returns:
            bool: True si se inició correctamente
        """
        if self.is_capturing:
            return True
            
        if not self.is_connected:
            if not self.connect():
                return False
                
        # Iniciar hilo de captura
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.is_capturing = True
        self.logger.info(f"Captura continua iniciada para cámara IP {self.camera_id}")
        return True
        
    def stop_capture(self) -> None:
        """Detiene la captura continua de fotogramas."""
        if not self.is_capturing:
            return
            
        # Detener hilo de captura
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
            
        self.is_capturing = False
        self.logger.info(f"Captura continua detenida para cámara IP {self.camera_id}")
        
    def set_config(self, config: Dict[str, Any]) -> bool:
        """Actualiza la configuración de la cámara IP.
        
        Args:
            config: Nueva configuración
            
        Returns:
            bool: True si se aplicó correctamente
        """
        was_capturing = self.is_capturing
        if was_capturing:
            self.stop_capture()
            
        was_connected = self.is_connected
        if was_connected:
            self.disconnect()
            
        # Actualizar URL si se proporciona
        if "url" in config:
            self.url = config["url"]
            
        result = super().set_config(config)
        
        # Configurar parámetros específicos
        if "connection_timeout" in config:
            self.connection_timeout = config["connection_timeout"]
        if "reconnect_attempts" in config:
            self.reconnect_attempts = config["reconnect_attempts"]
            
        # Reconectar si estaba conectado
        if was_connected:
            self.connect()
            if was_capturing:
                self.start_capture()
                
        return result


class FileCamera(CameraDevice):
    """Clase para cargar imágenes desde archivos."""
    
    def __init__(self, camera_id: str, file_path: str = "", config: Dict[str, Any] = None):
        """Inicializa una cámara basada en archivos.
        
        Args:
            camera_id: Identificador único para la cámara
            file_path: Ruta al archivo o directorio de imágenes
            config: Configuración
        """
        super().__init__(camera_id, config)
        self.file_path = file_path
        self.files = []
        self.current_file_index = 0
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.is_directory = os.path.isdir(file_path) if file_path else False
        
    def connect(self) -> bool:
        """Conecta con la fuente de archivos.
        
        Returns:
            bool: True si la conexión fue exitosa
        """
        if self.is_connected:
            return True
            
        try:
            # Si es un directorio, listar archivos de imagen
            if self.is_directory:
                self._scan_directory()
                if not self.files:
                    self.logger.error(f"No se encontraron imágenes en {self.file_path}")
                    return False
            # Si es un archivo, verificar que existe
            elif self.file_path:
                if not os.path.isfile(self.file_path):
                    self.logger.error(f"Archivo no encontrado: {self.file_path}")
                    return False
                self.files = [self.file_path]
            else:
                self.logger.error("No se especificó archivo o directorio")
                return False
                
            self.is_connected = True
            self.logger.info(f"Cámara de archivos {self.camera_id} conectada ({len(self.files)} imágenes)")
            return True
        except Exception as e:
            self.logger.error(f"Error al conectar cámara de archivos {self.camera_id}: {str(e)}")
            return False
            
    def _scan_directory(self) -> None:
        """Escanea el directorio en busca de archivos de imagen."""
        self.files = []
        
        if not self.file_path or not os.path.isdir(self.file_path):
            return
            
        # Extensiones de imagen soportadas
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Buscar archivos en el directorio
        for file in os.listdir(self.file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.join(self.file_path, file)
                self.files.append(full_path)
                
        # Ordenar por nombre
        self.files.sort()
        self.current_file_index = 0
        
    def disconnect(self) -> None:
        """Desconecta la cámara de archivos."""
        if self.is_capturing:
            self.stop_capture()
            
        self.is_connected = False
        self.logger.info(f"Cámara de archivos {self.camera_id} desconectada")
        
    def capture_frame(self) -> Optional[np.ndarray]:
        """Captura un fotograma (carga una imagen).
        
        Returns:
            Optional[np.ndarray]: Imagen cargada o None si falla
        """
        if not self.is_connected:
            if not self.connect():
                return None
                
        if not self.files:
            return None
            
        # Modo secuencial o aleatorio
        if self.config.get("random", False):
            import random
            file_index = random.randint(0, len(self.files) - 1)
        else:
            file_index = self.current_file_index
            self.current_file_index = (self.current_file_index + 1) % len(self.files)
            
        # Cargar imagen
        try:
            file_path = self.files[file_index]
            frame = cv2.imread(file_path)
            
            if frame is not None:
                self.last_frame = frame
                self.last_frame_time = time.time()
                self.frame_count += 1
                return frame
            else:
                self.logger.warning(f"Error al cargar imagen: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error al cargar imagen: {str(e)}")
            return None
            
    def _capture_loop(self) -> None:
        """Bucle de captura continua ejecutado en un hilo separado."""
        self.logger.info(f"Iniciando bucle de captura para cámara de archivos {self.camera_id}")
        
        while not self.stop_event.is_set():
            frame = self.capture_frame()
            
            # Pausa según FPS configurados
            delay = 1.0
            if "fps" in self.config and self.config["fps"] > 0:
                delay = 1.0 / self.config["fps"]
                
            time.sleep(delay)
                
        self.logger.info(f"Bucle de captura detenido para cámara de archivos {self.camera_id}")
        
    def start_capture(self) -> bool:
        """Inicia la captura continua de fotogramas en un hilo separado.
        
        Returns:
            bool: True si se inició correctamente
        """
        if self.is_capturing:
            return True
            
        if not self.is_connected:
            if not self.connect():
                return False
                
        # Iniciar hilo de captura
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.is_capturing = True
        self.logger.info(f"Captura continua iniciada para cámara de archivos {self.camera_id}")
        return True
        
    def stop_capture(self) -> None:
        """Detiene la captura continua de fotogramas."""
        if not self.is_capturing:
            return
            
        # Detener hilo de captura
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
            
        self.is_capturing = False
        self.logger.info(f"Captura continua detenida para cámara de archivos {self.camera_id}")
        
    def set_config(self, config: Dict[str, Any]) -> bool:
        """Actualiza la configuración de la cámara de archivos.
        
        Args:
            config: Nueva configuración
            
        Returns:
            bool: True si se aplicó correctamente
        """
        was_capturing = self.is_capturing
        if was_capturing:
            self.stop_capture()
            
        # Actualizar ruta de archivo si se proporciona
        if "file_path" in config:
            self.file_path = config["file_path"]
            self.is_directory = os.path.isdir(self.file_path) if self.file_path else False
            self.is_connected = False
            
        result = super().set_config(config)
        
        # Reconectar y reiniciar captura si es necesario
        if was_capturing:
            self.connect()
            self.start_capture()
            
        return result


class CameraManager:
    """Clase para gestionar múltiples cámaras."""
    
    def __init__(self):
        """Inicializa el gestor de cámaras."""
        self.logger = logging.getLogger('system_logger')
        self.cameras = {}  # Diccionario de cámaras por ID
        self.default_camera_id = None
        
    def add_camera(self, camera: CameraDevice) -> bool:
        """Agrega una cámara al gestor.
        
        Args:
            camera: Instancia de cámara
            
        Returns:
            bool: True si se agregó correctamente
        """
        camera_id = camera.camera_id
        
        if camera_id in self.cameras:
            self.logger.warning(f"Cámara con ID {camera_id} ya existe, reemplazando")
            self.remove_camera(camera_id)
            
        self.cameras[camera_id] = camera
        
        # Si es la primera cámara, establecer como predeterminada
        if self.default_camera_id is None:
            self.default_camera_id = camera_id
            
        self.logger.info(f"Cámara {camera_id} agregada al gestor")
        return True
        
    def remove_camera(self, camera_id: str) -> bool:
        """Elimina una cámara del gestor.
        
        Args:
            camera_id: ID de la cámara a eliminar
            
        Returns:
            bool: True si se eliminó correctamente
        """
        if camera_id not in self.cameras:
            self.logger.warning(f"Cámara {camera_id} no encontrada")
            return False
            
        # Desconectar la cámara
        camera = self.cameras[camera_id]
        if camera.is_connected:
            camera.disconnect()
            
        # Eliminar la cámara
        del self.cameras[camera_id]
        
        # Ajustar cámara predeterminada si es necesario
        if self.default_camera_id == camera_id:
            if self.cameras:
                self.default_camera_id = next(iter(self.cameras))
            else:
                self.default_camera_id = None
                
        self.logger.info(f"Cámara {camera_id} eliminada del gestor")
        return True
        
    def get_camera(self, camera_id: str = None) -> Optional[CameraDevice]:
        """Obtiene una cámara por su ID.
        
        Args:
            camera_id: ID de la cámara o None para la predeterminada
            
        Returns:
            Optional[CameraDevice]: Instancia de cámara o None si no existe
        """
        if camera_id is None:
            camera_id = self.default_camera_id
            
        if camera_id not in self.cameras:
            self.logger.warning(f"Cámara {camera_id} no encontrada")
            return None
            
        return self.cameras[camera_id]
        
    def get_all_cameras(self) -> Dict[str, CameraDevice]:
        """Obtiene todas las cámaras registradas.
        
        Returns:
            Dict[str, CameraDevice]: Diccionario de cámaras
        """
        return self.cameras
        
    def set_default_camera(self, camera_id: str) -> bool:
        """Establece la cámara predeterminada.
        
        Args:
            camera_id: ID de la cámara
            
        Returns:
            bool: True si se estableció correctamente
        """
        if camera_id not in self.cameras:
            self.logger.warning(f"Cámara {camera_id} no encontrada")
            return False
            
        self.default_camera_id = camera_id
        self.logger.info(f"Cámara predeterminada establecida: {camera_id}")
        return True
        
    def capture_from_camera(self, camera_id: str = None) -> Optional[np.ndarray]:
        """Captura un fotograma de una cámara específica.
        
        Args:
            camera_id: ID de la cámara o None para la predeterminada
            
        Returns:
            Optional[np.ndarray]: Fotograma capturado o None si falla
        """
        camera = self.get_camera(camera_id)
        if camera:
            return camera.capture_frame()
        return None
        
    def connect_all_cameras(self) -> Dict[str, bool]:
        """Conecta todas las cámaras registradas.
        
        Returns:
            Dict[str, bool]: Diccionario con resultados de conexión
        """
        results = {}
        
        for camera_id, camera in self.cameras.items():
            results[camera_id] = camera.connect()
            
        return results
        
    def disconnect_all_cameras(self) -> None:
        """Desconecta todas las cámaras registradas."""
        for camera_id, camera in self.cameras.items():
            if camera.is_connected:
                camera.disconnect()
                
        self.logger.info("Todas las cámaras desconectadas")
        
    def get_camera_info(self, camera_id: str = None) -> Optional[Dict[str, Any]]:
        """Obtiene información sobre una cámara.
        
        Args:
            camera_id: ID de la cámara o None para la predeterminada
            
        Returns:
            Optional[Dict[str, Any]]: Información de la cámara
        """
        camera = self.get_camera(camera_id)
        if camera:
            return camera.get_camera_info()
        return None
        
    def save_image_from_camera(self, output_path: str, camera_id: str = None) -> bool:
        """Captura y guarda una imagen desde una cámara.
        
        Args:
            output_path: Ruta de salida para la imagen
            camera_id: ID de la cámara o None para la predeterminada
            
        Returns:
            bool: True si se guardó correctamente
        """
        frame = self.capture_from_camera(camera_id)
        
        if frame is None:
            self.logger.error(f"No se pudo capturar imagen de cámara {camera_id or self.default_camera_id}")
            return False
            
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Guardar imagen
            cv2.imwrite(output_path, frame)
            self.logger.info(f"Imagen guardada en {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar imagen: {str(e)}")
            return False


def create_usb_camera(camera_id: str, device_index: int = 0, 
                     resolution: Tuple[int, int] = (1280, 720),
                     fps: int = 30) -> USBCamera:
    """Crea y configura una cámara USB.
    
    Args:
        camera_id: Identificador para la cámara
        device_index: Índice del dispositivo
        resolution: Tupla (ancho, alto) para resolución
        fps: Fotogramas por segundo
        
    Returns:
        USBCamera: Instancia configurada
    """
    config = {
        "width": resolution[0],
        "height": resolution[1],
        "fps": fps
    }
    
    return USBCamera(camera_id, device_index, config)


def create_ip_camera(camera_id: str, url: str, 
                    connection_timeout: int = 10,
                    reconnect_attempts: int = 3,
                    fps: int = 15) -> IPCamera:
    """Crea y configura una cámara IP.
    
    Args:
        camera_id: Identificador para la cámara
        url: URL de la transmisión
        connection_timeout: Timeout para conexión en segundos
        reconnect_attempts: Número de intentos de reconexión
        fps: Fotogramas por segundo deseados
        
    Returns:
        IPCamera: Instancia configurada
    """
    config = {
        "connection_timeout": connection_timeout,
        "reconnect_attempts": reconnect_attempts,
        "fps": fps
    }
    
    return IPCamera(camera_id, url, config)


def create_file_camera(camera_id: str, file_path: str, 
                      fps: int = 1, 
                      random: bool = False) -> FileCamera:
    """Crea y configura una cámara basada en archivos.
    
    Args:
        camera_id: Identificador para la cámara
        file_path: Ruta a archivo o directorio de imágenes
        fps: Fotogramas por segundo simulados
        random: Si se deben seleccionar imágenes al azar
        
    Returns:
        FileCamera: Instancia configurada
    """
    config = {
        "fps": fps,
        "random": random
    }
    
    return FileCamera(camera_id, file_path, config)

    
def list_available_cameras() -> List[int]:
    """Lista los índices de cámaras USB disponibles.
    
    Returns:
        List[int]: Lista de índices de cámaras disponibles
    """
    available_cameras = []
    
    # Probar más dispositivos (aumentar a 20 para asegurar)
    for i in range(20):  # Probar más índices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Intentar leer un frame para verificar que funciona
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
            else:
                cap.release()
        except:
            pass
    
    return available_cameras
