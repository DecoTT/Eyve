#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilidades Generales
------------------
Proporciona funciones y clases de utilidad para el sistema de inspección visual.
"""

import os
import re
import cv2
import json
import time
import random
import string
import hashlib
import logging
import datetime
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable


def generate_random_id(length: int = 8) -> str:
    """Genera un ID aleatorio de longitud específica.
    
    Args:
        length: Longitud del ID a generar
        
    Returns:
        str: ID aleatorio generado
    """
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def generate_timestamp_id(prefix: str = "") -> str:
    """Genera un ID basado en timestamp actual.
    
    Args:
        prefix: Prefijo opcional para el ID
        
    Returns:
        str: ID generado con formato [prefijo]_AAAAMMDD_HHMMSS
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}" if prefix else timestamp


def safe_filename(name: str) -> str:
    """Convierte un string en un nombre de archivo seguro.
    
    Args:
        name: Nombre original
        
    Returns:
        str: Nombre seguro para usar como archivo
    """
    # Reemplazar caracteres no seguros
    safe_name = re.sub(r'[^\w\-_. ]', '_', name)
    # Eliminar espacios adicionales
    safe_name = re.sub(r'\s+', '_', safe_name)
    return safe_name


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Crea un hash seguro para una contraseña.
    
    Args:
        password: Contraseña a hashear
        salt: Sal opcional (se genera si no se proporciona)
        
    Returns:
        Tuple[str, str]: (hash de contraseña, sal utilizada)
    """
    if salt is None:
        salt = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(16))
        
    # Combinar contraseña y sal
    salted = password + salt
    
    # Crear hash usando SHA-256
    hash_obj = hashlib.sha256(salted.encode())
    hashed_password = hash_obj.hexdigest()
    
    return hashed_password, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifica si una contraseña coincide con un hash almacenado.
    
    Args:
        password: Contraseña a verificar
        stored_hash: Hash almacenado
        salt: Sal utilizada
        
    Returns:
        bool: True si la contraseña coincide
    """
    # Combinar contraseña y sal
    salted = password + salt
    
    # Crear hash usando SHA-256
    hash_obj = hashlib.sha256(salted.encode())
    hashed_password = hash_obj.hexdigest()
    
    return hashed_password == stored_hash


def load_json_file(file_path: str, default: Any = None) -> Any:
    """Carga datos desde un archivo JSON.
    
    Args:
        file_path: Ruta al archivo JSON
        default: Valor por defecto si falla la carga
        
    Returns:
        Any: Datos cargados o valor por defecto
    """
    if not os.path.exists(file_path):
        return default
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger = logging.getLogger('system_logger')
        logger.error(f"Error al cargar archivo JSON {file_path}: {str(e)}")
        return default


def save_json_file(file_path: str, data: Any, indent: int = 4) -> bool:
    """Guarda datos en un archivo JSON.
    
    Args:
        file_path: Ruta donde guardar el archivo
        data: Datos a guardar
        indent: Nivel de indentación
        
    Returns:
        bool: True si se guardó correctamente
    """
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logger = logging.getLogger('system_logger')
        logger.error(f"Error al guardar archivo JSON {file_path}: {str(e)}")
        return False


def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None, 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """Redimensiona una imagen a las dimensiones especificadas.
    
    Args:
        image: Imagen a redimensionar (numpy array)
        width: Ancho objetivo (None para calcular a partir de la altura)
        height: Altura objetivo (None para calcular a partir del ancho)
        keep_aspect_ratio: Si se debe mantener la relación de aspecto
        
    Returns:
        np.ndarray: Imagen redimensionada
    """
    # Obtener dimensiones originales
    h, w = image.shape[:2]
    
    # Si no se especifica ni ancho ni alto, devolver la imagen original
    if width is None and height is None:
        return image
        
    # Si se mantiene la relación de aspecto
    if keep_aspect_ratio:
        # Si solo se especifica ancho, calcular altura proporcionalmente
        if height is None:
            height = int(h * width / w)
        # Si solo se especifica alto, calcular ancho proporcionalmente
        elif width is None:
            width = int(w * height / h)
        # Si se especifican ambos, ajustar para mantener relación de aspecto
        else:
            # Calcular relación de aspecto de destino y original
            target_ratio = width / height
            original_ratio = w / h
            
            # Ajustar según cuál dimensión queda limitante
            if target_ratio > original_ratio:
                # La altura es limitante
                width = int(height * original_ratio)
            else:
                # El ancho es limitante
                height = int(width / original_ratio)
    
    # Redimensionar la imagen
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Recorta una región de una imagen.
    
    Args:
        image: Imagen a recortar (numpy array)
        x: Coordenada X de la esquina superior izquierda
        y: Coordenada Y de la esquina superior izquierda
        width: Ancho de la región a recortar
        height: Alto de la región a recortar
        
    Returns:
        np.ndarray: Región recortada
    """
    # Obtener dimensiones de la imagen
    h, w = image.shape[:2]
    
    # Asegurar que las coordenadas están dentro de la imagen
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    
    # Asegurar que el ancho y alto no se salgan de la imagen
    width = min(width, w - x)
    height = min(height, h - y)
    
    # Recortar la imagen
    return image[y:y+height, x:x+width]


def draw_text_with_background(image: np.ndarray, text: str, position: Tuple[int, int], 
                             font_scale: float = 0.7, font_thickness: int = 1,
                             text_color: Tuple[int, int, int] = (255, 255, 255),
                             bg_color: Tuple[int, int, int] = (0, 0, 0),
                             bg_alpha: float = 0.7) -> np.ndarray:
    """Dibuja texto con fondo semitransparente en una imagen.
    
    Args:
        image: Imagen donde dibujar (numpy array)
        text: Texto a dibujar
        position: Posición (x, y) del texto
        font_scale: Escala de la fuente
        font_thickness: Grosor de la fuente
        text_color: Color del texto (BGR)
        bg_color: Color del fondo (BGR)
        bg_alpha: Opacidad del fondo (0.0 a 1.0)
        
    Returns:
        np.ndarray: Imagen con el texto dibujado
    """
    # Crear una copia de la imagen
    result = image.copy()
    
    # Obtener dimensiones del texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    padding = 5
    
    # Coordenadas del rectángulo
    x, y = position
    x1, y1 = x - padding, y - text_height - padding
    x2, y2 = x + text_width + padding, y + padding
    
    # Crear capa para el fondo semitransparente
    overlay = result.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    
    # Combinar la imagen original con la capa
    cv2.addWeighted(overlay, bg_alpha, result, 1 - bg_alpha, 0, result)
    
    # Dibujar el texto
    cv2.putText(result, text, (x, y), font, font_scale, text_color, font_thickness)
    
    return result


def draw_detection_boxes(image: np.ndarray, detections: List[Dict[str, Any]], 
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2, show_labels: bool = True) -> np.ndarray:
    """Dibuja cajas de detección en una imagen.
    
    Args:
        image: Imagen donde dibujar (numpy array)
        detections: Lista de detecciones, cada una con 'bbox' y opcionalmente 'label'
        color: Color de las cajas (BGR)
        thickness: Grosor de las líneas
        show_labels: Si se deben mostrar etiquetas
        
    Returns:
        np.ndarray: Imagen con las cajas dibujadas
    """
    # Crear una copia de la imagen
    result = image.copy()
    
    # Dibujar cada detección
    for detection in detections:
        # Obtener coordenadas de la caja
        if 'bbox' not in detection:
            continue
            
        bbox = detection['bbox']
        if len(bbox) != 4:
            continue
            
        x, y, w, h = bbox
        
        # Dibujar rectángulo
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        # Dibujar etiqueta si está disponible y se solicita
        if show_labels and 'label' in detection:
            label = detection['label']
            confidence = detection.get('confidence', None)
            
            # Formatear etiqueta con confianza si está disponible
            if confidence is not None:
                label_text = f"{label}: {confidence:.2f}"
            else:
                label_text = label
                
            # Dibujar etiqueta con fondo
            result = draw_text_with_background(
                result, label_text, (x, y - 5), 
                bg_color=color, text_color=(255, 255, 255)
            )
            
    return result


def draw_grid(image: np.ndarray, grid_size: int = 50, 
             color: Tuple[int, int, int] = (128, 128, 128),
             thickness: int = 1, alpha: float = 0.3) -> np.ndarray:
    """Dibuja una cuadrícula en una imagen.
    
    Args:
        image: Imagen donde dibujar (numpy array)
        grid_size: Tamaño de las celdas de la cuadrícula
        color: Color de las líneas (BGR)
        thickness: Grosor de las líneas
        alpha: Opacidad de la cuadrícula
        
    Returns:
        np.ndarray: Imagen con la cuadrícula dibujada
    """
    # Crear una copia de la imagen
    result = image.copy()
    
    # Obtener dimensiones de la imagen
    h, w = image.shape[:2]
    
    # Crear una imagen para la cuadrícula
    grid = np.zeros_like(image)
    
    # Dibujar líneas horizontales
    for y in range(0, h, grid_size):
        cv2.line(grid, (0, y), (w, y), color, thickness)
        
    # Dibujar líneas verticales
    for x in range(0, w, grid_size):
        cv2.line(grid, (x, 0), (x, h), color, thickness)
        
    # Combinar la imagen original con la cuadrícula
    cv2.addWeighted(result, 1.0, grid, alpha, 0, result)
    
    return result


def enhance_image_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """Mejora el contraste de una imagen.
    
    Args:
        image: Imagen a mejorar (numpy array)
        method: Método a utilizar ('clahe', 'histogram_eq', 'gamma')
        
    Returns:
        np.ndarray: Imagen mejorada
    """
    # Crear una copia de la imagen
    result = image.copy()
    
    # Si la imagen es a color, convertir a LAB
    if len(image.shape) == 3 and image.shape[2] == 3:
        if method == 'clahe':
            # Convertir a LAB
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            
            # Separar canales
            l, a, b = cv2.split(lab)
            
            # Aplicar CLAHE al canal L
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Combinar canales
            lab = cv2.merge((l, a, b))
            
            # Convertir de vuelta a BGR
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif method == 'histogram_eq':
            # Convertir a YUV
            yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
            
            # Aplicar ecualización de histograma al canal Y
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            
            # Convertir de vuelta a BGR
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
        elif method == 'gamma':
            # Aplicar corrección gamma
            gamma = 1.5
            inv_gamma = 1.0 / gamma
            
            # Crear tabla de búsqueda
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            
            # Aplicar la tabla de búsqueda
            result = cv2.LUT(result, table)
            
    else:
        # Imagen en escala de grises
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(result)
            
        elif method == 'histogram_eq':
            result = cv2.equalizeHist(result)
            
        elif method == 'gamma':
            # Aplicar corrección gamma
            gamma = 1.5
            inv_gamma = 1.0 / gamma
            
            # Crear tabla de búsqueda
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            
            # Aplicar la tabla de búsqueda
            result = cv2.LUT(result, table)
            
    return result


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """Calcula el hash de un archivo.
    
    Args:
        file_path: Ruta al archivo
        algorithm: Algoritmo de hash a utilizar ('md5', 'sha1', 'sha256')
        
    Returns:
        str: Hash calculado
    """
    if not os.path.exists(file_path):
        return ""
        
    # Seleccionar algoritmo
    if algorithm == 'md5':
        hash_func = hashlib.md5()
    elif algorithm == 'sha1':
        hash_func = hashlib.sha1()
    else:  # sha256 predeterminado
        hash_func = hashlib.sha256()
        
    # Leer archivo por bloques
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
            
    return hash_func.hexdigest()


class TimerContext:
    """Clase para medir tiempo de ejecución de bloques de código."""
    
    def __init__(self, name: str = None, logger: logging.Logger = None):
        """Inicializa el contexto de temporizador.
        
        Args:
            name: Nombre para identificar este temporizador
            logger: Logger opcional para registrar tiempo
        """
        self.name = name or "Timer"
        self.logger = logger or logging.getLogger('system_logger')
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        """Inicia el temporizador al entrar en el bloque with."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finaliza el temporizador al salir del bloque with."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.logger.info(f"{self.name}: {elapsed:.4f} segundos")
        
    def elapsed(self) -> float:
        """Calcula el tiempo transcurrido.
        
        Returns:
            float: Tiempo transcurrido en segundos
        """
        if self.start_time is None:
            return 0.0
            
        end = self.end_time or time.time()
        return end - self.start_time


class LimitedQueue:
    """Cola con tamaño limitado que elimina elementos antiguos."""
    
    def __init__(self, max_size: int = 100):
        """Inicializa la cola limitada.
        
        Args:
            max_size: Tamaño máximo de la cola
        """
        self.max_size = max_size
        self.items = []
        
    def push(self, item: Any) -> None:
        """Añade un elemento a la cola.
        
        Args:
            item: Elemento a añadir
        """
        self.items.append(item)
        
        # Si se supera el tamaño máximo, eliminar el elemento más antiguo
        if len(self.items) > self.max_size:
            self.items.pop(0)
            
    def pop(self) -> Any:
        """Saca el elemento más antiguo de la cola.
        
        Returns:
            Any: Elemento extraído o None si la cola está vacía
        """
        if not self.items:
            return None
            
        return self.items.pop(0)
        
    def peek(self) -> Any:
        """Obtiene el elemento más antiguo sin sacarlo.
        
        Returns:
            Any: Primer elemento o None si la cola está vacía
        """
        if not self.items:
            return None
            
        return self.items[0]
        
    def clear(self) -> None:
        """Vacía la cola."""
        self.items = []
        
    def is_empty(self) -> bool:
        """Comprueba si la cola está vacía.
        
        Returns:
            bool: True si la cola está vacía
        """
        return len(self.items) == 0
        
    def size(self) -> int:
        """Obtiene el número de elementos en la cola.
        
        Returns:
            int: Número de elementos
        """
        return len(self.items)
        
    def get_all(self) -> List[Any]:
        """Obtiene todos los elementos de la cola.
        
        Returns:
            List[Any]: Lista con todos los elementos
        """
        return list(self.items)


class MovingAverage:
    """Clase para calcular el promedio móvil de una serie de valores."""
    
    def __init__(self, window_size: int = 10):
        """Inicializa el promedio móvil.
        
        Args:
            window_size: Tamaño de la ventana
        """
        self.window_size = window_size
        self.values = []
        self.sum = 0.0
        
    def add(self, value: float) -> None:
        """Añade un nuevo valor.
        
        Args:
            value: Valor a añadir
        """
        self.values.append(value)
        self.sum += value
        
        # Eliminar valor más antiguo si se supera la ventana
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
            
    def get_average(self) -> float:
        """Obtiene el promedio actual.
        
        Returns:
            float: Promedio o 0.0 si no hay valores
        """
        if not self.values:
            return 0.0
            
        return self.sum / len(self.values)
        
    def reset(self) -> None:
        """Reinicia el promedio móvil."""
        self.values = []
        self.sum = 0.0
        
    def get_values(self) -> List[float]:
        """Obtiene los valores actuales.
        
        Returns:
            List[float]: Lista de valores
        """
        return list(self.values)


# Función principal para pruebas
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    # Probar algunas funciones
    print("Generar ID aleatorio:", generate_random_id())
    print("Generar ID con timestamp:", generate_timestamp_id("test"))
    
    # Probar hash de contraseña
    pwd_hash, salt = hash_password("contraseña123")
    print("Hash de contraseña:", pwd_hash)
    print("Salt:", salt)
    
    # Verificar contraseña
    verify_result = verify_password("contraseña123", pwd_hash, salt)
    print("Verificación correcta:", verify_result)
    
    verify_result = verify_password("contraseña_incorrecta", pwd_hash, salt)
    print("Verificación incorrecta:", verify_result)
    
    # Probar medición de tiempo
    with TimerContext("Test") as timer:
        # Simular operación
        time.sleep(1.5)
        
    print(f"Tiempo transcurrido: {timer.elapsed():.4f} segundos")
