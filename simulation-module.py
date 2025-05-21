#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Simulación
-----------------
Proporciona funcionalidades para simular componentes del sistema,
especialmente útil para pruebas sin hardware real.
"""

import cv2
import numpy as np
import os
import random
import time
import logging
from typing import Dict, List, Tuple, Any, Optional, Generator
from datetime import datetime
import threading
import queue


class CameraSimulator:
    """Simula una cámara para pruebas sin hardware físico."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el simulador de cámara.
        
        Args:
            config: Configuración del simulador
        """
        self.logger = logging.getLogger('system_logger')
        self.config = config or {}
        
        # Resolver rutas de imágenes
        self.image_dir = self.config.get("image_dir", "simulation/images")
        self.reference_images = self._load_images()
        
        # Parámetros de imagen
        self.width = self.config.get("width", 640)
        self.height = self.config.get("height", 480)
        self.fps = self.config.get("fps", 30)
        
        # Estado de conexión
        self.is_connected = False
        self.frame_index = 0
        
        # Modo de simulación
        self.mode = self.config.get("mode", "sequence")  # "sequence", "random", "generate"
        
        # Parametros para imágenes generadas
        self.defect_probability = self.config.get("defect_probability", 0.3)
        self.noise_level = self.config.get("noise_level", 0.05)
        
        # Para streaming simulado
        self.streaming = False
        self.stream_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        
        self.logger.info("Simulador de cámara inicializado")
        
    def _load_images(self) -> List[np.ndarray]:
        """Carga imágenes de muestra desde el directorio configurado.
        
        Returns:
            List[np.ndarray]: Lista de imágenes cargadas
        """
        images = []
        
        if not os.path.exists(self.image_dir):
            self.logger.warning(f"Directorio de imágenes de simulación no encontrado: {self.image_dir}")
            
            # Crear imágenes de ejemplo si no hay directorio
            images = self._create_sample_images()
            return images
        
        # Listar archivos de imagen
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for filename in os.listdir(self.image_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                try:
                    image_path = os.path.join(self.image_dir, filename)
                    img = cv2.imread(image_path)
                    
                    if img is not None:
                        # Redimensionar si es necesario
                        if img.shape[1] != self.width or img.shape[0] != self.height:
                            img = cv2.resize(img, (self.width, self.height))
                        
                        images.append(img)
                        self.logger.debug(f"Imagen cargada: {filename}")
                except Exception as e:
                    self.logger.error(f"Error al cargar imagen {filename}: {str(e)}")
        
        if not images:
            self.logger.warning("No se encontraron imágenes válidas, creando imágenes de ejemplo")
            images = self._create_sample_images()
            
        return images
    
    def _create_sample_images(self) -> List[np.ndarray]:
        """Crea imágenes de muestra si no hay disponibles.
        
        Returns:
            List[np.ndarray]: Lista de imágenes generadas
        """
        images = []
        
        # Crear directorio si no existe
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Pieza correcta (rectángulo azul)
        good_part = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(good_part, (self.width//4, self.height//4), 
                     (3*self.width//4, 3*self.height//4), (255, 0, 0), -1)
        
        # Pieza con defecto de color (rectángulo rojo)
        color_defect = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(color_defect, (self.width//4, self.height//4), 
                     (3*self.width//4, 3*self.height//4), (0, 0, 255), -1)
        
        # Pieza con defecto de forma (rectángulo azul con esquina dañada)
        shape_defect = good_part.copy()
        # Añadir triángulo negro (defecto) en la esquina superior derecha
        pts = np.array([[3*self.width//4, self.height//4], 
                       [3*self.width//4-50, self.height//4], 
                       [3*self.width//4, self.height//4+50]], np.int32)
        cv2.fillPoly(shape_defect, [pts], (0, 0, 0))
        
        # Pieza con defecto de dimensión (rectángulo azul más pequeño)
        dimension_defect = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(dimension_defect, (self.width//3, self.height//3), 
                     (2*self.width//3, 2*self.height//3), (255, 0, 0), -1)
        
        # Guardar imágenes
        images = [good_part, color_defect, shape_defect, dimension_defect]
        
        for i, img in enumerate(images):
            filename = os.path.join(self.image_dir, f"sample_{i+1}.jpg")
            cv2.imwrite(filename, img)
            self.logger.info(f"Imagen de muestra creada: {filename}")
            
        return images
    
    def connect(self) -> bool:
        """Simula la conexión con la cámara.
        
        Returns:
            bool: True si la conexión fue exitosa
        """
        if not self.reference_images:
            self.logger.error("No hay imágenes de referencia para simular la cámara")
            return False
            
        self.frame_index = 0
        self.is_connected = True
        self.logger.info("Cámara simulada conectada")
        return True
    
    def disconnect(self) -> None:
        """Simula la desconexión de la cámara."""
        if self.streaming:
            self.stop_streaming()
            
        self.is_connected = False
        self.logger.info("Cámara simulada desconectada")
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Obtiene un frame de la cámara simulada.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (éxito, frame)
        """
        if not self.is_connected:
            return False, None
            
        if self.mode == "sequence":
            # Modo secuencia: recorrer imágenes en orden
            frame = self.reference_images[self.frame_index].copy()
            self.frame_index = (self.frame_index + 1) % len(self.reference_images)
            
        elif self.mode == "random":
            # Modo aleatorio: seleccionar imagen al azar
            frame = random.choice(self.reference_images).copy()
            
        elif self.mode == "generate":
            # Modo generación: crear imagen con parámetros
            frame = self._generate_frame()
        else:
            self.logger.error(f"Modo de simulación no reconocido: {self.mode}")
            return False, None
            
        # Agregar ruido si está configurado
        if self.noise_level > 0:
            frame = self._add_noise(frame, self.noise_level)
            
        # Simular retraso para fps realistas
        time.sleep(1.0 / self.fps)
        
        return True, frame
    
    def _generate_frame(self) -> np.ndarray:
        """Genera un frame sintético para simulación.
        
        Returns:
            np.ndarray: Frame generado
        """
        # Crear imagen base (rectángulo azul en fondo negro)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calcular dimensiones del rectángulo
        rect_width = self.width // 2
        rect_height = self.height // 2
        
        # Calcular posición centrada
        x = (self.width - rect_width) // 2
        y = (self.height - rect_height) // 2
        
        # Decidir si se genera un defecto
        has_defect = random.random() < self.defect_probability
        
        if has_defect:
            # Seleccionar tipo de defecto
            defect_type = random.choice(["color", "shape", "dimension"])
            
            if defect_type == "color":
                # Defecto de color: usar rojo en lugar de azul
                color = (0, 0, 255)  # BGR: rojo
            elif defect_type == "shape":
                # Defecto de forma: añadir "mordida" al rectángulo
                color = (255, 0, 0)  # BGR: azul
                cv2.rectangle(frame, (x, y), (x + rect_width, y + rect_height), color, -1)
                
                # Añadir defecto (triángulo negro en una esquina)
                corner = random.randint(0, 3)
                if corner == 0:  # Superior izquierda
                    pts = np.array([[x, y], [x+50, y], [x, y+50]], np.int32)
                elif corner == 1:  # Superior derecha
                    pts = np.array([[x+rect_width, y], [x+rect_width-50, y], [x+rect_width, y+50]], np.int32)
                elif corner == 2:  # Inferior izquierda
                    pts = np.array([[x, y+rect_height], [x+50, y+rect_height], [x, y+rect_height-50]], np.int32)
                else:  # Inferior derecha
                    pts = np.array([[x+rect_width, y+rect_height], [x+rect_width-50, y+rect_height], 
                                   [x+rect_width, y+rect_height-50]], np.int32)
                
                cv2.fillPoly(frame, [pts], (0, 0, 0))
                return frame
                
            elif defect_type == "dimension":
                # Defecto de dimensión: cambiar tamaño del rectángulo
                factor = random.choice([0.7, 1.3])  # Más pequeño o más grande
                rect_width = int(rect_width * factor)
                rect_height = int(rect_height * factor)
                
                # Recalcular posición centrada
                x = (self.width - rect_width) // 2
                y = (self.height - rect_height) // 2
                
                color = (255, 0, 0)  # BGR: azul
        else:
            # Sin defecto: rectángulo azul normal
            color = (255, 0, 0)  # BGR: azul
            
        # Dibujar rectángulo
        cv2.rectangle(frame, (x, y), (x + rect_width, y + rect_height), color, -1)
        
        return frame
    
    def _add_noise(self, image: np.ndarray, level: float) -> np.ndarray:
        """Añade ruido a la imagen para simular condiciones reales.
        
        Args:
            image: Imagen original
            level: Nivel de ruido (0.0 - 1.0)
            
        Returns:
            np.ndarray: Imagen con ruido
        """
        if level <= 0:
            return image
            
        # Copia para no modificar original
        result = image.copy()
        
        # Ruido gaussiano
        mean = 0
        stddev = level * 50  # Ajustar según nivel
        noise = np.random.normal(mean, stddev, image.shape).astype(np.int16)
        
        # Añadir ruido a la imagen
        result = cv2.add(result, noise.astype(np.int8))
        
        return result
    
    def get_properties(self) -> Dict[str, Any]:
        """Obtiene las propiedades simuladas de la cámara.
        
        Returns:
            Dict[str, Any]: Propiedades simuladas
        """
        if not self.is_connected:
            return {}
            
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'mode': self.mode,
            'images_count': len(self.reference_images),
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'defect_probability': self.defect_probability,
            'noise_level': self.noise_level
        }
    
    def set_property(self, prop_name: str, value: Any) -> bool:
        """Establece una propiedad en la cámara simulada.
        
        Args:
            prop_name: Nombre de la propiedad
            value: Valor a establecer
            
        Returns:
            bool: True si se estableció correctamente
        """
        if not self.is_connected:
            return False
            
        try:
            if prop_name == "width":
                self.width = int(value)
            elif prop_name == "height":
                self.height = int(value)
            elif prop_name == "fps":
                self.fps = float(value)
            elif prop_name == "mode":
                if value in ["sequence", "random", "generate"]:
                    self.mode = value
                else:
                    return False
            elif prop_name == "defect_probability":
                self.defect_probability = float(value)
            elif prop_name == "noise_level":
                self.noise_level = float(value)
            else:
                return False
                
            return True
        except Exception:
            return False
    
    def start_streaming(self) -> bool:
        """Inicia streaming simulado en un hilo separado.
        
        Returns:
            bool: True si se inició correctamente
        """
        if self.streaming:
            return True
            
        if not self.is_connected:
            return False
            
        # Iniciar hilo
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        
        self.logger.info("Streaming simulado iniciado")
        return True
    
    def stop_streaming(self) -> None:
        """Detiene el streaming simulado."""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1.0)
            self.stream_thread = None
            
        # Limpiar cola
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        self.logger.info("Streaming simulado detenido")
    
    def _stream_worker(self) -> None:
        """Función de trabajo para el hilo de streaming."""
        while self.streaming and self.is_connected:
            try:
                # Obtener frame
                ret, frame = self.get_frame()
                
                if ret and frame is not None:
                    # Intentar poner en la cola, descartar si está llena
                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        pass
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error en streaming simulado: {str(e)}")
                time.sleep(0.5)
    
    def get_streaming_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Obtiene el frame más reciente del streaming.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (éxito, frame)
        """
        if not self.streaming or not self.is_connected:
            return False, None
            
        try:
            frame = self.frame_queue.get(block=False)
            return True, frame
        except queue.Empty:
            return False, None


class ProductSimulator:
    """Simula productos y defectos para pruebas del sistema."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el simulador de productos.
        
        Args:
            config: Configuración del simulador
        """
        self.logger = logging.getLogger('system_logger')
        self.config = config or {}
        
        # Parámetros de simulación
        self.sku_id = self.config.get("sku_id", "DEMO001")
        self.defect_probability = self.config.get("defect_probability", 0.3)
        self.defect_types = self.config.get("defect_types", ["color", "dimension", "shape"])
        
        # Parámetros nominales del producto
        self.nominal_dimensions = self.config.get("nominal_dimensions", {
            "width": 100.0,  # mm
            "height": 50.0,  # mm
            "tolerance": 5.0  # mm
        })
        
        # Colores aceptables
        self.acceptable_colors = self.config.get("acceptable_colors", [
            {"name": "blue", "lower": [100, 100, 100], "upper": [130, 255, 255]},
            {"name": "black", "lower": [0, 0, 0], "upper": [180, 255, 30]}
        ])
        
        self.logger.info("Simulador de productos inicializado")
    
    def generate_product(self) -> Dict[str, Any]:
        """Genera un producto simulado (con o sin defectos).
        
        Returns:
            Dict[str, Any]: Información del producto generado
        """
        # Decidir si se genera un defecto
        has_defect = random.random() < self.defect_probability
        
        product = {
            "sku_id": self.sku_id,
            "timestamp": datetime.now().isoformat(),
            "has_defect": has_defect,
            "defects": [],
            "dimensions": {},
            "color": {}
        }
        
        # Generar dimensiones
        width = self.nominal_dimensions["width"]
        height = self.nominal_dimensions["height"]
        tolerance = self.nominal_dimensions["tolerance"]
        
        if has_defect and "dimension" in self.defect_types and random.random() < 0.5:
            # Generar defecto de dimensión
            factor = random.choice([0.7, 1.3])  # Más pequeño o más grande
            
            if random.random() < 0.5:
                # Defecto en ancho
                width = width * factor
                product["defects"].append({
                    "type": "dimension",
                    "parameter": "width",
                    "value": width,
                    "nominal": self.nominal_dimensions["width"],
                    "deviation": width - self.nominal_dimensions["width"]
                })
            else:
                # Defecto en alto
                height = height * factor
                product["defects"].append({
                    "type": "dimension",
                    "parameter": "height",
                    "value": height,
                    "nominal": self.nominal_dimensions["height"],
                    "deviation": height - self.nominal_dimensions["height"]
                })
        else:
            # Dimensiones normales con variación aleatoria
            width += random.uniform(-tolerance/2, tolerance/2)
            height += random.uniform(-tolerance/2, tolerance/2)
        
        product["dimensions"] = {
            "width": width,
            "height": height,
            "in_tolerance_width": abs(width - self.nominal_dimensions["width"]) <= tolerance,
            "in_tolerance_height": abs(height - self.nominal_dimensions["height"]) <= tolerance
        }
        
        # Generar color
        if has_defect and "color" in self.defect_types and random.random() < 0.5:
            # Defecto de color
            # Generar un color fuera de los rangos aceptables
            hue = random.randint(0, 179)
            # Buscar un hue que esté fuera de todos los rangos aceptables
            while any(color["lower"][0] <= hue <= color["upper"][0] for color in self.acceptable_colors):
                hue = random.randint(0, 179)
                
            product["color"] = {
                "name": "unknown",
                "hsv": [hue, random.randint(100, 255), random.randint(100, 255)],
                "in_range": False
            }
            
            product["defects"].append({
                "type": "color",
                "parameter": "hue",
                "value": hue,
                "description": "Color fuera de rango aceptable"
            })
        else:
            # Color normal
            acceptable_color = random.choice(self.acceptable_colors)
            hue_range = acceptable_color["upper"][0] - acceptable_color["lower"][0]
            hue = acceptable_color["lower"][0] + random.randint(0, hue_range)
            sat = random.randint(acceptable_color["lower"][1], acceptable_color["upper"][1])
            val = random.randint(acceptable_color["lower"][2], acceptable_color["upper"][2])
            
            product["color"] = {
                "name": acceptable_color["name"],
                "hsv": [hue, sat, val],
                "in_range": True
            }
        
        # Defectos de forma
        if has_defect and "shape" in self.defect_types and random.random() < 0.5:
            # Simular defecto de forma
            corner = random.randint(0, 3)
            corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
            
            product["defects"].append({
                "type": "shape",
                "parameter": "corner",
                "location": corner_names[corner],
                "description": f"Defecto en esquina {corner_names[corner]}"
            })
        
        return product
    
    def generate_product_stream(self, count: int = None, interval: float = 1.0) -> Generator[Dict[str, Any], None, None]:
        """Genera un flujo continuo de productos simulados.
        
        Args:
            count: Número de productos a generar (None para infinito)
            interval: Intervalo entre productos (segundos)
            
        Yields:
            Dict[str, Any]: Información del producto generado
        """
        generated = 0
        
        while count is None or generated < count:
            product = self.generate_product()
            yield product
            
            generated += 1
            time.sleep(interval)
    
    def get_product_config(self) -> Dict[str, Any]:
        """Obtiene la configuración del producto simulado.
        
        Returns:
            Dict[str, Any]: Configuración del producto
        """
        return {
            "sku_id": self.sku_id,
            "nominal_dimensions": self.nominal_dimensions,
            "acceptable_colors": self.acceptable_colors,
            "defect_probability": self.defect_probability,
            "defect_types": self.defect_types
        }
    
    def set_defect_probability(self, probability: float) -> None:
        """Establece la probabilidad de defectos.
        
        Args:
            probability: Probabilidad (0.0 - 1.0)
        """
        self.defect_probability = max(0.0, min(1.0, probability))
        self.logger.info(f"Probabilidad de defectos establecida a {self.defect_probability:.2f}")


class SimulationEnvironment:
    """Entorno completo de simulación que integra todos los simuladores."""
    
    def __init__(self, config_path: str = None):
        """Inicializa el entorno de simulación.
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
        """
        self.logger = logging.getLogger('system_logger')
        
        # Cargar configuración si se proporciona
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                self.logger.error(f"Error al cargar configuración de simulación: {str(e)}")
                self.config = {}
        
        # Inicializar componentes
        camera_config = self.config.get("camera", {})
        self.camera = CameraSimulator(camera_config)
        
        product_config = self.config.get("product", {})
        self.product = ProductSimulator(product_config)
        
        # Estado de la simulación
        self.running = False
        self.simulation_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.product_queue = queue.Queue(maxsize=10)
        
        self.logger.info("Entorno de simulación inicializado")
    
    def start(self) -> bool:
        """Inicia la simulación completa.
        
        Returns:
            bool: True si se inició correctamente
        """
        if self.running:
            return True
            
        # Conectar cámara
        if not self.camera.connect():
            self.logger.error("No se pudo conectar la cámara simulada")
            return False
            
        # Iniciar simulación en hilo separado
        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulation_worker, daemon=True)
        self.simulation_thread.start()
        
        self.logger.info("Simulación iniciada")
        return True
    
    def stop(self) -> None:
        """Detiene la simulación."""
        self.running = False
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
            self.simulation_thread = None
            
        # Desconectar cámara
        self.camera.disconnect()
        
        # Limpiar colas
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.product_queue.empty():
            try:
                self.product_queue.get_nowait()
            except queue.Empty:
                break
                
        self.logger.info("Simulación detenida")
    
    def _simulation_worker(self) -> None:
        """Función de trabajo para el hilo de simulación."""
        interval = self.config.get("simulation_interval", 1.0)
        
        while self.running:
            try:
                # Generar producto
                product = self.product.generate_product()
                
                # Obtener frame de la cámara
                ret, frame = self.camera.get_frame()
                
                if ret and frame is not None:
                    # Intentar poner en las colas, descartar si están llenas
                    try:
                        self.frame_queue.put(frame, block=False)
                        self.product_queue.put(product, block=False)
                    except queue.Full:
                        pass
                
                # Esperar antes de siguiente iteración
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error en simulación: {str(e)}")
                time.sleep(0.5)
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Obtiene el frame más reciente de la simulación.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (éxito, frame)
        """
        if not self.running:
            return False, None
            
        try:
            frame = self.frame_queue.get(block=False)
            return True, frame
        except queue.Empty:
            # Si la cola está vacía, intentar obtener un frame directamente
            return self.camera.get_frame()
    
    def get_product(self) -> Optional[Dict[str, Any]]:
        """Obtiene la información del producto más reciente.
        
        Returns:
            Optional[Dict[str, Any]]: Información del producto o None si no hay
        """
        if not self.running:
            return None
            
        try:
            product = self.product_queue.get(block=False)
            return product
        except queue.Empty:
            return None
    
    def set_defect_probability(self, probability: float) -> None:
        """Establece la probabilidad de defectos para la simulación.
        
        Args:
            probability: Probabilidad (0.0 - 1.0)
        """
        self.product.set_defect_probability(probability)
        self.camera.defect_probability = probability
        
    def save_simulation_config(self, config_path: str) -> bool:
        """Guarda la configuración actual de la simulación.
        
        Args:
            config_path: Ruta donde guardar la configuración
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            # Recopilar configuración
            config = {
                "camera": self.camera.config,
                "product": self.product.config,
                "simulation_interval": self.config.get("simulation_interval", 1.0)
            }
            
            # Actualizar valores actuales
            config["camera"]["defect_probability"] = self.camera.defect_probability
            config["camera"]["noise_level"] = self.camera.noise_level
            config["camera"]["mode"] = self.camera.mode
            
            config["product"]["defect_probability"] = self.product.defect_probability
            config["product"]["sku_id"] = self.product.sku_id
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Guardar como JSON
            import json
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            self.logger.info(f"Configuración de simulación guardada en {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar configuración de simulación: {str(e)}")
            return False


# Para pruebas directas del módulo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('system_logger')
    
    # Opciones de prueba
    test_camera = True
    test_product = True
    test_environment = True
    
    # Probar simulador de cámara
    if test_camera:
        logger.info("Probando simulador de cámara...")
        camera_sim = CameraSimulator({
            "width": 640,
            "height": 480,
            "fps": 10,
            "mode": "generate",
            "defect_probability": 0.4
        })
        
        if camera_sim.connect():
            # Capturar frames
            for i in range(5):
                ret, frame = camera_sim.get_frame()
                if ret:
                    cv2.imwrite(f"test_cam_frame_{i+1}.jpg", frame)
                    logger.info(f"Frame {i+1} guardado")
            
            # Probar streaming
            camera_sim.start_streaming()
            time.sleep(2)  # Dejar que se generen algunos frames
            
            ret, stream_frame = camera_sim.get_streaming_frame()
            if ret:
                cv2.imwrite("test_stream_frame.jpg", stream_frame)
                logger.info("Frame de streaming guardado")
                
            camera_sim.stop_streaming()
            camera_sim.disconnect()
    
    # Probar simulador de productos
    if test_product:
        logger.info("Probando simulador de productos...")
        product_sim = ProductSimulator({
            "sku_id": "TEST001",
            "defect_probability": 0.5,
            "nominal_dimensions": {
                "width": 100.0,
                "height": 50.0,
                "tolerance": 5.0
            }
        })
        
        # Generar productos
        for i in range(5):
            product = product_sim.generate_product()
            logger.info(f"Producto {i+1}: {'Con defectos' if product['has_defect'] else 'Sin defectos'}")
            if product["defects"]:
                for defect in product["defects"]:
                    logger.info(f"  - Defecto: {defect['type']}")
        
        # Probar stream
        logger.info("Probando stream de productos...")
        for i, product in enumerate(product_sim.generate_product_stream(count=3, interval=0.5)):
            logger.info(f"Producto stream {i+1}: {'Con defectos' if product['has_defect'] else 'Sin defectos'}")
    
    # Probar entorno completo
    if test_environment:
        logger.info("Probando entorno de simulación completo...")
        
        # Crear directorio para configuración
        os.makedirs("simulation/config", exist_ok=True)
        
        # Crear configuración de ejemplo
        example_config = {
            "camera": {
                "width": 800,
                "height": 600,
                "fps": 15,
                "mode": "generate",
                "defect_probability": 0.3,
                "noise_level": 0.1
            },
            "product": {
                "sku_id": "SIMTEST",
                "defect_probability": 0.3,
                "defect_types": ["color", "dimension", "shape"]
            },
            "simulation_interval": 0.5
        }
        
        config_path = "simulation/config/test_sim.json"
        with open(config_path, 'w') as f:
            json.dump(example_config, f, indent=2)
        
        # Iniciar simulación
        env = SimulationEnvironment(config_path)
        if env.start():
            # Ejecutar por unos segundos
            for i in range(5):
                time.sleep(1)
                
                # Obtener frame y producto
                ret, frame = env.get_frame()
                product = env.get_product()
                
                if ret and product:
                    logger.info(f"Iteración {i+1}: Frame recibido, Producto: {product['sku_id']}, " +
                              f"{'Con defectos' if product['has_defect'] else 'Sin defectos'}")
                    
                    # Guardar último frame
                    if i == 4:
                        cv2.imwrite("test_env_frame.jpg", frame)
            
            # Guardar configuración actualizada
            env.save_simulation_config("simulation/config/updated_sim.json")
            
            # Detener simulación
            env.stop()
    
    logger.info("Pruebas de simulación completadas")