#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Inspección
------------------
Contiene las clases y funciones necesarias para realizar la inspección visual
y la detección de defectos en las imágenes capturadas.
"""

import cv2
import numpy as np
import logging
import os
import json
from typing import Dict, List, Tuple, Any, Optional

class InspectionModule:
    """Clase base para los módulos de inspección."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Inicializa el módulo de inspección.
        
        Args:
            name: Nombre del módulo de inspección
            config: Configuración del módulo
        """
        self.logger = logging.getLogger('system_logger')
        self.name = name
        self.config = config or {}
        self.enabled = True
        
    def inspect(self, image: np.ndarray) -> Dict[str, Any]:
        """Método principal para realizar la inspección.
        
        Debe ser implementado por las clases hijas.
        
        Args:
            image: Imagen a inspeccionar en formato numpy array (BGR)
            
        Returns:
            Dict[str, Any]: Resultados de la inspección
        """
        raise NotImplementedError("El método inspect debe ser implementado por la clase hija")
        
    def enable(self) -> None:
        """Habilita el módulo de inspección."""
        self.enabled = True
        self.logger.info(f"Módulo de inspección '{self.name}' habilitado")
        
    def disable(self) -> None:
        """Deshabilita el módulo de inspección."""
        self.enabled = False
        self.logger.info(f"Módulo de inspección '{self.name}' deshabilitado")
        
    def is_enabled(self) -> bool:
        """Verifica si el módulo está habilitado.
        
        Returns:
            bool: True si el módulo está habilitado
        """
        return self.enabled
        
    def get_name(self) -> str:
        """Obtiene el nombre del módulo.
        
        Returns:
            str: Nombre del módulo
        """
        return self.name
        
    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración actual del módulo.
        
        Returns:
            Dict[str, Any]: Configuración del módulo
        """
        return self.config
        
    def set_config(self, config: Dict[str, Any]) -> None:
        """Establece la configuración del módulo.
        
        Args:
            config: Nueva configuración
        """
        self.config.update(config)
        self.logger.info(f"Configuración actualizada para el módulo '{self.name}'")


class ColorDetectionModule(InspectionModule):
    """Módulo para detectar colores fuera de rango en la imagen."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el módulo de detección de color.
        
        Args:
            config: Configuración del módulo con rangos de colores
        """
        super().__init__("ColorDetection", config)
        
        # Valores predeterminados si no se proporciona configuración
        if not self.config.get("color_ranges"):
            self.config["color_ranges"] = {
                "red": {"lower": [0, 100, 100], "upper": [10, 255, 255]},
                "blue": {"lower": [100, 100, 100], "upper": [130, 255, 255]}
            }
        
        self.logger.info("Módulo de detección de color inicializado")
        
    def inspect(self, image: np.ndarray) -> Dict[str, Any]:
        """Realiza la inspección de color en la imagen.
        
        Args:
            image: Imagen a inspeccionar (BGR)
            
        Returns:
            Dict[str, Any]: Resultados de la inspección
        """
        if not self.enabled:
            return {"status": "disabled"}
            
        # Convertir la imagen de BGR a HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        results = {"status": "pass", "detections": {}}
        
        # Verificar cada rango de color
        for color_name, ranges in self.config["color_ranges"].items():
            lower = np.array(ranges["lower"])
            upper = np.array(ranges["upper"])
            
            # Crear máscara para este rango de color
            mask = cv2.inRange(hsv_image, lower, upper)
            
            # Encontrar contornos en la máscara
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar por área mínima
            min_area = self.config.get("min_area", 100)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            if valid_contours:
                results["status"] = "fail"
                results["detections"][color_name] = {
                    "count": len(valid_contours),
                    "areas": [cv2.contourArea(cnt) for cnt in valid_contours],
                    "bounding_boxes": [cv2.boundingRect(cnt) for cnt in valid_contours]
                }
        
        return results


class DefectDetectionModule(InspectionModule):
    """Módulo para detectar defectos visuales en imágenes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el módulo de detección de defectos.
        
        Args:
            config: Configuración del módulo
        """
        super().__init__("DefectDetection", config)
        
        # Valores predeterminados
        if not self.config.get("threshold"):
            self.config["threshold"] = 127
        if not self.config.get("min_area"):
            self.config["min_area"] = 100
        if not self.config.get("reference_image"):
            self.config["reference_image"] = None
            
        # Cargar imagen de referencia si se proporciona
        self.reference_image = None
        if self.config["reference_image"] and os.path.exists(self.config["reference_image"]):
            self.reference_image = cv2.imread(self.config["reference_image"])
            if self.reference_image is not None:
                self.logger.info(f"Imagen de referencia cargada: {self.config['reference_image']}")
            else:
                self.logger.error(f"No se pudo cargar la imagen de referencia: {self.config['reference_image']}")
                
        self.logger.info("Módulo de detección de defectos inicializado")
        
    def set_reference_image(self, image: np.ndarray) -> None:
        """Establece una nueva imagen de referencia.
        
        Args:
            image: Imagen de referencia
        """
        self.reference_image = image.copy()
        self.logger.info("Nueva imagen de referencia establecida")
        
    def inspect(self, image: np.ndarray) -> Dict[str, Any]:
        """Realiza la inspección de defectos en la imagen.
        
        Args:
            image: Imagen a inspeccionar (BGR)
            
        Returns:
            Dict[str, Any]: Resultados de la inspección
        """
        if not self.enabled:
            return {"status": "disabled"}
            
        results = {"status": "pass", "defects": []}
        
        # Si hay una imagen de referencia, realizar comparación
        if self.reference_image is not None:
            # Asegurar que las imágenes tienen el mismo tamaño
            if image.shape != self.reference_image.shape:
                image = cv2.resize(image, (self.reference_image.shape[1], self.reference_image.shape[0]))
                
            # Calcular la diferencia absoluta
            diff = cv2.absdiff(image, self.reference_image)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Aplicar umbral
            _, thresh = cv2.threshold(gray_diff, self.config["threshold"], 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar por área mínima
            defects = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.config["min_area"]:
                    x, y, w, h = cv2.boundingRect(cnt)
                    defects.append({
                        "area": area,
                        "bbox": (x, y, w, h),
                        "confidence": float(np.mean(gray_diff[y:y+h, x:x+w]) / 255.0)
                    })
            
            if defects:
                results["status"] = "fail"
                results["defects"] = defects
        else:
            # Sin imagen de referencia, usar detección de bordes
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Encontrar contornos en la imagen de bordes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analizar contornos buscando formas irregulares
            defects = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.config["min_area"]:
                    # Calcular circularidad para detectar formas irregulares
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Si la circularidad es muy baja, podría ser un defecto
                    if circularity < 0.5:
                        x, y, w, h = cv2.boundingRect(cnt)
                        defects.append({
                            "area": area,
                            "bbox": (x, y, w, h),
                            "circularity": circularity
                        })
            
            if defects:
                results["status"] = "fail"
                results["defects"] = defects
                
        return results


class DimensionMeasurementModule(InspectionModule):
    """Módulo para medir dimensiones de objetos en imágenes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el módulo de medición de dimensiones.
        
        Args:
            config: Configuración del módulo
        """
        super().__init__("DimensionMeasurement", config)
        
        # Valores predeterminados
        if not self.config.get("pixels_per_mm"):
            self.config["pixels_per_mm"] = 1.0  # Calibración: píxeles por milímetro
        if not self.config.get("target_dimensions"):
            self.config["target_dimensions"] = {
                "width": {"min": 90, "max": 110},  # mm
                "height": {"min": 90, "max": 110}  # mm
            }
        if not self.config.get("threshold"):
            self.config["threshold"] = 127
            
        self.logger.info("Módulo de medición de dimensiones inicializado")
        
    def calibrate(self, image: np.ndarray, known_distance_mm: float) -> float:
        """Calibra la relación píxeles/mm usando un objeto de referencia.
        
        Args:
            image: Imagen con el objeto de referencia
            known_distance_mm: Distancia conocida en mm
            
        Returns:
            float: Píxeles por mm calibrados
        """
        # Procesamiento básico para detectar el objeto de referencia
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.config["threshold"], 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.logger.error("No se detectaron objetos para calibración")
            return self.config["pixels_per_mm"]
            
        # Usar el contorno más grande como referencia
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calcular píxeles por mm (usando la dimensión más grande)
        object_size_pixels = max(w, h)
        pixels_per_mm = object_size_pixels / known_distance_mm
        
        self.config["pixels_per_mm"] = pixels_per_mm
        self.logger.info(f"Calibración completada: {pixels_per_mm:.2f} píxeles/mm")
        
        return pixels_per_mm
        
    def inspect(self, image: np.ndarray) -> Dict[str, Any]:
        """Realiza la medición de dimensiones en la imagen.
        
        Args:
            image: Imagen a inspeccionar (BGR)
            
        Returns:
            Dict[str, Any]: Resultados de la inspección
        """
        if not self.enabled:
            return {"status": "disabled"}
            
        # Procesar la imagen para detectar el objeto
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, self.config["threshold"], 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = {"status": "pass", "measurements": []}
        
        if not contours:
            results["status"] = "error"
            results["error"] = "No se detectaron objetos"
            return results
            
        # Analizar cada contorno
        for cnt in contours:
            # Obtener rectángulo delimitador
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Convertir píxeles a mm
            width_mm = w / self.config["pixels_per_mm"]
            height_mm = h / self.config["pixels_per_mm"]
            
            # Verificar si las dimensiones están dentro de los rangos aceptables
            width_in_range = (self.config["target_dimensions"]["width"]["min"] <= width_mm <= 
                             self.config["target_dimensions"]["width"]["max"])
            height_in_range = (self.config["target_dimensions"]["height"]["min"] <= height_mm <= 
                              self.config["target_dimensions"]["height"]["max"])

            measurement = {
                "width_mm": width_mm,
                "height_mm": height_mm,
                "width_in_range": width_in_range,
                "height_in_range": height_in_range,
                "bbox": (x, y, w, h)
            }
            
            results["measurements"].append(measurement)
            
            # Si alguna medida está fuera de rango, marcar como fallo
            if not (width_in_range and height_in_range):
                results["status"] = "fail"
        
        return results


class TextureAnalysisModule(InspectionModule):
    """Módulo para analizar texturas y detectar irregularidades."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el módulo de análisis de textura.
        
        Args:
            config: Configuración del módulo
        """
        super().__init__("TextureAnalysis", config)
        
        # Valores predeterminados
        if not self.config.get("glcm_distance"):
            self.config["glcm_distance"] = 5
        if not self.config.get("glcm_angles"):
            self.config["glcm_angles"] = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        if not self.config.get("threshold_contrast"):
            self.config["threshold_contrast"] = 0.8
        if not self.config.get("threshold_homogeneity"):
            self.config["threshold_homogeneity"] = 0.7
            
        self.logger.info("Módulo de análisis de textura inicializado")
        
    def _calculate_glcm_features(self, image: np.ndarray) -> Dict[str, float]:
        """Calcula características GLCM (Gray-Level Co-occurrence Matrix).
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Dict[str, float]: Características de textura
        """
        # Escalar imagen a 8 niveles de gris para reducir cálculos
        bins = 8
        img_scaled = (image // (256 // bins)).astype(np.uint8)
        
        # Calcular matriz de co-ocurrencia
        glcm_features = {
            "contrast": 0.0,
            "dissimilarity": 0.0,
            "homogeneity": 0.0,
            "energy": 0.0,
            "correlation": 0.0
        }
        
        # Simulación simple de características GLCM
        # En una implementación real se usaría skimage.feature.graycomatrix
        h, w = img_scaled.shape
        contrast = 0.0
        homogeneity = 0.0
        energy = 0.0
        
        # Calcular características de manera simplificada (versión básica)
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Contrast: diferencia con píxeles vecinos
                diff = abs(int(img_scaled[i, j]) - int(img_scaled[i, j-1])) + \
                       abs(int(img_scaled[i, j]) - int(img_scaled[i, j+1])) + \
                       abs(int(img_scaled[i, j]) - int(img_scaled[i-1, j])) + \
                       abs(int(img_scaled[i, j]) - int(img_scaled[i+1, j]))
                contrast += diff / 4.0
                
                # Homogeneity: similitud con píxeles vecinos
                homogeneity += 1.0 / (1.0 + diff / 4.0)
                
                # Energy: suma de cuadrados
                energy += img_scaled[i, j] ** 2
        
        # Normalizar
        total_pixels = (h-2) * (w-2)
        if total_pixels > 0:
            glcm_features["contrast"] = contrast / total_pixels / bins
            glcm_features["homogeneity"] = homogeneity / total_pixels
            glcm_features["energy"] = energy / total_pixels / (bins ** 2)
        
        return glcm_features
        
    def inspect(self, image: np.ndarray) -> Dict[str, Any]:
        """Realiza el análisis de textura en la imagen.
        
        Args:
            image: Imagen a inspeccionar (BGR)
            
        Returns:
            Dict[str, Any]: Resultados de la inspección
        """
        if not self.enabled:
            return {"status": "disabled"}
            
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Dividir la imagen en regiones
        h, w = gray.shape
        region_size = min(100, h // 2, w // 2)  # Tamaño máximo de región
        
        results = {
            "status": "pass",
            "overall_features": {},
            "regions": []
        }
        
        # Calcular características para toda la imagen
        overall_features = self._calculate_glcm_features(gray)
        results["overall_features"] = overall_features
        
        # Analizar por regiones
        anomalies = []
        
        for y in range(0, h - region_size + 1, region_size):
            for x in range(0, w - region_size + 1, region_size):
                region = gray[y:y+region_size, x:x+region_size]
                region_features = self._calculate_glcm_features(region)
                
                # Detectar anomalías en textura
                is_anomaly = (
                    region_features["contrast"] > self.config["threshold_contrast"] or
                    region_features["homogeneity"] < self.config["threshold_homogeneity"]
                )
                
                region_result = {
                    "x": x,
                    "y": y,
                    "width": region_size,
                    "height": region_size,
                    "features": region_features,
                    "is_anomaly": is_anomaly
                }
                
                results["regions"].append(region_result)
                
                if is_anomaly:
                    anomalies.append(region_result)
        
        # Si hay anomalías, marcar como fallo
        if anomalies:
            results["status"] = "fail"
            results["anomalies"] = anomalies
        
        return results


class BarcodeQRDetectionModule(InspectionModule):
    """Módulo para detectar y validar códigos de barras y QR."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el módulo de detección de códigos.
        
        Args:
            config: Configuración del módulo
        """
        super().__init__("BarcodeQRDetection", config)
        
        # Valores predeterminados
        if not self.config.get("expected_formats"):
            self.config["expected_formats"] = ["QR", "EAN-13", "CODE-128"]
        if not self.config.get("expected_values"):
            self.config["expected_values"] = []
            
        # Verificar si podemos usar zbar (en una implementación real)
        self.zbar_available = False
        try:
            # Esta línea simula la verificación de disponibilidad de pyzbar
            # En un sistema real: from pyzbar import pyzbar
            self.zbar_available = True
        except ImportError:
            self.logger.warning("Librería pyzbar no disponible. Funcionalidad limitada.")
            
        self.logger.info("Módulo de detección de códigos inicializado")
        
    def inspect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecta y valida códigos de barras y QR en la imagen.
        
        Args:
            image: Imagen a inspeccionar (BGR)
            
        Returns:
            Dict[str, Any]: Resultados de la inspección
        """
        if not self.enabled:
            return {"status": "disabled"}
            
        results = {
            "status": "pass",
            "detected_codes": []
        }
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simular detección de códigos
        # En una implementación real, utilizaríamos pyzbar:
        # decoded_objects = pyzbar.decode(gray)
        
        # Simulación básica de detección con OpenCV
        # Aplicar umbral adaptativo
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Encontrar contornos que podrían ser códigos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_codes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Filtrar contornos pequeños
                # Aproximar contorno a un polígono
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                
                # Si es un cuadrilátero, podría ser un QR o código de barras
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Extraer la región
                    roi = gray[y:y+h, x:x+w]
                    
                    # Simular un código detectado
                    # En una aplicación real, esto vendría de pyzbar o similar
                    code_value = f"SAMPLE-{len(detected_codes)+1}"
                    code_type = "QR" if 0.8 < w/h < 1.2 else "BARCODE"
                    
                    code_info = {
                        "bbox": (x, y, w, h),
                        "value": code_value,
                        "type": code_type,
                        "valid": True
                    }
                    
                    # Validar contra valores esperados
                    if self.config["expected_values"] and code_value not in self.config["expected_values"]:
                        code_info["valid"] = False
                        results["status"] = "fail"
                    
                    detected_codes.append(code_info)
        
        # Verificar si se esperaban códigos pero no se encontraron
        if not detected_codes and self.config["expected_values"]:
            results["status"] = "fail"
            results["error"] = "No se detectaron códigos esperados"
        
        results["detected_codes"] = detected_codes
        return results


class InspectionController:
    """Controlador que gestiona múltiples módulos de inspección."""
    
    def __init__(self):
        """Inicializa el controlador de inspección."""
        self.logger = logging.getLogger('system_logger')
        self.modules = {}
        
    def add_module(self, module: InspectionModule) -> None:
        """Agrega un módulo al controlador.
        
        Args:
            module: Módulo de inspección a agregar
        """
        self.modules[module.get_name()] = module
        self.logger.info(f"Módulo '{module.get_name()}' agregado al controlador")
        
    def remove_module(self, module_name: str) -> bool:
        """Elimina un módulo del controlador.
        
        Args:
            module_name: Nombre del módulo a eliminar
            
        Returns:
            bool: True si se eliminó correctamente
        """
        if module_name in self.modules:
            del self.modules[module_name]
            self.logger.info(f"Módulo '{module_name}' eliminado del controlador")
            return True
        else:
            self.logger.warning(f"Módulo '{module_name}' no encontrado")
            return False
        
    def get_module(self, module_name: str) -> Optional[InspectionModule]:
        """Obtiene un módulo por su nombre.
        
        Args:
            module_name: Nombre del módulo
            
        Returns:
            Optional[InspectionModule]: Módulo solicitado o None si no existe
        """
        return self.modules.get(module_name)
        
    def get_all_modules(self) -> Dict[str, InspectionModule]:
        """Obtiene todos los módulos registrados.
        
        Returns:
            Dict[str, InspectionModule]: Diccionario de módulos
        """
        return self.modules
        
    def inspect_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Ejecuta todos los módulos de inspección activos sobre una imagen.
        
        Args:
            image: Imagen a inspeccionar
            
        Returns:
            Dict[str, Any]: Resultados combinados de todos los módulos
        """
        results = {
            "status": "pass",
            "timestamp": np.datetime64('now').astype(str),
            "module_results": {}
        }
        
        # Ejecutar todos los módulos habilitados
        for name, module in self.modules.items():
            if module.is_enabled():
                try:
                    module_result = module.inspect(image)
                    results["module_results"][name] = module_result
                    
                    # Si algún módulo falla, el resultado general es fallo
                    if module_result.get("status") == "fail":
                        results["status"] = "fail"
                        
                except Exception as e:
                    self.logger.error(f"Error en módulo '{name}': {str(e)}")
                    results["module_results"][name] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        return results
        
    def load_configuration(self, config_file: str) -> bool:
        """Carga la configuración para todos los módulos desde un archivo.
        
        Args:
            config_file: Ruta al archivo de configuración JSON
            
        Returns:
            bool: True si se cargó correctamente
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Aplicar configuración a cada módulo
            for module_name, module_config in config.items():
                if module_name in self.modules:
                    self.modules[module_name].set_config(module_config)
                    self.logger.info(f"Configuración cargada para '{module_name}'")
                    
            return True
        except Exception as e:
            self.logger.error(f"Error al cargar configuración: {str(e)}")
            return False
            
    def save_configuration(self, config_file: str) -> bool:
        """Guarda la configuración de todos los módulos en un archivo.
        
        Args:
            config_file: Ruta al archivo de configuración JSON
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            config = {}
            for name, module in self.modules.items():
                config[name] = module.get_config()
                
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
                
            self.logger.info(f"Configuración guardada en '{config_file}'")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar configuración: {str(e)}")
            return False


# Función de ayuda para crear un controlador con los módulos estándar
def create_standard_inspection_controller() -> InspectionController:
    """Crea un controlador con los módulos estándar de inspección.
    
    Returns:
        InspectionController: Controlador configurado
    """
    controller = InspectionController()
    
    # Agregar módulos estándar
    controller.add_module(ColorDetectionModule())
    controller.add_module(DefectDetectionModule())
    controller.add_module(DimensionMeasurementModule())
    controller.add_module(TextureAnalysisModule())
    controller.add_module(BarcodeQRDetectionModule())
    
    return controller