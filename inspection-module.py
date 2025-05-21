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
            
            # Crear medición
            measurement = {
                "width_pixels": w,
                "height_pixels": h,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "position": (x, y),
                "in_tolerance": width_in_range and height_in_range
            }
            
            results["measurements"].append(measurement)
            
            # Si alguna medición está fuera de tolerancia, fallar la inspección
            if not (width_in_range and height_in_range):
                results["status"] = "fail"
        
        return results


class TextVerificationModule(InspectionModule):
    """Módulo para verificar texto en imágenes mediante OCR."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el módulo de verificación de texto.
        
        Args:
            config: Configuración del módulo
        """
        super().__init__("TextVerification", config)
        
        # Verificar si pytesseract está instalado
        try:
            import pytesseract
            self.ocr_available = True
        except ImportError:
            self.logger.warning("pytesseract no está instalado. Funcionalidad OCR limitada.")
            self.ocr_available = False
        
        # Valores predeterminados
        if not self.config.get("target_text"):
            self.config["target_text"] = ""
        if not self.config.get("min_confidence"):
            self.config["min_confidence"] = 0.7
        if not self.config.get("preprocess"):
            self.config["preprocess"] = True
            
        self.logger.info("Módulo de verificación de texto inicializado")
        
    def inspect(self, image: np.ndarray) -> Dict[str, Any]:
        """Realiza la verificación de texto en la imagen.
        
        Args:
            image: Imagen a inspeccionar (BGR)
            
        Returns:
            Dict[str, Any]: Resultados de la inspección
        """
        if not self.enabled:
            return {"status": "disabled"}
            
        if not self.ocr_available:
            return {"status": "error", "error": "OCR no disponible"}
            
        import pytesseract
        
        results = {"status": "pass", "detected_text": "", "confidence": 0.0, "matches_target": False}
        
        # Preprocesar la imagen para mejorar OCR
        if self.config["preprocess"]:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Realizar OCR
        try:
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            # Procesar resultados del OCR
            texts = []
            confidences = []
            
            for i in range(len(ocr_data["text"])):
                if int(ocr_data["conf"][i]) > 0:  # Solo considerar resultados con confianza > 0
                    text = ocr_data["text"][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(float(ocr_data["conf"][i]) / 100.0)  # Convertir a escala 0-1
            
            # Combinar resultados
            if texts:
                results["detected_text"] = " ".join(texts)
                results["confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Verificar si el texto coincide con el objetivo
                target = self.config["target_text"].strip().lower()
                detected = results["detected_text"].lower()
                
                # Comprobar si el texto objetivo está contenido en el detectado
                if target in detected:
                    results["matches_target"] = True
                # Alternativa: verificar similitud usando distancia de Levenshtein
                elif target and self._string_similarity(target, detected) > 0.8:
                    results["matches_target"] = True
                    results["similarity"] = self._string_similarity(target, detected)
                
                # Establecer estado según la coincidencia y confianza
                if not results["matches_target"] or results["confidence"] < self.config["min_confidence"]:
                    results["status"] = "fail"
            else:
                results["status"] = "fail"
                results["error"] = "No se detectó texto"
                
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            
        return results
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calcula la similitud entre dos cadenas usando distancia de Levenshtein.
        
        Args:
            s1: Primera cadena
            s2: Segunda cadena
            
        Returns:
            float: Similitud entre 0 (totalmente diferente) y 1 (idéntico)
        """
        # Implementación simple de distancia de Levenshtein
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
            
        # Matriz para cálculo de distancia
        d = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
        
        # Inicializar primera fila y columna
        for i in range(len(s1) + 1):
            d[i][0] = i
        for j in range(len(s2) + 1):
            d[0][j] = j
            
        # Calcular distancia
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,      # eliminación
                    d[i][j-1] + 1,      # inserción
                    d[i-1][j-1] + cost  # sustitución
                )
                
        # Convertir distancia a similitud
        max_len = max(len(s1), len(s2))
        distance = d[len(s1)][len(s2)]
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0
        
        return similarity


class InspectionManager:
    """Clase para gestionar y coordinar múltiples módulos de inspección."""
    
    def __init__(self):
        """Inicializa el gestor de inspección."""
        self.logger = logging.getLogger('system_logger')
        self.modules = {}
        self.results_history = []
        self.max_history_size = 100
        
    def add_module(self, module: InspectionModule) -> None:
        """Agrega un módulo de inspección.
        
        Args:
            module: Módulo de inspección a agregar
        """
        self.modules[module.get_name()] = module
        self.logger.info(f"Módulo '{module.get_name()}' agregado al gestor de inspección")
        
    def remove_module(self, module_name: str) -> bool:
        """Elimina un módulo de inspección.
        
        Args:
            module_name: Nombre del módulo a eliminar
            
        Returns:
            bool: True si se eliminó correctamente
        """
        if module_name in self.modules:
            del self.modules[module_name]
            self.logger.info(f"Módulo '{module_name}' eliminado del gestor de inspección")
            return True
        return False
        
    def get_module(self, module_name: str) -> Optional[InspectionModule]:
        """Obtiene un módulo de inspección por nombre.
        
        Args:
            module_name: Nombre del módulo
            
        Returns:
            Optional[InspectionModule]: Módulo de inspección o None si no existe
        """
        return self.modules.get(module_name)
        
    def inspect_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Realiza la inspección completa de una imagen con todos los módulos.
        
        Args:
            image: Imagen a inspeccionar (BGR)
            
        Returns:
            Dict[str, Any]: Resultados combinados de todos los módulos
        """
        timestamp = datetime.now().isoformat()
        
        combined_results = {
            "timestamp": timestamp,
            "overall_status": "pass",
            "module_results": {}
        }
        
        # Ejecutar cada módulo de inspección
        for name, module in self.modules.items():
            if module.is_enabled():
                results = module.inspect(image)
                combined_results["module_results"][name] = results
                
                # Si algún módulo falla, el resultado general es fallo
                if results.get("status") == "fail":
                    combined_results["overall_status"] = "fail"
                # Si algún módulo tiene error pero ninguno ha fallado aún, marcar como error
                elif results.get("status") == "error" and combined_results["overall_status"] != "fail":
                    combined_results["overall_status"] = "error"
        
        # Guardar en historial
        self.results_history.append(combined_results)
        
        # Limitar tamaño del historial
        if len(self.results_history) > self.max_history_size:
            self.results_history.pop(0)
            
        return combined_results
    
    def load_config_from_sku(self, sku_config: Dict[str, Any]) -> None:
        """Carga la configuración para todos los módulos desde un SKU.
        
        Args:
            sku_config: Configuración del SKU
        """
        if "inspection_modules" not in sku_config:
            self.logger.error("Configuración de SKU inválida: 'inspection_modules' no encontrado")
            return
            
        # Crear y configurar módulos según la configuración del SKU
        modules_config = sku_config["inspection_modules"]
        
        # Color detection
        if "color" in modules_config:
            color_config = modules_config["color"]
            if self.get_module("ColorDetection"):
                self.get_module("ColorDetection").set_config(color_config)
            else:
                color_module = ColorDetectionModule(color_config)
                self.add_module(color_module)
                
            # Habilitar/deshabilitar según configuración
            if not color_config.get("enabled", True):
                self.get_module("ColorDetection").disable()
        
        # Defect detection
        if "defect" in modules_config:
            defect_config = modules_config["defect"]
            if self.get_module("DefectDetection"):
                self.get_module("DefectDetection").set_config(defect_config)
            else:
                defect_module = DefectDetectionModule(defect_config)
                self.add_module(defect_module)
                
            if not defect_config.get("enabled", True):
                self.get_module("DefectDetection").disable()
        
        # Dimension measurement
        if "dimensions" in modules_config:
            dim_config = modules_config["dimensions"]
            if self.get_module("DimensionMeasurement"):
                self.get_module("DimensionMeasurement").set_config(dim_config)
            else:
                dim_module = DimensionMeasurementModule(dim_config)
                self.add_module(dim_module)
                
            if not dim_config.get("enabled", True):
                self.get_module("DimensionMeasurement").disable()
        
        # Text verification
        if "text_verification" in modules_config:
            text_config = modules_config["text_verification"]
            if self.get_module("TextVerification"):
                self.get_module("TextVerification").set_config(text_config)
            else:
                text_module = TextVerificationModule(text_config)
                self.add_module(text_module)
                
            if not text_config.get("enabled", True):
                self.get_module("TextVerification").disable()
                
        self.logger.info(f"Configuración cargada para {len(self.modules)} módulos de inspección")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calcula estadísticas basadas en los resultados históricos.
        
        Returns:
            Dict[str, Any]: Estadísticas calculadas
        """
        if not self.results_history:
            return {}
            
        total = len(self.results_history)
        passed = sum(1 for r in self.results_history if r["overall_status"] == "pass")
        failed = sum(1 for r in self.results_history if r["overall_status"] == "fail")
        errors = sum(1 for r in self.results_history if r["overall_status"] == "error")
        
        # Estadísticas por módulo
        module_stats = {}
        for module_name in self.modules:
            module_results = [
                r["module_results"].get(module_name, {"status": "unknown"}) 
                for r in self.results_history 
                if module_name in r["module_results"]
            ]
            
            if module_results:
                module_passed = sum(1 for r in module_results if r["status"] == "pass")
                module_failed = sum(1 for r in module_results if r["status"] == "fail")
                module_errors = sum(1 for r in module_results if r["status"] == "error")
                
                module_stats[module_name] = {
                    "total": len(module_results),
                    "passed": module_passed,
                    "failed": module_failed,
                    "errors": module_errors,
                    "pass_rate": module_passed / len(module_results) if module_results else 0
                }
        
        return {
            "total_inspections": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total else 0,
            "failure_rate": failed / total if total else 0,
            "error_rate": errors / total if total else 0,
            "modules": module_stats
        }


# Función para crear una instancia de gestor de inspección preconfigurada
def create_inspection_manager_from_config(config_path: str) -> InspectionManager:
    """Crea un gestor de inspección configurado desde un archivo.
    
    Args:
        config_path: Ruta al archivo de configuración JSON
        
    Returns:
        InspectionManager: Gestor de inspección configurado
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        manager = InspectionManager()
        manager.load_config_from_sku(config)
        return manager
    except Exception as e:
        logging.getLogger('system_logger').error(f"Error al crear gestor de inspección: {str(e)}")
        return InspectionManager()


# Para pruebas directas del módulo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('system_logger')
    
    # Probar el módulo de detección de color
    logger.info("Probando módulo de detección de color...")
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    # Crear un cuadrado rojo en el medio
    test_image[100:200, 100:200] = [0, 0, 255]  # BGR: rojo
    
    color_module = ColorDetectionModule()
    color_results = color_module.inspect(test_image)
    logger.info(f"Resultados de detección de color: {color_results}")
    
    # Probar el módulo de medición de dimensiones
    logger.info("Probando módulo de medición de dimensiones...")
    dim_module = DimensionMeasurementModule({"pixels_per_mm": 10})
    dim_results = dim_module.inspect(test_image)
    logger.info(f"Resultados de medición de dimensiones: {dim_results}")
    
    # Probar el gestor de inspección
    logger.info("Probando gestor de inspección...")
    manager = InspectionManager()
    manager.add_module(color_module)
    manager.add_module(dim_module)
    
    combined_results = manager.inspect_image(test_image)
    logger.info(f"Resultados combinados: {combined_results}")
    
    # Mostrar estadísticas
    stats = manager.get_statistics()
    logger.info(f"Estadísticas: {stats}")
