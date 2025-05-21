#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptador de Visión Artificial
---------------------------
Proporciona integración con diferentes frameworks de visión artificial
y procesamiento de imágenes avanzado.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import os

# Intentar importar bibliotecas opcionales
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from skimage import feature, filters, morphology, measure, segmentation
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class VisionAdapter:
    """Clase base para adaptadores de visión artificial."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el adaptador de visión.
        
        Args:
            config: Configuración del adaptador
        """
        self.logger = logging.getLogger('system_logger')
        self.config = config or {}
        
        self.logger.info("Adaptador de visión artificial inicializado")
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Realiza preprocesamiento básico en la imagen.
        
        Args:
            image: Imagen a procesar (BGR)
            
        Returns:
            np.ndarray: Imagen preprocesada
        """
        # Aplicar desenfoque para reducir ruido
        if self.config.get("apply_blur", True):
            kernel_size = self.config.get("blur_kernel_size", 5)
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Convertir a escala de grises si se necesita
        if self.config.get("convert_grayscale", False):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Ecualización de histograma para mejorar contraste
        if self.config.get("equalize_hist", False) and len(image.shape) == 2:
            image = cv2.equalizeHist(image)
            
        return image
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detecta bordes en la imagen.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            np.ndarray: Imagen con bordes detectados
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Aplicar algoritmo de detección de bordes
        edge_detector = self.config.get("edge_detector", "canny")
        
        if edge_detector == "canny":
            low_threshold = self.config.get("canny_low_threshold", 50)
            high_threshold = self.config.get("canny_high_threshold", 150)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
        elif edge_detector == "sobel":
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.magnitude(sobel_x, sobel_y)
            # Normalizar a 0-255 y convertir a uint8
            edges = np.uint8(255 * edges / np.max(edges))
        elif edge_detector == "laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        else:
            self.logger.warning(f"Detector de bordes no reconocido: {edge_detector}, usando Canny")
            edges = cv2.Canny(gray, 50, 150)
            
        return edges
    
    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Segmenta la imagen para detectar objetos.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: Imagen segmentada y lista de objetos detectados
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Aplicar umbral para binarizar la imagen
        threshold_method = self.config.get("threshold_method", "otsu")
        
        if threshold_method == "simple":
            threshold_value = self.config.get("threshold_value", 127)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_method == "adaptive":
            block_size = self.config.get("adaptive_block_size", 11)
            c = self.config.get("adaptive_c", 2)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, block_size, c)
        else:
            self.logger.warning(f"Método de umbral no reconocido: {threshold_method}, usando Otsu")
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        # Operaciones morfológicas para mejorar segmentación
        if self.config.get("apply_morphology", True):
            kernel_size = self.config.get("morph_kernel_size", 5)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if self.config.get("morph_operation", "opening") == "opening":
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            elif self.config.get("morph_operation") == "closing":
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            elif self.config.get("morph_operation") == "gradient":
                binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
                
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área mínima
        min_area = self.config.get("min_contour_area", 100)
        objects = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                # Calcular rectángulo delimitador
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Calcular centro
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                # Extraer información adicional
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                objects.append({
                    "contour": cnt,
                    "area": area,
                    "perimeter": perimeter,
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "circularity": circularity
                })
        
        # Crear imagen segmentada para visualización
        segmented = np.zeros_like(image)
        if len(image.shape) == 3:
            # Colorear cada objeto con un color aleatorio
            for i, obj in enumerate(objects):
                color = np.random.randint(0, 255, 3).tolist()
                cv2.drawContours(segmented, [obj["contour"]], -1, color, -1)
        else:
            # Imagen en escala de grises
            cv2.drawContours(segmented, [obj["contour"] for obj in objects], -1, 255, -1)
            
        return segmented, objects
    
    def detect_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecta características en la imagen.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            Dict[str, Any]: Características detectadas
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        features = {}
        
        # Detección de esquinas Harris
        if self.config.get("detect_corners", False):
            corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
            # Dilatar para marcar las esquinas
            corners = cv2.dilate(corners, None)
            # Umbral para obtener puntos de interés
            threshold = self.config.get("corner_threshold", 0.01)
            corner_points = np.where(corners > threshold * corners.max())
            features["corners"] = list(zip(corner_points[1], corner_points[0]))  # x, y coordinates
            
        # Detección de características SIFT/ORB
        feature_detector = self.config.get("feature_detector", None)
        if feature_detector:
            if feature_detector == "sift":
                detector = cv2.SIFT_create()
            elif feature_detector == "orb":
                detector = cv2.ORB_create()
            else:
                self.logger.warning(f"Detector de características no reconocido: {feature_detector}")
                detector = None
                
            if detector:
                keypoints, descriptors = detector.detectAndCompute(gray, None)
                features["keypoints"] = keypoints
                features["descriptors"] = descriptors
                
        # Histograma de la imagen
        if self.config.get("compute_histogram", False):
            if len(image.shape) == 3:
                # Histograma para cada canal
                hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
                hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
                features["histogram"] = {
                    "blue": hist_b.flatten(),
                    "green": hist_g.flatten(),
                    "red": hist_r.flatten()
                }
            else:
                # Histograma para escala de grises
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                features["histogram"] = hist.flatten()
                
        return features


class TensorFlowAdapter(VisionAdapter):
    """Adaptador para integrar con TensorFlow para detección de objetos y clasificación."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el adaptador de TensorFlow.
        
        Args:
            config: Configuración del adaptador
        """
        super().__init__(config)
        
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow no disponible. Instale tensorflow para usar este adaptador.")
            self.model = None
            return
            
        # Cargar modelo si se especifica en la configuración
        model_path = self.config.get("model_path")
        if model_path and os.path.exists(model_path):
            try:
                self.model = tf.saved_model.load(model_path)
                self.logger.info(f"Modelo TensorFlow cargado desde {model_path}")
            except Exception as e:
                self.logger.error(f"Error al cargar modelo TensorFlow: {str(e)}")
                self.model = None
        else:
            self.logger.warning("No se especificó ruta a modelo TensorFlow o no existe")
            self.model = None
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta objetos en la imagen utilizando un modelo TensorFlow.
        
        Args:
            image: Imagen a procesar (BGR)
            
        Returns:
            List[Dict[str, Any]]: Lista de objetos detectados
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            self.logger.error("TensorFlow no disponible o modelo no cargado")
            return []
            
        # Convertir de BGR a RGB (TensorFlow usa RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar si es necesario
        input_size = self.config.get("input_size", (224, 224))
        if image_rgb.shape[:2] != input_size:
            image_rgb = cv2.resize(image_rgb, input_size)
            
        # Preprocesar imagen según el modelo
        input_tensor = tf.convert_to_tensor(image_rgb)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        # Realizar inferencia
        try:
            detections = self.model(input_tensor)
            
            # Procesar resultados basados en el tipo de modelo
            model_type = self.config.get("model_type", "detection")
            
            if model_type == "detection":
                # Para modelos de detección de objetos (tipo SSD, RCNN, etc.)
                boxes = detections['detection_boxes'][0].numpy()
                classes = detections['detection_classes'][0].numpy().astype(np.int32)
                scores = detections['detection_scores'][0].numpy()
                
                # Convertir a coordenadas de imagen
                height, width = image.shape[:2]
                
                results = []
                min_score = self.config.get("min_score", 0.5)
                
                for i in range(min(len(boxes), 100)):  # Limitar a 100 detecciones max
                    if scores[i] >= min_score:
                        ymin, xmin, ymax, xmax = boxes[i]
                        # Convertir a coordenadas de píxeles
                        xmin = int(xmin * width)
                        xmax = int(xmax * width)
                        ymin = int(ymin * height)
                        ymax = int(ymax * height)
                        
                        results.append({
                            "bbox": (xmin, ymin, xmax - xmin, ymax - ymin),
                            "class_id": int(classes[i]),
                            "score": float(scores[i]),
                            "class_name": self._get_class_name(classes[i])
                        })
                
                return results
                
            elif model_type == "classification":
                # Para modelos de clasificación
                predictions = detections.numpy()
                top_k = self.config.get("top_k", 5)
                
                # Obtener los top-k índices y probabilidades
                indices = np.argsort(predictions[0])[-top_k:][::-1]
                probabilities = predictions[0][indices]
                
                results = []
                for i, (idx, prob) in enumerate(zip(indices, probabilities)):
                    results.append({
                        "class_id": int(idx),
                        "class_name": self._get_class_name(idx),
                        "probability": float(prob),
                        "rank": i + 1
                    })
                    
                return results
            else:
                self.logger.error(f"Tipo de modelo no soportado: {model_type}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error en inferencia de TensorFlow: {str(e)}")
            return []
    
    def _get_class_name(self, class_id: int) -> str:
        """Obtiene el nombre de la clase a partir de su ID.
        
        Args:
            class_id: ID de la clase
            
        Returns:
            str: Nombre de la clase o ID como string si no se encuentra
        """
        # Buscar el nombre de la clase en el mapa de clases si está disponible
        class_map = self.config.get("class_map", {})
        return class_map.get(str(class_id), str(class_id))


class SkimageAdapter(VisionAdapter):
    """Adaptador para algoritmos avanzados de scikit-image."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializa el adaptador de scikit-image.
        
        Args:
            config: Configuración del adaptador
        """
        super().__init__(config)
        
        if not SKIMAGE_AVAILABLE:
            self.logger.warning("scikit-image no disponible. Instale skimage para usar este adaptador.")
    
    def detect_blobs(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta blobs (manchas) en la imagen.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            List[Dict[str, Any]]: Lista de blobs detectados
        """
        if not SKIMAGE_AVAILABLE:
            self.logger.error("scikit-image no disponible")
            return []
            
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Configurar parámetros de detección
        min_sigma = self.config.get("min_sigma", 1)
        max_sigma = self.config.get("max_sigma", 30)
        num_sigma = self.config.get("num_sigma", 10)
        threshold = self.config.get("blob_threshold", 0.05)
        
        # Detectar blobs
        blobs = feature.blob_log(gray, min_sigma=min_sigma, max_sigma=max_sigma, 
                               num_sigma=num_sigma, threshold=threshold)
        
        # Convertir a lista de resultados
        results = []
        for blob in blobs:
            y, x, r = blob
            results.append({
                "center": (int(x), int(y)),
                "radius": int(r * np.sqrt(2)),  # Convertir sigma a radio
                "sigma": float(r)
            })
            
        return results
    
    def watershed_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Realiza segmentación watershed en la imagen.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: Imagen segmentada y propiedades de regiones
        """
        if not SKIMAGE_AVAILABLE:
            self.logger.error("scikit-image no disponible")
            return image, []
            
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Calcular gradiente
        gradient = filters.sobel(gray)
        
        # Umbralización
        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh
        
        # Operaciones morfológicas
        binary = morphology.opening(binary, morphology.disk(3))
        
        # Distancia
        distance = ndimage.distance_transform_edt(binary)
        
        # Máximos locales para marcadores
        local_max = feature.peak_local_max(distance, min_distance=20, 
                                         labels=binary)
        markers = np.zeros_like(distance, dtype=np.int32)
        markers[tuple(local_max.T)] = range(1, len(local_max) + 1)
        
        # Watershed
        labels = segmentation.watershed(-distance, markers, mask=binary)
        
        # Extraer propiedades de regiones
        props = measure.regionprops(labels)
        
        # Crear imagen segmentada para visualización
        segmented = np.zeros_like(image)
        
        # Información de regiones
        regions = []
        
        for prop in props:
            # Obtener propiedades
            label = prop.label
            area = prop.area
            centroid = prop.centroid
            bbox = prop.bbox
            
            # Convertir coordenadas
            y0, x0, y1, x1 = bbox
            
            regions.append({
                "label": int(label),
                "area": float(area),
                "centroid": (float(centroid[1]), float(centroid[0])),  # x, y
                "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            })
            
            # Colorear región en la imagen segmentada
            if len(image.shape) == 3:
                color = np.random.randint(0, 255, 3).tolist()
                mask = labels == label
                segmented[mask] = color
            else:
                segmented[labels == label] = 255
                
        return segmented, regions


# Funciones utilitarias para integración con otros sistemas

def create_vision_adapter(adapter_type: str, config: Dict[str, Any] = None) -> VisionAdapter:
    """Crea un adaptador de visión artificial del tipo especificado.
    
    Args:
        adapter_type: Tipo de adaptador ("opencv", "tensorflow", "skimage")
        config: Configuración del adaptador
        
    Returns:
        VisionAdapter: Instancia del adaptador
    """
    if adapter_type.lower() == "tensorflow":
        return TensorFlowAdapter(config)
    elif adapter_type.lower() == "skimage":
        return SkimageAdapter(config)
    else:
        # OpenCV es el adaptador por defecto
        return VisionAdapter(config)


def apply_vision_pipeline(image: np.ndarray, pipeline: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Aplica un pipeline de procesamiento de visión artificial a una imagen.
    
    Args:
        image: Imagen a procesar
        pipeline: Lista de operaciones a aplicar
        
    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Imagen procesada y resultados
    """
    processed_image = image.copy()
    results = {}
    
    # Crear un adaptador básico
    adapter = VisionAdapter()
    
    # Procesar cada operación en el pipeline
    for operation in pipeline:
        op_type = operation.get("type", "")
        config = operation.get("config", {})
        
        if op_type == "preprocess":
            processed_image = adapter.preprocess_image(processed_image)
            
        elif op_type == "detect_edges":
            edges = adapter.detect_edges(processed_image)
            results["edges"] = edges
            if operation.get("update_image", False):
                processed_image = edges
                
        elif op_type == "segment":
            segmented, objects = adapter.segment_image(processed_image)
            results["segmentation"] = {
                "objects_count": len(objects),
                "objects": objects
            }
            if operation.get("update_image", False):
                processed_image = segmented
                
        elif op_type == "detect_features":
            features = adapter.detect_features(processed_image)
            results["features"] = features
            
        elif op_type == "tensorflow":
            # Crear adaptador específico para TensorFlow
            tf_adapter = TensorFlowAdapter(config)
            detections = tf_adapter.detect_objects(processed_image)
            results["detections"] = detections
            
            # Dibujar detecciones en la imagen si se solicita
            if operation.get("draw_detections", False) and detections:
                for det in detections:
                    if "bbox" in det:
                        x, y, w, h = det["bbox"]
                        label = det.get("class_name", "")
                        score = det.get("score", 0)
                        
                        # Dibujar rectángulo y etiqueta
                        cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        text = f"{label}: {score:.2f}"
                        cv2.putText(processed_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
        elif op_type == "skimage":
            # Crear adaptador específico para scikit-image
            skim_adapter = SkimageAdapter(config)
            
            # Determinar operación específica
            skim_op = operation.get("operation", "")
            
            if skim_op == "detect_blobs":
                blobs = skim_adapter.detect_blobs(processed_image)
                results["blobs"] = blobs
                
                # Dibujar blobs en la imagen si se solicita
                if operation.get("draw_blobs", False) and blobs:
                    blob_image = processed_image.copy()
                    for blob in blobs:
                        x, y = blob["center"]
                        r = blob["radius"]
                        cv2.circle(blob_image, (x, y), r, (0, 0, 255), 2)
                    
                    if operation.get("update_image", False):
                        processed_image = blob_image
                        
            elif skim_op == "watershed":
                watershed_image, regions = skim_adapter.watershed_segmentation(processed_image)
                results["watershed"] = {
                    "regions_count": len(regions),
                    "regions": regions
                }
                
                if operation.get("update_image", False):
                    processed_image = watershed_image
        
        # Registro de operación
        results[f"operation_{len(results)}"] = {
            "type": op_type,
            "config": config
        }
    
    return processed_image, results


# Para pruebas directas del módulo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('system_logger')
    
    # Crear imagen de prueba
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (250, 250), (0, 0, 255), -1)  # Rectángulo rojo
    cv2.circle(test_image, (150, 150), 50, (0, 255, 0), -1)  # Círculo verde
    
    # Probar adaptador básico
    logger.info("Probando adaptador básico...")
    adapter = VisionAdapter({"apply_blur": True, "blur_kernel_size": 3})
    
    # Detectar bordes
    edges = adapter.detect_edges(test_image)
    cv2.imwrite("test_edges.jpg", edges)
    
    # Segmentar imagen
    segmented, objects = adapter.segment_image(test_image)
    cv2.imwrite("test_segmented.jpg", segmented)
    
    logger.info(f"Se detectaron {len(objects)} objetos")
    for i, obj in enumerate(objects):
        logger.info(f"Objeto {i+1}: Área = {obj['area']}, Circularidad = {obj['circularity']:.2f}")
    
    # Probar pipeline
    pipeline = [
        {"type": "preprocess", "config": {"apply_blur": True}},
        {"type": "detect_edges", "update_image": True},
        {"type": "segment", "update_image": False}
    ]
    
    processed, results = apply_vision_pipeline(test_image, pipeline)
    cv2.imwrite("test_pipeline.jpg", processed)
    
    logger.info("Pruebas completadas")
