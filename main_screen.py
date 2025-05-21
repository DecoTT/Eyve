#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pantalla Principal
-----------------
Interfaz gráfica principal del sistema de inspección visual, que muestra
la vista de cámara, resultados de inspección y controles del sistema.
"""

import os
import sys
import time
import threading
import datetime
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QComboBox, QTabWidget, QSplitter, QFrame, QGroupBox, 
                            QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                            QAction, QToolBar, QStatusBar, QMenu, QDialog, QFileDialog,
                            QLineEdit, QTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor, QPalette, QFont
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, pyqtSlot, QThread, QDateTime

# En una implementación real, se importarían los componentes del sistema
try:
    from system_manager import get_system_manager
except ImportError:
    pass


class CameraViewWidget(QWidget):
    """Widget para mostrar la vista de cámara en tiempo real."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger('system_logger')
        
        # Configuración
        self.camera_id = None
        self.refresh_rate = 30  # ms
        self.frame = None
        self.results = None
        self.overlay_enabled = True
        self.last_frame_time = 0
        
        # Interfaz
        self.layout = QVBoxLayout(self)
        
        # Etiqueta para mostrar la imagen
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid #cccccc; background-color: #000000;")
        
        # Barra de estado de cámara
        self.status_bar = QWidget()
        self.status_layout = QHBoxLayout(self.status_bar)
        self.status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_label = QLabel("Cámara: No seleccionada")
        self.fps_label = QLabel("FPS: -")
        self.resolution_label = QLabel("Resolución: -")
        
        self.status_layout.addWidget(self.camera_label)
        self.status_layout.addWidget(self.fps_label)
        self.status_layout.addWidget(self.resolution_label)
        self.status_layout.addStretch()
        
        # Agregar widgets
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.status_bar)
        
        # Timer para actualizar la imagen
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_frame)
        
    def start_capture(self, camera_id=None):
        """Inicia la captura desde la cámara especificada.
        
        Args:
            camera_id: ID de la cámara a utilizar
        """
        # Primero detener cualquier captura activa
        self.stop_capture()
        
        if camera_id:
            self.camera_id = camera_id
            
        # Mostrar mensaje de estado mientras se conecta
        self.image_label.setText("Conectando con la cámara...")
        QApplication.processEvents()  # Actualizar la interfaz
        
        try:
            system_manager = get_system_manager()
            if system_manager and system_manager.camera_manager:
                # Conectar la cámara
                camera = system_manager.camera_manager.get_camera(self.camera_id)
                if camera and not camera.is_connected:
                    camera.connect()
            
            self.camera_label.setText(f"Cámara: {self.camera_id or 'No seleccionada'}")
            self.update_timer.start(self.refresh_rate)
            self.logger.info(f"Iniciada captura de cámara {self.camera_id}")
        except Exception as e:
            self.logger.error(f"Error al iniciar captura: {str(e)}")
            self.image_label.setText(f"Error al conectar con la cámara: {str(e)}")
        
    def stop_capture(self):
        """Detiene la captura de imágenes."""
        self.update_timer.stop()
        self.logger.info("Detenida captura de cámara")
        
    def set_refresh_rate(self, rate_ms):
        """Establece la tasa de refresco.
        
        Args:
            rate_ms: Tasa de refresco en milisegundos
        """
        self.refresh_rate = rate_ms
        if self.update_timer.isActive():
            self.update_timer.start(self.refresh_rate)
            
    def set_overlay_enabled(self, enabled):
        """Activa o desactiva la superposición de resultados.
        
        Args:
            enabled: True para activar, False para desactivar
        """
        self.overlay_enabled = enabled
        
    def update_frame(self):
        """Actualiza el fotograma mostrado desde la cámara seleccionada."""
        try:
            # Obtener el gestor del sistema
            system_manager = get_system_manager()
            if not system_manager or not system_manager.camera_manager:
                return
                
            # Obtener la cámara
            camera = system_manager.camera_manager.get_camera(self.camera_id)
            if not camera:
                self.image_label.setText(f"Cámara no encontrada: {self.camera_id}")
                return
                
            # Comprobar conexión
            if not camera.is_connected:
                self.image_label.setText(f"Cámara {self.camera_id} no conectada. Reconectando...")
                try:
                    camera.connect()
                except Exception as e:
                    self.image_label.setText(f"Error al conectar: {str(e)}")
                    return

            # Obtener el último fotograma directamente mediante una captura fresca
            # Esto es más confiable que get_last_frame en algunos sistemas
            frame = None
            if hasattr(camera, 'capture') and camera.capture is not None:
                ret, direct_frame = camera.capture.read()
                if ret and direct_frame is not None and direct_frame.size > 0:
                    frame = direct_frame
                
            # Si no se pudo capturar directamente, intentar con get_last_frame
            if frame is None:
                frame = camera.get_last_frame()
                
            # Verificar que tenemos un frame válido
            if frame is None or frame.size == 0:
                self.image_label.setText(f"No hay fotograma disponible de {self.camera_id}")
                return
                
            # Verificar que el frame tiene el formato correcto
            if len(frame.shape) < 2:
                self.image_label.setText("Formato de imagen no válido")
                return
                
            # Agregar diagnóstico de frame
            h, w = frame.shape[:2]
            self.logger.info(f"Frame recibido: {w}x{h}, tipo: {frame.dtype}, forma: {frame.shape}")
            
            # Calcular FPS
            current_time = time.time()
            if self.last_frame_time > 0:
                fps = 1.0 / max(0.001, current_time - self.last_frame_time)
                self.fps_label.setText(f"FPS: {fps:.1f}")
            self.last_frame_time = current_time
            
            # Actualizar etiqueta de resolución
            self.resolution_label.setText(f"Resolución: {w}x{h}")
            
            # Guardar frame para referencia
            self.frame = frame.copy()
            
            # Dibujar resultados si están disponibles y está habilitada la superposición
            if self.results and self.overlay_enabled:
                frame = self.draw_results(frame.copy(), self.results)
            
            # Asegurarse de que el frame está en formato BGR (OpenCV usa BGR por defecto)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convertir de BGR a RGB para Qt
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Si es en escala de grises, convertir a RGB
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
            # Crear QImage y luego QPixmap
            h, w = rgb_image.shape[:2]
            bytes_per_line = 3 * w  # 3 bytes por píxel (RGB)
            
            try:
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                # Escalar manteniendo relación de aspecto
                label_size = self.image_label.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # Mostrar la imagen
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
                
                # Registrar éxito
                self.logger.debug(f"Frame mostrado correctamente: {w}x{h}")
            except Exception as e:
                self.logger.error(f"Error al crear QImage/QPixmap: {str(e)}")
                self.image_label.setText(f"Error al procesar imagen: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error al actualizar frame: {str(e)}")
            self.image_label.setText(f"Error: {str(e)}")
            
    def set_results(self, results):
        """Establece los resultados de inspección para mostrar en superposición.
        
        Args:
            results: Resultados de la inspección
        """
        self.results = results
        
    def draw_results(self, frame, results):
        """Dibuja los resultados de la inspección sobre el fotograma.
        
        Args:
            frame: Fotograma sobre el que dibujar
            results: Resultados de la inspección
            
        Returns:
            numpy.ndarray: Fotograma con los resultados dibujados
        """
        try:
            # Dibujar estado general en la esquina superior izquierda
            status = results.get("status", "unknown")
            color = (0, 255, 0) if status == "pass" else (0, 0, 255)  # Verde para pass, rojo para fail
            cv2.putText(frame, f"Estado: {status.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Dibujar resultados por módulo
            module_results = results.get("module_results", {})
            y_pos = 70
            
            for module_name, module_result in module_results.items():
                # Dibujar nombre del módulo
                cv2.putText(frame, f"{module_name}:", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 30
                
                module_status = module_result.get("status", "unknown")
                color = (0, 255, 0) if module_status == "pass" else (0, 0, 255)
                
                # Dibujar estado del módulo
                cv2.putText(frame, f"  Estado: {module_status.upper()}", (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 30
                
                # Dibujar defectos si los hay
                if module_status == "fail":
                    # Módulo de detección de color
                    if "detections" in module_result:
                        for color_name, detection in module_result["detections"].items():
                            boxes = detection.get("bounding_boxes", [])
                            for box in boxes:
                                x, y, w, h = box
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                cv2.putText(frame, f"{color_name}", (x, y-5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Módulo de detección de defectos
                    if "defects" in module_result:
                        for defect in module_result["defects"]:
                            if "bbox" in defect:
                                x, y, w, h = defect["bbox"]
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                                confidence = defect.get("confidence", 0) * 100
                                cv2.putText(frame, f"Defecto: {confidence:.1f}%", (x, y-5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    
                    # Módulo de medición de dimensiones
                    if "measurements" in module_result:
                        for measurement in module_result["measurements"]:
                            if "bbox" in measurement:
                                x, y, w, h = measurement["bbox"]
                                color = (0, 255, 0) if measurement.get("width_in_range", True) and measurement.get("height_in_range", True) else (0, 0, 255)
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                
                                # Mostrar medidas
                                width_mm = measurement.get("width_mm", 0)
                                height_mm = measurement.get("height_mm", 0)
                                cv2.putText(frame, f"{width_mm:.1f}x{height_mm:.1f}mm", (x, y-5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                y_pos += 20
            
            return frame
        except Exception as e:
            self.logger.error(f"Error al dibujar resultados: {str(e)}")
            return frame
            
    def capture_current_frame(self):
        """Captura el fotograma actual.
        
        Returns:
            numpy.ndarray: Fotograma capturado o None si no hay
        """
        return self.frame.copy() if self.frame is not None else None


class ResultsWidget(QWidget):
    """Widget para mostrar los resultados de inspección."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger('system_logger')
        
        # Interfaz
        self.layout = QVBoxLayout(self)
        
        # Cabecera con información general
        self.header_frame = QFrame()
        self.header_frame.setFrameShape(QFrame.StyledPanel)
        self.header_layout = QHBoxLayout(self.header_frame)
        
        self.status_label = QLabel("Estado:")
        self.status_value = QLabel("No inspeccionado")
        self.status_value.setStyleSheet("font-weight: bold; color: gray;")
        
        self.timestamp_label = QLabel("Timestamp:")
        self.timestamp_value = QLabel("-")
        
        self.header_layout.addWidget(self.status_label)
        self.header_layout.addWidget(self.status_value)
        self.header_layout.addWidget(self.timestamp_label)
        self.header_layout.addWidget(self.timestamp_value)
        self.header_layout.addStretch()
        
        # Detalles en un widget de tabla
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(2)
        self.details_table.setHorizontalHeaderLabels(["Módulo", "Resultado"])
        self.details_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        
        # Botones
        self.button_layout = QHBoxLayout()
        self.save_button = QPushButton("Guardar Resultados")
        self.save_button.clicked.connect(self.save_results)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addStretch()
        
        # Agregar widgets
        self.layout.addWidget(self.header_frame)
        self.layout.addWidget(self.details_table)
        self.layout.addLayout(self.button_layout)
        
    def set_results(self, results):
        """Establece y muestra los resultados de inspección.
        
        Args:
            results: Resultados de la inspección
        """
        if not results:
            self.clear_results()
            return
            
        try:
            # Actualizar estado general
            status = results.get("status", "unknown")
            self.status_value.setText(status.upper())
            
            if status == "pass":
                self.status_value.setStyleSheet("font-weight: bold; color: green;")
            else:
                self.status_value.setStyleSheet("font-weight: bold; color: red;")
                
            # Actualizar timestamp
            timestamp = results.get("timestamp", datetime.datetime.now().isoformat())
            self.timestamp_value.setText(timestamp)
            
            # Actualizar tabla de detalles
            self.details_table.setRowCount(0)  # Limpiar tabla
            
            module_results = results.get("module_results", {})
            for module_name, module_result in module_results.items():
                row = self.details_table.rowCount()
                self.details_table.insertRow(row)
                
                # Nombre del módulo
                module_item = QTableWidgetItem(module_name)
                self.details_table.setItem(row, 0, module_item)
                
                # Resultado
                module_status = module_result.get("status", "unknown")
                result_item = QTableWidgetItem(module_status.upper())
                
                if module_status == "pass":
                    result_item.setForeground(QColor("green"))
                elif module_status == "fail":
                    result_item.setForeground(QColor("red"))
                elif module_status == "disabled":
                    result_item.setForeground(QColor("gray"))
                    
                self.details_table.setItem(row, 1, result_item)
                
        except Exception as e:
            self.logger.error(f"Error al mostrar resultados: {str(e)}")
            
    def clear_results(self):
        """Limpia todos los resultados mostrados."""
        self.status_value.setText("No inspeccionado")
        self.status_value.setStyleSheet("font-weight: bold; color: gray;")
        self.timestamp_value.setText("-")
        self.details_table.setRowCount(0)
        
    def save_results(self):
        """Guarda los resultados actuales en un archivo."""
        try:
            system_manager = get_system_manager()
            if not system_manager:
                return
                
            with system_manager.status_lock:
                results = system_manager.system_status.get("last_inspection_result")
                
            if not results:
                QMessageBox.warning(self, "Guardar Resultados", 
                                   "No hay resultados para guardar.")
                return
                
            # Mostrar diálogo para seleccionar archivo
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Guardar Resultados", 
                os.path.join(os.getcwd(), "resultados_inspeccion.json"),
                "Archivos JSON (*.json)"
            )
            
            if not file_path:
                return
                
            # Guardar resultados en formato JSON
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                
            QMessageBox.information(self, "Guardar Resultados", 
                                   f"Resultados guardados en:\n{file_path}")
            
        except Exception as e:
            self.logger.error(f"Error al guardar resultados: {str(e)}")
            QMessageBox.critical(self, "Error", 
                               f"Error al guardar resultados:\n{str(e)}")


class ControlPanel(QWidget):
    """Panel de control para inspección y gestión de lotes."""
    
    # Señales
    inspection_started = pyqtSignal()
    inspection_stopped = pyqtSignal()
    results_available = pyqtSignal(dict)
    camera_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger('system_logger')
        
        # Estado interno
        self.auto_inspect_active = False
        self.auto_inspect_interval = 1000  # ms
        
        # Interfaz
        self.layout = QVBoxLayout(self)
        
        # Grupo de selección de cámara
        self.camera_group = QGroupBox("Cámara")
        self.camera_layout = QVBoxLayout(self.camera_group)
        
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        self.refresh_cameras_button = QPushButton("Actualizar Cámaras")
        self.refresh_cameras_button.clicked.connect(self.refresh_cameras)
        
        self.camera_layout.addWidget(QLabel("Seleccionar Cámara:"))
        self.camera_layout.addWidget(self.camera_combo)
        self.camera_layout.addWidget(self.refresh_cameras_button)
        
        # Grupo de inspección
        self.inspection_group = QGroupBox("Inspección")
        self.inspection_layout = QVBoxLayout(self.inspection_group)
        
        self.inspect_button = QPushButton("Inspeccionar")
        self.inspect_button.clicked.connect(self.on_inspect)
        self.inspect_button.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.auto_inspect_check = QCheckBox("Inspección Automática")
        self.auto_inspect_check.toggled.connect(self.on_auto_inspect_toggled)
        
        self.interval_layout = QHBoxLayout()
        self.interval_layout.addWidget(QLabel("Intervalo (ms):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(100, 10000)
        self.interval_spin.setValue(self.auto_inspect_interval)
        self.interval_spin.setSingleStep(100)
        self.interval_spin.valueChanged.connect(self.on_interval_changed)
        self.interval_layout.addWidget(self.interval_spin)
        
        self.inspection_layout.addWidget(self.inspect_button)
        self.inspection_layout.addWidget(self.auto_inspect_check)
        self.inspection_layout.addLayout(self.interval_layout)
        
        # Grupo de lote
        self.batch_group = QGroupBox("Lote de Producción")
        self.batch_layout = QVBoxLayout(self.batch_group)
        
        self.batch_info_label = QLabel("Sin lote activo")
        
        self.batch_layout.addWidget(self.batch_info_label)
        
        self.batch_button_layout = QHBoxLayout()
        self.new_batch_button = QPushButton("Nuevo Lote")
        self.new_batch_button.clicked.connect(self.on_new_batch)
        self.close_batch_button = QPushButton("Cerrar Lote")
        self.close_batch_button.clicked.connect(self.on_close_batch)
        self.close_batch_button.setEnabled(False)
        
        self.batch_button_layout.addWidget(self.new_batch_button)
        self.batch_button_layout.addWidget(self.close_batch_button)
        
        self.batch_layout.addLayout(self.batch_button_layout)
        
        # Grupo de estadísticas
        self.stats_group = QGroupBox("Estadísticas")
        self.stats_layout = QVBoxLayout(self.stats_group)
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        
        self.stats_layout.addWidget(self.stats_table)
        
        # Timer para inspección automática
        self.inspect_timer = QTimer(self)
        self.inspect_timer.timeout.connect(self.on_inspect)
        
        # Agregar widgets a layout principal
        self.layout.addWidget(self.camera_group)
        self.layout.addWidget(self.inspection_group)
        self.layout.addWidget(self.batch_group)
        self.layout.addWidget(self.stats_group)
        self.layout.addStretch()
        
        # Inicializar
        self.refresh_cameras()
        self.update_stats()
        
        # Timer para actualizar estadísticas periódicamente
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(5000)  # Actualizar cada 5 segundos
        
    def refresh_cameras(self):
        """Actualiza la lista de cámaras disponibles."""
        try:
            system_manager = get_system_manager()
            if not system_manager or not system_manager.camera_manager:
                return
                
            # Guardar selección actual
            current_camera = self.camera_combo.currentText()
            
            # Actualizar lista
            self.camera_combo.clear()
            
            cameras = system_manager.camera_manager.get_all_cameras()
            for camera_id in cameras.keys():
                self.camera_combo.addItem(camera_id)
                
            # Restaurar selección si es posible
            if current_camera and self.camera_combo.findText(current_camera) >= 0:
                self.camera_combo.setCurrentText(current_camera)
                
        except Exception as e:
            self.logger.error(f"Error al actualizar cámaras: {str(e)}")
            
    def on_camera_changed(self, index):
        """Maneja el cambio de cámara seleccionada.
        
        Args:
            index: Índice de la cámara seleccionada
        """
        if index < 0:
            return
            
        camera_id = self.camera_combo.currentText()
        self.camera_changed.emit(camera_id)
        
    def on_inspect(self):
        """Realiza una inspección manual."""
        try:
            system_manager = get_system_manager()
            if not system_manager:
                return
                
            camera_id = self.camera_combo.currentText()
            results = system_manager.inspect_current_frame(camera_id)
            
            if results:
                self.results_available.emit(results)
                
        except Exception as e:
            self.logger.error(f"Error al inspeccionar: {str(e)}")
            
    def on_auto_inspect_toggled(self, checked):
        """Activa o desactiva la inspección automática.
        
        Args:
            checked: True si está activada, False si no
        """
        self.auto_inspect_active = checked
        
        if checked:
            self.inspect_timer.start(self.auto_inspect_interval)
            self.inspection_started.emit()
        else:
            self.inspect_timer.stop()
            self.inspection_stopped.emit()
            
    def on_interval_changed(self, value):
        """Actualiza el intervalo de inspección automática.
        
        Args:
            value: Nuevo valor del intervalo en milisegundos
        """
        self.auto_inspect_interval = value
        
        if self.inspect_timer.isActive():
            self.inspect_timer.start(self.auto_inspect_interval)
            
    def on_new_batch(self):
        """Inicia un nuevo lote de producción."""
        try:
            system_manager = get_system_manager()
            if not system_manager:
                return
                
            # Verificar si hay usuario logueado
            current_user = system_manager.get_current_user()
            if not current_user:
                QMessageBox.warning(self, "Nuevo Lote", 
                                   "Debe iniciar sesión para crear un lote.")
                return
                
            # Verificar si hay producto seleccionado
            with system_manager.status_lock:
                current_product = system_manager.system_status.get("current_product")
                
            if not current_product:
                QMessageBox.warning(self, "Nuevo Lote", 
                                   "Debe seleccionar un producto para crear un lote.")
                return
                
            # Mostrar diálogo para crear lote
            dialog = NewBatchDialog(current_product, self)
            if dialog.exec_() == QDialog.Accepted:
                batch_code = dialog.batch_code
                notes = dialog.notes
                
                batch_id = system_manager.start_batch(batch_code, notes)
                
                if batch_id:
                    self.batch_info_label.setText(f"Lote activo: {batch_code}")
                    self.new_batch_button.setEnabled(False)
                    self.close_batch_button.setEnabled(True)
                    
        except Exception as e:
            self.logger.error(f"Error al crear lote: {str(e)}")
            
    def on_close_batch(self):
        """Cierra el lote actual."""
        try:
            system_manager = get_system_manager()
            if not system_manager:
                return
                
            # Verificar si hay lote activo
            with system_manager.status_lock:
                current_batch = system_manager.system_status.get("current_batch")
                
            if not current_batch:
                return
                
            # Confirmar cierre
            reply = QMessageBox.question(
                self, "Cerrar Lote", 
                "¿Está seguro de que desea cerrar el lote actual?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if system_manager.close_batch():
                    self.batch_info_label.setText("Sin lote activo")
                    self.new_batch_button.setEnabled(True)
                    self.close_batch_button.setEnabled(False)
                    
        except Exception as e:
            self.logger.error(f"Error al cerrar lote: {str(e)}")
            
    def update_stats(self):
        """Actualiza las estadísticas mostradas."""
        try:
            system_manager = get_system_manager()
            if not system_manager:
                return
                
            # Obtener estado del sistema
            status = system_manager.get_system_status()
            
            # Obtener estadísticas del lote si hay uno activo
            batch_stats = system_manager.get_batch_statistics()
            
            # Preparar datos para la tabla
            stats_data = []
            
            # Datos generales
            stats_data.append(("Total inspecciones", str(status.get("inspection_count", 0))))
            stats_data.append(("Tiempo activo", status.get("uptime_formatted", "0h 0m 0s")))
            
            # Datos del lote si existe
            if batch_stats:
                stats_data.append(("Lote", batch_stats.get("batch_code", "-")))
                stats_data.append(("Inspecciones", str(batch_stats.get("total_inspections", 0))))
                stats_data.append(("Aprobados", str(batch_stats.get("passed", 0))))
                stats_data.append(("Rechazados", str(batch_stats.get("failed", 0))))
                
                pass_rate = batch_stats.get("pass_rate", 0)
                stats_data.append(("Tasa de aprobación", f"{pass_rate:.1f}%"))
                
                # Tipos de defectos
                defect_types = batch_stats.get("defect_types", {})
                for defect_type, count in defect_types.items():
                    stats_data.append((f"Defecto: {defect_type}", str(count)))
                    
            # Actualizar tabla
            self.stats_table.setRowCount(len(stats_data))
            
            for i, (metric, value) in enumerate(stats_data):
                self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
                self.stats_table.setItem(i, 1, QTableWidgetItem(value))
                
        except Exception as e:
            self.logger.error(f"Error al actualizar estadísticas: {str(e)}")


class NewBatchDialog(QDialog):
    """Diálogo para crear un nuevo lote."""
    
    def __init__(self, product, parent=None):
        super().__init__(parent)
        self.product = product
        self.batch_code = ""
        self.notes = ""
        
        self.setWindowTitle("Nuevo Lote")
        self.setMinimumWidth(400)
        
        # Layout
        layout = QVBoxLayout(self)
        
        # Información del producto
        product_layout = QHBoxLayout()
        product_layout.addWidget(QLabel("Producto:"))
        product_label = QLabel(f"{product.get('name', '')} (SKU: {product.get('sku', '')})")
        product_label.setStyleSheet("font-weight: bold;")
        product_layout.addWidget(product_label)
        
        # Código del lote
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Código del Lote:"))
        self.batch_edit = QLineEdit()
        batch_layout.addWidget(self.batch_edit)
        
        # Generar código predeterminado
        today = datetime.datetime.now().strftime("%Y%m%d")
        sku = product.get('sku', 'SKU')
        self.batch_edit.setText(f"{sku}-{today}")
        
        # Notas
        notes_layout = QVBoxLayout()
        notes_layout.addWidget(QLabel("Notas:"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        notes_layout.addWidget(self.notes_edit)
        
        # Botones
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        self.create_button = QPushButton("Crear Lote")
        self.create_button.clicked.connect(self.accept)
        self.create_button.setDefault(True)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.create_button)
        
        # Agregar layouts
        layout.addLayout(product_layout)
        layout.addLayout(batch_layout)
        layout.addLayout(notes_layout)
        layout.addLayout(button_layout)
        
    def accept(self):
        """Acepta el diálogo y guarda los datos."""
        self.batch_code = self.batch_edit.text().strip()
        self.notes = self.notes_edit.toPlainText().strip()
        
        if not self.batch_code:
            QMessageBox.warning(self, "Error", "El código del lote no puede estar vacío.")
            return
            
        super().accept()


class MainScreen(QMainWindow):
    """Ventana principal del sistema de inspección visual."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('system_logger')
        
        # Configurar ventana
        self.setWindowTitle("Eyve Inspection - Sistema de Inspección Visual")
        self.setMinimumSize(1200, 800)
        
        # Widgets principales
        self.camera_view = CameraViewWidget()
        self.results_widget = ResultsWidget()
        self.control_panel = ControlPanel()
        
        # Conectar señales
        self.control_panel.results_available.connect(self.on_results_available)
        self.control_panel.camera_changed.connect(self.on_camera_changed)
        self.control_panel.inspection_started.connect(self.on_inspection_started)
        self.control_panel.inspection_stopped.connect(self.on_inspection_stopped)
        
        # Configurar layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Panel izquierdo (cámara y resultados)
        left_panel = QSplitter(Qt.Vertical)
        left_panel.addWidget(self.camera_view)
        left_panel.addWidget(self.results_widget)
        left_panel.setSizes([600, 200])
        
        # Agregar paneles
        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(self.control_panel, 1)
        
        # Configurar barra de menú
        self.create_menu_bar()
        
        # Configurar barra de estado
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("Sistema inicializado")
        self.statusBar.addPermanentWidget(self.status_label)
        
        # Timer para actualizar barra de estado
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status_bar)
        self.status_timer.start(1000)  # Actualizar cada segundo
        
        # Inicializar sistema
        self.init_system()
        
    def create_menu_bar(self):
        """Crea la barra de menú."""
        menu_bar = self.menuBar()
        
        # Menú Sistema
        system_menu = menu_bar.addMenu("Sistema")
        
        login_action = QAction("Iniciar Sesión", self)
        login_action.triggered.connect(self.on_login)
        system_menu.addAction(login_action)
        
        logout_action = QAction("Cerrar Sesión", self)
        logout_action.triggered.connect(self.on_logout)
        system_menu.addAction(logout_action)
        
        system_menu.addSeparator()
        
        backup_action = QAction("Crear Copia de Seguridad", self)
        backup_action.triggered.connect(self.on_backup)
        system_menu.addAction(backup_action)
        
        system_menu.addSeparator()
        
        exit_action = QAction("Salir", self)
        exit_action.triggered.connect(self.close)
        system_menu.addAction(exit_action)
        
        # Menú Productos
        products_menu = menu_bar.addMenu("Productos")
        
        select_product_action = QAction("Seleccionar Producto", self)
        select_product_action.triggered.connect(self.on_select_product)
        products_menu.addAction(select_product_action)
        
        # Menú Configuración
        config_menu = menu_bar.addMenu("Configuración")
        
        camera_config_action = QAction("Configurar Cámaras", self)
        camera_config_action.triggered.connect(self.on_camera_config)
        config_menu.addAction(camera_config_action)
        
        inspection_config_action = QAction("Configurar Inspección", self)
        inspection_config_action.triggered.connect(self.on_inspection_config)
        config_menu.addAction(inspection_config_action)
        
        # Menú Ayuda
        help_menu = menu_bar.addMenu("Ayuda")
        
        about_action = QAction("Acerca de", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
        
    def init_system(self):
        """Inicializa el sistema."""
        try:
            # Obtener gestor del sistema
            system_manager = get_system_manager()
            if not system_manager:
                QMessageBox.critical(self, "Error", "No se pudo inicializar el gestor del sistema")
                return
                
            # Iniciar sistema
            if not system_manager.initialize():
                QMessageBox.critical(self, "Error", "Error al inicializar el sistema")
                return
                
            if not system_manager.start():
                QMessageBox.critical(self, "Error", "Error al iniciar el sistema")
                return
                
            # Iniciar cámara
            self.control_panel.refresh_cameras()
            if self.control_panel.camera_combo.count() > 0:
                camera_id = self.control_panel.camera_combo.currentText()
                self.camera_view.start_capture(camera_id)
                
            self.statusBar.showMessage("Sistema inicializado correctamente", 5000)
            
        except Exception as e:
            self.logger.error(f"Error al inicializar sistema: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error al inicializar sistema:\n{str(e)}")
            
    def on_results_available(self, results):
        """Maneja los resultados de inspección disponibles.
        
        Args:
            results: Resultados de la inspección
        """
        # Actualizar widget de resultados
        self.results_widget.set_results(results)
        
        # Actualizar superposición en vista de cámara
        self.camera_view.set_results(results)
        
        # Actualizar barra de estado
        status = results.get("status", "unknown")
        if status == "pass":
            self.statusBar.showMessage("Inspección completada: APROBADO", 5000)
        else:
            self.statusBar.showMessage("Inspección completada: RECHAZADO", 5000)
            
    def on_camera_changed(self, camera_id):
        """Maneja el cambio de cámara seleccionada.
        
        Args:
            camera_id: ID de la cámara seleccionada
        """
        # Reiniciar la captura con la nueva cámara
        self.camera_view.stop_capture()
        self.camera_view.start_capture(camera_id)
        
    def on_inspection_started(self):
        """Maneja el inicio de la inspección automática."""
        self.statusBar.showMessage("Inspección automática iniciada", 5000)
        
    def on_inspection_stopped(self):
        """Maneja la detención de la inspección automática."""
        self.statusBar.showMessage("Inspección automática detenida", 5000)
        
    def on_login(self):
        """Maneja la acción de inicio de sesión."""
        try:
            # Mostrar diálogo de login
            from login_screen import LoginDialog
            dialog = LoginDialog(self)
            
            if dialog.exec_() == QDialog.Accepted:
                # Actualizar interfaz con información de usuario
                system_manager = get_system_manager()
                if system_manager:
                    user = system_manager.get_current_user()
                    if user:
                        self.statusBar.showMessage(f"Sesión iniciada como {user.get('username')}", 5000)
                        
        except Exception as e:
            self.logger.error(f"Error en inicio de sesión: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error en inicio de sesión:\n{str(e)}")
            
    def on_logout(self):
        """Maneja la acción de cierre de sesión."""
        try:
            system_manager = get_system_manager()
            if system_manager:
                system_manager.logout_user()
                self.statusBar.showMessage("Sesión cerrada", 5000)
                
        except Exception as e:
            self.logger.error(f"Error en cierre de sesión: {str(e)}")
            
    def on_backup(self):
        """Maneja la acción de crear copia de seguridad."""
        try:
            system_manager = get_system_manager()
            if not system_manager:
                return
                
            # Crear directorio para backups
            backup_dir = os.path.join(system_manager.data_path, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Crear backup de BD
            db_backup = system_manager.backup_database()
            
            # Crear backup de configuración
            config_backup = system_manager.backup_configuration()
            
            if db_backup or config_backup:
                msg = "Copias de seguridad creadas:\n"
                if db_backup:
                    msg += f"Base de datos: {os.path.basename(db_backup)}\n"
                if config_backup:
                    msg += f"Configuración: {os.path.basename(config_backup)}\n"
                    
                QMessageBox.information(self, "Copia de Seguridad", msg)
            else:
                QMessageBox.warning(self, "Copia de Seguridad", 
                                   "No se pudieron crear copias de seguridad")
                
        except Exception as e:
            self.logger.error(f"Error al crear copia de seguridad: {str(e)}")
            QMessageBox.critical(self, "Error", 
                               f"Error al crear copia de seguridad:\n{str(e)}")
            
    def on_select_product(self):
        """Maneja la acción de seleccionar producto."""
        try:
            system_manager = get_system_manager()
            if not system_manager or not system_manager.product_manager:
                return
                
            # Obtener lista de productos
            products = system_manager.product_manager.list_products()
            
            if not products:
                QMessageBox.information(self, "Seleccionar Producto", 
                                       "No hay productos disponibles")
                return
                
            # Mostrar diálogo de selección
            dialog = ProductSelectionDialog(products, self)
            if dialog.exec_() == QDialog.Accepted:
                product_id = dialog.selected_product_id
                
                if product_id:
                    product = system_manager.select_product(product_id)
                    if product:
                        self.statusBar.showMessage(
                            f"Producto seleccionado: {product.get('name')} (SKU: {product.get('sku')})", 
                            5000
                        )
                        
        except Exception as e:
            self.logger.error(f"Error al seleccionar producto: {str(e)}")
            QMessageBox.critical(self, "Error", 
                               f"Error al seleccionar producto:\n{str(e)}")
            
    def on_camera_config(self):
        """Maneja la acción de configurar cámaras."""
        # Esta funcionalidad requiere implementación adicional
        QMessageBox.information(self, "Configurar Cámaras",
                              "Funcionalidad no implementada")
            
    def on_inspection_config(self):
        """Maneja la acción de configurar inspección."""
        # Esta funcionalidad requiere implementación adicional
        QMessageBox.information(self, "Configurar Inspección",
                              "Funcionalidad no implementada")
            
    def on_about(self):
        """Muestra información sobre la aplicación."""
        QMessageBox.about(self, "Acerca de",
                        "Eyve Inspection v1.0\n\n"
                        "Sistema de inspección visual para control de calidad.\n\n"
                        "© 2025 Eyve Inspection")
            
    def update_status_bar(self):
        """Actualiza la barra de estado con información del sistema."""
        try:
            system_manager = get_system_manager()
            if not system_manager:
                return
                
            # Obtener usuario actual
            user = system_manager.get_current_user()
            user_text = f"Usuario: {user.get('username')}" if user else "No hay sesión iniciada"
            
            # Obtener información del producto actual
            with system_manager.status_lock:
                product = system_manager.system_status.get("current_product")
                batch_id = system_manager.system_status.get("current_batch")
                
            product_text = f"Producto: {product.get('name')}" if product else "Ningún producto seleccionado"
            batch_text = f"Lote: Activo (ID: {batch_id})" if batch_id else "Sin lote activo"
            
            # Actualizar etiqueta de estado
            self.status_label.setText(f"{user_text} | {product_text} | {batch_text}")
            
        except Exception as e:
            self.logger.error(f"Error al actualizar barra de estado: {str(e)}")
            
    def closeEvent(self, event):
        """Maneja el evento de cierre de la ventana.
        
        Args:
            event: Evento de cierre
        """
        reply = QMessageBox.question(
            self, "Salir", 
            "¿Está seguro de que desea salir?\nSe detendrán todas las operaciones en curso.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Detener timers
                self.status_timer.stop()
                
                # Detener captura de cámara
                self.camera_view.stop_capture()
                
                # Detener sistema
                system_manager = get_system_manager()
                if system_manager:
                    system_manager.stop()
                    
                event.accept()
            except Exception as e:
                self.logger.error(f"Error al cerrar aplicación: {str(e)}")
                event.accept()
        else:
            event.ignore()


class ProductSelectionDialog(QDialog):
    """Diálogo para seleccionar un producto."""
    
    def __init__(self, products, parent=None):
        super().__init__(parent)
        self.products = products
        self.selected_product_id = None
        
        self.setWindowTitle("Seleccionar Producto")
        self.setMinimumWidth(500)
        
        # Layout
        layout = QVBoxLayout(self)
        
        # Tabla de productos
        self.products_table = QTableWidget()
        self.products_table.setColumnCount(3)
        self.products_table.setHorizontalHeaderLabels(["ID", "SKU", "Nombre"])
        self.products_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.products_table.setSelectionMode(QTableWidget.SingleSelection)
        
        # Configurar ancho de columnas
        self.products_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.products_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        
        # Llenar tabla
        self.products_table.setRowCount(len(products))
        for i, product in enumerate(products):
            self.products_table.setItem(i, 0, QTableWidgetItem(str(product["id"])))
            self.products_table.setItem(i, 1, QTableWidgetItem(product["sku"]))
            self.products_table.setItem(i, 2, QTableWidgetItem(product["name"]))
        
        # Botones
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        self.select_button = QPushButton("Seleccionar")
        self.select_button.clicked.connect(self.accept)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.select_button)
        
        # Agregar widgets
        layout.addWidget(QLabel("Seleccione un producto:"))
        layout.addWidget(self.products_table)
        layout.addLayout(button_layout)
        
    def accept(self):
        """Acepta el diálogo y guarda el producto seleccionado."""
        selected_items = self.products_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Debe seleccionar un producto.")
            return
            
        row = selected_items[0].row()
        product_id_item = self.products_table.item(row, 0)
        
        if product_id_item:
            try:
                self.selected_product_id = int(product_id_item.text())
                super().accept()
            except ValueError:
                QMessageBox.warning(self, "Error", "ID de producto inválido.")
        else:
            QMessageBox.warning(self, "Error", "Error al obtener el producto seleccionado.")


# Función principal para ejecutar la aplicación
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Estilo moderno y consistente
    
    # Configurar paleta de colores
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    main_window = MainScreen()
    main_window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
