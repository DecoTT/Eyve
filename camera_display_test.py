#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para probar la visualización de cámaras con PyQt
"""

import sys
import cv2
import time
import logging
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QComboBox, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class CameraTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Configuración de la ventana
        self.setWindowTitle("Test de Visualización de Cámara")
        self.setGeometry(100, 100, 800, 600)
        
        # Widgets
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Selector de cámara
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        layout.addWidget(self.camera_combo)
        
        # Botón para refrescar lista de cámaras
        refresh_btn = QPushButton("Refrescar Cámaras")
        refresh_btn.clicked.connect(self.refresh_cameras)
        layout.addWidget(refresh_btn)
        
        # Etiqueta para mostrar la imagen
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Seleccione una cámara y presione 'Iniciar'")
        self.image_label.setStyleSheet("background-color: black; color: white; font-size: 18px;")
        layout.addWidget(self.image_label)
        
        # Botones de control
        self.start_button = QPushButton("Iniciar")
        self.start_button.clicked.connect(self.start_camera)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Detener")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        # Timer para actualizar la imagen
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_frame)
        self.update_interval = 33  # ~30 FPS
        
        # Variables para la cámara
        self.capture = None
        self.camera_index = -1
        
        # Estadísticas
        self.frame_count = 0
        self.start_time = 0
        self.status_label = QLabel("Estado: Listo")
        layout.addWidget(self.status_label)
        
    def refresh_cameras(self):
        """Actualiza la lista de cámaras disponibles."""
        self.camera_combo.clear()
        
        available_cameras = []
        # Probar los primeros 10 índices
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    # Intentar leer un frame para verificar
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        available_cameras.append(i)
                        self.camera_combo.addItem(f"Cámara {i}", i)
                else:
                    cap.release()
            except Exception as e:
                logger.error(f"Error al probar cámara {i}: {e}")
                
        if not available_cameras:
            self.camera_combo.addItem("No hay cámaras disponibles")
            self.start_button.setEnabled(False)
        else:
            self.start_button.setEnabled(True)
            
        logger.info(f"Cámaras disponibles: {available_cameras}")
        
    def start_camera(self):
        """Inicia la captura de la cámara seleccionada."""
        if self.capture is not None:
            self.stop_camera()
            
        # Obtener índice de cámara seleccionada
        self.camera_index = self.camera_combo.currentData()
        if self.camera_index is None:
            self.status_label.setText("Error: No hay cámara seleccionada válida")
            return
            
        try:
            # Iniciar captura con DirectShow
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            # Configurar resolución
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            if not self.capture.isOpened():
                self.status_label.setText(f"Error: No se pudo abrir la cámara {self.camera_index}")
                return
                
            # Leer un frame para verificar que funciona
            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.status_label.setText(f"Error: La cámara {self.camera_index} no devuelve frames válidos")
                self.capture.release()
                self.capture = None
                return
                
            # Iniciar timer para actualizar frames
            self.frame_count = 0
            self.start_time = time.time()
            self.update_timer.start(self.update_interval)
            
            # Actualizar interfaz
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.camera_combo.setEnabled(False)
            
            self.status_label.setText(f"Capturando desde cámara {self.camera_index}")
            logger.info(f"Cámara {self.camera_index} iniciada con éxito")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            logger.error(f"Error al iniciar cámara: {e}")
            if self.capture:
                self.capture.release()
                self.capture = None
                
    def stop_camera(self):
        """Detiene la captura de la cámara."""
        self.update_timer.stop()
        
        if self.capture:
            self.capture.release()
            self.capture = None
            
        # Actualizar interfaz
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_combo.setEnabled(True)
        
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            self.status_label.setText(f"Detenido. Promedio FPS: {fps:.1f}")
        else:
            self.status_label.setText("Detenido")
            
        logger.info("Cámara detenida")
        
    def update_frame(self):
        """Actualiza el frame mostrado desde la cámara."""
        if not self.capture:
            return
            
        try:
            # Capturar frame
            ret, frame = self.capture.read()
            
            if not ret or frame is None:
                self.status_label.setText("Error: Frame no válido, reconectando...")
                # Intentar reconectar
                self.capture.release()
                self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                return
                
            # Convertir a RGB para Qt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crear QImage y luego QPixmap
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Escalar manteniendo relación de aspecto
            label_size = self.image_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Mostrar imagen
            self.image_label.setPixmap(scaled_pixmap)
            
            # Actualizar estadísticas
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                fps = self.frame_count / elapsed
                self.status_label.setText(f"FPS: {fps:.1f}, Frames: {self.frame_count}, Tamaño: {w}x{h}")
                
        except Exception as e:
            self.status_label.setText(f"Error al actualizar frame: {str(e)}")
            logger.error(f"Error al actualizar frame: {e}")
            
    def closeEvent(self, event):
        """Maneja el evento de cierre de la ventana."""
        self.stop_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = CameraTestWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()