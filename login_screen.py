#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pantalla de Inicio de Sesión
---------------------------
Proporciona la interfaz gráfica para la autenticación de usuarios en el sistema.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (QDialog, QApplication, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QLineEdit, QMessageBox, QFrame, QFormLayout,
                             QCheckBox, QComboBox)
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QSize

# En una implementación real, se importarían los componentes del sistema
try:
    from system_manager import get_system_manager
except ImportError:
    pass


class LoginDialog(QDialog):
    """Diálogo de inicio de sesión en el sistema."""
    
    # Señal emitida cuando un usuario inicia sesión correctamente
    login_successful = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """Inicializa el diálogo de inicio de sesión.
        
        Args:
            parent: Widget padre
        """
        super().__init__(parent)
        self.logger = logging.getLogger('system_logger')
        
        # Configuración de la ventana
        self.setWindowTitle("Inicio de Sesión - Eyve Inspection")
        self.setFixedSize(450, 350)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        
        # Crear interfaz
        self.init_ui()
        
    def init_ui(self):
        """Crea y configura los elementos de la interfaz de usuario."""
        # Layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Logo y título
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        # En una aplicación real, cargaría una imagen real
        # logo_pixmap = QPixmap("assets/logo.png")
        # logo_label.setPixmap(logo_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Como alternativa, usamos un texto con estilo
        logo_label.setText("E")
        logo_label.setStyleSheet("""
            font-size: 48px;
            font-weight: bold;
            color: #3498db;
            background-color: #ecf0f1;
            border-radius: 40px;
            min-width: 80px;
            min-height: 80px;
            qproperty-alignment: AlignCenter;
        """)
        logo_label.setFixedSize(80, 80)
        
        title_layout = QVBoxLayout()
        title_label = QLabel("Eyve Inspection")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50;")
        subtitle_label = QLabel("Sistema de Inspección Visual")
        subtitle_label.setStyleSheet("font-size: 14px; color: #7f8c8d;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        
        header_layout.addWidget(logo_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # Formulario de inicio de sesión
        form_frame = QFrame()
        form_frame.setFrameShape(QFrame.StyledPanel)
        form_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #e0e0e0;
            }
        """)
        form_layout = QVBoxLayout(form_frame)
        
        # Campos del formulario
        login_form = QFormLayout()
        login_form.setSpacing(10)
        login_form.setContentsMargins(20, 20, 20, 20)
        
        # Usuario
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Nombre de usuario")
        self.username_edit.setMinimumHeight(35)
        self.username_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        login_form.addRow("Usuario:", self.username_edit)
        
        # Contraseña
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Contraseña")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setMinimumHeight(35)
        self.password_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        login_form.addRow("Contraseña:", self.password_edit)
        
        # Recordar credenciales
        self.remember_checkbox = QCheckBox("Recordar mis credenciales")
        self.remember_checkbox.setStyleSheet("""
            QCheckBox {
                color: #7f8c8d;
            }
        """)
        
        # Botones
        buttons_layout = QHBoxLayout()
        
        self.login_button = QPushButton("Iniciar Sesión")
        self.login_button.setMinimumHeight(40)
        self.login_button.setCursor(Qt.PointingHandCursor)
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
        """)
        self.login_button.clicked.connect(self.handle_login)
        
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setMinimumHeight(40)
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #2c3e50;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #bdc3c7;
            }
            QPushButton:pressed {
                background-color: #95a5a6;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.login_button)
        
        # Agregar elementos al layout del formulario
        form_layout.addLayout(login_form)
        form_layout.addWidget(self.remember_checkbox)
        form_layout.addLayout(buttons_layout)
        
        # Footer con información
        footer_label = QLabel("Para asistencia técnica, contacte al administrador del sistema.")
        footer_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        footer_label.setAlignment(Qt.AlignCenter)
        
        # Agregar todos los elementos al layout principal
        main_layout.addLayout(header_layout)
        main_layout.addWidget(form_frame)
        main_layout.addWidget(footer_label)
        
        # Enfoque inicial
        self.username_edit.setFocus()
        
        # Cargar credenciales guardadas (si hay)
        self.load_saved_credentials()
        
    def load_saved_credentials(self):
        """Carga credenciales guardadas si están disponibles."""
        try:
            settings_file = os.path.join(os.path.expanduser("~"), ".eyve_credentials")
            if os.path.exists(settings_file):
                with open(settings_file, "r") as f:
                    username = f.readline().strip()
                    
                if username:
                    self.username_edit.setText(username)
                    self.remember_checkbox.setChecked(True)
                    self.password_edit.setFocus()
        except Exception as e:
            self.logger.error(f"Error al cargar credenciales guardadas: {str(e)}")
            
    def save_credentials(self, username: str):
        """Guarda las credenciales del usuario.
        
        Args:
            username: Nombre de usuario a guardar
        """
        try:
            settings_file = os.path.join(os.path.expanduser("~"), ".eyve_credentials")
            with open(settings_file, "w") as f:
                f.write(username)
        except Exception as e:
            self.logger.error(f"Error al guardar credenciales: {str(e)}")
            
    def clear_saved_credentials(self):
        """Elimina las credenciales guardadas."""
        try:
            settings_file = os.path.join(os.path.expanduser("~"), ".eyve_credentials")
            if os.path.exists(settings_file):
                os.remove(settings_file)
        except Exception as e:
            self.logger.error(f"Error al eliminar credenciales guardadas: {str(e)}")
            
    def handle_login(self):
        """Maneja el proceso de inicio de sesión."""
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        
        if not username or not password:
            QMessageBox.warning(self, "Error de Inicio de Sesión", 
                               "Por favor, introduzca nombre de usuario y contraseña.")
            return
            
        try:
            # Obtener el gestor del sistema
            system_manager = get_system_manager()
            if not system_manager:
                QMessageBox.critical(self, "Error", "No se pudo inicializar el gestor del sistema.")
                return
                
            # Autenticar usuario
            user_data = system_manager.authenticate_user(username, password)
            
            if user_data:
                # Recordar credenciales si está marcado
                if self.remember_checkbox.isChecked():
                    self.save_credentials(username)
                else:
                    self.clear_saved_credentials()
                    
                # Emitir señal y cerrar diálogo
                self.login_successful.emit(user_data)
                self.accept()
            else:
                QMessageBox.warning(self, "Error de Inicio de Sesión", 
                                   "Nombre de usuario o contraseña incorrectos.")
                
        except Exception as e:
            self.logger.error(f"Error durante el inicio de sesión: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error durante el inicio de sesión:\n{str(e)}")
            
    def keyPressEvent(self, event):
        """Maneja eventos de teclado.
        
        Args:
            event: Evento de teclado
        """
        # Presionar Enter en cualquier campo inicia el login
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.handle_login()
        else:
            super().keyPressEvent(event)


class UserRegistrationDialog(QDialog):
    """Diálogo para registrar nuevos usuarios en el sistema."""
    
    def __init__(self, parent=None):
        """Inicializa el diálogo de registro de usuarios.
        
        Args:
            parent: Widget padre
        """
        super().__init__(parent)
        self.logger = logging.getLogger('system_logger')
        
        # Configuración de la ventana
        self.setWindowTitle("Registro de Usuario - Eyve Inspection")
        self.setFixedSize(500, 450)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        
        # Crear interfaz
        self.init_ui()
        
    def init_ui(self):
        """Crea y configura los elementos de la interfaz de usuario."""
        # Layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Título
        title_label = QLabel("Registro de Nuevo Usuario")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        title_label.setAlignment(Qt.AlignCenter)
        
        # Formulario de registro
        form_frame = QFrame()
        form_frame.setFrameShape(QFrame.StyledPanel)
        form_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #e0e0e0;
            }
        """)
        form_layout = QVBoxLayout(form_frame)
        
        # Campos del formulario
        registration_form = QFormLayout()
        registration_form.setSpacing(10)
        registration_form.setContentsMargins(20, 20, 20, 20)
        
        # Usuario
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Nombre de usuario")
        self.username_edit.setMinimumHeight(35)
        self.username_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        registration_form.addRow("Usuario:", self.username_edit)
        
        # Nombre completo
        self.fullname_edit = QLineEdit()
        self.fullname_edit.setPlaceholderText("Nombre completo")
        self.fullname_edit.setMinimumHeight(35)
        self.fullname_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        registration_form.addRow("Nombre completo:", self.fullname_edit)
        
        # Contraseña
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Contraseña")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setMinimumHeight(35)
        self.password_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        registration_form.addRow("Contraseña:", self.password_edit)
        
        # Confirmar contraseña
        self.confirm_password_edit = QLineEdit()
        self.confirm_password_edit.setPlaceholderText("Confirmar contraseña")
        self.confirm_password_edit.setEchoMode(QLineEdit.Password)
        self.confirm_password_edit.setMinimumHeight(35)
        self.confirm_password_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        registration_form.addRow("Confirmar contraseña:", self.confirm_password_edit)
        
        # Rol
        self.role_combo = QComboBox()
        self.role_combo.addItems(["operador", "supervisor", "admin"])
        self.role_combo.setMinimumHeight(35)
        self.role_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            QComboBox:focus {
                border: 1px solid #3498db;
            }
        """)
        registration_form.addRow("Rol:", self.role_combo)
        
        # Botones
        buttons_layout = QHBoxLayout()
        
        self.register_button = QPushButton("Registrar Usuario")
        self.register_button.setMinimumHeight(40)
        self.register_button.setCursor(Qt.PointingHandCursor)
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.register_button.clicked.connect(self.handle_registration)
        
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setMinimumHeight(40)
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #2c3e50;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #bdc3c7;
            }
            QPushButton:pressed {
                background-color: #95a5a6;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.register_button)
        
        # Agregar elementos al layout del formulario
        form_layout.addLayout(registration_form)
        form_layout.addLayout(buttons_layout)
        
        # Nota informativa
        info_label = QLabel(
            "Nota: La creación de usuarios requiere privilegios de administrador.\n"
            "Los nuevos usuarios deben ser aprobados por un administrador."
        )
        info_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        info_label.setAlignment(Qt.AlignCenter)
        
        # Agregar todos los elementos al layout principal
        main_layout.addWidget(title_label)
        main_layout.addWidget(form_frame)
        main_layout.addWidget(info_label)
        
        # Enfoque inicial
        self.username_edit.setFocus()
        
    def handle_registration(self):
        """Maneja el proceso de registro de usuario."""
        # Validar campos
        username = self.username_edit.text().strip()
        fullname = self.fullname_edit.text().strip()
        password = self.password_edit.text()
        confirm_password = self.confirm_password_edit.text()
        role = self.role_combo.currentText()
        
        # Verificar campos obligatorios
        if not username or not fullname or not password:
            QMessageBox.warning(self, "Error de Registro", 
                               "Por favor, complete todos los campos obligatorios.")
            return
            
        # Verificar coincidencia de contraseñas
        if password != confirm_password:
            QMessageBox.warning(self, "Error de Registro", 
                               "Las contraseñas no coinciden.")
            return
            
        try:
            # Obtener el gestor del sistema
            system_manager = get_system_manager()
            if not system_manager:
                QMessageBox.critical(self, "Error", "No se pudo inicializar el gestor del sistema.")
                return
                
            # Verificar privilegios
            current_user = system_manager.get_current_user()
            if not current_user or current_user.get("role") != "admin":
                QMessageBox.warning(self, "Permisos Insuficientes", 
                                   "Se requieren privilegios de administrador para crear usuarios.")
                return
                
            # Crear usuario
            success = system_manager.user_manager.create_user(
                username=username,
                password=password,
                full_name=fullname,
                role=role
            )
            
            if success:
                QMessageBox.information(self, "Registro Exitoso", 
                                       f"Usuario '{username}' creado correctamente.")
                self.accept()
            else:
                QMessageBox.warning(self, "Error de Registro", 
                                   "No se pudo crear el usuario. Puede que el nombre de usuario ya exista.")
                
        except Exception as e:
            self.logger.error(f"Error durante el registro de usuario: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error durante el registro:\n{str(e)}")


class PasswordChangeDialog(QDialog):
    """Diálogo para cambiar la contraseña de un usuario."""
    
    def __init__(self, parent=None):
        """Inicializa el diálogo de cambio de contraseña.
        
        Args:
            parent: Widget padre
        """
        super().__init__(parent)
        self.logger = logging.getLogger('system_logger')
        
        # Configuración de la ventana
        self.setWindowTitle("Cambio de Contraseña - Eyve Inspection")
        self.setFixedSize(400, 300)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        
        # Crear interfaz
        self.init_ui()
        
    def init_ui(self):
        """Crea y configura los elementos de la interfaz de usuario."""
        # Layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Título
        title_label = QLabel("Cambio de Contraseña")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        title_label.setAlignment(Qt.AlignCenter)
        
        # Formulario
        form_frame = QFrame()
        form_frame.setFrameShape(QFrame.StyledPanel)
        form_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #e0e0e0;
            }
        """)
        form_layout = QVBoxLayout(form_frame)
        
        # Campos del formulario
        password_form = QFormLayout()
        password_form.setSpacing(10)
        password_form.setContentsMargins(20, 20, 20, 20)
        
        # Contraseña actual
        self.current_password_edit = QLineEdit()
        self.current_password_edit.setPlaceholderText("Contraseña actual")
        self.current_password_edit.setEchoMode(QLineEdit.Password)
        self.current_password_edit.setMinimumHeight(35)
        self.current_password_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        password_form.addRow("Contraseña actual:", self.current_password_edit)
        
        # Nueva contraseña
        self.new_password_edit = QLineEdit()
        self.new_password_edit.setPlaceholderText("Nueva contraseña")
        self.new_password_edit.setEchoMode(QLineEdit.Password)
        self.new_password_edit.setMinimumHeight(35)
        self.new_password_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        password_form.addRow("Nueva contraseña:", self.new_password_edit)
        
        # Confirmar nueva contraseña
        self.confirm_password_edit = QLineEdit()
        self.confirm_password_edit.setPlaceholderText("Confirmar nueva contraseña")
        self.confirm_password_edit.setEchoMode(QLineEdit.Password)
        self.confirm_password_edit.setMinimumHeight(35)
        self.confirm_password_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        password_form.addRow("Confirmar:", self.confirm_password_edit)
        
        # Botones
        buttons_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Guardar Cambios")
        self.save_button.setMinimumHeight(40)
        self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
        """)
        self.save_button.clicked.connect(self.handle_password_change)
        
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setMinimumHeight(40)
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #2c3e50;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #bdc3c7;
            }
            QPushButton:pressed {
                background-color: #95a5a6;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.save_button)
        
        # Agregar elementos al layout del formulario
        form_layout.addLayout(password_form)
        form_layout.addLayout(buttons_layout)
        
        # Agregar todos los elementos al layout principal
        main_layout.addWidget(title_label)
        main_layout.addWidget(form_frame)
        
        # Enfoque inicial
        self.current_password_edit.setFocus()
        
    def handle_password_change(self):
        """Maneja el proceso de cambio de contraseña."""
        # Validar campos
        current_password = self.current_password_edit.text()
        new_password = self.new_password_edit.text()
        confirm_password = self.confirm_password_edit.text()
        
        # Verificar campos obligatorios
        if not current_password or not new_password or not confirm_password:
            QMessageBox.warning(self, "Error", 
                               "Por favor, complete todos los campos.")
            return
            
        # Verificar coincidencia de contraseñas
        if new_password != confirm_password:
            QMessageBox.warning(self, "Error", 
                               "Las nuevas contraseñas no coinciden.")
            return
            
        # Verificar que la nueva contraseña sea diferente
        if current_password == new_password:
            QMessageBox.warning(self, "Error", 
                               "La nueva contraseña debe ser diferente a la actual.")
            return
            
        try:
            # Obtener el gestor del sistema
            system_manager = get_system_manager()
            if not system_manager:
                QMessageBox.critical(self, "Error", "No se pudo inicializar el gestor del sistema.")
                return
                
            # Verificar que haya un usuario logueado
            current_user = system_manager.get_current_user()
            if not current_user:
                QMessageBox.warning(self, "Error", 
                                   "No hay usuario con sesión iniciada.")
                return
                
            # Cambiar contraseña
            success = system_manager.user_manager.change_password(
                user_id=current_user["id"],
                current_password=current_password,
                new_password=new_password
            )
            
            if success:
                QMessageBox.information(self, "Cambio de Contraseña", 
                                       "Contraseña actualizada correctamente.")
                self.accept()
            else:
                QMessageBox.warning(self, "Error", 
                                   "No se pudo cambiar la contraseña. Verifique la contraseña actual.")
                
        except Exception as e:
            self.logger.error(f"Error durante el cambio de contraseña: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error durante el cambio de contraseña:\n{str(e)}")


# Función para mostrar el diálogo de inicio de sesión
def show_login_dialog():
    """Muestra el diálogo de inicio de sesión y devuelve el usuario autenticado.
    
    Returns:
        Optional[Dict[str, Any]]: Datos del usuario autenticado o None si se cancela
    """
    dialog = LoginDialog()
    result = dialog.exec_()
    
    if result == QDialog.Accepted:
        system_manager = get_system_manager()
        if system_manager:
            return system_manager.get_current_user()
    
    return None


# Para pruebas
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Crear un logger de prueba
    logging.basicConfig(level=logging.INFO)
    
    # Mostrar diálogo de login
    login_dialog = LoginDialog()
    if login_dialog.exec_() == QDialog.Accepted:
        print("Login exitoso")
        
        # Mostrar diálogo de cambio de contraseña
        password_dialog = PasswordChangeDialog()
        password_dialog.exec_()
    else:
        print("Login cancelado")
    
    sys.exit(0)
