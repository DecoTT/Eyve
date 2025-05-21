#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Base de Datos
------------------
Proporciona clases y funciones para interactuar con la base de datos
del sistema de inspección visual.
"""

import os
import json
import sqlite3
import logging
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union


class DatabaseManager:
    """Clase para gestionar la conexión y operaciones con la base de datos."""
    
    def __init__(self, db_path: str = "data/inspection_system.db"):
        """Inicializa el gestor de base de datos.
        
        Args:
            db_path: Ruta al archivo de base de datos SQLite
        """
        self.logger = logging.getLogger('system_logger')
        self.db_path = db_path
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Inicializar conexión
        self.connection = None
        self._connect()
        
        # Crear tablas si no existen
        self._create_tables()
        
    def _connect(self) -> None:
        """Establece la conexión con la base de datos."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Para obtener resultados como diccionarios
            self.logger.info(f"Conexión establecida con la base de datos: {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Error al conectar con la base de datos: {str(e)}")
            raise
    
    def _create_tables(self) -> None:
        """Crea las tablas necesarias si no existen."""
        cursor = self.connection.cursor()
        
        # Tabla de usuarios
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
        ''')
        
        # Tabla de productos/SKUs
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            config_path TEXT,
            reference_image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
        ''')
        
        # Tabla de lotes de producción
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_code TEXT UNIQUE NOT NULL,
            product_id INTEGER NOT NULL,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            status TEXT DEFAULT 'pending',
            operator_id INTEGER,
            notes TEXT,
            FOREIGN KEY (product_id) REFERENCES products (id),
            FOREIGN KEY (operator_id) REFERENCES users (id)
        )
        ''')
        
        # Tabla de inspecciones
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result TEXT,
            image_path TEXT,
            data JSON,
            operator_id INTEGER,
            FOREIGN KEY (batch_id) REFERENCES batches (id),
            FOREIGN KEY (operator_id) REFERENCES users (id)
        )
        ''')
        
        # Tabla de defectos
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS defects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER NOT NULL,
            defect_type TEXT NOT NULL,
            severity TEXT,
            position_x REAL,
            position_y REAL,
            width REAL,
            height REAL,
            confidence REAL,
            description TEXT,
            FOREIGN KEY (inspection_id) REFERENCES inspections (id)
        )
        ''')
        
        # Tabla de calibraciones
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS calibrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            camera_id TEXT,
            pixels_per_mm REAL,
            reference_distance_mm REAL,
            operator_id INTEGER,
            notes TEXT,
            FOREIGN KEY (operator_id) REFERENCES users (id)
        )
        ''')
        
        # Tabla de configuraciones
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_key TEXT UNIQUE NOT NULL,
            config_value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
        ''')
        
        # Tabla de registro de eventos del sistema
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            level TEXT NOT NULL,
            module TEXT,
            message TEXT,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        self.connection.commit()
        self.logger.info("Tablas verificadas/creadas correctamente")
    
    def close(self) -> None:
        """Cierra la conexión con la base de datos."""
        if self.connection:
            self.connection.close()
            self.logger.info("Conexión con la base de datos cerrada")
    
    def execute_query(self, query: str, params: Tuple = ()) -> Optional[sqlite3.Cursor]:
        """Ejecuta una consulta SQL.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta
            
        Returns:
            Optional[sqlite3.Cursor]: Cursor con resultados o None si hay error
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            return cursor
        except sqlite3.Error as e:
            self.logger.error(f"Error al ejecutar consulta: {str(e)}")
            self.logger.debug(f"Query: {query}, Params: {params}")
            return None
    
    def commit(self) -> bool:
        """Confirma los cambios en la base de datos.
        
        Returns:
            bool: True si se confirmaron los cambios correctamente
        """
        try:
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Error al confirmar cambios: {str(e)}")
            return False
    
    def rollback(self) -> None:
        """Revierte los cambios no confirmados en la base de datos."""
        try:
            self.connection.rollback()
            self.logger.info("Cambios revertidos")
        except sqlite3.Error as e:
            self.logger.error(f"Error al revertir cambios: {str(e)}")


class UserManager:
    """Clase para gestionar usuarios en la base de datos."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Inicializa el gestor de usuarios.
        
        Args:
            db_manager: Instancia del gestor de base de datos
        """
        self.db = db_manager
        self.logger = logging.getLogger('system_logger')
        
    def create_user(self, username: str, password: str, full_name: str, 
                   role: str = 'operator') -> bool:
        """Crea un nuevo usuario en el sistema.
        
        Args:
            username: Nombre de usuario único
            password: Contraseña del usuario
            full_name: Nombre completo del usuario
            role: Rol del usuario (admin, supervisor, operator)
            
        Returns:
            bool: True si se creó correctamente
        """
        try:
            # En una implementación real se utilizaría hashlib para el hash de contraseñas
            # Aquí simplemente simulamos un hash y salt
            salt = "salt_simulado_123"  # En implementación real sería aleatorio
            password_hash = f"hash_{password}_{salt}"  # Simulación de hash
            
            query = '''
            INSERT INTO users (username, password_hash, salt, full_name, role)
            VALUES (?, ?, ?, ?, ?)
            '''
            
            cursor = self.db.execute_query(query, 
                                        (username, password_hash, salt, full_name, role))
            if cursor:
                self.db.commit()
                self.logger.info(f"Usuario '{username}' creado correctamente")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error al crear usuario: {str(e)}")
            self.db.rollback()
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Autentica un usuario en el sistema.
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            
        Returns:
            Optional[Dict[str, Any]]: Datos del usuario o None si falla la autenticación
        """
        query = '''
        SELECT id, username, password_hash, salt, full_name, role, last_login, is_active
        FROM users
        WHERE username = ?
        '''
        
        cursor = self.db.execute_query(query, (username,))
        if cursor:
            user_data = cursor.fetchone()
            
            if user_data and user_data['is_active']:
                # Simular verificación de contraseña
                stored_hash = user_data['password_hash']
                salt = user_data['salt']
                calculated_hash = f"hash_{password}_{salt}"
                
                if calculated_hash == stored_hash:
                    # Actualizar último inicio de sesión
                    update_query = '''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                    '''
                    self.db.execute_query(update_query, (user_data['id'],))
                    self.db.commit()
                    
                    # Convertir Row a diccionario
                    user_dict = dict(user_data)
                    del user_dict['password_hash']  # No devolver el hash de contraseña
                    del user_dict['salt']  # No devolver el salt
                    
                    self.logger.info(f"Usuario '{username}' autenticado correctamente")
                    return user_dict
            
        self.logger.warning(f"Intento fallido de autenticación para '{username}'")
        return None
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene los datos de un usuario por su ID.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Optional[Dict[str, Any]]: Datos del usuario o None si no existe
        """
        query = '''
        SELECT id, username, full_name, role, last_login, created_at, is_active
        FROM users
        WHERE id = ?
        '''
        
        cursor = self.db.execute_query(query, (user_id,))
        if cursor:
            user_data = cursor.fetchone()
            if user_data:
                return dict(user_data)
        return None
    
    def update_user(self, user_id: int, data: Dict[str, Any]) -> bool:
        """Actualiza los datos de un usuario.
        
        Args:
            user_id: ID del usuario a actualizar
            data: Diccionario con los campos a actualizar
            
        Returns:
            bool: True si se actualizó correctamente
        """
        allowed_fields = ["full_name", "role", "is_active"]
        
        # Filtrar campos permitidos
        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        if not update_data:
            self.logger.warning("No hay campos válidos para actualizar")
            return False
        
        # Construir consulta dinámica
        set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
        values = list(update_data.values())
        values.append(user_id)
        
        query = f'''
        UPDATE users SET {set_clause}
        WHERE id = ?
        '''
        
        cursor = self.db.execute_query(query, tuple(values))
        if cursor:
            self.db.commit()
            self.logger.info(f"Usuario ID {user_id} actualizado correctamente")
            return True
        return False
    
    def change_password(self, user_id: int, current_password: str, 
                        new_password: str) -> bool:
        """Cambia la contraseña de un usuario.
        
        Args:
            user_id: ID del usuario
            current_password: Contraseña actual
            new_password: Nueva contraseña
            
        Returns:
            bool: True si se cambió correctamente
        """
        # Verificar contraseña actual
        query = '''
        SELECT password_hash, salt
        FROM users
        WHERE id = ?
        '''
        
        cursor = self.db.execute_query(query, (user_id,))
        if cursor:
            user_data = cursor.fetchone()
            
            if user_data:
                # Simular verificación de contraseña
                stored_hash = user_data['password_hash']
                salt = user_data['salt']
                calculated_hash = f"hash_{current_password}_{salt}"
                
                if calculated_hash == stored_hash:
                    # Generar nuevo hash para la nueva contraseña
                    new_hash = f"hash_{new_password}_{salt}"
                    
                    # Actualizar contraseña
                    update_query = '''
                    UPDATE users 
                    SET password_hash = ?
                    WHERE id = ?
                    '''
                    
                    update_cursor = self.db.execute_query(update_query, (new_hash, user_id))
                    if update_cursor:
                        self.db.commit()
                        self.logger.info(f"Contraseña actualizada para usuario ID {user_id}")
                        return True
        
        self.logger.warning(f"Fallo al cambiar contraseña para usuario ID {user_id}")
        return False
    
    def list_users(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Lista todos los usuarios del sistema.
        
        Args:
            active_only: Si es True, solo devuelve usuarios activos
            
        Returns:
            List[Dict[str, Any]]: Lista de usuarios
        """
        query = '''
        SELECT id, username, full_name, role, last_login, created_at, is_active
        FROM users
        '''
        
        if active_only:
            query += " WHERE is_active = 1"
            
        cursor = self.db.execute_query(query)
        if cursor:
            return [dict(row) for row in cursor.fetchall()]
        return []


class ProductManager:
    """Clase para gestionar productos en la base de datos."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Inicializa el gestor de productos.
        
        Args:
            db_manager: Instancia del gestor de base de datos
        """
        self.db = db_manager
        self.logger = logging.getLogger('system_logger')
    
    def create_product(self, sku: str, name: str, description: str = "", 
                      config_path: str = "", reference_image_path: str = "") -> int:
        """Crea un nuevo producto en el sistema.
        
        Args:
            sku: Código SKU único del producto
            name: Nombre del producto
            description: Descripción del producto
            config_path: Ruta al archivo de configuración
            reference_image_path: Ruta a la imagen de referencia
            
        Returns:
            int: ID del producto creado o -1 si falla
        """
        query = '''
        INSERT INTO products (sku, name, description, config_path, reference_image_path, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        '''
        
        cursor = self.db.execute_query(query, 
                                    (sku, name, description, config_path, reference_image_path))
        if cursor:
            self.db.commit()
            product_id = cursor.lastrowid
            self.logger.info(f"Producto '{name}' (SKU: {sku}) creado con ID {product_id}")
            return product_id
        return -1
    
    def get_product(self, product_id: int = None, sku: str = None) -> Optional[Dict[str, Any]]:
        """Obtiene un producto por su ID o SKU.
        
        Args:
            product_id: ID del producto (opcional)
            sku: SKU del producto (opcional)
            
        Returns:
            Optional[Dict[str, Any]]: Datos del producto o None si no existe
        """
        if product_id is None and sku is None:
            self.logger.error("Debe proporcionar ID o SKU para buscar un producto")
            return None
            
        query = '''
        SELECT id, sku, name, description, config_path, reference_image_path, 
               created_at, updated_at, is_active
        FROM products
        WHERE '''
        
        if product_id is not None:
            query += "id = ?"
            param = product_id
        else:
            query += "sku = ?"
            param = sku
            
        cursor = self.db.execute_query(query, (param,))
        if cursor:
            product_data = cursor.fetchone()
            if product_data:
                return dict(product_data)
        return None
    
    def update_product(self, product_id: int, data: Dict[str, Any]) -> bool:
        """Actualiza los datos de un producto.
        
        Args:
            product_id: ID del producto a actualizar
            data: Diccionario con los campos a actualizar
            
        Returns:
            bool: True si se actualizó correctamente
        """
        allowed_fields = ["name", "description", "config_path", 
                          "reference_image_path", "is_active"]
        
        # Filtrar campos permitidos
        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        if not update_data:
            self.logger.warning("No hay campos válidos para actualizar")
            return False
        
        # Añadir actualización de timestamp
        update_data["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Construir consulta dinámica
        set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
        values = list(update_data.values())
        values.append(product_id)
        
        query = f'''
        UPDATE products SET {set_clause}
        WHERE id = ?
        '''
        
        cursor = self.db.execute_query(query, tuple(values))
        if cursor:
            self.db.commit()
            self.logger.info(f"Producto ID {product_id} actualizado correctamente")
            return True
        return False
    
    def list_products(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Lista todos los productos del sistema.
        
        Args:
            active_only: Si es True, solo devuelve productos activos
            
        Returns:
            List[Dict[str, Any]]: Lista de productos
        """
        query = '''
        SELECT id, sku, name, description, created_at, updated_at, is_active
        FROM products
        '''
        
        if active_only:
            query += " WHERE is_active = 1"
            
        cursor = self.db.execute_query(query)
        if cursor:
            return [dict(row) for row in cursor.fetchall()]
        return []


class InspectionDataManager:
    """Clase para gestionar datos de inspecciones en la base de datos."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Inicializa el gestor de datos de inspección.
        
        Args:
            db_manager: Instancia del gestor de base de datos
        """
        self.db = db_manager
        self.logger = logging.getLogger('system_logger')
    
    def create_batch(self, batch_code: str, product_id: int, 
                    operator_id: int, notes: str = "") -> int:
        """Crea un nuevo lote de producción.
        
        Args:
            batch_code: Código único del lote
            product_id: ID del producto
            operator_id: ID del operador que crea el lote
            notes: Notas sobre el lote
            
        Returns:
            int: ID del lote creado o -1 si falla
        """
        query = '''
        INSERT INTO batches (batch_code, product_id, start_time, status, operator_id, notes)
        VALUES (?, ?, CURRENT_TIMESTAMP, 'active', ?, ?)
        '''
        
        cursor = self.db.execute_query(query, (batch_code, product_id, operator_id, notes))
        if cursor:
            self.db.commit()
            batch_id = cursor.lastrowid
            self.logger.info(f"Lote '{batch_code}' creado con ID {batch_id}")
            return batch_id
        return -1
    
    def close_batch(self, batch_id: int, status: str = "completed") -> bool:
        """Cierra un lote de producción.
        
        Args:
            batch_id: ID del lote a cerrar
            status: Estado final del lote (completed, aborted, etc.)
            
        Returns:
            bool: True si se cerró correctamente
        """
        query = '''
        UPDATE batches 
        SET end_time = CURRENT_TIMESTAMP, status = ?
        WHERE id = ?
        '''
        
        cursor = self.db.execute_query(query, (status, batch_id))
        if cursor:
            self.db.commit()
            self.logger.info(f"Lote ID {batch_id} cerrado con estado '{status}'")
            return True
        return False
    
    def record_inspection(self, batch_id: int, result: str, 
                         image_path: str, data: Dict[str, Any],
                         operator_id: int) -> int:
        """Registra una inspección realizada.
        
        Args:
            batch_id: ID del lote
            result: Resultado de la inspección (pass/fail)
            image_path: Ruta a la imagen inspeccionada
            data: Datos de la inspección (JSON)
            operator_id: ID del operador
            
        Returns:
            int: ID de la inspección registrada o -1 si falla
        """
        # Convertir datos a JSON
        data_json = json.dumps(data)
        
        query = '''
        INSERT INTO inspections (batch_id, timestamp, result, image_path, data, operator_id)
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
        '''
        
        cursor = self.db.execute_query(query, 
                                    (batch_id, result, image_path, data_json, operator_id))
        if cursor:
            inspection_id = cursor.lastrowid
            
            # Registrar defectos si es "fail" y hay datos de defectos
            if result == "fail" and "defects" in data:
                self._record_defects(inspection_id, data["defects"])
                
            self.db.commit()
            self.logger.info(f"Inspección registrada con ID {inspection_id}, resultado: {result}")
            return inspection_id
        return -1
    
    def _record_defects(self, inspection_id: int, defects_data: List[Dict[str, Any]]) -> None:
        """Registra los defectos de una inspección.
        
        Args:
            inspection_id: ID de la inspección
            defects_data: Lista de defectos detectados
        """
        for defect in defects_data:
            defect_type = defect.get("type", "unknown")
            severity = defect.get("severity", "medium")
            
            # Obtener datos de posición y dimensión
            bbox = defect.get("bbox", (0, 0, 0, 0))
            if len(bbox) == 4:
                x, y, w, h = bbox
            else:
                x, y, w, h = 0, 0, 0, 0
                
            confidence = defect.get("confidence", 0.0)
            description = defect.get("description", "")
            
            query = '''
            INSERT INTO defects (
                inspection_id, defect_type, severity, position_x, position_y, 
                width, height, confidence, description
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            self.db.execute_query(query, 
                               (inspection_id, defect_type, severity, x, y, 
                                w, h, confidence, description))
    
    def get_inspection(self, inspection_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene los datos completos de una inspección.
        
        Args:
            inspection_id: ID de la inspección
            
        Returns:
            Optional[Dict[str, Any]]: Datos de la inspección con sus defectos
        """
        # Obtener datos básicos de la inspección
        query = '''
        SELECT i.id, i.batch_id, i.timestamp, i.result, i.image_path, i.data,
               b.batch_code, p.sku, p.name as product_name
        FROM inspections i
        JOIN batches b ON i.batch_id = b.id
        JOIN products p ON b.product_id = p.id
        WHERE i.id = ?
        '''
        
        cursor = self.db.execute_query(query, (inspection_id,))
        if not cursor:
            return None
            
        inspection_data = cursor.fetchone()
        if not inspection_data:
            return None
            
        # Convertir a diccionario
        result = dict(inspection_data)
        
        # Cargar JSON de datos
        try:
            result["data"] = json.loads(result["data"])
        except:
            result["data"] = {}
            
        # Obtener defectos asociados
        defects_query = '''
        SELECT id, defect_type, severity, position_x, position_y, 
               width, height, confidence, description
        FROM defects
        WHERE inspection_id = ?
        '''
        
        defects_cursor = self.db.execute_query(defects_query, (inspection_id,))
        if defects_cursor:
            result["defects"] = [dict(row) for row in defects_cursor.fetchall()]
        else:
            result["defects"] = []
            
        return result
    
    def get_batch_statistics(self, batch_id: int) -> Dict[str, Any]:
        """Obtiene estadísticas de un lote.
        
        Args:
            batch_id: ID del lote
            
        Returns:
            Dict[str, Any]: Estadísticas del lote
        """
        stats = {
            "batch_id": batch_id,
            "total_inspections": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "defect_types": {},
            "start_time": None,
            "end_time": None,
            "duration_minutes": 0
        }
        
        # Obtener datos básicos del lote
        batch_query = '''
        SELECT batch_code, product_id, start_time, end_time, status
        FROM batches
        WHERE id = ?
        '''
        
        batch_cursor = self.db.execute_query(batch_query, (batch_id,))
        if not batch_cursor:
            return stats
            
        batch_data = batch_cursor.fetchone()
        if not batch_data:
            return stats
            
        # Actualizar datos básicos
        stats["batch_code"] = batch_data["batch_code"]
        stats["product_id"] = batch_data["product_id"]
        stats["status"] = batch_data["status"]
        stats["start_time"] = batch_data["start_time"]
        stats["end_time"] = batch_data["end_time"]
        
        # Calcular duración si hay fecha de finalización
        if stats["start_time"] and stats["end_time"]:
            start = datetime.datetime.fromisoformat(stats["start_time"].replace("Z", "+00:00"))
            end = datetime.datetime.fromisoformat(stats["end_time"].replace("Z", "+00:00"))
            stats["duration_minutes"] = (end - start).total_seconds() / 60
            
        # Obtener conteo de inspecciones
        counts_query = '''
        SELECT result, COUNT(*) as count
        FROM inspections
        WHERE batch_id = ?
        GROUP BY result
        '''
        
        counts_cursor = self.db.execute_query(counts_query, (batch_id,))
        if counts_cursor:
            for row in counts_cursor.fetchall():
                if row["result"] == "pass":
                    stats["passed"] = row["count"]
                elif row["result"] == "fail":
                    stats["failed"] = row["count"]
            
            stats["total_inspections"] = stats["passed"] + stats["failed"]
            if stats["total_inspections"] > 0:
                stats["pass_rate"] = (stats["passed"] / stats["total_inspections"]) * 100
        
        # Obtener tipos de defectos
        defects_query = '''
        SELECT d.defect_type, COUNT(*) as count
        FROM defects d
        JOIN inspections i ON d.inspection_id = i.id
        WHERE i.batch_id = ?
        GROUP BY d.defect_type
        '''
        
        defects_cursor = self.db.execute_query(defects_query, (batch_id,))
        if defects_cursor:
            for row in defects_cursor.fetchall():
                stats["defect_types"][row["defect_type"]] = row["count"]
                
        return stats


class SystemConfigManager:
    """Clase para gestionar configuraciones del sistema en la base de datos."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Inicializa el gestor de configuraciones.
        
        Args:
            db_manager: Instancia del gestor de base de datos
        """
        self.db = db_manager
        self.logger = logging.getLogger('system_logger')
        
        # Inicializar configuraciones por defecto si no existen
        self._init_default_configs()
    
    def _init_default_configs(self) -> None:
        """Inicializa las configuraciones por defecto si no existen."""
        default_configs = {
            "inspection_timeout": "30",  # segundos
            "save_all_images": "true",
            "image_storage_path": "data/images",
            "backup_enabled": "true",
            "backup_interval": "86400",  # segundos (1 día)
            "ui_theme": "light",
            "log_level": "info"
        }
        
        for key, value in default_configs.items():
            # Verificar si la configuración ya existe
            query = "SELECT id FROM system_configs WHERE config_key = ?"
            cursor = self.db.execute_query(query, (key,))
            
            if cursor and cursor.fetchone() is None:
                # Insertar configuración predeterminada
                insert_query = '''
                INSERT INTO system_configs (config_key, config_value, description)
                VALUES (?, ?, ?)
                '''
                
                description = f"Configuración predeterminada: {key}"
                self.db.execute_query(insert_query, (key, value, description))
                
        self.db.commit()
        self.logger.info("Configuraciones predeterminadas inicializadas")
    
    def get_config(self, key: str, default_value: str = None) -> str:
        """Obtiene el valor de una configuración.
        
        Args:
            key: Clave de la configuración
            default_value: Valor por defecto si no existe
            
        Returns:
            str: Valor de la configuración
        """
        query = "SELECT config_value FROM system_configs WHERE config_key = ?"
        cursor = self.db.execute_query(query, (key,))
        
        if cursor:
            row = cursor.fetchone()
            if row:
                return row["config_value"]
                
        return default_value
    
    def set_config(self, key: str, value: str, description: str = None) -> bool:
        """Establece el valor de una configuración.
        
        Args:
            key: Clave de la configuración
            value: Nuevo valor
            description: Descripción opcional
            
        Returns:
            bool: True si se estableció correctamente
        """
        # Verificar si la configuración existe
        check_query = "SELECT id FROM system_configs WHERE config_key = ?"
        check_cursor = self.db.execute_query(check_query, (key,))
        
        if check_cursor and check_cursor.fetchone():
            # Actualizar
            update_query = '''
            UPDATE system_configs
            SET config_value = ?, updated_at = CURRENT_TIMESTAMP
            '''
            
            if description:
                update_query += ", description = ?"
                params = (value, description, key)
            else:
                params = (value, key)
                
            update_query += " WHERE config_key = ?"
            
            cursor = self.db.execute_query(update_query, params)
        else:
            # Insertar nueva
            insert_query = '''
            INSERT INTO system_configs (config_key, config_value, description)
            VALUES (?, ?, ?)
            '''
            
            desc = description or f"Configuración: {key}"
            cursor = self.db.execute_query(insert_query, (key, value, desc))
            
        if cursor:
            self.db.commit()
            self.logger.info(f"Configuración '{key}' actualizada a '{value}'")
            return True
            
        return False
    
    def get_all_configs(self) -> Dict[str, str]:
        """Obtiene todas las configuraciones del sistema.
        
        Returns:
            Dict[str, str]: Diccionario con todas las configuraciones
        """
        query = "SELECT config_key, config_value FROM system_configs"
        cursor = self.db.execute_query(query)
        
        configs = {}
        if cursor:
            for row in cursor.fetchall():
                configs[row["config_key"]] = row["config_value"]
                
        return configs


class SystemLogger:
    """Clase para registrar eventos del sistema en la base de datos."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Inicializa el registrador del sistema.
        
        Args:
            db_manager: Instancia del gestor de base de datos
        """
        self.db = db_manager
        self.logger = logging.getLogger('system_logger')
    
    def log_event(self, level: str, message: str, module: str = None, 
                 user_id: int = None) -> bool:
        """Registra un evento del sistema en la base de datos.
        
        Args:
            level: Nivel del evento (info, warning, error, critical)
            message: Mensaje del evento
            module: Módulo que genera el evento
            user_id: ID del usuario relacionado
            
        Returns:
            bool: True si se registró correctamente
        """
        query = '''
        INSERT INTO system_logs (timestamp, level, module, message, user_id)
        VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?)
        '''
        
        cursor = self.db.execute_query(query, (level, module, message, user_id))
        if cursor:
            self.db.commit()
            return True
            
        return False
    
    def get_recent_logs(self, limit: int = 100, 
                       level: str = None, 
                       module: str = None) -> List[Dict[str, Any]]:
        """Obtiene los eventos recientes del sistema.
        
        Args:
            limit: Número máximo de eventos a obtener
            level: Filtrar por nivel (opcional)
            module: Filtrar por módulo (opcional)
            
        Returns:
            List[Dict[str, Any]]: Lista de eventos
        """
        query = '''
        SELECT id, timestamp, level, module, message, user_id
        FROM system_logs
        '''
        
        conditions = []
        params = []
        
        if level:
            conditions.append("level = ?")
            params.append(level)
            
        if module:
            conditions.append("module = ?")
            params.append(module)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.db.execute_query(query, tuple(params))
        if cursor:
            return [dict(row) for row in cursor.fetchall()]
            
        return []


# Función para crear una instancia del gestor de base de datos
def get_database_manager(db_path: str = "data/inspection_system.db") -> DatabaseManager:
    """Crea y devuelve una instancia del gestor de base de datos.
    
    Args:
        db_path: Ruta al archivo de base de datos
        
    Returns:
        DatabaseManager: Instancia del gestor de base de datos
    """
    return DatabaseManager(db_path)
