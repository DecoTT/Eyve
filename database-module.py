#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Base de Datos
---------------------
Proporciona funcionalidades para la conexión y gestión de la base de datos
del sistema de inspección visual.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union


class DatabaseManager:
    """Clase para gestionar la conexión y operaciones con la base de datos."""
    
    def __init__(self, db_path: str = "database/inspection_system.db"):
        """Inicializa el gestor de base de datos.
        
        Args:
            db_path: Ruta al archivo de base de datos SQLite
        """
        self.logger = logging.getLogger('system_logger')
        self.db_path = db_path
        
        # Asegurar que el directorio exista
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Inicializar la base de datos
        self._init_db()
        
    def _init_db(self) -> None:
        """Inicializa la estructura de la base de datos si no existe."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabla de usuarios
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                full_name TEXT,
                role TEXT NOT NULL,
                active INTEGER DEFAULT 1,
                last_login TEXT,
                created_date TEXT NOT NULL
            )
            ''')
            
            # Tabla de SKUs/trabajos
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS skus (
                sku_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                config_path TEXT,
                active INTEGER DEFAULT 1,
                created_date TEXT NOT NULL,
                modified_date TEXT NOT NULL
            )
            ''')
            
            # Tabla de sesiones de inspección
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS inspection_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT,
                total_inspections INTEGER DEFAULT 0,
                pass_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                FOREIGN KEY (sku_id) REFERENCES skus (sku_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            # Tabla de resultados de inspección
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS inspection_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                overall_status TEXT NOT NULL,
                details TEXT,
                FOREIGN KEY (session_id) REFERENCES inspection_sessions (session_id)
            )
            ''')
            
            # Tabla de log de actividad
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                details TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            # Insertar usuario administrador por defecto si no existe
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
            if cursor.fetchone()[0] == 0:
                # Contraseña por defecto: "admin123"
                # En producción, debería usar una contraseña fuerte y un método de hash seguro
                cursor.execute('''
                INSERT INTO users (username, password, full_name, role, created_date)
                VALUES (?, ?, ?, ?, ?)
                ''', ('admin', 'admin123', 'Administrador', 'admin', datetime.now().isoformat()))
            
            conn.commit()
            self.logger.info("Base de datos inicializada correctamente")
            
        except Exception as e:
            self.logger.error(f"Error al inicializar la base de datos: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    def connect(self) -> sqlite3.Connection:
        """Establece una conexión con la base de datos.
        
        Returns:
            sqlite3.Connection: Objeto de conexión a la base de datos
        """
        return sqlite3.connect(self.db_path)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Autentica un usuario en el sistema.
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            
        Returns:
            Optional[Dict[str, Any]]: Información del usuario si la autenticación es exitosa, None en caso contrario
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Buscar usuario
            cursor.execute('''
            SELECT user_id, username, full_name, role
            FROM users
            WHERE username = ? AND password = ? AND active = 1
            ''', (username, password))
            
            user_data = cursor.fetchone()
            
            if user_data:
                # Actualizar último login
                cursor.execute('''
                UPDATE users SET last_login = ? WHERE user_id = ?
                ''', (datetime.now().isoformat(), user_data[0]))
                
                # Registrar actividad
                self.log_activity(user_data[0], "login", "Inicio de sesión exitoso")
                
                conn.commit()
                
                return {
                    "user_id": user_data[0],
                    "username": user_data[1],
                    "full_name": user_data[2],
                    "role": user_data[3]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error de autenticación: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_skus(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Obtiene la lista de SKUs/trabajos disponibles.
        
        Args:
            active_only: Si es True, solo devuelve SKUs activos
            
        Returns:
            List[Dict[str, Any]]: Lista de SKUs
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            query = '''
            SELECT sku_id, name, description, config_path, active, created_date, modified_date
            FROM skus
            '''
            
            if active_only:
                query += " WHERE active = 1"
            
            cursor.execute(query)
            skus = []
            
            for row in cursor.fetchall():
                skus.append({
                    "sku_id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "config_path": row[3],
                    "active": bool(row[4]),
                    "created_date": row[5],
                    "modified_date": row[6]
                })
                
            return skus
            
        except Exception as e:
            self.logger.error(f"Error al obtener SKUs: {str(e)}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_sku_by_id(self, sku_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un SKU por su ID.
        
        Args:
            sku_id: ID del SKU
            
        Returns:
            Optional[Dict[str, Any]]: Información del SKU o None si no existe
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT sku_id, name, description, config_path, active, created_date, modified_date
            FROM skus
            WHERE sku_id = ?
            ''', (sku_id,))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    "sku_id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "config_path": row[3],
                    "active": bool(row[4]),
                    "created_date": row[5],
                    "modified_date": row[6]
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error al obtener SKU {sku_id}: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
    
    def start_inspection_session(self, sku_id: str, user_id: int) -> Optional[int]:
        """Inicia una nueva sesión de inspección.
        
        Args:
            sku_id: ID del SKU a inspeccionar
            user_id: ID del usuario que inicia la sesión
            
        Returns:
            Optional[int]: ID de la sesión creada o None en caso de error
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            start_time = datetime.now().isoformat()
            
            cursor.execute('''
            INSERT INTO inspection_sessions (sku_id, user_id, start_time, status)
            VALUES (?, ?, ?, ?)
            ''', (sku_id, user_id, start_time, "active"))
            
            session_id = cursor.lastrowid
            
            # Registrar actividad
            self.log_activity(
                user_id, 
                "start_session", 
                f"Sesión de inspección iniciada para SKU {sku_id}"
            )
            
            conn.commit()
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error al iniciar sesión de inspección: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
    
    def end_inspection_session(self, session_id: int, status: str = "completed") -> bool:
        """Finaliza una sesión de inspección.
        
        Args:
            session_id: ID de la sesión
            status: Estado final de la sesión (completed, aborted, error)
            
        Returns:
            bool: True si se finalizó correctamente
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            end_time = datetime.now().isoformat()
            
            cursor.execute('''
            UPDATE inspection_sessions
            SET end_time = ?, status = ?
            WHERE session_id = ?
            ''', (end_time, status, session_id))
            
            # Obtener información de la sesión para el log
            cursor.execute('''
            SELECT user_id, sku_id FROM inspection_sessions WHERE session_id = ?
            ''', (session_id,))
            
            session_info = cursor.fetchone()
            
            if session_info:
                self.log_activity(
                    session_info[0],
                    "end_session",
                    f"Sesión {session_id} finalizada con estado '{status}' para SKU {session_info[1]}"
                )
            
            conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error al finalizar sesión {session_id}: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
    
    def save_inspection_result(self, session_id: int, result: Dict[str, Any], image_path: str = None) -> Optional[int]:
        """Guarda el resultado de una inspección en la base de datos.
        
        Args:
            session_id: ID de la sesión de inspección
            result: Diccionario con los resultados
            image_path: Ruta a la imagen capturada (opcional)
            
        Returns:
            Optional[int]: ID del resultado guardado o None en caso de error
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Convertir detalles a JSON
            details_json = json.dumps(result)
            
            cursor.execute('''
            INSERT INTO inspection_results (session_id, timestamp, image_path, overall_status, details)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                session_id, 
                result.get("timestamp", datetime.now().isoformat()),
                image_path,
                result.get("overall_status", "unknown"),
                details_json
            ))
            
            result_id = cursor.lastrowid
            
            # Actualizar contadores en la sesión
            status = result.get("overall_status", "")
            if status == "pass":
                cursor.execute('''
                UPDATE inspection_sessions 
                SET total_inspections = total_inspections + 1,
                    pass_count = pass_count + 1
                WHERE session_id = ?
                ''', (session_id,))
            elif status == "fail":
                cursor.execute('''
                UPDATE inspection_sessions 
                SET total_inspections = total_inspections + 1,
                    fail_count = fail_count + 1
                WHERE session_id = ?
                ''', (session_id,))
            else:
                cursor.execute('''
                UPDATE inspection_sessions 
                SET total_inspections = total_inspections + 1
                WHERE session_id = ?
                ''', (session_id,))
            
            conn.commit()
            
            return result_id
            
        except Exception as e:
            self.logger.error(f"Error al guardar resultado de inspección: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_session_results(self, session_id: int) -> List[Dict[str, Any]]:
        """Obtiene todos los resultados de una sesión de inspección.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            List[Dict[str, Any]]: Lista de resultados
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT result_id, timestamp, image_path, overall_status, details
            FROM inspection_results
            WHERE session_id = ?
            ORDER BY timestamp
            ''', (session_id,))
            
            results = []
            
            for row in cursor.fetchall():
                # Parsear detalles JSON
                details = json.loads(row[4]) if row[4] else {}
                
                results.append({
                    "result_id": row[0],
                    "timestamp": row[1],
                    "image_path": row[2],
                    "overall_status": row[3],
                    "details": details
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error al obtener resultados de sesión {session_id}: {str(e)}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_session_summary(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene un resumen de una sesión de inspección.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Optional[Dict[str, Any]]: Resumen de la sesión o None en caso de error
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT s.session_id, s.sku_id, s.user_id, s.start_time, s.end_time, 
                   s.status, s.total_inspections, s.pass_count, s.fail_count,
                   u.username, k.name as sku_name
            FROM inspection_sessions s
            JOIN users u ON s.user_id = u.user_id
            JOIN skus k ON s.sku_id = k.sku_id
            WHERE s.session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    "session_id": row[0],
                    "sku_id": row[1],
                    "sku_name": row[10],
                    "user_id": row[2],
                    "username": row[9],
                    "start_time": row[3],
                    "end_time": row[4],
                    "status": row[5],
                    "total_inspections": row[6],
                    "pass_count": row[7],
                    "fail_count": row[8],
                    "pass_rate": row[7] / row[6] if row[6] > 0 else 0,
                    "duration": self._calculate_duration(row[3], row[4]) if row[4] else "En progreso"
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error al obtener resumen de sesión {session_id}: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
    
    def log_activity(self, user_id: Union[int, None], action: str, details: str = None) -> bool:
        """Registra una actividad en el log.
        
        Args:
            user_id: ID del usuario (puede ser None para actividades del sistema)
            action: Tipo de acción realizada
            details: Detalles adicionales (opcional)
            
        Returns:
            bool: True si se registró correctamente
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
            INSERT INTO activity_log (user_id, action, timestamp, details)
            VALUES (?, ?, ?, ?)
            ''', (user_id, action, timestamp, details))
            
            conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error al registrar actividad en log: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_user_activity(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene el historial de actividad de un usuario.
        
        Args:
            user_id: ID del usuario
            limit: Número máximo de registros a devolver
            
        Returns:
            List[Dict[str, Any]]: Lista de actividades
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT log_id, action, timestamp, details
            FROM activity_log
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (user_id, limit))
            
            activities = []
            
            for row in cursor.fetchall():
                activities.append({
                    "log_id": row[0],
                    "action": row[1],
                    "timestamp": row[2],
                    "details": row[3]
                })
                
            return activities
            
        except Exception as e:
            self.logger.error(f"Error al obtener actividad del usuario {user_id}: {str(e)}")
            return []
        finally:
            if conn:
                conn.close()
    
    def add_user(self, username: str, password: str, full_name: str, role: str) -> Optional[int]:
        """Agrega un nuevo usuario al sistema.
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            full_name: Nombre completo
            role: Rol (admin, operator, viewer)
            
        Returns:
            Optional[int]: ID del usuario creado o None en caso de error
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Verificar si el usuario ya existe
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
            if cursor.fetchone()[0] > 0:
                self.logger.warning(f"El usuario {username} ya existe")
                return None
            
            created_date = datetime.now().isoformat()
            
            cursor.execute('''
            INSERT INTO users (username, password, full_name, role, created_date)
            VALUES (?, ?, ?, ?, ?)
            ''', (username, password, full_name, role, created_date))
            
            user_id = cursor.lastrowid
            
            conn.commit()
            
            self.logger.info(f"Usuario {username} creado con ID {user_id}")
            return user_id
            
        except Exception as e:
            self.logger.error(f"Error al crear usuario: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
    
    def add_sku(self, sku_id: str, name: str, description: str = None, config_path: str = None) -> bool:
        """Agrega un nuevo SKU al sistema.
        
        Args:
            sku_id: ID único del SKU
            name: Nombre del SKU
            description: Descripción (opcional)
            config_path: Ruta al archivo de configuración (opcional)
            
        Returns:
            bool: True si se agregó correctamente
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Verificar si el SKU ya existe
            cursor.execute("SELECT COUNT(*) FROM skus WHERE sku_id = ?", (sku_id,))
            if cursor.fetchone()[0] > 0:
                self.logger.warning(f"El SKU {sku_id} ya existe")
                return False
            
            current_time = datetime.now().isoformat()
            
            cursor.execute('''
            INSERT INTO skus (sku_id, name, description, config_path, created_date, modified_date)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (sku_id, name, description, config_path, current_time, current_time))
            
            conn.commit()
            
            self.logger.info(f"SKU {sku_id} creado correctamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al crear SKU: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
    
    def _calculate_duration(self, start_time: str, end_time: str) -> str:
        """Calcula la duración entre dos timestamps ISO.
        
        Args:
            start_time: Timestamp de inicio en formato ISO
            end_time: Timestamp de fin en formato ISO
            
        Returns:
            str: Duración formateada (HH:MM:SS)
        """
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            
            duration = end - start
            seconds = duration.total_seconds()
            
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        except Exception:
            return "Desconocido"


# Para pruebas directas del módulo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('system_logger')
    
    # Crear instancia de prueba
    db_manager = DatabaseManager("test_db.sqlite")
    
    # Probar funcionalidades
    logger.info("Agregando usuario de prueba...")
    user_id = db_manager.add_user("test_user", "password123", "Usuario de Prueba", "operator")
    
    logger.info("Agregando SKU de prueba...")
    db_manager.add_sku("TEST001", "Producto de Prueba", "Descripción del producto", "config/TEST001.json")
    
    logger.info("Iniciando sesión de inspección...")
    session_id = db_manager.start_inspection_session("TEST001", user_id)
    
    logger.info("Guardando resultados de inspección...")
    result1 = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "pass",
        "module_results": {
            "color": {"status": "pass"},
            "dimensions": {"status": "pass"}
        }
    }
    
    result2 = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "fail",
        "module_results": {
            "color": {"status": "pass"},
            "dimensions": {"status": "fail", "measurements": [{"width_mm": 95, "height_mm": 40}]}
        }
    }
    
    db_manager.save_inspection_result(session_id, result1)
    db_manager.save_inspection_result(session_id, result2, "images/test_image.jpg")
    
    logger.info("Finalizando sesión...")
    db_manager.end_inspection_session(session_id)
    
    logger.info("Obteniendo resumen de sesión...")
    summary = db_manager.get_session_summary(session_id)
    logger.info(f"Resumen: {summary}")
    
    logger.info("Obteniendo resultados de la sesión...")
    results = db_manager.get_session_results(session_id)
    logger.info(f"Número de resultados: {len(results)}")
    
    logger.info("Pruebas completadas")
