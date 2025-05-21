#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Configuración
--------------------
Proporciona funcionalidades para cargar y gestionar configuraciones del sistema.
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime


class ConfigManager:
    """Clase para gestionar las configuraciones del sistema."""
    
    def __init__(self, config_dir: str = "config"):
        """Inicializa el gestor de configuración.
        
        Args:
            config_dir: Directorio base de las configuraciones
        """
        self.logger = logging.getLogger('system_logger')
        self.config_dir = config_dir
        self.system_config_path = os.path.join(config_dir, "system.json")
        self.sku_config_dir = os.path.join(config_dir, "skus")
        
        # Configuración del sistema cargada
        self.system_config = {}
        
        # Crear directorios si no existen
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.sku_config_dir, exist_ok=True)
        
        # Cargar configuración del sistema
        self._load_system_config()
    
    def _load_system_config(self) -> None:
        """Carga la configuración general del sistema."""
        if os.path.exists(self.system_config_path):
            try:
                with open(self.system_config_path, 'r') as f:
                    self.system_config = json.load(f)
                self.logger.info("Configuración del sistema cargada correctamente")
            except Exception as e:
                self.logger.error(f"Error al cargar configuración del sistema: {str(e)}")
                # Crear configuración por defecto
                self._create_default_system_config()
        else:
            self.logger.warning("No se encontró archivo de configuración del sistema, creando uno por defecto")
            self._create_default_system_config()
    
    def _create_default_system_config(self) -> None:
        """Crea una configuración del sistema por defecto."""
        self.system_config = {
            "app_name": "Sistema de Inspección Visual",
            "version": "1.0.0",
            "logging": {
                "level": "INFO",
                "file": "logs/system.log",
                "max_size_mb": 10,
                "backup_count": 5
            },
            "database": {
                "path": "database/inspection_system.db"
            },
            "storage": {
                "images_dir": "storage/images",
                "temp_dir": "storage/temp",
                "results_dir": "storage/results"
            },
            "gui": {
                "theme": "clam",
                "main_window_size": "1280x800",
                "login_window_size": "600x400",
                "font_family": "Arial",
                "default_font_size": 10
            },
            "inspection": {
                "default_fps": 30,
                "save_failed_inspections": True,
                "auto_capture_interval_ms": 2000
            },
            "created_date": datetime.now().isoformat(),
            "modified_date": datetime.now().isoformat()
        }
        
        try:
            with open(self.system_config_path, 'w') as f:
                json.dump(self.system_config, f, indent=2)
            self.logger.info("Configuración del sistema por defecto creada")
        except Exception as e:
            self.logger.error(f"Error al crear configuración por defecto: {str(e)}")
    
    def get_system_config(self) -> Dict[str, Any]:
        """Obtiene la configuración completa del sistema.
        
        Returns:
            Dict[str, Any]: Configuración del sistema
        """
        return self.system_config
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor específico de la configuración del sistema.
        
        Soporta notación de punto para acceder a valores anidados.
        Ejemplo: "gui.theme" para obtener el tema de la GUI.
        
        Args:
            key: Clave a obtener
            default: Valor por defecto si la clave no existe
            
        Returns:
            Any: Valor de la configuración o el valor por defecto
        """
        if "." in key:
            # Manejar claves anidadas
            parts = key.split(".")
            config = self.system_config
            
            for part in parts[:-1]:
                if part in config:
                    config = config[part]
                else:
                    return default
            
            return config.get(parts[-1], default)
        else:
            # Clave simple
            return self.system_config.get(key, default)
    
    def set_config_value(self, key: str, value: Any) -> bool:
        """Establece un valor en la configuración del sistema.
        
        Soporta notación de punto para acceder a valores anidados.
        
        Args:
            key: Clave a establecer
            value: Valor a establecer
            
        Returns:
            bool: True si se estableció correctamente
        """
        try:
            if "." in key:
                # Manejar claves anidadas
                parts = key.split(".")
                config = self.system_config
                
                # Navegar a la ubicación correcta
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]
                
                # Establecer el valor
                config[parts[-1]] = value
            else:
                # Clave simple
                self.system_config[key] = value
            
            # Actualizar fecha de modificación
            self.system_config["modified_date"] = datetime.now().isoformat()
            
            # Guardar la configuración actualizada
            self._save_system_config()
            return True
        except Exception as e:
            self.logger.error(f"Error al establecer valor de configuración '{key}': {str(e)}")
            return False
    
    def _save_system_config(self) -> bool:
        """Guarda la configuración actual del sistema en el archivo.
        
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            with open(self.system_config_path, 'w') as f:
                json.dump(self.system_config, f, indent=2)
            self.logger.info("Configuración del sistema guardada")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar configuración del sistema: {str(e)}")
            return False
    
    def load_sku_config(self, sku_id: str) -> Optional[Dict[str, Any]]:
        """Carga la configuración de un SKU específico.
        
        Args:
            sku_id: ID del SKU
            
        Returns:
            Optional[Dict[str, Any]]: Configuración del SKU o None si no existe
        """
        sku_config_path = os.path.join(self.sku_config_dir, f"{sku_id}.json")
        
        if not os.path.exists(sku_config_path):
            self.logger.warning(f"No se encontró configuración para SKU {sku_id}")
            return None
        
        try:
            with open(sku_config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Configuración de SKU {sku_id} cargada correctamente")
            return config
        except Exception as e:
            self.logger.error(f"Error al cargar configuración de SKU {sku_id}: {str(e)}")
            return None
    
    def save_sku_config(self, sku_id: str, config: Dict[str, Any]) -> bool:
        """Guarda la configuración de un SKU.
        
        Args:
            sku_id: ID del SKU
            config: Configuración a guardar
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            # Asegurar que el directorio existe
            os.makedirs(self.sku_config_dir, exist_ok=True)
            
            # Actualizar fechas
            if "modified_date" in config:
                config["modified_date"] = datetime.now().isoformat()
            
            sku_config_path = os.path.join(self.sku_config_dir, f"{sku_id}.json")
            
            with open(sku_config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            self.logger.info(f"Configuración de SKU {sku_id} guardada correctamente")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar configuración de SKU {sku_id}: {str(e)}")
            return False
    
    def get_available_skus(self) -> List[str]:
        """Obtiene la lista de SKUs disponibles en el directorio de configuración.
        
        Returns:
            List[str]: Lista de IDs de SKUs
        """
        try:
            skus = []
            
            # Listar archivos JSON en el directorio de SKUs
            for filename in os.listdir(self.sku_config_dir):
                if filename.endswith(".json"):
                    sku_id = os.path.splitext(filename)[0]
                    skus.append(sku_id)
                    
            return skus
        except Exception as e:
            self.logger.error(f"Error al obtener lista de SKUs: {str(e)}")
            return []
    
    def create_default_sku_config(self, sku_id: str, name: str, description: str = "") -> bool:
        """Crea una configuración por defecto para un nuevo SKU.
        
        Args:
            sku_id: ID del SKU
            name: Nombre del SKU
            description: Descripción del SKU
            
        Returns:
            bool: True si se creó correctamente
        """
        default_config = {
            "sku_id": sku_id,
            "name": name,
            "description": description,
            "created_date": datetime.now().isoformat(),
            "modified_date": datetime.now().isoformat(),
            "version": "1.0",
            
            "camera_config": {
                "main_camera": {
                    "camera_id": 0,
                    "width": 1280,
                    "height": 720,
                    "fps": 30
                }
            },
            
            "inspection_modules": {
                "color": {
                    "enabled": True,
                    "threshold": 127,
                    "min_area": 100,
                    "color_ranges": {
                        "red": {"lower": [0, 100, 100], "upper": [10, 255, 255]}
                    }
                },
                
                "defect": {
                    "enabled": True,
                    "threshold": 50,
                    "min_area": 100
                },
                
                "dimensions": {
                    "enabled": True,
                    "pixels_per_mm": 10.0,
                    "threshold": 127,
                    "target_dimensions": {
                        "width": {"min": 90, "max": 110},
                        "height": {"min": 90, "max": 110}
                    }
                }
            },
            
            "inspection_settings": {
                "auto_capture": True,
                "capture_interval_ms": 2000,
                "save_failed_inspections": True
            }
        }
        
        return self.save_sku_config(sku_id, default_config)
    
    def delete_sku_config(self, sku_id: str) -> bool:
        """Elimina la configuración de un SKU.
        
        Args:
            sku_id: ID del SKU
            
        Returns:
            bool: True si se eliminó correctamente
        """
        try:
            sku_config_path = os.path.join(self.sku_config_dir, f"{sku_id}.json")
            
            if os.path.exists(sku_config_path):
                os.remove(sku_config_path)
                self.logger.info(f"Configuración de SKU {sku_id} eliminada")
                return True
            else:
                self.logger.warning(f"No se encontró configuración para SKU {sku_id}")
                return False
        except Exception as e:
            self.logger.error(f"Error al eliminar configuración de SKU {sku_id}: {str(e)}")
            return False
    
    def export_sku_config(self, sku_id: str, export_path: str, format: str = "json") -> bool:
        """Exporta la configuración de un SKU a un archivo.
        
        Args:
            sku_id: ID del SKU
            export_path: Ruta donde guardar el archivo exportado
            format: Formato de exportación ('json' o 'yaml')
            
        Returns:
            bool: True si se exportó correctamente
        """
        config = self.load_sku_config(sku_id)
        
        if not config:
            return False
        
        try:
            if format.lower() == "json":
                with open(export_path, 'w') as f:
                    json.dump(config, f, indent=2)
            elif format.lower() == "yaml":
                with open(export_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            else:
                self.logger.error(f"Formato de exportación no soportado: {format}")
                return False
            
            self.logger.info(f"Configuración de SKU {sku_id} exportada a {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error al exportar configuración de SKU {sku_id}: {str(e)}")
            return False
    
    def import_sku_config(self, import_path: str, format: str = "json") -> Optional[str]:
        """Importa la configuración de un SKU desde un archivo.
        
        Args:
            import_path: Ruta del archivo a importar
            format: Formato del archivo ('json' o 'yaml')
            
        Returns:
            Optional[str]: ID del SKU importado o None en caso de error
        """
        try:
            if format.lower() == "json":
                with open(import_path, 'r') as f:
                    config = json.load(f)
            elif format.lower() == "yaml":
                with open(import_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                self.logger.error(f"Formato de importación no soportado: {format}")
                return None
            
            # Verificar que la configuración tenga un ID de SKU
            if "sku_id" not in config:
                self.logger.error("La configuración importada no tiene un ID de SKU")
                return None
            
            sku_id = config["sku_id"]
            
            # Guardar la configuración
            if self.save_sku_config(sku_id, config):
                self.logger.info(f"Configuración de SKU {sku_id} importada correctamente")
                return sku_id
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error al importar configuración de SKU: {str(e)}")
            return None


# Para pruebas directas del módulo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('system_logger')
    
    # Crear instancia de prueba
    config_manager = ConfigManager("test_config")
    
    # Probar funcionalidades básicas
    logger.info("Obteniendo valores de configuración...")
    app_name = config_manager.get_config_value("app_name")
    theme = config_manager.get_config_value("gui.theme")
    non_existent = config_manager.get_config_value("non_existent", "valor_default")
    
    logger.info(f"App name: {app_name}")
    logger.info(f"Theme: {theme}")
    logger.info(f"Non-existent (with default): {non_existent}")
    
    # Modificar un valor
    logger.info("Modificando valor de configuración...")
    config_manager.set_config_value("gui.theme", "default")
    new_theme = config_manager.get_config_value("gui.theme")
    logger.info(f"New theme: {new_theme}")
    
    # Crear un SKU de prueba
    logger.info("Creando configuración de SKU de prueba...")
    config_manager.create_default_sku_config("TEST001", "Producto de Prueba", "Descripción de prueba")
    
    # Cargar configuración de SKU
    logger.info("Cargando configuración de SKU...")
    sku_config = config_manager.load_sku_config("TEST001")
    if sku_config:
        logger.info(f"Configuración cargada para SKU: {sku_config['name']}")
    
    # Listar SKUs disponibles
    logger.info("Listando SKUs disponibles...")
    skus = config_manager.get_available_skus()
    logger.info(f"SKUs disponibles: {skus}")
    
    # Exportar configuración
    logger.info("Exportando configuración...")
    config_manager.export_sku_config("TEST001", "test_config/TEST001_export.json")
    
    logger.info("Pruebas completadas")
