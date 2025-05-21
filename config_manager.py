#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de Gestión de Configuración
----------------------------------
Proporciona clases y funciones para gestionar la configuración del sistema
de inspección visual, incluyendo carga/guardado de configuraciones por producto
y gestión de perfiles.
"""

import os
import json
import logging
import shutil
import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class ConfigManager:
    """Clase principal para gestionar configuraciones del sistema."""
    
    def __init__(self, config_dir: str = "config"):
        """Inicializa el gestor de configuración.
        
        Args:
            config_dir: Directorio base para almacenar archivos de configuración
        """
        self.logger = logging.getLogger('system_logger')
        self.config_dir = config_dir
        self.system_config_file = os.path.join(config_dir, "system_config.json")
        self.products_dir = os.path.join(config_dir, "products")
        self.profiles_dir = os.path.join(config_dir, "profiles")
        
        # Crear directorios si no existen
        for directory in [config_dir, self.products_dir, self.profiles_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.info(f"Directorio creado: {directory}")
        
        # Cargar configuración del sistema
        self.system_config = self._load_system_config()
        
        self.logger.info("Gestor de configuración inicializado")
    
    def _load_system_config(self) -> Dict[str, Any]:
        """Carga la configuración del sistema.
        
        Returns:
            Dict[str, Any]: Configuración del sistema
        """
        # Configuración predeterminada
        default_config = {
            "version": "1.0.0",
            "inspection": {
                "save_all_images": True,
                "image_storage_path": "data/images",
                "timeout": 30  # segundos
            },
            "camera": {
                "default_camera_id": 0,
                "resolution": {
                    "width": 1280,
                    "height": 720
                },
                "fps": 30
            },
            "gui": {
                "theme": "light",
                "language": "es",
                "fullscreen": False,
                "display_results_timeout": 5  # segundos
            },
            "logging": {
                "level": "info",
                "log_file": "logs/inspection_system.log",
                "max_size_mb": 10,
                "backup_count": 5
            },
            "database": {
                "path": "data/inspection_system.db",
                "backup_enabled": True,
                "backup_interval_hours": 24
            }
        }
        
        # Intentar cargar el archivo de configuración existente
        if os.path.exists(self.system_config_file):
            try:
                with open(self.system_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"Configuración del sistema cargada desde {self.system_config_file}")
                
                # Actualizar con valores predeterminados faltantes
                self._update_nested_dict(default_config, config)
                return config
            except Exception as e:
                self.logger.error(f"Error al cargar la configuración del sistema: {str(e)}")
                self.logger.info("Usando configuración predeterminada")
                
                # Hacer una copia de seguridad del archivo corrupto
                if os.path.getsize(self.system_config_file) > 0:
                    backup_file = f"{self.system_config_file}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                    shutil.copy2(self.system_config_file, backup_file)
                    self.logger.info(f"Copia de seguridad creada: {backup_file}")
        else:
            self.logger.info(f"Archivo de configuración no encontrado, se creará uno nuevo")
            
        # Guardar la configuración predeterminada
        self._save_system_config(default_config)
        return default_config
    
    def _save_system_config(self, config: Dict[str, Any]) -> bool:
        """Guarda la configuración del sistema en archivo.
        
        Args:
            config: Configuración a guardar
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            with open(self.system_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Configuración del sistema guardada en {self.system_config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar la configuración del sistema: {str(e)}")
            return False
    
    def _update_nested_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Actualiza recursivamente un diccionario anidado con valores de otro diccionario.
        
        Args:
            target: Diccionario destino (se modificará)
            source: Diccionario fuente (sus valores se copiarán a target)
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_nested_dict(target[key], value)
            else:
                target[key] = value
    
    def get_system_config(self) -> Dict[str, Any]:
        """Obtiene la configuración completa del sistema.
        
        Returns:
            Dict[str, Any]: Configuración del sistema
        """
        return self.system_config
    
    def update_system_config(self, config_updates: Dict[str, Any]) -> bool:
        """Actualiza la configuración del sistema.
        
        Args:
            config_updates: Diccionario con actualizaciones
            
        Returns:
            bool: True si se actualizó correctamente
        """
        # Actualizar la configuración actual con los nuevos valores
        self._update_nested_dict(self.system_config, config_updates)
        
        # Guardar la configuración actualizada
        return self._save_system_config(self.system_config)
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """Obtiene un valor específico de la configuración usando una ruta de acceso.
        
        Args:
            path: Ruta de acceso separada por puntos (ej: "gui.theme")
            default: Valor por defecto si no se encuentra
            
        Returns:
            Any: Valor de configuración o valor por defecto
        """
        config = self.system_config
        keys = path.split('.')
        
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
                
        return config
    
    def set_config_value(self, path: str, value: Any) -> bool:
        """Establece un valor específico en la configuración usando una ruta de acceso.
        
        Args:
            path: Ruta de acceso separada por puntos (ej: "gui.theme")
            value: Nuevo valor
            
        Returns:
            bool: True si se estableció correctamente
        """
        config = self.system_config
        keys = path.split('.')
        
        # Navegar a través de la estructura hasta el penúltimo nivel
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        # Establecer el valor en el último nivel
        config[keys[-1]] = value
        
        # Guardar la configuración actualizada
        return self._save_system_config(self.system_config)
    
    def get_product_config_path(self, sku: str) -> str:
        """Obtiene la ruta al archivo de configuración de un producto.
        
        Args:
            sku: Código SKU del producto
            
        Returns:
            str: Ruta al archivo de configuración
        """
        return os.path.join(self.products_dir, f"{sku}.json")
    
    def load_product_config(self, sku: str) -> Optional[Dict[str, Any]]:
        """Carga la configuración de un producto.
        
        Args:
            sku: Código SKU del producto
            
        Returns:
            Optional[Dict[str, Any]]: Configuración del producto o None si no existe
        """
        config_file = self.get_product_config_path(sku)
        
        if not os.path.exists(config_file):
            self.logger.warning(f"Configuración para producto '{sku}' no encontrada")
            return None
            
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Configuración cargada para producto '{sku}'")
            return config
        except Exception as e:
            self.logger.error(f"Error al cargar configuración para producto '{sku}': {str(e)}")
            return None
    
    def save_product_config(self, sku: str, config: Dict[str, Any]) -> bool:
        """Guarda la configuración de un producto.
        
        Args:
            sku: Código SKU del producto
            config: Configuración a guardar
            
        Returns:
            bool: True si se guardó correctamente
        """
        config_file = self.get_product_config_path(sku)
        
        try:
            # Añadir metadatos
            if "metadata" not in config:
                config["metadata"] = {}
                
            config["metadata"]["updated_at"] = datetime.datetime.now().isoformat()
            config["metadata"]["sku"] = sku
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Configuración guardada para producto '{sku}'")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar configuración para producto '{sku}': {str(e)}")
            return False
    
    def create_product_config(self, sku: str, name: str, 
                             inspection_modules: List[str] = None) -> Dict[str, Any]:
        """Crea una configuración básica para un nuevo producto.
        
        Args:
            sku: Código SKU del producto
            name: Nombre del producto
            inspection_modules: Lista de módulos de inspección a habilitar
            
        Returns:
            Dict[str, Any]: Configuración básica creada
        """
        # Valores predeterminados para módulos
        if inspection_modules is None:
            inspection_modules = ["ColorDetection", "DefectDetection", "DimensionMeasurement"]
            
        # Plantilla básica de configuración
        config = {
            "metadata": {
                "sku": sku,
                "name": name,
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "version": "1.0"
            },
            "inspection": {
                "enabled_modules": inspection_modules,
                "reference_image": "",
                "timeout": 30
            },
            "modules": {
                "ColorDetection": {
                    "enabled": True,
                    "color_ranges": {
                        "red": {"lower": [0, 100, 100], "upper": [10, 255, 255]},
                        "blue": {"lower": [100, 100, 100], "upper": [130, 255, 255]}
                    },
                    "min_area": 100
                },
                "DefectDetection": {
                    "enabled": True,
                    "threshold": 127,
                    "min_area": 100,
                    "reference_image": ""
                },
                "DimensionMeasurement": {
                    "enabled": True,
                    "pixels_per_mm": 1.0,
                    "target_dimensions": {
                        "width": {"min": 90, "max": 110},
                        "height": {"min": 90, "max": 110}
                    },
                    "threshold": 127
                },
                "TextureAnalysis": {
                    "enabled": False,
                    "glcm_distance": 5,
                    "threshold_contrast": 0.8,
                    "threshold_homogeneity": 0.7
                },
                "BarcodeQRDetection": {
                    "enabled": False,
                    "expected_formats": ["QR", "EAN-13", "CODE-128"],
                    "expected_values": []
                }
            }
        }
        
        # Guardar la configuración
        if self.save_product_config(sku, config):
            return config
        else:
            return {}
    
    def list_product_configs(self) -> List[Dict[str, Any]]:
        """Lista todas las configuraciones de productos disponibles.
        
        Returns:
            List[Dict[str, Any]]: Lista de metadatos de configuraciones
        """
        result = []
        
        try:
            for filename in os.listdir(self.products_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.products_dir, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            
                        # Extraer metadatos básicos
                        metadata = config.get("metadata", {})
                        sku = metadata.get("sku", filename.replace('.json', ''))
                        
                        result.append({
                            "sku": sku,
                            "name": metadata.get("name", sku),
                            "version": metadata.get("version", "1.0"),
                            "updated_at": metadata.get("updated_at", ""),
                            "modules": config.get("inspection", {}).get("enabled_modules", [])
                        })
                    except Exception as e:
                        self.logger.error(f"Error al leer configuración '{filename}': {str(e)}")
        except Exception as e:
            self.logger.error(f"Error al listar configuraciones de productos: {str(e)}")
            
        return result
    
    def delete_product_config(self, sku: str) -> bool:
        """Elimina la configuración de un producto.
        
        Args:
            sku: Código SKU del producto
            
        Returns:
            bool: True si se eliminó correctamente
        """
        config_file = self.get_product_config_path(sku)
        
        if not os.path.exists(config_file):
            self.logger.warning(f"Configuración para producto '{sku}' no encontrada")
            return False
            
        try:
            # Crear copia de seguridad
            backup_dir = os.path.join(self.config_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            backup_file = os.path.join(backup_dir, f"{sku}_{timestamp}.json.bak")
            
            shutil.copy2(config_file, backup_file)
            os.remove(config_file)
            
            self.logger.info(f"Configuración para producto '{sku}' eliminada (backup en {backup_file})")
            return True
        except Exception as e:
            self.logger.error(f"Error al eliminar configuración para producto '{sku}': {str(e)}")
            return False
    
    def save_inspection_profile(self, profile_name: str, 
                               module_configs: Dict[str, Dict[str, Any]]) -> bool:
        """Guarda un perfil de configuración de inspección reutilizable.
        
        Args:
            profile_name: Nombre del perfil
            module_configs: Configuraciones de módulos
            
        Returns:
            bool: True si se guardó correctamente
        """
        profile_file = os.path.join(self.profiles_dir, f"{profile_name}.json")
        
        profile_data = {
            "name": profile_name,
            "created_at": datetime.datetime.now().isoformat(),
            "modules": module_configs
        }
        
        try:
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Perfil de inspección '{profile_name}' guardado")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar perfil '{profile_name}': {str(e)}")
            return False
    
    def load_inspection_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Carga un perfil de configuración de inspección.
        
        Args:
            profile_name: Nombre del perfil
            
        Returns:
            Optional[Dict[str, Any]]: Configuración del perfil o None si no existe
        """
        profile_file = os.path.join(self.profiles_dir, f"{profile_name}.json")
        
        if not os.path.exists(profile_file):
            self.logger.warning(f"Perfil de inspección '{profile_name}' no encontrado")
            return None
            
        try:
            with open(profile_file, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            self.logger.info(f"Perfil de inspección '{profile_name}' cargado")
            return profile
        except Exception as e:
            self.logger.error(f"Error al cargar perfil '{profile_name}': {str(e)}")
            return None
    
    def list_inspection_profiles(self) -> List[Dict[str, Any]]:
        """Lista todos los perfiles de inspección disponibles.
        
        Returns:
            List[Dict[str, Any]]: Lista de metadatos de perfiles
        """
        result = []
        
        try:
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.profiles_dir, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            profile = json.load(f)
                            
                        name = profile.get("name", filename.replace('.json', ''))
                        modules = list(profile.get("modules", {}).keys())
                        
                        result.append({
                            "name": name,
                            "created_at": profile.get("created_at", ""),
                            "modules": modules
                        })
                    except Exception as e:
                        self.logger.error(f"Error al leer perfil '{filename}': {str(e)}")
        except Exception as e:
            self.logger.error(f"Error al listar perfiles de inspección: {str(e)}")
            
        return result
    
    def apply_profile_to_product(self, profile_name: str, sku: str) -> bool:
        """Aplica un perfil de inspección a un producto existente.
        
        Args:
            profile_name: Nombre del perfil a aplicar
            sku: Código SKU del producto
            
        Returns:
            bool: True si se aplicó correctamente
        """
        # Cargar perfil
        profile = self.load_inspection_profile(profile_name)
        if not profile:
            return False
            
        # Cargar configuración del producto
        product_config = self.load_product_config(sku)
        if not product_config:
            return False
            
        # Actualizar módulos con el perfil
        if "modules" in profile:
            for module_name, module_config in profile["modules"].items():
                if "modules" in product_config and module_name in product_config["modules"]:
                    product_config["modules"][module_name].update(module_config)
                    
                    # Asegurarse de que el módulo esté habilitado
                    if module_name not in product_config["inspection"]["enabled_modules"]:
                        product_config["inspection"]["enabled_modules"].append(module_name)
                        
        # Guardar la configuración actualizada
        return self.save_product_config(sku, product_config)
    
    def backup_all_configs(self) -> str:
        """Crea una copia de seguridad de todas las configuraciones.
        
        Returns:
            str: Ruta al archivo de copia de seguridad
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            backup_dir = os.path.join(self.config_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_filename = f"config_backup_{timestamp}.zip"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Crear archivo ZIP
            shutil.make_archive(
                backup_path.replace('.zip', ''),  # Quitar extensión
                'zip',
                self.config_dir,
                '.'  # Directorio raíz para comprimir
            )
            
            self.logger.info(f"Copia de seguridad creada en {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Error al crear copia de seguridad: {str(e)}")
            return ""
    
    def restore_config_backup(self, backup_path: str, 
                             restore_system: bool = True,
                             restore_products: bool = True,
                             restore_profiles: bool = True) -> bool:
        """Restaura una copia de seguridad de configuraciones.
        
        Args:
            backup_path: Ruta al archivo ZIP de copia de seguridad
            restore_system: Si se debe restaurar la configuración del sistema
            restore_products: Si se deben restaurar las configuraciones de productos
            restore_profiles: Si se deben restaurar los perfiles de inspección
            
        Returns:
            bool: True si se restauró correctamente
        """
        if not os.path.exists(backup_path) or not backup_path.endswith('.zip'):
            self.logger.error(f"Archivo de copia de seguridad no válido: {backup_path}")
            return False
            
        try:
            # Crear directorio temporal para extracción
            import tempfile
            temp_dir = tempfile.mkdtemp()
            
            # Extraer ZIP
            shutil.unpack_archive(backup_path, temp_dir)
            
            # Restaurar según opciones
            if restore_system and os.path.exists(os.path.join(temp_dir, "system_config.json")):
                shutil.copy2(
                    os.path.join(temp_dir, "system_config.json"),
                    self.system_config_file
                )
                # Recargar configuración del sistema
                self.system_config = self._load_system_config()
                
            if restore_products and os.path.exists(os.path.join(temp_dir, "products")):
                # Crear copia de seguridad de productos actuales
                if os.path.exists(self.products_dir):
                    products_backup = f"{self.products_dir}_bak_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                    shutil.copytree(self.products_dir, products_backup)
                    
                # Restaurar productos
                for file in os.listdir(os.path.join(temp_dir, "products")):
                    if file.endswith('.json'):
                        shutil.copy2(
                            os.path.join(temp_dir, "products", file),
                            os.path.join(self.products_dir, file)
                        )
                        
            if restore_profiles and os.path.exists(os.path.join(temp_dir, "profiles")):
                # Crear copia de seguridad de perfiles actuales
                if os.path.exists(self.profiles_dir):
                    profiles_backup = f"{self.profiles_dir}_bak_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                    shutil.copytree(self.profiles_dir, profiles_backup)
                    
                # Restaurar perfiles
                for file in os.listdir(os.path.join(temp_dir, "profiles")):
                    if file.endswith('.json'):
                        shutil.copy2(
                            os.path.join(temp_dir, "profiles", file),
                            os.path.join(self.profiles_dir, file)
                        )
            
            # Limpiar
            shutil.rmtree(temp_dir)
            
            self.logger.info(f"Configuración restaurada desde {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error al restaurar configuración: {str(e)}")
            return False


# Función para obtener una instancia del gestor de configuración
def get_config_manager(config_dir: str = "config") -> ConfigManager:
    """Crea y devuelve una instancia del gestor de configuración.
    
    Args:
        config_dir: Directorio base para archivos de configuración
        
    Returns:
        ConfigManager: Instancia del gestor de configuración
    """
    return ConfigManager(config_dir)
