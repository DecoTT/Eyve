#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestor de Resultados
-----------------
Gestiona el almacenamiento, exportación y análisis de resultados de inspección.
"""

import os
import json
import csv
import logging
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO


class ResultsManager:
    """Clase para gestionar resultados de inspección."""
    
    def __init__(self, results_dir: str = "storage/results"):
        """Inicializa el gestor de resultados.
        
        Args:
            results_dir: Directorio base para almacenamiento de resultados
        """
        self.logger = logging.getLogger('system_logger')
        self.results_dir = results_dir
        
        # Crear directorio si no existe
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Historial de resultados en memoria
        self.results_history = []
        self.max_history_size = 1000
        
    def save_inspection_result(self, result: Dict[str, Any], sku_id: str, image: Optional[np.ndarray] = None) -> str:
        """Guarda un resultado de inspección.
        
        Args:
            result: Diccionario con resultados de inspección
            sku_id: ID del SKU inspeccionado
            image: Imagen de la inspección (opcional)
            
        Returns:
            str: Ruta al directorio donde se guardaron los resultados
        """
        # Obtener timestamp del resultado o generar uno nuevo
        timestamp = result.get("timestamp", datetime.now().isoformat())
        timestamp_str = timestamp.replace(":", "-").replace(".", "-")
        
        # Crear estructura de directorios
        date_str = timestamp[:10]  # Formato YYYY-MM-DD
        result_dir = os.path.join(self.results_dir, date_str, sku_id, timestamp_str)
        os.makedirs(result_dir, exist_ok=True)
        
        # Guardar resultado en formato JSON
        result_path = os.path.join(result_dir, "result.json")
        try:
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error al guardar resultado JSON: {str(e)}")
        
        # Guardar imagen si se proporciona
        if image is not None:
            # Determinar formato de imagen según status
            format_suffix = "_fail.jpg" if result.get("overall_status") == "fail" else ".jpg"
            image_path = os.path.join(result_dir, f"inspection{format_suffix}")
            
            try:
                cv2.imwrite(image_path, image)
            except Exception as e:
                self.logger.error(f"Error al guardar imagen: {str(e)}")
        
        # Guardar en historial
        self.results_history.append(result)
        
        # Limitar tamaño del historial
        if len(self.results_history) > self.max_history_size:
            self.results_history.pop(0)
            
        self.logger.info(f"Resultado guardado en {result_dir}")
        return result_dir
    
    def create_inspection_report(self, session_results: List[Dict[str, Any]], output_path: str, 
                             report_format: str = "csv") -> bool:
        """Crea un informe con los resultados de una sesión de inspección.
        
        Args:
            session_results: Lista de resultados de inspección
            output_path: Ruta donde guardar el informe
            report_format: Formato del informe ("csv", "json", "excel")
            
        Returns:
            bool: True si se creó correctamente
        """
        if not session_results:
            self.logger.warning("No hay resultados para generar informe")
            return False
            
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convertir resultados a formato tabular
            tabular_data = []
            
            for result in session_results:
                # Extraer campos comunes
                row = {
                    "timestamp": result.get("timestamp", ""),
                    "overall_status": result.get("overall_status", "unknown"),
                }
                
                # Extraer resultados de módulos
                module_results = result.get("module_results", {})
                for module_name, module_data in module_results.items():
                    row[f"{module_name}_status"] = module_data.get("status", "unknown")
                    
                    # Extraer detalles específicos por tipo de módulo
                    if module_name == "color":
                        detections = module_data.get("detections", {})
                        row[f"{module_name}_detections"] = len(detections)
                        
                    elif module_name == "defect":
                        defects = module_data.get("defects", [])
                        row[f"{module_name}_count"] = len(defects)
                        if defects:
                            row[f"{module_name}_max_area"] = max(d.get("area", 0) for d in defects)
                            
                    elif module_name == "dimensions":
                        measurements = module_data.get("measurements", [])
                        if measurements:
                            # Tomar la primera medición (asumiendo un objeto principal)
                            measurement = measurements[0]
                            row[f"{module_name}_width_mm"] = measurement.get("width_mm", 0)
                            row[f"{module_name}_height_mm"] = measurement.get("height_mm", 0)
                            row[f"{module_name}_in_tolerance"] = measurement.get("in_tolerance", False)
                
                tabular_data.append(row)
                
            # Guardar en el formato solicitado
            if report_format.lower() == "csv":
                # Determinar todas las columnas posibles
                all_columns = set()
                for row in tabular_data:
                    all_columns.update(row.keys())
                    
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(all_columns))
                    writer.writeheader()
                    writer.writerows(tabular_data)
                    
            elif report_format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(tabular_data, f, indent=2)
                    
            elif report_format.lower() == "excel":
                df = pd.DataFrame(tabular_data)
                df.to_excel(output_path, index=False)
                
            else:
                self.logger.error(f"Formato de informe no soportado: {report_format}")
                return False
                
            self.logger.info(f"Informe creado en {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al crear informe: {str(e)}")
            return False
    
    def generate_statistics(self, results: List[Dict[str, Any]], include_charts: bool = False) -> Dict[str, Any]:
        """Genera estadísticas a partir de resultados de inspección.
        
        Args:
            results: Lista de resultados de inspección
            include_charts: Si es True, incluye gráficos en las estadísticas
            
        Returns:
            Dict[str, Any]: Estadísticas generadas
        """
        if not results:
            return {"error": "No hay resultados para analizar"}
            
        try:
            # Estadísticas generales
            total = len(results)
            passed = sum(1 for r in results if r.get("overall_status") == "pass")
            failed = sum(1 for r in results if r.get("overall_status") == "fail")
            errors = sum(1 for r in results if r.get("overall_status") == "error")
            
            # Tasas
            pass_rate = passed / total if total > 0 else 0
            fail_rate = failed / total if total > 0 else 0
            error_rate = errors / total if total > 0 else 0
            
            # Tendencia temporal
            timestamps = [r.get("timestamp", "") for r in results]
            sorted_results = sorted(zip(timestamps, results), key=lambda x: x[0])
            
            # Estadísticas por módulo
            module_stats = {}
            
            # Determinar qué módulos están presentes
            all_modules = set()
            for result in results:
                module_results = result.get("module_results", {})
                all_modules.update(module_results.keys())
                
            # Recopilar estadísticas por módulo
            for module_name in all_modules:
                module_data = []
                
                for result in results:
                    module_results = result.get("module_results", {})
                    if module_name in module_results:
                        module_data.append(module_results[module_name])
                        
                if module_data:
                    module_passed = sum(1 for d in module_data if d.get("status") == "pass")
                    module_failed = sum(1 for d in module_data if d.get("status") == "fail")
                    module_errors = sum(1 for d in module_data if d.get("status") == "error")
                    
                    module_stats[module_name] = {
                        "total": len(module_data),
                        "passed": module_passed,
                        "failed": module_failed,
                        "errors": module_errors,
                        "pass_rate": module_passed / len(module_data)
                    }
                    
                    # Estadísticas específicas por tipo de módulo
                    if module_name == "dimensions" and module_data:
                        # Recopilar mediciones
                        all_width_mm = []
                        all_height_mm = []
                        
                        for data in module_data:
                            measurements = data.get("measurements", [])
                            for m in measurements:
                                if "width_mm" in m:
                                    all_width_mm.append(m["width_mm"])
                                if "height_mm" in m:
                                    all_height_mm.append(m["height_mm"])
                                    
                        if all_width_mm:
                            module_stats[module_name]["width_stats"] = {
                                "min": min(all_width_mm),
                                "max": max(all_width_mm),
                                "mean": sum(all_width_mm) / len(all_width_mm),
                                "std": np.std(all_width_mm) if len(all_width_mm) > 1 else 0
                            }
                            
                        if all_height_mm:
                            module_stats[module_name]["height_stats"] = {
                                "min": min(all_height_mm),
                                "max": max(all_height_mm),
                                "mean": sum(all_height_mm) / len(all_height_mm),
                                "std": np.std(all_height_mm) if len(all_height_mm) > 1 else 0
                            }
            
            # Generar estadísticas completas
            statistics = {
                "total_inspections": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "pass_rate": pass_rate,
                "fail_rate": fail_rate,
                "error_rate": error_rate,
                "modules": module_stats,
                "generated_at": datetime.now().isoformat()
            }
            
            # Generar gráficos si se solicita
            if include_charts:
                charts = {}
                
                # Gráfico de pastel para resultados generales
                pie_fig = plt.figure(figsize=(8, 6))
                ax = pie_fig.add_subplot(111)
                ax.pie([passed, failed, errors], 
                     labels=["Pasó", "Falló", "Error"],
                     autopct='%1.1f%%', 
                     colors=['green', 'red', 'orange'])
                ax.set_title("Distribución de Resultados de Inspección")
                
                # Guardar gráfico en buffer
                buf = BytesIO()
                pie_fig.savefig(buf, format='png')
                buf.seek(0)
                
                # Convertir a base64 para inclusión en informes web
                import base64
                pie_chart_data = base64.b64encode(buf.read()).decode('utf-8')
                charts["results_pie"] = pie_chart_data
                
                # Gráfico temporal de tasa de éxito
                if len(sorted_results) > 1:
                    # Agrupar resultados por fecha
                    date_results = {}
                    for timestamp, result in sorted_results:
                        date = timestamp[:10]  # YYYY-MM-DD
                        if date not in date_results:
                            date_results[date] = {"total": 0, "passed": 0}
                            
                        date_results[date]["total"] += 1
                        if result.get("overall_status") == "pass":
                            date_results[date]["passed"] += 1
                    
                    # Calcular tasas de éxito por fecha
                    dates = []
                    rates = []
                    for date, counts in sorted(date_results.items()):
                        dates.append(date)
                        rates.append(counts["passed"] / counts["total"] if counts["total"] > 0 else 0)
                    
                    # Crear gráfico
                    if dates and rates:
                        trend_fig = plt.figure(figsize=(10, 6))
                        ax = trend_fig.add_subplot(111)
                        ax.plot(dates, rates, marker='o', linestyle='-', color='blue')
                        ax.set_title("Tendencia de Tasa de Éxito por Día")
                        ax.set_xlabel("Fecha")
                        ax.set_ylabel("Tasa de Éxito")
                        ax.set_ylim(0, 1)
                        ax.grid(True)
                        
                        # Rotar etiquetas de fecha si hay muchas
                        if len(dates) > 5:
                            plt.xticks(rotation=45)
                            
                        plt.tight_layout()
                        
                        # Guardar gráfico
                        buf = BytesIO()
                        trend_fig.savefig(buf, format='png')
                        buf.seek(0)
                        
                        trend_chart_data = base64.b64encode(buf.read()).decode('utf-8')
                        charts["trend_chart"] = trend_chart_data
                
                statistics["charts"] = charts
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error al generar estadísticas: {str(e)}")
            return {"error": str(e)}
    
    def export_results_to_external_system(self, results: List[Dict[str, Any]], system_config: Dict[str, Any]) -> bool:
        """Exporta resultados a un sistema externo (ERP, MES, etc.).
        
        Args:
            results: Lista de resultados a exportar
            system_config: Configuración del sistema externo
            
        Returns:
            bool: True si se exportó correctamente
        """
        if not results:
            self.logger.warning("No hay resultados para exportar")
            return False
            
        system_type = system_config.get("type", "").lower()
        
        try:
            if system_type == "file":
                # Exportar a archivo
                file_path = system_config.get("path", "export/results.json")
                format = system_config.get("format", "json").lower()
                
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                if format == "json":
                    with open(file_path, 'w') as f:
                        json.dump(results, f, indent=2)
                        
                elif format == "csv":
                    # Aplanar resultados para formato CSV
                    rows = self._flatten_results_for_csv(results)
                    
                    with open(file_path, 'w', newline='') as f:
                        if rows:
                            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                            writer.writeheader()
                            writer.writerows(rows)
                            
                elif format == "excel":
                    # Aplanar resultados para formato tabular
                    rows = self._flatten_results_for_csv(results)
                    df = pd.DataFrame(rows)
                    df.to_excel(file_path, index=False)
                    
                else:
                    self.logger.error(f"Formato de exportación no soportado: {format}")
                    return False
                    
                self.logger.info(f"Resultados exportados a {file_path}")
                return True
                
            elif system_type == "database":
                # La implementación dependerá de la base de datos específica
                self.logger.warning("Exportación a base de datos no implementada")
                return False
                
            elif system_type == "api":
                # Implementación básica para exportar a API REST
                import requests
                
                url = system_config.get("url", "")
                headers = system_config.get("headers", {})
                auth = system_config.get("auth", None)
                
                if not url:
                    self.logger.error("URL no especificada para exportación API")
                    return False
                    
                # Convertir resultados al formato requerido
                payload = results
                if system_config.get("flatten", False):
                    payload = self._flatten_results_for_csv(results)
                    
                # Enviar datos
                response = requests.post(url, json=payload, headers=headers, auth=auth)
                
                if response.status_code in [200, 201, 202]:
                    self.logger.info(f"Resultados exportados a API: {url}")
                    return True
                else:
                    self.logger.error(f"Error al exportar a API: {response.status_code} - {response.text}")
                    return False
                    
            else:
                self.logger.error(f"Tipo de sistema externo no soportado: {system_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error al exportar resultados: {str(e)}")
            return False
    
    def _flatten_results_for_csv(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convierte resultados anidados a formato plano para CSV/Excel.
        
        Args:
            results: Lista de resultados
            
        Returns:
            List[Dict[str, Any]]: Lista de resultados aplanados
        """
        flattened_rows = []
        
        for result in results:
            row = {
                "timestamp": result.get("timestamp", ""),
                "overall_status": result.get("overall_status", "")
            }
            
            # Procesar módulos
            module_results = result.get("module_results", {})
            for module_name, data in module_results.items():
                # Agregar estado del módulo
                row[f"{module_name}_status"] = data.get("status", "")
                
                # Agregar datos específicos según el tipo de módulo
                if module_name == "dimensions" and "measurements" in data:
                    # Para mediciones, tomar el primer objeto
                    if data["measurements"]:
                        measurement = data["measurements"][0]
                        for key, value in measurement.items():
                            row[f"{module_name}_{key}"] = value
                            
                elif module_name == "color" and "detections" in data:
                    # Para detecciones de color
                    row[f"{module_name}_count"] = len(data["detections"])
                    for color, detection in data["detections"].items():
                        row[f"{module_name}_{color}_count"] = detection.get("count", 0)
                        
                elif module_name == "defect" and "defects" in data:
                    # Para defectos
                    defects = data["defects"]
                    row[f"{module_name}_count"] = len(defects)
                    if defects:
                        # Estadísticas de defectos
                        areas = [d.get("area", 0) for d in defects]
                        row[f"{module_name}_min_area"] = min(areas) if areas else 0
                        row[f"{module_name}_max_area"] = max(areas) if areas else 0
                        row[f"{module_name}_avg_area"] = sum(areas) / len(areas) if areas else 0
            
            flattened_rows.append(row)
            
        return flattened_rows
    
    def create_report_pdf(self, results: List[Dict[str, Any]], output_path: str, 
                      sku_info: Dict[str, Any] = None, include_images: bool = True) -> bool:
        """Crea un informe PDF con los resultados de inspección.
        
        Args:
            results: Lista de resultados de inspección
            output_path: Ruta de salida para el PDF
            sku_info: Información del SKU (opcional)
            include_images: Si es True, incluye imágenes en el informe
            
        Returns:
            bool: True si se creó correctamente
        """
        try:
            # Verificar si reportlab está disponible
            import reportlab
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
        except ImportError:
            self.logger.error("reportlab no está instalado. Instale con: pip install reportlab")
            return False
            
        if not results:
            self.logger.warning("No hay resultados para generar informe PDF")
            return False
            
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Crear documento
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Elementos del documento
            elements = []
            
            # Título
            title_style = styles["Heading1"]
            elements.append(Paragraph("Informe de Inspección Visual", title_style))
            elements.append(Spacer(1, 0.25 * inch))
            
            # Información del SKU
            if sku_info:
                elements.append(Paragraph("Información del Producto", styles["Heading2"]))
                sku_data = [
                    ["SKU ID:", sku_info.get("sku_id", "")],
                    ["Nombre:", sku_info.get("name", "")],
                    ["Descripción:", sku_info.get("description", "")]
                ]
                sku_table = Table(sku_data, colWidths=[1.5 * inch, 4 * inch])
                sku_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(sku_table)
                elements.append(Spacer(1, 0.25 * inch))
            
            # Estadísticas generales
            elements.append(Paragraph("Resumen de Inspección", styles["Heading2"]))
            
            stats = self.generate_statistics(results)
            stats_data = [
                ["Total de inspecciones:", str(stats.get("total_inspections", 0))],
                ["Aprobadas:", f"{stats.get('passed', 0)} ({stats.get('pass_rate', 0) * 100:.1f}%)"],
                ["Fallidas:", f"{stats.get('failed', 0)} ({stats.get('fail_rate', 0) * 100:.1f}%)"],
                ["Con errores:", f"{stats.get('errors', 0)} ({stats.get('error_rate', 0) * 100:.1f}%)"]
            ]
            
            stats_table = Table(stats_data, colWidths=[2 * inch, 3.5 * inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(stats_table)
            elements.append(Spacer(1, 0.25 * inch))
            
            # Detalles de resultados
            elements.append(Paragraph("Detalle de Inspecciones", styles["Heading2"]))
            
            # Cabecera de tabla de resultados
            results_data = [["#", "Fecha/Hora", "Resultado", "Detalles"]]
            
            # Añadir filas para cada resultado
            for i, result in enumerate(results[:50]):  # Limitar a 50 resultados para evitar PDFs enormes
                timestamp = result.get("timestamp", "")
                status = result.get("overall_status", "unknown")
                
                # Formatear detalles
                details = []
                for module_name, module_data in result.get("module_results", {}).items():
                    module_status = module_data.get("status", "unknown")
                    details.append(f"{module_name}: {module_status}")
                
                detail_text = ", ".join(details)
                
                # Añadir fila
                results_data.append([str(i+1), timestamp, status, detail_text])
            
            # Crear tabla
            results_table = Table(results_data, colWidths=[0.5 * inch, 2 * inch, 1 * inch, 3 * inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            # Colorear filas según resultado
            for i, result in enumerate(results[:50]):
                status = result.get("overall_status", "unknown")
                if status == "pass":
                    results_table.setStyle(TableStyle([
                        ('BACKGROUND', (2, i+1), (2, i+1), colors.lightgreen)
                    ]))
                elif status == "fail":
                    results_table.setStyle(TableStyle([
                        ('BACKGROUND', (2, i+1), (2, i+1), colors.lightcoral)
                    ]))
                elif status == "error":
                    results_table.setStyle(TableStyle([
                        ('BACKGROUND', (2, i+1), (2, i+1), colors.lightyellow)
                    ]))
            
            elements.append(results_table)
            
            # Crear PDF
            doc.build(elements)
            
            self.logger.info(f"Informe PDF creado en {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al crear informe PDF: {str(e)}")
            return False


# Para pruebas directas del módulo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('system_logger')
    
    # Crear instancia de prueba
    results_manager = ResultsManager("test_results")
    
    # Crear algunos resultados de prueba
    test_results = []
    
    # Resultado aprobado
    pass_result = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "pass",
        "module_results": {
            "color": {
                "status": "pass",
                "detections": {}
            },
            "dimensions": {
                "status": "pass",
                "measurements": [
                    {"width_mm": 100.2, "height_mm": 50.1, "in_tolerance": True}
                ]
            }
        }
    }
    test_results.append(pass_result)
    
    # Resultado fallido
    fail_result = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "fail",
        "module_results": {
            "color": {
                "status": "fail",
                "detections": {
                    "red": {"count": 2, "areas": [120, 150]}
                }
            },
            "dimensions": {
                "status": "pass",
                "measurements": [
                    {"width_mm": 99.5, "height_mm": 49.8, "in_tolerance": True}
                ]
            }
        }
    }
    test_results.append(fail_result)
    
    # Guardar resultados
    logger.info("Guardando resultados de prueba...")
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (250, 250), (0, 0, 255), -1)  # Rectángulo rojo
    
    results_manager.save_inspection_result(pass_result, "TEST001", test_image)
    results_manager.save_inspection_result(fail_result, "TEST001", test_image)
    
    # Generar informe
    logger.info("Generando informe CSV...")
    results_manager.create_inspection_report(test_results, "test_results/informe.csv")
    
    # Generar estadísticas
    logger.info("Generando estadísticas...")
    stats = results_manager.generate_statistics(test_results, include_charts=True)
    logger.info(f"Estadísticas: {stats}")
    
    # Exportar resultados
    logger.info("Exportando resultados...")
    file_config = {
        "type": "file",
        "path": "test_results/export.json",
        "format": "json"
    }
    results_manager.export_results_to_external_system(test_results, file_config)
    
    # Crear informe PDF
    try:
        logger.info("Creando informe PDF...")
        sku_info = {
            "sku_id": "TEST001",
            "name": "Producto de Prueba",
            "description": "Componente metálico rectangular para pruebas"
        }
        results_manager.create_report_pdf(test_results, "test_results/informe.pdf", sku_info)
    except Exception as e:
        logger.error(f"No se pudo crear PDF (es posible que falte reportlab): {str(e)}")
    
    logger.info("Pruebas completadas")