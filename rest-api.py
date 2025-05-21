#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API REST para Sistema de Inspección Visual
----------------------------------------
Proporciona una API REST para integrar el sistema con aplicaciones externas.
"""

import os
import json
import base64
import logging
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS

# Importar componentes del sistema
# Asegurarse de que estos módulos estén en el path de importación
import sys
sys.path.append('.')

# Intentar importar los componentes del sistema
try:
    from database.database_manager import DatabaseManager
    from config.config_manager import ConfigManager
    from inspection_modules.inspection_module import InspectionManager
    from simulation.simulation_module import SimulationEnvironment
except ImportError as e:
    print(f"Error al importar módulos del sistema: {e}")
    print("Ejecute el servidor desde el directorio raíz del proyecto")
    sys.exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('api_logger')

# Crear directorio de logs si no existe
os.makedirs("logs", exist_ok=True)

# Inicializar aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Inicializar componentes del sistema
config_manager = ConfigManager()
db_manager = DatabaseManager(config_manager.get_config_value("database.path"))

# Modo de simulación (para pruebas sin hardware real)
SIMULATION_MODE = os.environ.get("SIMULATION_MODE", "False").lower() in ("true", "1", "yes")
simulation_env = None

if SIMULATION_MODE:
    logger.info("Iniciando en modo simulación")
    simulation_config = os.environ.get("SIMULATION_CONFIG", "config/simulation/default.json")
    simulation_env = SimulationEnvironment(simulation_config)
    simulation_env.start()
else:
    logger.info("Iniciando en modo normal (sin simulación)")

# Variables globales para estado de la aplicación
current_session_id = None
current_sku_id = None
inspection_manager = None


@app.route('/api/status', methods=['GET'])
def get_status():
    """Obtiene el estado actual del sistema."""
    status = {
        "status": "online",
        "version": config_manager.get_config_value("version", "1.0.0"),
        "simulation_mode": SIMULATION_MODE,
        "current_session": current_session_id,
        "current_sku": current_sku_id,
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(status)


@app.route('/api/login', methods=['POST'])
def login():
    """Autenticar un usuario en el sistema."""
    data = request.json
    
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Se requiere nombre de usuario y contraseña"}), 400
    
    username = data['username']
    password = data['password']
    
    # Autenticar usuario
    user = db_manager.authenticate_user(username, password)
    
    if user:
        return jsonify({
            "success": True,
            "user": user,
            "message": "Autenticación exitosa"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Credenciales inválidas"
        }), 401


@app.route('/api/skus', methods=['GET'])
def get_skus():
    """Obtiene la lista de SKUs disponibles."""
    active_only = request.args.get('active_only', 'true').lower() == 'true'
    skus = db_manager.get_skus(active_only)
    return jsonify({
        "success": True,
        "skus": skus,
        "count": len(skus)
    })


@app.route('/api/skus/<sku_id>', methods=['GET'])
def get_sku(sku_id):
    """Obtiene información de un SKU específico."""
    sku = db_manager.get_sku_by_id(sku_id)
    
    if sku:
        # Cargar configuración del SKU
        sku_config = config_manager.load_sku_config(sku_id)
        if sku_config:
            sku['config'] = sku_config
            
        return jsonify({
            "success": True,
            "sku": sku
        })
    else:
        return jsonify({
            "success": False,
            "message": f"SKU {sku_id} no encontrado"
        }), 404


@app.route('/api/sessions/start', methods=['POST'])
def start_session():
    """Inicia una nueva sesión de inspección."""
    global current_session_id, current_sku_id, inspection_manager
    
    data = request.json
    
    if not data or 'sku_id' not in data or 'user_id' not in data:
        return jsonify({"error": "Se requiere SKU ID y User ID"}), 400
    
    sku_id = data['sku_id']
    user_id = data['user_id']
    
    # Verificar que el SKU existe
    sku = db_manager.get_sku_by_id(sku_id)
    if not sku:
        return jsonify({
            "success": False,
            "message": f"SKU {sku_id} no encontrado"
        }), 404
    
    # Iniciar sesión en la base de datos
    session_id = db_manager.start_inspection_session(sku_id, user_id)
    
    if session_id:
        # Cargar configuración del SKU
        sku_config = config_manager.load_sku_config(sku_id)
        
        # Inicializar gestor de inspección
        inspection_manager = InspectionManager()
        if sku_config:
            inspection_manager.load_config_from_sku(sku_config)
        
        # Actualizar variables globales
        current_session_id = session_id
        current_sku_id = sku_id
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "sku_id": sku_id,
            "message": "Sesión iniciada correctamente"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Error al iniciar sesión"
        }), 500


@app.route('/api/sessions/end', methods=['POST'])
def end_session():
    """Finaliza la sesión de inspección actual."""
    global current_session_id, current_sku_id, inspection_manager
    
    data = request.json
    
    if not data or 'session_id' not in data:
        return jsonify({"error": "Se requiere Session ID"}), 400
    
    session_id = data['session_id']
    status = data.get('status', 'completed')
    
    # Finalizar sesión en la base de datos
    success = db_manager.end_inspection_session(session_id, status)
    
    if success:
        # Limpiar estado global si corresponde
        if current_session_id == session_id:
            current_session_id = None
            current_sku_id = None
            inspection_manager = None
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "Sesión finalizada correctamente"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Error al finalizar sesión"
        }), 500


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Obtiene información de una sesión específica."""
    session_summary = db_manager.get_session_summary(int(session_id))
    
    if session_summary:
        return jsonify({
            "success": True,
            "session": session_summary
        })
    else:
        return jsonify({
            "success": False,
            "message": f"Sesión {session_id} no encontrada"
        }), 404


@app.route('/api/sessions/<session_id>/results', methods=['GET'])
def get_session_results(session_id):
    """Obtiene los resultados de una sesión específica."""
    results = db_manager.get_session_results(int(session_id))
    
    return jsonify({
        "success": True,
        "session_id": session_id,
        "results": results,
        "count": len(results)
    })


@app.route('/api/inspect', methods=['POST'])
def perform_inspection():
    """Realiza una inspección con una imagen proporcionada."""
    global current_session_id, current_sku_id, inspection_manager
    
    # Verificar que hay una sesión activa
    if not current_session_id or not inspection_manager:
        return jsonify({
            "success": False,
            "message": "No hay una sesión de inspección activa"
        }), 400
    
    # Procesar imagen
    if 'image' in request.files:
        # Recibir imagen desde archivo
        file = request.files['image']
        img_data = file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    elif request.json and 'image_base64' in request.json:
        # Recibir imagen en base64
        base64_data = request.json['image_base64']
        img_data = base64.b64decode(base64_data.split(',')[1] if ',' in base64_data else base64_data)
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    elif SIMULATION_MODE and simulation_env:
        # En modo simulación, obtener imagen del simulador
        ret, image = simulation_env.get_frame()
        if not ret or image is None:
            return jsonify({
                "success": False,
                "message": "No se pudo obtener imagen del simulador"
            }), 500
    else:
        return jsonify({
            "success": False,
            "message": "No se proporcionó imagen para inspección"
        }), 400
    
    # Realizar inspección
    try:
        # Inspeccionar imagen
        result = inspection_manager.inspect_image(image)
        
        # En modo simulación, añadir información del producto
        if SIMULATION_MODE and simulation_env:
            product_info = simulation_env.get_product()
            if product_info:
                result["simulation_info"] = product_info
        
        # Guardar resultado en la base de datos
        image_path = None
        if result["overall_status"] != "pass" or config_manager.get_config_value("inspection.save_all_inspections", False):
            # Guardar imagen solo si falló o si está configurado para guardar todas
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_dir = f"storage/images/{current_sku_id}/{timestamp}"
            os.makedirs(image_dir, exist_ok=True)
            image_path = f"{image_dir}/inspection.jpg"
            cv2.imwrite(image_path, image)
        
        # Guardar en base de datos
        result_id = db_manager.save_inspection_result(current_session_id, result, image_path)
        
        # Añadir IDs a la respuesta
        result["result_id"] = result_id
        result["session_id"] = current_session_id
        
        return jsonify({
            "success": True,
            "result": result
        })
    except Exception as e:
        logger.error(f"Error en inspección: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error en inspección: {str(e)}"
        }), 500


@app.route('/api/statistics/<session_id>', methods=['GET'])
def get_statistics(session_id):
    """Obtiene estadísticas de una sesión específica."""
    results = db_manager.get_session_results(int(session_id))
    
    if not results:
        return jsonify({
            "success": False,
            "message": f"No hay resultados para la sesión {session_id}"
        }), 404
    
    # Crear gestor de resultados para generar estadísticas
    try:
        from utils.results_manager import ResultsManager
        results_manager = ResultsManager()
        
        # Generar estadísticas
        include_charts = request.args.get('charts', 'false').lower() == 'true'
        stats = results_manager.generate_statistics(results, include_charts)
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "statistics": stats
        })
    except Exception as e:
        logger.error(f"Error al generar estadísticas: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error al generar estadísticas: {str(e)}"
        }), 500


@app.route('/api/reports/<session_id>', methods=['GET'])
def generate_report(session_id):
    """Genera un informe para una sesión específica."""
    format = request.args.get('format', 'csv').lower()
    results = db_manager.get_session_results(int(session_id))
    
    if not results:
        return jsonify({
            "success": False,
            "message": f"No hay resultados para la sesión {session_id}"
        }), 404
    
    try:
        from utils.results_manager import ResultsManager
        results_manager = ResultsManager()
        
        # Crear directorio temporal para el informe
        report_dir = f"storage/reports/{session_id}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Determinar ruta y formato de salida
        if format == 'csv':
            output_path = f"{report_dir}/report_{session_id}.csv"
            results_manager.create_inspection_report(results, output_path, 'csv')
            mimetype = 'text/csv'
        elif format == 'excel':
            output_path = f"{report_dir}/report_{session_id}.xlsx"
            results_manager.create_inspection_report(results, output_path, 'excel')
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif format == 'json':
            output_path = f"{report_dir}/report_{session_id}.json"
            results_manager.create_inspection_report(results, output_path, 'json')
            mimetype = 'application/json'
        elif format == 'pdf':
            output_path = f"{report_dir}/report_{session_id}.pdf"
            
            # Obtener información del SKU
            session_info = db_manager.get_session_summary(int(session_id))
            sku_info = None
            if session_info and 'sku_id' in session_info:
                sku_info = db_manager.get_sku_by_id(session_info['sku_id'])
            
            # Generar PDF
            results_manager.create_report_pdf(results, output_path, sku_info)
            mimetype = 'application/pdf'
        else:
            return jsonify({
                "success": False,
                "message": f"Formato de informe no soportado: {format}"
            }), 400
        
        # Enviar archivo
        return send_file(
            output_path,
            mimetype=mimetype,
            as_attachment=True,
            download_name=os.path.basename(output_path)
        )
        
    except Exception as e:
        logger.error(f"Error al generar informe: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error al generar informe: {str(e)}"
        }), 500


@app.route('/api/config/<sku_id>', methods=['GET'])
def get_sku_config(sku_id):
    """Obtiene la configuración de un SKU específico."""
    sku_config = config_manager.load_sku_config(sku_id)
    
    if sku_config:
        return jsonify({
            "success": True,
            "sku_id": sku_id,
            "config": sku_config
        })
    else:
        return jsonify({
            "success": False,
            "message": f"Configuración para SKU {sku_id} no encontrada"
        }), 404


@app.route('/api/config/<sku_id>', methods=['POST'])
def update_sku_config(sku_id):
    """Actualiza la configuración de un SKU específico."""
    data = request.json
    
    if not data:
        return jsonify({"error": "No se proporcionaron datos de configuración"}), 400
    
    # Verificar que el SKU existe en la base de datos
    sku = db_manager.get_sku_by_id(sku_id)
    if not sku:
        return jsonify({
            "success": False,
            "message": f"SKU {sku_id} no encontrado"
        }), 404
    
    # Actualizar configuración
    success = config_manager.save_sku_config(sku_id, data)
    
    if success:
        return jsonify({
            "success": True,
            "sku_id": sku_id,
            "message": "Configuración actualizada correctamente"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Error al actualizar configuración"
        }), 500


@app.route('/api/simulation/status', methods=['GET'])
def get_simulation_status():
    """Obtiene el estado actual de la simulación."""
    if not SIMULATION_MODE or not simulation_env:
        return jsonify({
            "success": False,
            "message": "Modo de simulación no habilitado"
        }), 400
    
    camera_props = simulation_env.camera.get_properties()
    product_config = simulation_env.product.get_product_config()
    
    status = {
        "running": simulation_env.running,
        "camera": camera_props,
        "product": product_config
    }
    
    return jsonify({
        "success": True,
        "status": status
    })


@app.route('/api/simulation/control', methods=['POST'])
def control_simulation():
    """Controla el estado de la simulación."""
    if not SIMULATION_MODE or not simulation_env:
        return jsonify({
            "success": False,
            "message": "Modo de simulación no habilitado"
        }), 400
    
    data = request.json
    
    if not data or 'action' not in data:
        return jsonify({"error": "Se requiere una acción"}), 400
    
    action = data['action'].lower()
    
    if action == 'start':
        success = simulation_env.start()
        message = "Simulación iniciada" if success else "Error al iniciar simulación"
    elif action == 'stop':
        simulation_env.stop()
        success = True
        message = "Simulación detenida"
    elif action == 'set_defect_probability':
        if 'probability' not in data:
            return jsonify({"error": "Se requiere el parámetro 'probability'"}), 400
        
        probability = float(data['probability'])
        simulation_env.set_defect_probability(probability)
        success = True
        message = f"Probabilidad de defectos establecida a {probability}"
    else:
        return jsonify({
            "success": False,
            "message": f"Acción no reconocida: {action}"
        }), 400
    
    return jsonify({
        "success": success,
        "message": message
    })


@app.route('/api/simulation/frame', methods=['GET'])
def get_simulation_frame():
    """Obtiene el frame actual de la simulación."""
    if not SIMULATION_MODE or not simulation_env:
        return jsonify({
            "success": False,
            "message": "Modo de simulación no habilitado"
        }), 400
    
    # Obtener frame de la simulación
    ret, frame = simulation_env.get_frame()
    
    if not ret or frame is None:
        return jsonify({
            "success": False,
            "message": "No se pudo obtener frame de la simulación"
        }), 500
    
    # Convertir a JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    
    # Crear respuesta con imagen
    response = make_response(frame_bytes)
    response.headers.set('Content-Type', 'image/jpeg')
    return response


# Función principal para ejecutar el servidor
def main():
    """Función principal para iniciar el servidor."""
    # Configurar puerto
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 'yes')
    
    logger.info(f"Iniciando servidor en puerto {port} (Debug: {debug})")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error al iniciar servidor: {str(e)}")
    finally:
        # Limpiar recursos al finalizar
        if SIMULATION_MODE and simulation_env:
            simulation_env.stop()
            logger.info("Simulación detenida")


if __name__ == '__main__':
    main()
