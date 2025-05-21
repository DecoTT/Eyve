import cv2
import time
import numpy as np

def test_camera(index):
    """Prueba una cámara específica y muestra información sobre ella"""
    print(f"\nProbando cámara con índice {index}...")
    
    # Intentar abrir la cámara
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Usar DirectShow en Windows
    
    if not cap.isOpened():
        print(f"  - No se pudo abrir la cámara {index}")
        return False
        
    # Obtener propiedades
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  - Cámara abierta con éxito")
    print(f"  - Resolución: {width}x{height}")
    print(f"  - FPS: {fps}")
    
    # Intentar capturar un frame
    ret, frame = cap.read()
    
    if ret:
        print(f"  - Captura de frame exitosa")
        print(f"  - Tamaño del frame: {frame.shape}")
        
        # Guardar el frame para verificación
        filename = f"camera_{index}_test.jpg"
        cv2.imwrite(filename, frame)
        print(f"  - Frame guardado como {filename}")
    else:
        print(f"  - No se pudo capturar el frame")
    
    # Liberar la cámara
    cap.release()
    return ret

# Probar cada índice de cámara de 0 a 9
print("DIAGNÓSTICO DE CÁMARAS")
print("=====================")

working_cameras = []

for i in range(10):
    if test_camera(i):
        working_cameras.append(i)
    time.sleep(1)  # Esperar entre pruebas

print("\nRESUMEN")
print("=======")
print(f"Cámaras funcionales detectadas: {len(working_cameras)}")
print(f"Índices: {working_cameras}")
print("\nRECOMENDACIÓN:")
if working_cameras:
    print(f"Usar el índice {working_cameras[0]} para la cámara principal")
else:
    print("No se detectaron cámaras funcionales. Verifique los drivers y conexiones.")