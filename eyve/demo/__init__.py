"""
Eyve demo — tela sintética para stand de expo.

Un patrón de impresión textil (azul sobre blanco) que pasa infinitamente
frente a una cámara que no existe: el visitante dibuja defectos sobre la
tela y Eyve los detecta, los sigue y los cuenta con el mismo pipeline que
usa en producción (YOLO → InstanceTracker → CountingModule).

Piezas:
    textile.TextilePattern   la tela: motivo, viaje infinito, capa de
                             defectos en coordenadas de tela
    source.SyntheticSource   origen de video compatible con VideoSource
                             (start/stop/read), sin cámara ni OBS
    dataset.build_dataset    genera el dataset etiquetado con las MISMAS
                             primitivas con las que dibuja el visitante
"""
from eyve.demo.textile import TextilePattern, Defect, DEFECT_CLASSES
from eyve.demo.source import SyntheticSource

__all__ = ["TextilePattern", "Defect", "DEFECT_CLASSES", "SyntheticSource"]
