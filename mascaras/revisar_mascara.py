# -*- coding: utf-8 -*-
"""
La ventana en la que el operador ve la mascara y decide.

Prototipo: no esta en Produccion.  Muestra vista() —la celda, sus zonas y la
mascara sobre el material— con dos botones, y guarda la decision en el .npz.

    python revisar_mascara.py mascara.npz cuadro.png
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eyve.modules.mascara import MascaraAprendida


class Revision:
    def __init__(self, master, mascara: MascaraAprendida, cuadro=None,
                 ruta=None, quien: str = "operador") -> None:
        self.mascara = mascara
        self.ruta = ruta
        self.quien = quien
        self.top = tk.Toplevel(master)
        self.top.title("Revisar máscara")
        rgb = cv2.cvtColor(mascara.vista(cuadro), cv2.COLOR_BGR2RGB)
        self._foto = ImageTk.PhotoImage(Image.fromarray(rgb), master=self.top)
        tk.Label(self.top, image=self._foto).pack()
        for aviso in mascara.informe.avisos:
            tk.Label(self.top, text=aviso, fg="#b00020").pack()
        fila = tk.Frame(self.top)
        fila.pack(pady=8)
        self.btn_aprobar = tk.Button(fila, text="Aprobar", width=16,
                                     command=self.aprobar)
        self.btn_rechazar = tk.Button(fila, text="Rechazar", width=16,
                                      command=self.rechazar)
        self.btn_aprobar.pack(side="left", padx=6)
        self.btn_rechazar.pack(side="left", padx=6)

    def _cerrar(self) -> None:
        if self.ruta:
            self.mascara.guardar(self.ruta)
        self.top.destroy()

    def aprobar(self) -> None:
        self.mascara.aprobar(self.quien)
        self._cerrar()

    def rechazar(self) -> None:
        self.mascara.rechazar()
        self._cerrar()


def main() -> None:
    ruta = Path(sys.argv[1])
    cuadro = cv2.imread(sys.argv[2]) if len(sys.argv) > 2 else None
    raiz = tk.Tk()
    raiz.withdraw()
    r = Revision(raiz, MascaraAprendida.cargar(ruta), cuadro, ruta)
    r.top.protocol("WM_DELETE_WINDOW", raiz.destroy)
    r.top.bind("<Destroy>", lambda e: raiz.after(1, raiz.destroy))
    raiz.mainloop()


if __name__ == "__main__":
    main()
