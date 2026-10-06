"""
lectura_escritura.py

Lectura del dataset de curvas I-V y escritura/gestión del log de test.
"""

import json
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Dataset de curvas I-V
# ---------------------------------------------------------------------------

def cargar_dataset(ruta_datos):
    """Carga el dataset de curvas I-V desde un archivo JSON.

    Lanza FileNotFoundError si no existe, o ValueError si no tiene el
    formato esperado. Devuelve el diccionario completo ya parseado.
    """
    if not os.path.isfile(ruta_datos):
        raise FileNotFoundError(f"No se encuentra el archivo de datos: {ruta_datos}")

    with open(ruta_datos, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if "simulaciones" not in dataset:
        raise ValueError(
            f"El archivo {ruta_datos} no tiene el formato esperado "
            "(falta la clave 'simulaciones')."
        )

    return dataset


def listar_simulaciones(dataset):
    """Devuelve la lista de curvas del dataset."""
    return dataset["simulaciones"]


def metadata_dataset(dataset):
    """Devuelve el bloque de metadata del dataset."""
    return dataset.get("metadata", {})


# ---------------------------------------------------------------------------
# Log de test
# ---------------------------------------------------------------------------

def ruta_log(directorio_logs, id_test):
    """Devuelve la ruta del log correspondiente a un ID de test."""
    nombre = f"test_{id_test}.log"
    return os.path.join(directorio_logs, nombre)


def abrir_log(directorio_logs, id_test):
    """Crea el directorio de logs si no existe y abre test_{ID}.log en
    modo escritura (sobrescribe uno existente con el mismo ID).

    Devuelve el file handle abierto; el llamador debe cerrarlo con
    cerrar_log().
    """
    os.makedirs(directorio_logs, exist_ok=True)
    fh = open(ruta_log(directorio_logs, id_test), "w", encoding="utf-8")
    return fh


def escribir_log(fh, texto, tambien_consola=True):
    """Escribe una línea en el log (con flush inmediato) y, si
    tambien_consola es True, la imprime también por consola.
    """
    if tambien_consola:
        print(texto)

    if not texto.endswith("\n"):
        texto += "\n"
    fh.write(texto)
    fh.flush()


def escribir_cabecera_log(fh, id_test, campos):
    """Escribe la cabecera del log: ID de test, fecha/hora de inicio y los
    campos de configuración de `campos` ({etiqueta: valor}).
    """
    etiquetas_fijas = ["Test ID", "Fecha/hora inicio"]
    ancho = max(len(e) for e in [*etiquetas_fijas, *campos.keys()])

    lineas = [
        "=" * 70,
        f"{'Test ID':<{ancho}} : {id_test}",
        f"{'Fecha/hora inicio':<{ancho}} : {datetime.now().isoformat(timespec='seconds')}",
    ]
    for etiqueta, valor in campos.items():
        lineas.append(f"{etiqueta:<{ancho}} : {valor}")
    lineas.append("=" * 70)

    for linea in lineas:
        escribir_log(fh, linea, tambien_consola=True)


def cerrar_log(fh):
    """Cierra el archivo de log."""
    try:
        fh.flush()
    finally:
        fh.close()
