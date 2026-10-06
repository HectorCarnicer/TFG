"""
lectura_escritura.py

Lectura (perezosa, curva a curva) del dataset de curvas I-V y
escritura/gestión del log de test.
"""

import json
import os
import re
from datetime import datetime


# ---------------------------------------------------------------------------
# Dataset de curvas I-V (carga perezosa)
# ---------------------------------------------------------------------------
#
# Los archivos .dat pueden ocupar varios GB (miles de puntos por curva x
# cientos de curvas), y json.load() completo los carga enteros en memoria
# como objetos Python, lo que puede agotar la RAM disponible. cargar_dataset
# indexa el archivo (posición de cada curva) sin cargar sus arrays V/I/P;
# cargar_curva() carga bajo demanda solo la curva que hace falta en cada
# momento, así el pico de memoria depende de una curva, no del dataset
# entero.

_TAM_BLOQUE = 1_048_576  # 1 MB, tamaño de bloque de lectura

_LLAVE_ABRE = ord("{")
_LLAVE_CIERRA = ord("}")
_CORCHETE_CIERRA = ord("]")
_COMILLA = ord('"')
_BACKSLASH = ord("\\")
_PATRON_ESTRUCTURA = re.compile(rb'["{}\]\\]')


def cargar_dataset(ruta_datos):
    """Indexa el dataset de curvas I-V sin cargar los arrays V/I/P en
    memoria: recorre el archivo una vez y guarda, para cada curva, sus
    metadatos ligeros (id, irradiancias, temperatura, mpp) junto con su
    posición en el archivo.

    Lanza FileNotFoundError si no existe, o ValueError si no tiene el
    formato esperado. Devuelve {"ruta", "metadata", "simulaciones"}, donde
    "simulaciones" es la lista de curvas indexadas (sin los arrays V/I/P
    de "curva"); usa cargar_curva() para obtenerlos bajo demanda.
    """
    if not os.path.isfile(ruta_datos):
        raise FileNotFoundError(f"No se encuentra el archivo de datos: {ruta_datos}")

    with open(ruta_datos, "rb") as f:
        cabecera = f.read(_TAM_BLOQUE)
        while b'"simulaciones"' not in cabecera:
            extra = f.read(_TAM_BLOQUE)
            if not extra:
                raise ValueError(
                    f"El archivo {ruta_datos} no tiene el formato esperado "
                    "(falta la clave 'simulaciones')."
                )
            cabecera += extra

        idx_clave = cabecera.find(b'"simulaciones"')
        while cabecera.find(b"[", idx_clave) == -1:
            extra = f.read(_TAM_BLOQUE)
            if not extra:
                raise ValueError(f"JSON incompleto tras la clave 'simulaciones' en {ruta_datos}.")
            cabecera += extra

        metadata = _extraer_metadata(cabecera[:idx_clave])
        offset_array = cabecera.find(b"[", idx_clave) + 1

    indice = []
    with open(ruta_datos, "rb") as f_escaneo, open(ruta_datos, "rb") as f_lectura:
        for inicio, fin in _iterar_curvas(f_escaneo, offset_array):
            f_lectura.seek(inicio)
            curva = json.loads(f_lectura.read(fin - inicio))
            info = {k: v for k, v in curva.items() if k != "curva"}
            info["_offset"] = inicio
            info["_longitud"] = fin - inicio
            indice.append(info)

    if not indice:
        raise ValueError(f"El archivo {ruta_datos} no contiene ninguna curva en 'simulaciones'.")

    return {"ruta": ruta_datos, "metadata": metadata, "simulaciones": indice}


def cargar_curva(dataset, simulacion):
    """Carga bajo demanda los arrays V/I/P de una curva indexada por
    cargar_dataset, leyendo solo su tramo del archivo.

    Devuelve el diccionario completo de la curva (incluye "curva").
    """
    with open(dataset["ruta"], "rb") as f:
        f.seek(simulacion["_offset"])
        bruto = f.read(simulacion["_longitud"])
    return json.loads(bruto)


def _fin_objeto(datos, inicio):
    """Devuelve la posición justo después del '}' que cierra el objeto
    JSON que empieza en datos[inicio] (debe ser '{'), respetando cadenas.
    """
    profundidad = 0
    en_cadena = False
    pos_escapado = -1
    for m in _PATRON_ESTRUCTURA.finditer(datos, inicio):
        pos = m.start()
        b = datos[pos]
        if en_cadena:
            if pos == pos_escapado:
                continue
            if b == _BACKSLASH:
                pos_escapado = pos + 1
            elif b == _COMILLA:
                en_cadena = False
            continue
        if b == _COMILLA:
            en_cadena = True
        elif b == _LLAVE_ABRE:
            profundidad += 1
        elif b == _LLAVE_CIERRA:
            profundidad -= 1
            if profundidad == 0:
                return pos + 1
    raise ValueError("JSON incompleto: no se ha encontrado el cierre esperado.")


def _extraer_metadata(prefijo):
    """Extrae el bloque "metadata" de los bytes anteriores a
    "simulaciones". Devuelve {} si el dataset no tiene ese bloque.
    """
    idx = prefijo.find(b'"metadata"')
    if idx == -1:
        return {}
    inicio = prefijo.find(b"{", idx)
    if inicio == -1:
        return {}
    fin = _fin_objeto(prefijo, inicio)
    return json.loads(prefijo[inicio:fin])


def _iterar_curvas(f, offset_inicial, tam_bloque=_TAM_BLOQUE):
    """Recorre el array "simulaciones" a partir de offset_inicial y genera
    (inicio, fin) para cada curva (objeto JSON), leyendo el archivo en
    bloques y sin cargar los objetos en memoria. Se detiene al cerrar el
    array.
    """
    f.seek(offset_inicial)
    profundidad = 0
    en_cadena = False
    pos_escapado = -1
    inicio = None
    base = offset_inicial

    while True:
        bloque = f.read(tam_bloque)
        if not bloque:
            return
        for m in _PATRON_ESTRUCTURA.finditer(bloque):
            pos = base + m.start()
            b = bloque[m.start()]

            if en_cadena:
                if pos == pos_escapado:
                    continue
                if b == _BACKSLASH:
                    pos_escapado = pos + 1
                elif b == _COMILLA:
                    en_cadena = False
                continue

            if b == _COMILLA:
                en_cadena = True
            elif b == _LLAVE_ABRE:
                if profundidad == 0:
                    inicio = pos
                profundidad += 1
            elif b == _LLAVE_CIERRA:
                profundidad -= 1
                if profundidad == 0:
                    yield inicio, pos + 1
            elif b == _CORCHETE_CIERRA and profundidad == 0:
                return
        base += len(bloque)


def listar_simulaciones(dataset):
    """Devuelve la lista de curvas indexadas del dataset (metadatos
    ligeros, sin los arrays V/I/P; usa cargar_curva() para obtenerlos).
    """
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
