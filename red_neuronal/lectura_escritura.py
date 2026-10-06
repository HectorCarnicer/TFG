"""
lectura_escritura.py
====================
Entrada/salida del predictor de MPP: lectura de las simulaciones I-V y
construccion del dataset de entrenamiento, particion train/val/test, y
exportacion/carga de modelos en formato TensorFlow Lite.

Cada muestra es un punto (V, I) de una curva; la etiqueta es el Vmp de esa
curva.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import tensorflow as tf

# El interprete TFLite puede estar en ai_edge_litert o en tf.lite segun la version
try:  # pragma: no cover
    from ai_edge_litert.interpreter import Interpreter as _Interpreter
except Exception:  # pragma: no cover
    _Interpreter = tf.lite.Interpreter


# --- Utilidades internas -------------------------------------------------------
def _buscar_clave(diccionario: dict, patron: str):
    """Devuelve el valor de la primera clave que contiene 'patron' (sin distinguir mayusculas).

    Lanza KeyError si no se encuentra ninguna coincidencia.
    """
    patron = patron.lower()
    for clave, valor in diccionario.items():
        if patron in clave.lower():
            return valor
    raise KeyError(f"No se ha encontrado ninguna clave que contenga '{patron}'")


def _indices_muestreo(n_disponibles: int, n_puntos: int, modo: str,
                      rng: np.random.Generator) -> np.ndarray:
    """Selecciona que indices de una curva se usan como muestras.

    n_puntos=None o >= n_disponibles: se usan todos los puntos.
    modo='uniforme': reparto uniforme a lo largo de la curva.
    modo='aleatorio': muestreo aleatorio sin reemplazo.
    """
    if n_puntos is None or n_puntos >= n_disponibles:
        return np.arange(n_disponibles)

    if modo == "uniforme":
        return np.unique(np.linspace(0, n_disponibles - 1, n_puntos).round().astype(int))
    if modo == "aleatorio":
        return np.sort(rng.choice(n_disponibles, size=n_puntos, replace=False))
    raise ValueError(f"Modo de muestreo desconocido: {modo!r} (usar 'uniforme' o 'aleatorio')")


# --- Lectura de datos ------------------------------------------------------------
def leer_datos_entrenamiento(ruta_json: str,
                             n_puntos: int | None = 100,
                             modo_muestreo: str = "uniforme",
                             descartar_corriente_negativa: bool = True,
                             semilla: int = 42,
                             devolver_curvas: bool = True,
                             verbose: bool = True):
    """Lee un JSON de curvas I-V y construye el dataset de entrenamiento.

    Parametros
    ----------
    ruta_json : ruta del JSON con 'metadata' y 'simulaciones'.
    n_puntos : puntos tomados de cada curva; None = todos los disponibles.
    modo_muestreo : 'uniforme' | 'aleatorio'.
    descartar_corriente_negativa : elimina los puntos con I < 0 o V < 0.
    semilla : semilla del muestreo aleatorio.
    devolver_curvas : si True, 'info' incluye las curvas completas.

    Devuelve
    --------
    X : ndarray (n_muestras, 2), columnas [V, I].
    y : ndarray (n_muestras, 1), Vmp de la curva de origen de cada muestra.
    grupos : ndarray (n_muestras,), id de curva de cada muestra.
    info : dict con metadata, curvas y estadisticas del dataset.
    """
    if not os.path.isfile(ruta_json):
        raise FileNotFoundError(f"No se encuentra el fichero de datos: {ruta_json}")

    with open(ruta_json, "r", encoding="utf-8") as fichero:
        datos = json.load(fichero)

    metadata = datos.get("metadata", {})
    simulaciones = datos.get("simulaciones", [])
    if not simulaciones:
        raise ValueError("El JSON no contiene la lista 'simulaciones'")

    rng = np.random.default_rng(semilla)
    lista_X, lista_y, lista_grupos, curvas = [], [], [], []

    for i, simulacion in enumerate(simulaciones):
        curva = simulacion["curva"]
        V = np.asarray(curva["V"], dtype=np.float32)
        I = np.asarray(curva["I"], dtype=np.float32)
        if V.size != I.size:
            raise ValueError(f"Simulacion {simulacion.get('id', i)}: V e I con distinta longitud")

        vmp = float(_buscar_clave(simulacion["mpp"], "vmp"))
        id_curva = int(simulacion.get("id", i))

        orden = np.argsort(V)
        V, I = V[orden], I[orden]
        if descartar_corriente_negativa:
            valida = (I >= 0.0) & (V >= 0.0)
            V, I = V[valida], I[valida]
        if V.size == 0:
            continue

        indices = _indices_muestreo(V.size, n_puntos, modo_muestreo, rng)
        lista_X.append(np.column_stack((V[indices], I[indices])))
        lista_y.append(np.full(indices.size, vmp, dtype=np.float32))
        lista_grupos.append(np.full(indices.size, id_curva, dtype=np.int32))

        if devolver_curvas:
            curvas.append({
                "id": id_curva,
                "V": V,
                "I": I,
                "vmp": vmp,
                "pmax": float(_buscar_clave(simulacion["mpp"], "pmax")),
                "imp": float(_buscar_clave(simulacion["mpp"], "imp")),
                "irradiancias": simulacion.get("irradiancias_substrings_Wm2", []),
            })

    X = np.concatenate(lista_X).astype(np.float32)
    y = np.concatenate(lista_y).astype(np.float32).reshape(-1, 1)
    grupos = np.concatenate(lista_grupos)

    n_curvas = len(lista_X)
    info = {
        "metadata": metadata,
        "curvas": curvas,
        "n_curvas": n_curvas,
        "n_muestras": X.shape[0],
        "puntos_por_curva": int(X.shape[0] / max(n_curvas, 1)),
        "rango_V": (float(X[:, 0].min()), float(X[:, 0].max())),
        "rango_I": (float(X[:, 1].min()), float(X[:, 1].max())),
        "rango_Vmp": (float(y.min()), float(y.max())),
    }

    if verbose:
        print(f"[datos] Fichero            : {ruta_json}")
        print(f"[datos] Curvas leidas      : {info['n_curvas']}")
        print(f"[datos] Muestras (V, I)    : {info['n_muestras']}  "
              f"(~{info['puntos_por_curva']} por curva)")
        print(f"[datos] Rango V   [V]      : {info['rango_V'][0]:.2f} .. {info['rango_V'][1]:.2f}")
        print(f"[datos] Rango I   [A]      : {info['rango_I'][0]:.2f} .. {info['rango_I'][1]:.2f}")
        print(f"[datos] Rango Vmp [V]      : {info['rango_Vmp'][0]:.2f} .. {info['rango_Vmp'][1]:.2f}")

    return X, y, grupos, info


def dividir_datos(X, y, grupos, fraccion_test=0.15, fraccion_val=0.15,
                  semilla=42, verbose=True):
    """Divide el dataset en train/val/test por id de curva (sin mezclar
    puntos de la misma curva entre conjuntos).

    Devuelve un dict {'train', 'val', 'test'}, cada uno (X, y, grupos).
    """
    ids = np.unique(grupos)
    rng = np.random.default_rng(semilla)
    rng.shuffle(ids)

    n = ids.size
    n_test = int(round(n * fraccion_test))
    n_val = int(round(n * fraccion_val))
    ids_test, ids_val, ids_train = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]

    def _subconjunto(ids_sel):
        mascara = np.isin(grupos, ids_sel)
        return X[mascara], y[mascara], grupos[mascara]

    particion = {
        "train": _subconjunto(ids_train),
        "val": _subconjunto(ids_val),
        "test": _subconjunto(ids_test),
    }

    if verbose:
        for nombre, (Xs, _, gs) in particion.items():
            print(f"[split] {nombre:5s}: {Xs.shape[0]:7d} muestras / {np.unique(gs).size:4d} curvas")

    return particion


# --- Escritura / lectura del modelo TensorFlow Lite -----------------------------
def exportar_tflite(modelo, ruta_salida: str, cuantizar: bool = False,
                    datos_representativos=None, verbose: bool = True) -> str:
    """Convierte un modelo Keras a .tflite y lo guarda en disco.

    cuantizar=False: pesos float32.
    cuantizar=True: cuantizacion por defecto; si se pasan
        'datos_representativos' (array de entradas reales) se calibra
        cuantizacion entera completa.

    Devuelve la ruta del fichero guardado.
    """
    directorio = os.path.dirname(os.path.abspath(ruta_salida))
    os.makedirs(directorio, exist_ok=True)

    try:
        convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
        if cuantizar:
            _configurar_cuantizacion(convertidor, datos_representativos)
        contenido = convertidor.convert()
    except Exception as error:  # Keras 3: se pasa por SavedModel
        if verbose:
            print(f"[tflite] Conversion directa fallida ({type(error).__name__}), "
                  f"reintentando via SavedModel...")
        with tempfile.TemporaryDirectory() as tmp:
            modelo.export(tmp)
            convertidor = tf.lite.TFLiteConverter.from_saved_model(tmp)
            if cuantizar:
                _configurar_cuantizacion(convertidor, datos_representativos)
            contenido = convertidor.convert()

    with open(ruta_salida, "wb") as fichero:
        fichero.write(contenido)

    if verbose:
        print(f"[tflite] Modelo guardado   : {ruta_salida} ({len(contenido) / 1024:.1f} kB)")
    return ruta_salida


def _configurar_cuantizacion(convertidor, datos_representativos):
    """Aplica las opciones de cuantizacion al convertidor TFLite."""
    convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
    if datos_representativos is None:
        return

    muestras = np.asarray(datos_representativos, dtype=np.float32)

    def generador():
        for muestra in muestras:
            yield [muestra.reshape(1, -1)]

    convertidor.representative_dataset = generador


def cargar_modelo_tflite(ruta_tflite: str, verbose: bool = True):
    """Carga un .tflite y devuelve el interprete con los tensores reservados."""
    if not os.path.isfile(ruta_tflite):
        raise FileNotFoundError(f"No se encuentra el modelo TFLite: {ruta_tflite}")

    interprete = _Interpreter(model_path=ruta_tflite)
    interprete.allocate_tensors()

    if verbose:
        entrada = interprete.get_input_details()[0]
        salida = interprete.get_output_details()[0]
        print(f"[tflite] Modelo cargado    : {ruta_tflite}")
        print(f"[tflite] Entrada           : {entrada['shape']} {entrada['dtype'].__name__}")
        print(f"[tflite] Salida            : {salida['shape']} {salida['dtype'].__name__}")

    return interprete


def guardar_resumen(ruta_json: str, resumen: dict, verbose: bool = True) -> str:
    """Guarda un dict como JSON junto al modelo. Devuelve la ruta de salida."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta_json)), exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as fichero:
        json.dump(resumen, fichero, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[tflite] Resumen guardado  : {ruta_json}")
    return ruta_json
