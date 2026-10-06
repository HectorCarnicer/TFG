"""
lectura_escritura.py
====================
Entrada/salida del predictor de MPP:

  * Lectura de ficheros JSON de curvas I-V y construccion del dataset
    (X = ventana de [V, I], y = Vmp de la curva).
  * Division train / validacion / test por curvas.
  * Exportacion a TensorFlow Lite y carga de un .tflite para inferencia.
  * Guardado del resumen JSON de un modelo.

Formato de los datos, ids globales de curva y division por curvas: README.md.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import tensorflow as tf

from muestreo_ventanas import construir_ventanas_curva

# El interprete de TFLite puede venir de ai-edge-litert; si no esta, de tf.lite.
try:  # pragma: no cover
    from ai_edge_litert.interpreter import Interpreter as _Interpreter
except Exception:  # pragma: no cover
    _Interpreter = tf.lite.Interpreter


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _buscar_clave(diccionario: dict, patron: str):
    """Devuelve el valor de la primera clave que contiene 'patron'.

    Las claves del JSON llevan unidades ("Vmp [V]"), por lo que no se buscan
    de forma exacta.

    Entrada: diccionario; patron (texto a buscar, sin distinguir mayusculas).
    Salida: valor asociado a la clave encontrada. Lanza KeyError si no hay ninguna.
    Globales: ninguna.
    """
    patron = patron.lower()
    for clave, valor in diccionario.items():
        if patron in clave.lower():
            return valor
    raise KeyError(f"No se ha encontrado ninguna clave que contenga '{patron}'")


# ---------------------------------------------------------------------------
# Lectura de datos
# ---------------------------------------------------------------------------
OFFSET_ID_DATOS = 1_000_000  # separacion entre ids de curva de distintos ficheros
                             # (debe superar el numero maximo de curvas por fichero)


def _leer_fichero_simulaciones(ruta_json: str) -> tuple[dict, list]:
    """Carga un JSON de simulaciones I-V.

    Entrada: ruta_json.
    Salida: (metadata, simulaciones). Lanza FileNotFoundError si el fichero no
        existe y ValueError si no contiene la lista 'simulaciones'.
    Globales: ninguna.
    """
    if not os.path.isfile(ruta_json):
        raise FileNotFoundError(f"No se encuentra el fichero de datos: {ruta_json}")

    with open(ruta_json, "r", encoding="utf-8") as fichero:
        datos = json.load(fichero)

    simulaciones = datos.get("simulaciones", [])
    if not simulaciones:
        raise ValueError(f"El JSON '{ruta_json}' no contiene la lista 'simulaciones'")
    return datos.get("metadata", {}), simulaciones


def _construir_ejemplos_curva(simulacion, indice_fichero, id_dato, ruta_json,
                              indice_simulacion, n_muestras, delta_v, ascendente,
                              n_ventanas, modo_muestreo, descartar_corriente_negativa,
                              rng, devolver_curvas, verbose, paso_fijo, ventana_ancho):
    """Construye las ventanas de entrenamiento de una curva.

    Ordena la curva por tension, descarta opcionalmente la cola no fisica
    (I < 0 o V < 0) y genera sus ventanas [V,I].

    Entrada: simulacion (dict del JSON); indice_fichero e id_dato (fichero de
        origen); ruta_json; indice_simulacion (posicion, id por defecto);
        n_muestras, delta_v, ascendente, n_ventanas, modo_muestreo,
        descartar_corriente_negativa, rng, paso_fijo, ventana_ancho (ver
        leer_datos_entrenamiento); devolver_curvas; verbose.
    Salida: (resultado, contar_descartada). resultado es
        (X_ventanas, y_vals, grupos_vals, info_curva), con info_curva=None si
        devolver_curvas=False, o None si la curva no produce ventanas.
        contar_descartada es True si la curva se descarta por rango
        insuficiente (suma a 'n_curvas_descartadas') y False si queda vacia
        tras quitar la cola no fisica. Lanza ValueError si V e I difieren en
        longitud.
    Globales: ninguna (avanza el estado de rng).
    """
    curva = simulacion["curva"]
    V = np.asarray(curva["V"], dtype=np.float32)
    I = np.asarray(curva["I"], dtype=np.float32)
    if V.size != I.size:
        raise ValueError(f"{ruta_json}: simulacion {simulacion.get('id', indice_simulacion)}: "
                         f"V e I con distinta longitud")

    vmp = float(_buscar_clave(simulacion["mpp"], "vmp"))
    id_original = int(simulacion.get("id", indice_simulacion))
    id_curva = indice_fichero * OFFSET_ID_DATOS + id_original

    # Orden creciente en tension y limpieza de la cola no fisica
    orden = np.argsort(V)
    V, I = V[orden], I[orden]
    if descartar_corriente_negativa:
        valida = (I >= 0.0) & (V >= 0.0)
        V, I = V[valida], I[valida]
    if V.size == 0:
        return None, False

    X_ventanas, _ = construir_ventanas_curva(
        V, I, n_muestras, delta_v, ascendente,
        n_ventanas=n_ventanas, modo_muestreo=modo_muestreo, rng=rng,
        paso_fijo=paso_fijo, ventana=ventana_ancho)

    if X_ventanas.shape[0] == 0:
        if verbose:
            sentido = "ascendente" if ascendente else "descendente"
            if paso_fijo:
                extension = f"{n_muestras} muestras x {delta_v:.2f} V"
            else:
                extension = f"{n_muestras} muestras dentro de {ventana_ancho:.2f} V"
            print(f"[datos] {ruta_json} curva {id_original}: rango insuficiente para "
                  f"una ventana de {extension} ({sentido}); se descarta.")
        return None, True

    y_vals = np.full(X_ventanas.shape[0], vmp, dtype=np.float32)
    grupos_vals = np.full(X_ventanas.shape[0], id_curva, dtype=np.int32)

    info_curva = None
    if devolver_curvas:
        info_curva = {
            "id": id_curva,
            "id_original": id_original,
            "id_dato": id_dato,
            "ruta_datos": ruta_json,
            "V": V,
            "I": I,
            "vmp": vmp,
            "pmax": float(_buscar_clave(simulacion["mpp"], "pmax")),
            "imp": float(_buscar_clave(simulacion["mpp"], "imp")),
            "irradiancias": simulacion.get("irradiancias_substrings_Wm2", []),
        }

    return (X_ventanas, y_vals, grupos_vals, info_curva), False


def _construir_info(rutas_json, ids_datos, metadatas, curvas, n_curvas,
                    curvas_descartadas, X, y, n_muestras, delta_v, ascendente,
                    paso_fijo, ventana_ancho, verbose):
    """Construye el diccionario 'info' de leer_datos_entrenamiento.

    Entrada: rutas_json, ids_datos, metadatas, curvas (listas de los ficheros
        leidos); n_curvas y curvas_descartadas (contadores); X, y (dataset);
        n_muestras, delta_v, ascendente, paso_fijo, ventana_ancho (ventana);
        verbose (imprime el resumen del dataset).
    Salida: dict 'info' (ver leer_datos_entrenamiento).
    Globales: ninguna.
    """
    V_columnas = X[:, 0::2]
    I_columnas = X[:, 1::2]

    info = {
        "rutas": rutas_json,
        "ids_datos": ids_datos,
        "metadata": metadatas,
        "curvas": curvas,
        "n_curvas": n_curvas,
        "n_curvas_descartadas": curvas_descartadas,
        "n_ejemplos": X.shape[0],
        "ventanas_por_curva": int(X.shape[0] / max(n_curvas, 1)),
        "ventana": {"n_muestras": n_muestras, "delta_v": delta_v, "ascendente": ascendente,
                   "paso_fijo": paso_fijo, "ventana": ventana_ancho},
        "rango_V": (float(V_columnas.min()), float(V_columnas.max())),
        "rango_I": (float(I_columnas.min()), float(I_columnas.max())),
        "rango_Vmp": (float(y.min()), float(y.max())),
    }

    if verbose:
        sentido = "ascendente" if ascendente else "descendente"
        if paso_fijo:
            descripcion_ventana = f"delta_V={delta_v:.2f} V"
        else:
            descripcion_ventana = f"ventana={ventana_ancho:.2f} V (tension aleatoria)"
        print(f"[datos] Ficheros           : {rutas_json}  (ids_datos={ids_datos})")
        print(f"[datos] Ventana P&O        : n={n_muestras}  {descripcion_ventana}  sentido={sentido}")
        print(f"[datos] Curvas leidas      : {info['n_curvas']}  (descartadas: {curvas_descartadas})")
        print(f"[datos] Ejemplos (ventanas): {info['n_ejemplos']}  "
              f"(~{info['ventanas_por_curva']} por curva)")
        print(f"[datos] Rango V   [V]      : {info['rango_V'][0]:.2f} .. {info['rango_V'][1]:.2f}")
        print(f"[datos] Rango I   [A]      : {info['rango_I'][0]:.2f} .. {info['rango_I'][1]:.2f}")
        print(f"[datos] Rango Vmp [V]      : {info['rango_Vmp'][0]:.2f} .. {info['rango_Vmp'][1]:.2f}")

    return info


def leer_datos_entrenamiento(rutas_json,
                             n_muestras: int = 5,
                             delta_v: float = 0.5,
                             ascendente: bool = True,
                             n_ventanas: int | None = None,
                             modo_muestreo: str = "uniforme",
                             descartar_corriente_negativa: bool = True,
                             semilla: int = 42,
                             devolver_curvas: bool = True,
                             ids_datos=None,
                             verbose: bool = True,
                             paso_fijo: bool = True,
                             ventana: float | None = None):
    """Lee uno o varios JSON de simulaciones I-V y construye el dataset de
    ventanas [V,I] -> Vmp, combinando todos los ficheros en un unico conjunto.

    Entrada:
        rutas_json: ruta o lista de rutas a JSON con 'metadata' y 'simulaciones'.
        n_muestras: pares [V_i, I_i] por ventana.
        delta_v: paso de tension [V] entre muestras (solo con paso_fijo=True).
        ascendente: sentido del barrido (True = V creciente).
        n_ventanas: ventanas tomadas por curva (None = todas las validas).
        modo_muestreo: 'uniforme' | 'aleatorio', reparto de esas ventanas.
        descartar_corriente_negativa: quita los puntos con I < 0 o V < 0.
        semilla: semilla del muestreo aleatorio.
        devolver_curvas: si True, info['curvas'] incluye las curvas completas.
        ids_datos: id de cada fichero (mismo orden y longitud que rutas_json),
            para trazabilidad; por defecto 0, 1, 2...
        verbose: imprime el resumen del dataset.
        paso_fijo: True = muestras a paso delta_v; False = a tension aleatoria
            dentro de una ventana de anchura 'ventana'.
        ventana: anchura [V]; obligatoria si paso_fijo=False.
    Salida: (X, y, grupos, info).
        X: (n_ejemplos, 2*n_muestras), ventanas [V_1,I_1,...,V_n,I_n].
        y: (n_ejemplos, 1), Vmp de la curva de origen.
        grupos: (n_ejemplos,), id global de la curva de origen
            (indice_fichero * OFFSET_ID_DATOS + id_original).
        info: dict con rutas, metadata (una entrada por fichero), curvas,
            contadores, ventana y rangos de V, I y Vmp.
        Lanza ValueError si rutas_json esta vacio, ids_datos no encaja o
        ninguna curva admite una ventana.
    Globales: ninguna.
    """
    if isinstance(rutas_json, (str, os.PathLike)):
        rutas_json = [rutas_json]
    rutas_json = list(rutas_json)
    if not rutas_json:
        raise ValueError("rutas_json no puede estar vacio")

    if ids_datos is None:
        ids_datos = list(range(len(rutas_json)))
    ids_datos = list(ids_datos)
    if len(ids_datos) != len(rutas_json):
        raise ValueError(f"ids_datos ({len(ids_datos)}) y rutas_json ({len(rutas_json)}) "
                         f"deben tener la misma longitud")

    rng = np.random.default_rng(semilla)
    lista_X, lista_y, lista_grupos, curvas = [], [], [], []
    curvas_descartadas = 0
    metadatas = []

    for indice_fichero, (ruta_json, id_dato) in enumerate(zip(rutas_json, ids_datos)):
        metadata, simulaciones = _leer_fichero_simulaciones(ruta_json)
        metadatas.append({"id_dato": id_dato, "ruta": ruta_json, **metadata})

        for indice_simulacion, simulacion in enumerate(simulaciones):
            resultado, contar_descartada = _construir_ejemplos_curva(
                simulacion, indice_fichero, id_dato, ruta_json, indice_simulacion,
                n_muestras, delta_v, ascendente, n_ventanas, modo_muestreo,
                descartar_corriente_negativa, rng, devolver_curvas, verbose,
                paso_fijo, ventana)

            if resultado is None:
                if contar_descartada:
                    curvas_descartadas += 1
                continue

            X_ventanas, y_vals, grupos_vals, info_curva = resultado
            lista_X.append(X_ventanas)
            lista_y.append(y_vals)
            lista_grupos.append(grupos_vals)
            if info_curva is not None:
                curvas.append(info_curva)

    if not lista_X:
        raise ValueError("Ninguna curva tiene rango suficiente para construir una ventana; "
                         "reduce n_muestras, delta_v o amplia la ventana")

    X = np.concatenate(lista_X).astype(np.float32)
    y = np.concatenate(lista_y).astype(np.float32).reshape(-1, 1)
    grupos = np.concatenate(lista_grupos)

    info = _construir_info(rutas_json, ids_datos, metadatas, curvas, len(lista_X),
                           curvas_descartadas, X, y, n_muestras, delta_v, ascendente,
                           paso_fijo, ventana, verbose)

    return X, y, grupos, info


def dividir_datos(X, y, grupos, fraccion_test=0.15, fraccion_val=0.15,
                  semilla=42, verbose=True):
    """Divide el dataset en train/val/test sin separar las ventanas de una curva.

    Entrada: X, y, grupos (salida de leer_datos_entrenamiento); fraccion_test y
        fraccion_val (fraccion de CURVAS para cada conjunto); semilla (baraja
        las curvas); verbose (imprime el tamano de cada conjunto).
    Salida: dict {'train', 'val', 'test'}, cada uno una tupla (X, y, grupos).
    Globales: ninguna.
    """
    ids = np.unique(grupos)
    rng = np.random.default_rng(semilla)
    rng.shuffle(ids)

    n = ids.size
    n_test = int(round(n * fraccion_test))
    n_val = int(round(n * fraccion_val))
    ids_test, ids_val, ids_train = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]

    def _subconjunto(ids_sel):
        """Devuelve (X, y, grupos) de las ventanas cuyas curvas estan en ids_sel."""
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


# ---------------------------------------------------------------------------
# Escritura / lectura del modelo TensorFlow Lite
# ---------------------------------------------------------------------------
def exportar_tflite(modelo, ruta_salida: str, cuantizar: bool = False,
                    datos_representativos=None, verbose: bool = True) -> str:
    """Convierte un modelo Keras a .tflite y lo guarda en disco.

    Si la conversion directa falla, la repite pasando por un SavedModel
    temporal (necesario en algunas versiones de Keras 3).

    Entrada: modelo Keras; ruta_salida (.tflite; se crea la carpeta si no
        existe); cuantizar (False = float32; True = optimizacion por defecto);
        datos_representativos (entradas reales para calibrar la cuantizacion
        entera, solo con cuantizar=True); verbose.
    Salida: ruta_salida.
    Globales: ninguna (escribe el fichero en ruta_salida).
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
    """Activa la cuantizacion en un convertidor TFLite.

    Entrada: convertidor (TFLiteConverter); datos_representativos (entradas
        reales, o None para cuantizar sin calibracion).
    Salida: ninguna; modifica el convertidor recibido.
    Globales: ninguna.
    """
    convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
    if datos_representativos is None:
        return

    muestras = np.asarray(datos_representativos, dtype=np.float32)

    def generador():
        """Entrega cada muestra representativa como entrada de un solo ejemplo."""
        for muestra in muestras:
            yield [muestra.reshape(1, -1)]

    convertidor.representative_dataset = generador


def cargar_modelo_tflite(ruta_tflite: str, verbose: bool = True):
    """Carga un .tflite y reserva sus tensores.

    Entrada: ruta_tflite; verbose (imprime ruta, entrada y salida).
    Salida: interprete listo para invocar. Lanza FileNotFoundError si el
        fichero no existe.
    Globales: ninguna.
    """
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
    """Guarda un diccionario como JSON (parametros y metricas de un modelo).

    Entrada: ruta_json (se crea la carpeta si no existe); resumen (dict
        serializable); verbose.
    Salida: ruta_json.
    Globales: ninguna (escribe el fichero en ruta_json).
    """
    os.makedirs(os.path.dirname(os.path.abspath(ruta_json)), exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as fichero:
        json.dump(resumen, fichero, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[tflite] Resumen guardado  : {ruta_json}")
    return ruta_json
