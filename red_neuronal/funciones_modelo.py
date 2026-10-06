"""
funciones_modelo.py
===================
Creacion, entrenamiento, prediccion y evaluacion de la red que estima la
tension de maxima potencia (Vmp) a partir de una ventana de n puntos [V, I]
(aplanada a 2*n valores).

Arquitectura por defecto:
    Entrada (V_1,I_1,...,V_n,I_n) -> Normalizacion -> 5 x Dense(64, sigmoid)
        -> Dense(1) -> Desnormalizacion -> Vmp [V]

Normalizacion y desnormalizacion van dentro del modelo: el .tflite acepta
voltios y amperios en crudo y devuelve voltios. Detalles: README.md.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from muestreo_ventanas import construir_ventanas_curva

OPTIMIZADORES = {
    "nadam": tf.keras.optimizers.Nadam,
    "adam": tf.keras.optimizers.Adam,
    "rmsprop": tf.keras.optimizers.RMSprop,
    "sgd": tf.keras.optimizers.SGD,
    "adamw": tf.keras.optimizers.AdamW,
}


# ---------------------------------------------------------------------------
# Creacion del modelo
# ---------------------------------------------------------------------------
def crear_modelo(n_entradas: int = 2,
                 n_capas: int = 5,
                 n_neuronas: int = 64,
                 activacion: str = "sigmoid",
                 optimizador: str = "nadam",
                 learning_rate: float = 1e-3,
                 perdida: str = "mse",
                 X_referencia: np.ndarray | None = None,
                 y_referencia: np.ndarray | None = None,
                 nombre: str = "predictor_mpp",
                 verbose: bool = True) -> tf.keras.Model:
    """Construye y compila la red densa.

    Entrada: n_entradas (2 * n_muestras); n_capas y n_neuronas (capas ocultas);
        activacion; optimizador (clave de OPTIMIZADORES); learning_rate;
        perdida; X_referencia / y_referencia (datos de ENTRENAMIENTO que fijan
        media y desviacion de las capas de normalizacion y desnormalizacion; si
        son None, esa capa no se anade); nombre del modelo; verbose (imprime
        el resumen).
    Salida: modelo Keras compilado (metrica MAE). Lanza ValueError si el
        optimizador no existe.
    Globales: ninguna (solo lee OPTIMIZADORES).
    """
    entradas = tf.keras.Input(shape=(n_entradas,), name="ventana_V_I")

    # --- Normalizacion de la entrada (x - mu) / sigma ----------------------
    if X_referencia is not None:
        capa_norm = tf.keras.layers.Normalization(axis=-1, name="normalizacion")
        capa_norm.adapt(np.asarray(X_referencia, dtype=np.float32))
        x = capa_norm(entradas)
    else:
        x = entradas

    # --- Capas ocultas ----------------------------------------------------
    for i in range(n_capas):
        x = tf.keras.layers.Dense(n_neuronas, activation=activacion,
                                  name=f"oculta_{i + 1}")(x)

    # --- Salida -----------------------------------------------------------
    salida = tf.keras.layers.Dense(1, activation="linear", name="vmp_norm")(x)

    if y_referencia is not None:
        y_ref = np.asarray(y_referencia, dtype=np.float32)
        media = float(y_ref.mean())
        sigma = float(y_ref.std()) or 1.0
        # Rescaling: y = x * sigma + media  (inversa de la normalizacion)
        salida = tf.keras.layers.Rescaling(scale=sigma, offset=media,
                                           name="vmp")(salida)

    modelo = tf.keras.Model(entradas, salida, name=nombre)

    clave = optimizador.lower()
    if clave not in OPTIMIZADORES:
        raise ValueError(f"Optimizador no soportado: {optimizador!r}. "
                         f"Opciones: {list(OPTIMIZADORES)}")
    modelo.compile(optimizer=OPTIMIZADORES[clave](learning_rate=learning_rate),
                   loss=perdida,
                   metrics=["mae"])

    if verbose:
        modelo.summary()
        print(f"[modelo] {n_capas} capas x {n_neuronas} neuronas | "
              f"activacion={activacion} | optimizador={optimizador} | lr={learning_rate}")

    return modelo


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def entrenar_modelo(modelo: tf.keras.Model,
                    X_train, y_train,
                    X_val=None, y_val=None,
                    epocas: int = 200,
                    batch_size: int = 256,
                    paciencia: int = 25,
                    reducir_lr: bool = True,
                    semilla: int = 42,
                    verbose: int = 1):
    """Entrena el modelo con parada temprana y reduccion del learning rate.

    Entrada: modelo compilado; X_train, y_train; X_val, y_val (opcionales, si
        faltan se monitoriza la perdida de entrenamiento); epocas maximas;
        batch_size; paciencia (epocas sin mejora antes de parar, 0 = sin
        parada temprana; con parada se restauran los mejores pesos);
        reducir_lr (ReduceLROnPlateau, factor 0.5); semilla; verbose (0, 1, 2).
    Salida: historial de Keras (History); los pesos de 'modelo' quedan entrenados.
    Globales: fija la semilla global de Python, NumPy y TensorFlow.
    """
    tf.keras.utils.set_random_seed(semilla)

    callbacks = []
    monitor = "val_loss" if X_val is not None else "loss"

    if paciencia and paciencia > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=paciencia,
            restore_best_weights=True, verbose=verbose))

    if reducir_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5,
            patience=max(paciencia // 3, 5), min_lr=1e-6, verbose=verbose))

    datos_val = (X_val, y_val) if X_val is not None else None

    historial = modelo.fit(X_train, y_train,
                           validation_data=datos_val,
                           epochs=epocas,
                           batch_size=batch_size,
                           callbacks=callbacks,
                           shuffle=True,
                           verbose=verbose)
    return historial


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------
def predecir(modelo: tf.keras.Model, X, batch_size: int = 1024) -> np.ndarray:
    """Predice Vmp con el modelo Keras.

    Entrada: modelo; X (ventanas, shape (n, 2*n_muestras) o una sola ventana);
        batch_size.
    Salida: array (n, 1) con Vmp en voltios.
    Globales: ninguna.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    return modelo.predict(X, batch_size=batch_size, verbose=0).reshape(-1, 1)


def predecir_tflite(interprete, X) -> np.ndarray:
    """Predice Vmp con un interprete TFLite ya cargado.

    Redimensiona el tensor de entrada al numero de ventanas y, si el modelo
    esta cuantizado a enteros, aplica escala y punto cero a entrada y salida.

    Entrada: interprete (ver cargar_modelo_tflite); X (ventanas, (n, 2*n_muestras)).
    Salida: array (n, 1) con Vmp en voltios.
    Globales: ninguna (cambia el tamano de entrada del interprete recibido).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))

    detalles_entrada = interprete.get_input_details()[0]
    detalles_salida = interprete.get_output_details()[0]

    interprete.resize_tensor_input(detalles_entrada["index"], X.shape, strict=False)
    interprete.allocate_tensors()

    # Cuantizacion de la entrada si procede
    entrada = X
    escala, cero = detalles_entrada["quantization"]
    if detalles_entrada["dtype"] != np.float32 and escala != 0:
        entrada = (X / escala + cero).astype(detalles_entrada["dtype"])

    interprete.set_tensor(detalles_entrada["index"], entrada)
    interprete.invoke()
    salida = interprete.get_tensor(detalles_salida["index"]).astype(np.float32)

    escala_s, cero_s = detalles_salida["quantization"]
    if detalles_salida["dtype"] != np.float32 and escala_s != 0:
        salida = (salida - cero_s) * escala_s

    return salida.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
def evaluar(y_real, y_pred, etiqueta: str = "", verbose: bool = True) -> dict:
    """Calcula MAE, RMSE, MAPE, error maximo y R2 entre prediccion y valor real.

    Entrada: y_real, y_pred (misma longitud, en voltios); etiqueta (texto de la
        linea impresa); verbose (imprime las metricas).
    Salida: dict con 'MAE [V]', 'RMSE [V]', 'MAPE [%]', 'Error max [V]' y 'R2'
        (nan si y_real es constante).
    Globales: ninguna.
    """
    y_real = np.asarray(y_real, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    error = y_pred - y_real

    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_real - y_real.mean()) ** 2))

    metricas = {
        "MAE [V]": float(np.mean(np.abs(error))),
        "RMSE [V]": float(np.sqrt(np.mean(error ** 2))),
        "MAPE [%]": float(np.mean(np.abs(error / np.maximum(np.abs(y_real), 1e-9))) * 100),
        "Error max [V]": float(np.max(np.abs(error))),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }

    if verbose:
        cabecera = f"[eval] {etiqueta}" if etiqueta else "[eval]"
        print(cabecera + "  " + " | ".join(f"{k}: {v:.4f}" for k, v in metricas.items()))

    return metricas


def predecir_vmp_curva(modelo, V, I, n_muestras: int, delta_v: float,
                       ascendente: bool, agregacion: str = "mediana",
                       tflite: bool = False, paso_fijo: bool = True,
                       ventana: float | None = None,
                       semilla: int | None = None) -> float:
    """Estima el Vmp de una curva I-V completa agregando la prediccion de
    todas sus ventanas validas.

    Entrada: modelo (Keras, o interprete TFLite si tflite=True); V, I (curva);
        n_muestras, delta_v, ascendente, paso_fijo, ventana (definicion de la
        ventana, igual que en entrenamiento); agregacion ('mediana' | 'media');
        semilla (fija las tensiones aleatorias con paso_fijo=False; sin ella
        cada llamada sortea ventanas distintas; se ignora con paso_fijo=True).
    Salida: Vmp estimado [V] (float). Lanza ValueError si la curva no tiene
        rango para ninguna ventana.
    Globales: ninguna.
    """
    funcion = predecir_tflite if tflite else predecir

    V = np.asarray(V, dtype=np.float32)
    I = np.asarray(I, dtype=np.float32)
    orden = np.argsort(V)
    V, I = V[orden], I[orden]

    rng = np.random.default_rng(semilla) if semilla is not None else None
    X, _ = construir_ventanas_curva(V, I, n_muestras, delta_v, ascendente,
                                    n_ventanas=None, modo_muestreo="uniforme",
                                    rng=rng, paso_fijo=paso_fijo, ventana=ventana)
    if X.shape[0] == 0:
        sentido = "ascendente" if ascendente else "descendente"
        extension = (f"{n_muestras} muestras x {delta_v:.2f} V" if paso_fijo
                    else f"{n_muestras} muestras dentro de {ventana:.2f} V")
        raise ValueError(
            f"La curva no tiene rango suficiente para una ventana de "
            f"{extension} ({sentido}).")

    predicciones = funcion(modelo, X).reshape(-1)
    return float(np.median(predicciones) if agregacion == "mediana"
                 else np.mean(predicciones))
