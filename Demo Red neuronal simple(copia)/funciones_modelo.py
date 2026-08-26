"""
funciones_modelo.py
===================
Creacion, entrenamiento, prediccion y evaluacion de la red que estima Vmp
a partir de un punto de operacion (V, I) del panel.

Arquitectura por defecto:
    (V, I) -> Normalizacion -> 5 x Dense(64, sigmoid) -> Dense(1)
           -> Desnormalizacion -> Vmp [V]

Normalizacion y desnormalizacion van dentro del modelo: el .tflite exportado
acepta voltios/amperios en crudo y devuelve voltios directamente.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

OPTIMIZADORES = {
    "nadam": tf.keras.optimizers.Nadam,
    "adam": tf.keras.optimizers.Adam,
    "rmsprop": tf.keras.optimizers.RMSprop,
    "sgd": tf.keras.optimizers.SGD,
    "adamw": tf.keras.optimizers.AdamW,
}


# --- Creacion del modelo --------------------------------------------------------
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

    X_referencia / y_referencia son los datos de entrenamiento usados para
    ajustar las capas de normalizacion/desnormalizacion; si no se pasan, la
    entrada/salida no se escalan.

    Devuelve el modelo compilado.
    """
    entradas = tf.keras.Input(shape=(n_entradas,), name="V_I")

    if X_referencia is not None:
        capa_norm = tf.keras.layers.Normalization(axis=-1, name="normalizacion")
        capa_norm.adapt(np.asarray(X_referencia, dtype=np.float32))
        x = capa_norm(entradas)
    else:
        x = entradas

    for i in range(n_capas):
        x = tf.keras.layers.Dense(n_neuronas, activation=activacion,
                                  name=f"oculta_{i + 1}")(x)

    salida = tf.keras.layers.Dense(1, activation="linear", name="vmp_norm")(x)

    if y_referencia is not None:
        y_ref = np.asarray(y_referencia, dtype=np.float32)
        media = float(y_ref.mean())
        sigma = float(y_ref.std()) or 1.0
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


# --- Entrenamiento --------------------------------------------------------------
def entrenar_modelo(modelo: tf.keras.Model,
                    X_train, y_train,
                    X_val=None, y_val=None,
                    epocas: int = 200,
                    batch_size: int = 256,
                    paciencia: int = 25,
                    reducir_lr: bool = True,
                    semilla: int = 42,
                    verbose: int = 1):
    """Entrena el modelo con early stopping y reduccion de learning rate opcionales.

    paciencia=0 desactiva el early stopping. Devuelve el historial de entrenamiento.
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


# --- Prediccion --------------------------------------------------------------
def predecir(modelo: tf.keras.Model, X, batch_size: int = 1024) -> np.ndarray:
    """Prediccion con el modelo Keras. Devuelve Vmp en voltios, shape (n, 1)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    return modelo.predict(X, batch_size=batch_size, verbose=0).reshape(-1, 1)


def predecir_tflite(interprete, X) -> np.ndarray:
    """Prediccion con un interprete TFLite ya cargado; admite lotes.

    Aplica automaticamente la cuantizacion de entrada/salida si el modelo
    esta cuantizado a enteros. Devuelve Vmp, shape (n, 1).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))

    detalles_entrada = interprete.get_input_details()[0]
    detalles_salida = interprete.get_output_details()[0]

    interprete.resize_tensor_input(detalles_entrada["index"], X.shape, strict=False)
    interprete.allocate_tensors()

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


# --- Evaluacion --------------------------------------------------------------
def evaluar(y_real, y_pred, etiqueta: str = "", verbose: bool = True) -> dict:
    """Calcula MAE, RMSE, MAPE, error maximo y R2. Devuelve un dict con las metricas."""
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


def predecir_vmp_curva(modelo, V, I, agregacion: str = "mediana",
                       tflite: bool = False) -> float:
    """Estima el Vmp de una curva completa evaluando el modelo en todos sus
    puntos y agregando las predicciones.

    agregacion : 'mediana' | 'media'.

    Devuelve el Vmp estimado.
    """
    funcion = predecir_tflite if tflite else predecir
    X = np.column_stack((np.asarray(V, dtype=np.float32),
                         np.asarray(I, dtype=np.float32)))
    predicciones = funcion(modelo, X).reshape(-1)
    return float(np.median(predicciones) if agregacion == "mediana"
                 else np.mean(predicciones))
