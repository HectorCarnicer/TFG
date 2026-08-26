"""
rn_main.py

@author: Héctor Carnicer Ull

Script principal de entrenamiento del predictor de MPP.

Flujo: lectura de datos -> particion train/val/test -> creacion del modelo
-> entrenamiento -> evaluacion -> exportacion a TFLite -> visualizacion.
"""

from __future__ import annotations

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from lectura_escritura import (leer_datos_entrenamiento, dividir_datos,
                               exportar_tflite, cargar_modelo_tflite,
                               guardar_resumen)
from funciones_modelo import (crear_modelo, entrenar_modelo, predecir,
                              predecir_tflite, evaluar, predecir_vmp_curva)

# --- PARAMETROS DE UTILIZACION ----------------------------------------------

# --- Datos -------------------------------------------------------------------
RUTA_DATOS = "sim_1.dat"              # JSON con metadata + simulaciones
N_PUNTOS_CURVA = None                 # puntos tomados de cada curva; None = todos
MODO_MUESTREO = "uniforme"            # 'uniforme' | 'aleatorio'
DESCARTAR_I_NEGATIVA = True           # elimina la cola posterior a Voc

# --- Particion del dataset -----------------------------------------------------
FRACCION_TEST = 0.15                  # fraccion de curvas para test
FRACCION_VALIDACION = 0.15            # fraccion de curvas para validacion
SEMILLA = 42                          # reproducibilidad (split + pesos)

# --- Arquitectura de la red ---------------------------------------------------
N_ENTRADAS = 2                        # [V, I], fijo por definicion del modelo
N_CAPAS = 5                           # numero de capas ocultas
N_NEURONAS = 64                       # neuronas por capa oculta
ACTIVACION = "sigmoid"
OPTIMIZADOR = "nadam"                 # nadam | adam | rmsprop | sgd | adamw
LEARNING_RATE = 1e-3
FUNCION_PERDIDA = "mse"

# --- Entrenamiento --------------------------------------------------------------
EPOCAS = 400
BATCH_SIZE = 256
PACIENCIA = 80                        # parada temprana; 0 = desactivada
REDUCIR_LR = True
VERBOSE_ENTRENAMIENTO = 1             # 0 silencioso | 1 barra | 2 una linea/epoca

# --- Estimacion de Vmp para una curva completa -----------------------------------
AGREGACION_CURVA = "mediana"          # 'mediana' | 'media'

# --- Modelo de salida -------------------------------------------------------------
ID_MODELO = "mpp_v1"                  # identificador del modelo generado
DIR_MODELOS = "modelos"               # carpeta de salida
CUANTIZAR = False                     # True -> .tflite cuantizado (int8)
VERIFICAR_TFLITE = True               # compara Keras vs TFLite tras exportar
GUARDAR_RESUMEN = True                # JSON con parametros y metricas

# --- Visualizacion ------------------------------------------------------------------
VISUALIZAR = True
MOSTRAR_FIGURAS = True                # plt.show() al final
GUARDAR_FIGURAS = True                # guarda los PNG en DIR_MODELOS
DIR_FIGURAS = "figuras"
ESCALA_LOG_PERDIDA = True             # eje Y logaritmico en la curva de perdida


def graficar_entrenamiento(historial, y_test, y_pred_test, ruta_png=None):
    """Figura 2x2 de perdida, MAE, dispersion prediccion-real e histograma del error.

    Devuelve la figura; si se indica ruta_png, la guarda tambien en disco.
    """
    hist = historial.history
    epocas = np.arange(1, len(hist["loss"]) + 1)

    figura, ejes = plt.subplots(2, 2, figsize=(13, 9))
    figura.suptitle(f"Entrenamiento del predictor de MPP  |  {ID_MODELO}",
                    fontsize=14, fontweight="bold")

    # Perdida
    ax = ejes[0, 0]
    ax.plot(epocas, hist["loss"], label="Entrenamiento", lw=1.6)
    if "val_loss" in hist:
        ax.plot(epocas, hist["val_loss"], label="Validacion", lw=1.6)
    if ESCALA_LOG_PERDIDA:
        ax.set_yscale("log")
    ax.set_xlabel("Epoca")
    ax.set_ylabel(f"Perdida ({FUNCION_PERDIDA.upper()})")
    ax.set_title("Evolucion de la perdida")
    ax.legend()
    ax.grid(alpha=0.3)

    # MAE
    ax = ejes[0, 1]
    ax.plot(epocas, hist["mae"], label="Entrenamiento", lw=1.6)
    if "val_mae" in hist:
        ax.plot(epocas, hist["val_mae"], label="Validacion", lw=1.6)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MAE [V]")
    ax.set_title("Error absoluto medio")
    ax.legend()
    ax.grid(alpha=0.3)

    # Prediccion frente a valor real (test)
    ax = ejes[1, 0]
    y_r = np.asarray(y_test).reshape(-1)
    y_p = np.asarray(y_pred_test).reshape(-1)
    ax.scatter(y_r, y_p, s=6, alpha=0.25, edgecolors="none")
    limites = [min(y_r.min(), y_p.min()), max(y_r.max(), y_p.max())]
    ax.plot(limites, limites, "r--", lw=1.4, label="Ideal (y = x)")
    ax.set_xlabel("Vmp real [V]")
    ax.set_ylabel("Vmp predicho [V]")
    ax.set_title("Prediccion vs. real (test)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Histograma del error
    ax = ejes[1, 1]
    error = y_p - y_r
    ax.hist(error, bins=60, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="r", ls="--", lw=1.4)
    ax.axvline(error.mean(), color="k", ls=":", lw=1.4,
               label=f"Media = {error.mean():.3f} V")
    ax.set_xlabel("Error de prediccion [V]")
    ax.set_ylabel("Numero de muestras")
    ax.set_title(f"Distribucion del error  (sigma = {error.std():.3f} V)")
    ax.legend()
    ax.grid(alpha=0.3)

    figura.tight_layout()
    if ruta_png:
        figura.savefig(ruta_png, dpi=140)
        print(f"[figura] Guardada          : {ruta_png}")
    return figura


def graficar_curva_ejemplo(modelo, curva, ruta_png=None):
    """Curvas I-V y P-V de una curva de test, con el Vmp real y el predicho.

    Devuelve la figura; si se indica ruta_png, la guarda tambien en disco.
    """
    V, I = curva["V"], curva["I"]
    P = V * I
    vmp_pred = predecir_vmp_curva(modelo, V, I, AGREGACION_CURVA)

    figura, ax_iv = plt.subplots(figsize=(9, 5.5))
    ax_iv.plot(V, I, color="tab:blue", lw=1.8, label="Curva I-V")
    ax_iv.set_xlabel("Tension [V]")
    ax_iv.set_ylabel("Corriente [A]", color="tab:blue")
    ax_iv.tick_params(axis="y", labelcolor="tab:blue")
    ax_iv.grid(alpha=0.3)

    ax_pv = ax_iv.twinx()
    ax_pv.plot(V, P, color="tab:orange", lw=1.8, label="Curva P-V")
    ax_pv.set_ylabel("Potencia [W]", color="tab:orange")
    ax_pv.tick_params(axis="y", labelcolor="tab:orange")

    ax_pv.axvline(curva["vmp"], color="green", ls="--", lw=1.6,
                  label=f"Vmp real = {curva['vmp']:.2f} V")
    ax_pv.axvline(vmp_pred, color="red", ls=":", lw=1.8,
                  label=f"Vmp predicho = {vmp_pred:.2f} V")

    lineas = ax_iv.get_lines() + ax_pv.get_lines()
    ax_iv.legend(lineas, [l.get_label() for l in lineas], loc="lower left", fontsize=9)
    irr = ", ".join(f"{g:.0f}" for g in curva.get("irradiancias", []))
    ax_iv.set_title(f"Curva de test id={curva['id']}  |  irradiancias: [{irr}] W/m2  "
                    f"|  error = {vmp_pred - curva['vmp']:+.2f} V")

    figura.tight_layout()
    if ruta_png:
        figura.savefig(ruta_png, dpi=140)
        print(f"[figura] Guardada          : {ruta_png}")
    return figura


def main():
    inicio = time.time()
    os.makedirs(DIR_MODELOS, exist_ok=True)
    ruta_tflite = os.path.join(DIR_MODELOS, f"{ID_MODELO}.tflite")

    # --- 1. Datos ------------------------------------------------------------
    print("\n=== 1. LECTURA DE DATOS ===")
    X, y, grupos, info = leer_datos_entrenamiento(
        RUTA_DATOS,
        n_puntos=N_PUNTOS_CURVA,
        modo_muestreo=MODO_MUESTREO,
        descartar_corriente_negativa=DESCARTAR_I_NEGATIVA,
        semilla=SEMILLA,
        devolver_curvas=True)

    particion = dividir_datos(X, y, grupos,
                              fraccion_test=FRACCION_TEST,
                              fraccion_val=FRACCION_VALIDACION,
                              semilla=SEMILLA)
    X_train, y_train, _ = particion["train"]
    X_val, y_val, _ = particion["val"]
    X_test, y_test, g_test = particion["test"]

    # --- 2. Modelo -------------------------------------------------------------
    print("\n=== 2. CREACION DEL MODELO ===")
    modelo = crear_modelo(n_entradas=N_ENTRADAS,
                          n_capas=N_CAPAS,
                          n_neuronas=N_NEURONAS,
                          activacion=ACTIVACION,
                          optimizador=OPTIMIZADOR,
                          learning_rate=LEARNING_RATE,
                          perdida=FUNCION_PERDIDA,
                          X_referencia=X_train,
                          y_referencia=y_train,
                          nombre=ID_MODELO)

    # --- 3. Entrenamiento -------------------------------------------------------
    print("\n=== 3. ENTRENAMIENTO ===")
    historial = entrenar_modelo(modelo, X_train, y_train, X_val, y_val,
                                epocas=EPOCAS,
                                batch_size=BATCH_SIZE,
                                paciencia=PACIENCIA,
                                reducir_lr=REDUCIR_LR,
                                semilla=SEMILLA,
                                verbose=VERBOSE_ENTRENAMIENTO)

    # --- 4. Evaluacion ---------------------------------------------------------
    print("\n=== 4. EVALUACION ===")
    y_pred_test = predecir(modelo, X_test)
    metricas_punto = evaluar(y_test, y_pred_test, "test (punto a punto)")

    # Metricas agregando las predicciones de todos los puntos de cada curva
    ids_test = set(np.unique(g_test).tolist())
    curvas_test = [c for c in info["curvas"] if c["id"] in ids_test]
    vmp_real = np.array([c["vmp"] for c in curvas_test])
    vmp_pred = np.array([predecir_vmp_curva(modelo, c["V"], c["I"], AGREGACION_CURVA)
                         for c in curvas_test])
    metricas_curva = evaluar(vmp_real, vmp_pred, f"test ({AGREGACION_CURVA} por curva)")

    # --- 5. Exportacion a TFLite -------------------------------------------------
    print("\n=== 5. EXPORTACION A TENSORFLOW LITE ===")
    representativos = X_train[np.random.default_rng(SEMILLA).choice(
        X_train.shape[0], size=min(500, X_train.shape[0]), replace=False)]
    exportar_tflite(modelo, ruta_tflite,
                    cuantizar=CUANTIZAR,
                    datos_representativos=representativos if CUANTIZAR else None)

    metricas_tflite = {}
    if VERIFICAR_TFLITE:
        interprete = cargar_modelo_tflite(ruta_tflite)
        y_pred_lite = predecir_tflite(interprete, X_test)
        metricas_tflite = evaluar(y_test, y_pred_lite, "test (modelo .tflite)")
        desviacion = np.max(np.abs(y_pred_lite - y_pred_test))
        print(f"[tflite] Desviacion max. respecto al modelo Keras: {desviacion:.6f} V")

    if GUARDAR_RESUMEN:
        guardar_resumen(os.path.join(DIR_MODELOS, f"{ID_MODELO}_resumen.json"), {
            "id_modelo": ID_MODELO,
            "datos": {"ruta": RUTA_DATOS,
                      "n_curvas": info["n_curvas"],
                      "n_muestras": info["n_muestras"],
                      "n_puntos_curva": N_PUNTOS_CURVA,
                      "modo_muestreo": MODO_MUESTREO,
                      "metadata": info["metadata"]},
            "arquitectura": {"n_entradas": N_ENTRADAS, "n_capas": N_CAPAS,
                             "n_neuronas": N_NEURONAS, "activacion": ACTIVACION,
                             "optimizador": OPTIMIZADOR, "learning_rate": LEARNING_RATE,
                             "perdida": FUNCION_PERDIDA},
            "entrenamiento": {"epocas_ejecutadas": len(historial.history["loss"]),
                              "batch_size": BATCH_SIZE, "paciencia": PACIENCIA,
                              "semilla": SEMILLA},
            "metricas": {"punto": metricas_punto,
                         "curva": metricas_curva,
                         "tflite": metricas_tflite},
            "cuantizado": CUANTIZAR,
        })

    # --- 6. Visualizacion --------------------------------------------------------
    if VISUALIZAR:
        print("\n=== 6. VISUALIZACION ===")
        directorio_figuras = os.path.join(DIR_MODELOS, DIR_FIGURAS)
        os.makedirs(directorio_figuras, exist_ok=True)
        ruta_1 = os.path.join(directorio_figuras, f"{ID_MODELO}_entrenamiento.png") if GUARDAR_FIGURAS else None
        ruta_2 = os.path.join(directorio_figuras, f"{ID_MODELO}_curva_ejemplo.png") if GUARDAR_FIGURAS else None

        graficar_entrenamiento(historial, y_test, y_pred_test, ruta_1)
        if curvas_test:
            graficar_curva_ejemplo(modelo, curvas_test[0], ruta_2)
        if MOSTRAR_FIGURAS:
            plt.show()

    print(f"\n=== FIN ({time.time() - inicio:.1f} s) ===")
    return modelo, historial


if __name__ == "__main__":
    main()
