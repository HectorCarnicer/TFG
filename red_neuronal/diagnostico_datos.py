"""
diagnostico_datos.py
====================
Evalua un modelo ya entrenado por rn_main.py sobre su propio conjunto de test.
Reconstruye los mismos datos, ventana y particion train/val/test con los que se
entreno (leidos del resumen del modelo) y ejecuta su .tflite sobre el test, con
metricas punto a punto y por curva y dos figuras. Detalles: README.md.

Uso:
    python diagnostico_datos.py            (usa ID)
    python diagnostico_datos.py 3          (diagnostica el modelo ID=3)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from lectura_escritura import leer_datos_entrenamiento, dividir_datos, cargar_modelo_tflite
from funciones_modelo import predecir_tflite, evaluar, predecir_vmp_curva

# --- Parametros --------------------------------------------------------------
ID = 4                        # ID del modelo (carpeta modelos/mpp_v{ID}) a diagnosticar
DIR_MODELOS = "modelos"       # misma carpeta raiz que usa rn_main.py
DIR_DATOS = "datos"           # carpeta con los ficheros sim_{ID}.dat
DIR_FIGURAS = "figuras"       # subcarpeta donde rn_main.py guarda sus figuras
IDS_DATOS = [1]               # fallback si el resumen no registra ids_datos
FRACCION_TEST = 0.15          # fallback si el resumen no registra la particion
FRACCION_VALIDACION = 0.15    # fallback si el resumen no registra la particion
DESCARTAR_I_NEGATIVA = True   # fallback si el resumen no registra este dato
SEMILLA = 42                  # fallback si el resumen no registra la semilla
AGREGACION_CURVA = "mediana"  # 'mediana' | 'media' de las predicciones de una curva
N_EJEMPLOS_CURVAS = 3         # curvas de test con mayor error que se dibujan
GUARDAR_FIGURAS = True
MOSTRAR_FIGURAS = True

_SIN_VALOR = object()  # centinela: distingue "clave ausente" de "clave con valor null"


def _con_fallback(diccionario: dict, clave: str, defecto, etiqueta: str, avisos: list):
    """Lee diccionario[clave]; si la clave no existe, devuelve 'defecto' y lo anota.

    Entrada: diccionario; clave; defecto (valor si falta); etiqueta (nombre que
        se anota); avisos (lista donde se anade "etiqueta=defecto").
    Salida: el valor de la clave (aunque sea None) o 'defecto' si no existe.
    Globales: ninguna (modifica la lista 'avisos' recibida).
    """
    valor = diccionario.get(clave, _SIN_VALOR)
    if valor is _SIN_VALOR:
        avisos.append(f"{etiqueta}={defecto!r}")
        return defecto
    return valor


def cargar_configuracion_modelo(dir_modelos: str, id_modelo: int, verbose: bool = True):
    """Localiza el modelo mpp_v{id_modelo} y lee de su resumen JSON la
    configuracion con la que se entreno.

    Si el resumen no registra algun campo, se usa el valor por defecto de este
    fichero (constantes de parametros) y se avisa por consola.

    Entrada: dir_modelos (carpeta raiz de los modelos); id_modelo (entero);
        verbose (imprime avisos y la ruta del modelo).
    Salida: (carpeta_modelo, config). config es un dict con 'ventana' (dict con
        n_muestras, delta_v, ascendente, paso_fijo y ventana), 'ids_datos',
        'n_ventanas_curva', 'modo_muestreo', 'descartar_i_negativa',
        'fraccion_test', 'fraccion_val' y 'semilla'. Lanza FileNotFoundError si
        no existe el modelo.
    Globales: lee IDS_DATOS, FRACCION_TEST, FRACCION_VALIDACION,
        DESCARTAR_I_NEGATIVA y SEMILLA (valores por defecto); no modifica ninguna.
    """
    nombre_modelo = f"mpp_v{id_modelo}"
    carpeta_modelo = os.path.join(dir_modelos, nombre_modelo)
    ruta_resumen = os.path.join(carpeta_modelo, f"{nombre_modelo}_resumen.json")
    ruta_tflite = os.path.join(carpeta_modelo, f"{nombre_modelo}.tflite")

    if not os.path.isfile(ruta_resumen) or not os.path.isfile(ruta_tflite):
        raise FileNotFoundError(
            f"No existe ningun modelo con ID={id_modelo} en '{carpeta_modelo}'. "
            f"Ejecuta primero rn_main.py con ID={id_modelo} antes de diagnosticar.")

    with open(ruta_resumen, "r", encoding="utf-8") as fichero:
        resumen = json.load(fichero)

    datos = resumen.get("datos", {})
    particion = resumen.get("particion", {})
    entrenamiento = resumen.get("entrenamiento", {})

    avisos = []
    ventana = dict(datos["ventana"])
    ventana["paso_fijo"] = _con_fallback(datos["ventana"], "paso_fijo", True,
                                         "paso_fijo", avisos)
    ventana["ventana"] = _con_fallback(datos["ventana"], "ventana", None,
                                       "ventana", avisos)

    config = {
        "ventana": ventana,
        "ids_datos": _con_fallback(datos, "ids_datos", list(IDS_DATOS), "ids_datos", avisos),
        "n_ventanas_curva": _con_fallback(datos, "n_ventanas_curva", None,
                                          "n_ventanas_curva", avisos),
        "modo_muestreo": _con_fallback(datos, "modo_muestreo", "uniforme",
                                       "modo_muestreo", avisos),
        "descartar_i_negativa": _con_fallback(datos, "descartar_corriente_negativa",
                                              DESCARTAR_I_NEGATIVA,
                                              "descartar_i_negativa", avisos),
        "fraccion_test": _con_fallback(particion, "fraccion_test", FRACCION_TEST,
                                       "fraccion_test", avisos),
        "fraccion_val": _con_fallback(particion, "fraccion_val", FRACCION_VALIDACION,
                                      "fraccion_val", avisos),
        "semilla": _con_fallback(entrenamiento, "semilla", SEMILLA, "semilla", avisos),
    }

    if avisos and verbose:
        print(f"[modelo] AVISO: el resumen de ID={id_modelo} no registra "
              f"{', '.join(avisos)}. La particion de test reconstruida podria "
              f"no coincidir exactamente con la del entrenamiento original.")

    if verbose:
        print(f"[modelo] ID={id_modelo}  ->  {ruta_tflite}")

    return carpeta_modelo, config


def evaluar_modelo(carpeta_modelo: str, id_modelo: int, config: dict,
                   dir_datos: str = DIR_DATOS, agregacion: str = AGREGACION_CURVA,
                   verbose: bool = True) -> dict:
    """Reconstruye el dataset y la particion originales del modelo y evalua su
    .tflite sobre el conjunto de test.

    Entrada: carpeta_modelo e id_modelo (ver cargar_configuracion_modelo);
        config (su salida); dir_datos (carpeta de los sim_{ID}.dat); agregacion
        ('mediana' | 'media', para la estimacion por curva); verbose.
    Salida: dict con 'id', parametros de la ventana, 'ids_datos', contadores de
        curvas, 'y_test' y 'y_pred_test' (punto a punto), 'curvas_test',
        'vmp_real' y 'vmp_pred' (por curva), 'metricas_punto' y
        'metricas_curva'. Lanza ValueError si la entrada del .tflite no
        coincide con la ventana del resumen.
    Globales: ninguna.
    """
    ventana = config["ventana"]
    n_muestras = ventana["n_muestras"]
    delta_v = ventana["delta_v"]
    ascendente = ventana["ascendente"]
    paso_fijo = ventana["paso_fijo"]
    ancho_ventana = ventana["ventana"]

    rutas_datos = [os.path.join(dir_datos, f"sim_{i}.dat") for i in config["ids_datos"]]
    if verbose:
        print(f"[datos] IDS_DATOS={config['ids_datos']}  ->  {rutas_datos}")

    X, y, grupos, info = leer_datos_entrenamiento(
        rutas_datos,
        n_muestras=n_muestras, delta_v=delta_v, ascendente=ascendente,
        n_ventanas=config["n_ventanas_curva"], modo_muestreo=config["modo_muestreo"],
        descartar_corriente_negativa=config["descartar_i_negativa"],
        semilla=config["semilla"], devolver_curvas=True,
        ids_datos=config["ids_datos"], verbose=verbose,
        paso_fijo=paso_fijo, ventana=ancho_ventana)

    particion = dividir_datos(X, y, grupos,
                              fraccion_test=config["fraccion_test"],
                              fraccion_val=config["fraccion_val"],
                              semilla=config["semilla"], verbose=verbose)
    X_test, y_test, g_test = particion["test"]

    nombre_modelo = f"mpp_v{id_modelo}"
    ruta_tflite = os.path.join(carpeta_modelo, f"{nombre_modelo}.tflite")
    interprete = cargar_modelo_tflite(ruta_tflite, verbose=verbose)

    entrada_esperada = 2 * n_muestras
    entrada_real = interprete.get_input_details()[0]["shape"][-1]
    if entrada_real != entrada_esperada:
        raise ValueError(
            f"El modelo ID={id_modelo} espera una entrada de {entrada_real} valores, pero "
            f"la ventana del resumen (n_muestras={n_muestras}) genera {entrada_esperada}. "
            f"El .tflite y el resumen no son coherentes entre si.")

    y_pred_test = predecir_tflite(interprete, X_test)
    metricas_punto = evaluar(y_test, y_pred_test, "test (punto a punto)", verbose=verbose)

    ids_test = set(np.unique(g_test).tolist())
    curvas_test = [c for c in info["curvas"] if c["id"] in ids_test]
    vmp_real = np.array([c["vmp"] for c in curvas_test])
    vmp_pred = np.array([predecir_vmp_curva(interprete, c["V"], c["I"],
                                            n_muestras, delta_v, ascendente,
                                            agregacion, tflite=True,
                                            paso_fijo=paso_fijo, ventana=ancho_ventana,
                                            semilla=config["semilla"])
                         for c in curvas_test])
    metricas_curva = evaluar(vmp_real, vmp_pred, f"test ({agregacion} por curva)", verbose=verbose)

    return {
        "id": id_modelo,
        "n_muestras": n_muestras, "delta_v": delta_v, "ascendente": ascendente,
        "paso_fijo": paso_fijo, "ventana": ancho_ventana,
        "ids_datos": config["ids_datos"],
        "n_curvas": info["n_curvas"], "n_curvas_descartadas": info["n_curvas_descartadas"],
        "y_test": y_test, "y_pred_test": y_pred_test,
        "curvas_test": curvas_test, "vmp_real": vmp_real, "vmp_pred": vmp_pred,
        "metricas_punto": metricas_punto, "metricas_curva": metricas_curva,
    }


def figura_evaluacion(y_test, y_pred_test, vmp_real, vmp_pred, ruta=None):
    """Dibuja dispersion prediccion-real e histograma del error, punto a punto
    (fila superior) y por curva (fila inferior).

    Entrada: y_test, y_pred_test (Vmp real y predicho por ventana); vmp_real,
        vmp_pred (por curva); ruta (si se indica, guarda la figura).
    Salida: la figura de matplotlib.
    Globales: crea una figura en el estado global de pyplot.
    """
    figura, ejes = plt.subplots(2, 2, figsize=(13, 9))

    def _panel(ax_disp, ax_hist, y_r, y_p, titulo):
        """Rellena un par de ejes (dispersion e histograma del error).

        Entrada: ax_disp, ax_hist (ejes); y_r, y_p (real y predicho); titulo.
        Salida: ninguna. Globales: ninguna.
        """
        y_r, y_p = np.asarray(y_r).reshape(-1), np.asarray(y_p).reshape(-1)
        ax_disp.scatter(y_r, y_p, s=6, alpha=0.3, edgecolors="none")
        limites = [min(y_r.min(), y_p.min()), max(y_r.max(), y_p.max())]
        ax_disp.plot(limites, limites, "r--", lw=1.4, label="Ideal (y = x)")
        ax_disp.set_xlabel("Vmp real [V]")
        ax_disp.set_ylabel("Vmp predicho [V]")
        ax_disp.set_title(f"Prediccion vs. real ({titulo})")
        ax_disp.legend()
        ax_disp.grid(alpha=0.3)

        error = y_p - y_r
        ax_hist.hist(error, bins=50, alpha=0.8, edgecolor="black", linewidth=0.4)
        ax_hist.axvline(0, color="r", ls="--", lw=1.4)
        ax_hist.axvline(error.mean(), color="k", ls=":", lw=1.4,
                        label=f"Media = {error.mean():.3f} V")
        ax_hist.set_xlabel("Error de prediccion [V]")
        ax_hist.set_ylabel("Numero de muestras")
        ax_hist.set_title(f"Distribucion del error ({titulo})  (sigma = {error.std():.3f} V)")
        ax_hist.legend()
        ax_hist.grid(alpha=0.3)

    _panel(ejes[0, 0], ejes[0, 1], y_test, y_pred_test, "punto a punto")
    _panel(ejes[1, 0], ejes[1, 1], vmp_real, vmp_pred, "por curva")

    figura.tight_layout()
    if ruta:
        figura.savefig(ruta, dpi=140)
        print(f"[figura] Guardada: {ruta}")
    return figura


def figura_curvas_ejemplo(curvas_test, vmp_pred, n_ejemplos=3, ruta=None):
    """Dibuja las curvas I-V y P-V de test con mayor error, con el Vmp real y el predicho.

    Entrada: curvas_test (lista de curvas); vmp_pred (Vmp estimado de cada una);
        n_ejemplos (cuantas curvas dibujar); ruta (si se indica, guarda la figura).
    Salida: la figura de matplotlib, o None si no hay curvas de test.
    Globales: crea una figura en el estado global de pyplot.
    """
    if not curvas_test:
        return None

    errores = np.abs(vmp_pred - np.array([c["vmp"] for c in curvas_test]))
    peores = np.argsort(-errores)[:n_ejemplos]

    figura, ejes = plt.subplots(1, len(peores), figsize=(6.5 * len(peores), 5), squeeze=False)
    for ax_iv, idx in zip(ejes[0], peores):
        curva = curvas_test[idx]
        V, I = np.asarray(curva["V"]), np.asarray(curva["I"])
        P = V * I
        vmp_p = vmp_pred[idx]

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
        ax_pv.axvline(vmp_p, color="red", ls=":", lw=1.8,
                      label=f"Vmp predicho = {vmp_p:.2f} V")

        lineas = ax_iv.get_lines() + ax_pv.get_lines()
        ax_iv.legend(lineas, [l.get_label() for l in lineas], loc="lower left", fontsize=8)
        ax_iv.set_title(f"curva id={curva['id']}  |  error = {vmp_p - curva['vmp']:+.2f} V")

    figura.tight_layout()
    if ruta:
        figura.savefig(ruta, dpi=140)
        print(f"[figura] Guardada: {ruta}")
    return figura


def main(id_modelo=ID):
    """Diagnostica un modelo: metricas por consola y figuras en su carpeta.

    Entrada: id_modelo (entero; por defecto ID).
    Salida: el dict de evaluar_modelo. Guarda las figuras en
        modelos/mpp_v{id_modelo}/figuras/ si GUARDAR_FIGURAS.
    Globales: lee ID, DIR_MODELOS, DIR_DATOS, DIR_FIGURAS, AGREGACION_CURVA,
        N_EJEMPLOS_CURVAS, GUARDAR_FIGURAS y MOSTRAR_FIGURAS; no modifica ninguna.
    """
    carpeta_modelo, config = cargar_configuracion_modelo(DIR_MODELOS, id_modelo)
    resultado = evaluar_modelo(carpeta_modelo, id_modelo, config, DIR_DATOS, AGREGACION_CURVA)

    vmp_real = resultado["vmp_real"]
    print("\n=== DISTRIBUCION DE Vmp (curvas de test) ===")
    print(f"media = {vmp_real.mean():.2f} V | std = {vmp_real.std():.2f} V | "
          f"min = {vmp_real.min():.2f} V | max = {vmp_real.max():.2f} V | "
          f"n = {vmp_real.size} curvas")

    nombre_modelo = f"mpp_v{id_modelo}"
    directorio = os.path.join(carpeta_modelo, DIR_FIGURAS)
    os.makedirs(directorio, exist_ok=True)
    figura_evaluacion(
        resultado["y_test"], resultado["y_pred_test"], vmp_real, resultado["vmp_pred"],
        os.path.join(directorio, f"{nombre_modelo}_diagnostico_evaluacion.png")
        if GUARDAR_FIGURAS else None)
    figura_curvas_ejemplo(
        resultado["curvas_test"], resultado["vmp_pred"], N_EJEMPLOS_CURVAS,
        os.path.join(directorio, f"{nombre_modelo}_diagnostico_curvas_ejemplo.png")
        if GUARDAR_FIGURAS else None)
    if MOSTRAR_FIGURAS:
        plt.show()

    return resultado


if __name__ == "__main__":
    _id = int(sys.argv[1]) if len(sys.argv) > 1 else ID
    main(_id)
