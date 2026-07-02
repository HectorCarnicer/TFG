"""
panel_funciones.py
==================

@author: Héctor Carnicer Ull

Funciones de lectura y simulación para un modelo de panel solar con PVlib,
usando el modelo de diodo único ajustado por el método CEC.

Incluye:
    - Lectura de los datos del panel desde un archivo JSON.
    - Ajuste del modelo de diodo único (fit_cec_sam).
    - Simulación de la curva IV del panel con irradiancia distinta por
      substring (sombreado parcial), incluyendo el efecto de los diodos
      de bypass.Cálculo del punto de máxima potencia (MPP).
    - Generación de N simulaciones aleatorias de sombreado parcial.
"""

import json
import random

import numpy as np

from pvlib.pvsystem import calcparams_cec
from pvlib.ivtools.sdm import fit_cec_sam
from pvlib.singlediode import bishop88_v_from_i, bishop88_i_from_v


# ---------------------------------------------------------------------------
# LECTURA DE DATOS DEL PANEL
# ---------------------------------------------------------------------------
def cargar_datasheet(ruta_json):
    """
    Parámetros:
    ----------
    ruta_json : str
        Ruta al archivo JSON con los datos del panel (ver panel_datos.json).

    Devuelve:
    -------
    dict con las claves: celltype, v_mp, i_mp, v_oc, i_sc, alpha_sc,
    beta_voc, gamma_pmp, cells_in_series, temp_ref, n_substrings,
    v_bypass, datasheet_referencia.
    """
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)

    datasheet = {
        "celltype": datos["celltype"],
        "v_mp": datos["v_mp"],
        "i_mp": datos["i_mp"],
        "v_oc": datos["v_oc"],
        "i_sc": datos["i_sc"],
        # Conversión de %/°C a A/°C y V/°C
        "alpha_sc": (datos["alpha_sc_pct"] / 100) * datos["i_sc"],
        "beta_voc": (datos["beta_voc_pct"] / 100) * datos["v_oc"],
        "gamma_pmp": datos["gamma_pmp"],
        "cells_in_series": datos["cells_in_series"],
        "temp_ref": datos["temp_ref"],
        "n_substrings": datos["n_substrings"],
        "v_bypass": datos["v_bypass"],
        "datasheet_referencia": datos.get("datasheet_referencia", {}),
    }
    return datasheet


# ---------------------------------------------------------------------------
# AJUSTE DEL MODELO DE DIODO ÚNICO (CEC)
# ---------------------------------------------------------------------------
def ajustar_modelo_cec(datasheet):
    """
    Ajusta los 5 parámetros del circuito equivalente de diodo único de PVlib
    (I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref) a partir de los datos
    del datasheet, usando el método CEC (función fit_cec_sam de PVlib).

    Parámetros:
    ----------
    datasheet : dict
        Diccionario obtenido de `cargar_datasheet`.

    Devuelve:
    -------
        El mismo `datasheet` extendido con los valores de: I_L_ref, I_o_ref,
        R_s, R_sh_ref, a_ref, Adjust. 
    """
    I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = fit_cec_sam(
        celltype=datasheet["celltype"],
        v_mp=datasheet["v_mp"],
        i_mp=datasheet["i_mp"],
        v_oc=datasheet["v_oc"],
        i_sc=datasheet["i_sc"],
        alpha_sc=datasheet["alpha_sc"],
        beta_voc=datasheet["beta_voc"],
        gamma_pmp=datasheet["gamma_pmp"],
        cells_in_series=datasheet["cells_in_series"],
        temp_ref=datasheet["temp_ref"],
    )

    modulo = {
        **datasheet,
        "I_L_ref": I_L_ref,
        "I_o_ref": I_o_ref,
        "R_s": R_s,
        "R_sh_ref": R_sh_ref,
        "a_ref": a_ref,
        "Adjust": Adjust,
    }
    return modulo

# ---------------------------------------------------------------------------
# SIMULACIÓN CON SOMBREADO PARCIAL: irradiancia distinta por substring
# ---------------------------------------------------------------------------
def parametros_substring(modulo, irradiancia, temp_celda):
    """
    Calcula IL, I0, Rs, Rsh e Isc a para un substring del panel (con su fracción de
    celdas correspondiente) a partir de los parámetros ya ajustados para
    el panel completo. En PVlib hay que suar las funciones del modelo
    de bishop88. En este caso i_from_v para la corriente de saturación.

    a_ref escala linealmente con el número de celdas en serie.
    La resistencia en serie y la resistencia Shunt escalan por la fracción de celdas.
    IL e I0 no cambian con el número de celdas en serie.
    """
    cells_per_substring = modulo["cells_in_series"] // modulo["n_substrings"]
    fraccion = cells_per_substring / modulo["cells_in_series"]

    IL, I0, Rs, Rsh, a = calcparams_cec(
        effective_irradiance=irradiancia,
        temp_cell=temp_celda,
        alpha_sc=modulo["alpha_sc"],
        a_ref=modulo["a_ref"] * fraccion,
        I_L_ref=modulo["I_L_ref"],
        I_o_ref=modulo["I_o_ref"],
        R_sh_ref=modulo["R_sh_ref"] * fraccion,
        R_s=modulo["R_s"] * fraccion,
        Adjust=modulo["Adjust"],
    )
    
    # Isc del substring: corriente cuando V=0.
    
    isc_sub = float(bishop88_i_from_v(np.array([0.0]), IL, I0, Rs, Rsh, a)[0])
    
    return IL, I0, Rs, Rsh, a, isc_sub


def curva_iv_panel_sombreado(modulo, irradiancias_substrings, temp_celda,
                              v_bypass=None, n_puntos=500):
    """
    Combina los substrings en serie para obtener la curva IV del panel
    completo bajo irradiancia no uniforme (sombreado parcial) usando el
    modelo bishop para diodos bypass

    Parámetros:
    - irradiancias_substrings:
        Irradiancia [W/m2] de cada uno de los `modulo["n_substrings"]`
        substrings, p.ej. [1000, 600, 1000]
    - v_bypass:
        Voltaje negativo al que conduce el diodo de bypass de cada
        substring. Si es None, se usa el valor definido en `modulo`
        (cargado originalmente desde el JSON)

    Devuelve:
    - v_total, i_array, p_total : 
        Curva IV combinada del panel completo
    - MPP : 
        {"Pmax [W]", "Vmp [V]", "Imp [A]"} en el punto de máxima potencia.
    - substrings : 
        Matriz (n_substrings x n_puntos) con el voltaje de cada substring.
    """
    n_substrings = modulo["n_substrings"]
    if v_bypass is None:
        v_bypass = modulo["v_bypass"]

    assert len(irradiancias_substrings) == n_substrings

    # Rango de corriente común para "barrer" todas las curvas: la corriente
    # es la misma para todos los substrings en serie (excepto cuando el
    # bypass de alguno se activa), así que barremos corriente y resolvemos
    # voltaje, en vez de al revés.
    i_max_ref = max(irradiancias_substrings) / 1000 * modulo["i_sc"] * 1.05
    i_array = np.linspace(0, i_max_ref, n_puntos)

    v_substrings = np.zeros((n_substrings, n_puntos))

    for idx, irr in enumerate(irradiancias_substrings):
        IL, I0, Rs, Rsh, a, isc_sub = parametros_substring(
            modulo, irr, temp_celda
        )

        # Máscara: puntos donde la corriente pedida SUPERA la Isc del
        # substring -> ahí el bypass conduce, fijamos v_bypass directamente
        # y nunca se llama al solver de diodo en esa zona.
        activa_bypass = i_array > isc_sub

        v_sub = np.empty_like(i_array)
        if np.any(~activa_bypass):
            v_sub[~activa_bypass] = bishop88_v_from_i(
                i_array[~activa_bypass], IL, I0, Rs, Rsh, a
            )
        v_sub[activa_bypass] = v_bypass

        v_substrings[idx, :] = v_sub

    v_total = v_substrings.sum(axis=0)
    p_total = v_total * i_array

    idx_mpp = np.argmax(p_total)
    MPP = {
        "Pmax [W]": round(float(p_total[idx_mpp]), 2),
        "Vmp [V]": round(float(v_total[idx_mpp]), 2),
        "Imp [A]": round(float(i_array[idx_mpp]), 2),
        "Índice MPP": int(idx_mpp)
    }

    return v_total, i_array, p_total, MPP, v_substrings


# ---------------------------------------------------------------------------
# GENERACIÓN DE SIMULACIONES ALEATORIAS
# ---------------------------------------------------------------------------
def generar_simulaciones_aleatorias(modulo, n_simulaciones, irr_min=150,
                                     irr_max=1000, temp_celda=35,
                                     semilla=None, n_puntos=500):
    """
    Genera N simulaciones de la curva IV del panel, donde cada substring
    recibe una irradiancia aleatoria uniforme e independiente en el
    rango [irr_min, irr_max] W/m².

    Parámetros:
    -modulo : 
        Modelo del panel ya ajustado (ver `ajustar_modelo_cec`).
    -n_simulaciones : 
        Número de curvas a generar.
    -irr_min, irr_max : 
        Rango de la distribución uniforme de irradiancia por substring [W/m2].
    -temp_celda : float
        Temperatura de celda usada en todas las simulaciones [°C].
    -semilla : 
        Semilla para `random`, para reproducibilidad. None = sin fijar.
    -n_puntos :
        Número de puntos de la curva IV de cada simulación.

    Devuelve:
    list[dict]
        Una entrada por simulación, con las claves: id,
        irradiancias_substrings_Wm2, temp_celda_C, mpp, curva
        (curva["V"], curva["I"], curva["P"] como listas de Python).
        El campo MPP:
        {"Pmax [W]", "Vmp [V]", "Imp [A]", "idx"}.
    """
    if semilla is not None:
        random.seed(semilla)

    n_substrings = modulo["n_substrings"]
    simulaciones = []

    for n in range(n_simulaciones):
        irr_substrings = [
            round(random.uniform(irr_min, irr_max), 2)
            for _ in range(n_substrings)
        ]

        v_t, i_t, p_t, MPP, _ = curva_iv_panel_sombreado(
            modulo, irr_substrings, temp_celda, n_puntos=n_puntos
        )

        simulaciones.append({
            "id": n,
            "irradiancias_substrings_Wm2": irr_substrings,
            "temp_celda_C": temp_celda,
            "mpp": MPP,
            "curva": {
                "V": v_t.round(5).tolist(),
                "I": i_t.round(5).tolist(),
                "P": p_t.round(5).tolist(),
            },
        })

    return simulaciones


def guardar_simulaciones(simulaciones, modulo, ruta_salida,
                          irr_min=150, irr_max=1000, temp_celda=35,
                          semilla=None):
    """
    Serializa una lista de simulaciones (generada por
    `generar_simulaciones_aleatorias`) a un archivo JSON, junto con
    metadatos del panel y de la generación.

    Parameters
    ----------
    simulaciones : list[dict]
        Salida de `generar_simulaciones_aleatorias`.
    modulo : dict
        Modelo del panel ya ajustado, usado para extraer metadatos.
    ruta_salida : str
        Ruta del archivo JSON a escribir (p.ej. "sim1.dat").
    """
    salida = {
        "metadata": {
            "n_simulaciones": len(simulaciones),
            "n_substrings": modulo["n_substrings"],
            "cells_per_substring": modulo["cells_in_series"] // modulo["n_substrings"],
            "rango_irradiancia_Wm2": [irr_min, irr_max],
            "temp_celda_C": temp_celda,
            "panel": {
                "v_oc_stc": modulo["v_oc"],
                "i_sc_stc": modulo["i_sc"],
                "v_mp_stc": modulo["v_mp"],
                "i_mp_stc": modulo["i_mp"],
            },
            "semilla_aleatoria": semilla,
        },
        "simulaciones": simulaciones,
    }

    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    return salida


def cargar_simulaciones(ruta_archivo):
    """
    Recarga un archivo de simulaciones generado por `guardar_simulaciones`.

    Returns
    -------
    dict con las claves "metadata" y "simulaciones".
    """
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        return json.load(f)
