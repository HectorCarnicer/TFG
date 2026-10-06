"""
panel_funciones.py

@author: Héctor Carnicer Ull

Funciones de lectura, ajuste de modelo, simulación y visualización de un
panel solar bajo sombreado parcial (modelo de diodo único, ajuste CEC, PVlib).
"""

import json
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from pvlib.pvsystem import calcparams_cec
from pvlib.ivtools.sdm import fit_cec_sam
from pvlib.singlediode import bishop88_v_from_i, bishop88_i_from_v


# Carpeta por defecto de salida de graficar_curvas_iv_pv.
CARPETA_GRAFICOS = "Graficos"


# LECTURA DE DATOS DEL PANEL
def cargar_datasheet(id_datos, carpeta_datos="datos_panel"):
    """Carga la ficha de datos de un panel desde
    carpeta_datos/datos_panel_<id_datos>.json.
    Lanza FileNotFoundError si el fichero no existe. Devuelve un dict con
    los datos del panel, con los coeficientes de temperatura ya
    convertidos a A/°C y V/°C.
    """
    ruta_json = os.path.join(carpeta_datos, f"datos_panel_{id_datos}.json")

    if not os.path.exists(ruta_json):
        raise FileNotFoundError(
            f"No se encontró '{ruta_json}'. Comprueba que exista un fichero "
            f"'datos_panel_{id_datos}.json' dentro de la carpeta "
            f"'{carpeta_datos}/' (o ajusta ID_DATOS / CARPETA_DATOS)."
        )

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


# AJUSTE DEL MODELO DE DIODO ÚNICO (CEC)
def ajustar_modelo_cec(datasheet):
    """Ajusta el modelo de diodo único (CEC, vía fit_cec_sam) al datasheet
    del panel. Devuelve el mismo datasheet extendido con los parámetros
    ajustados (I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust).
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

# SIMULACIÓN CON SOMBREADO PARCIAL: irradiancia distinta por substring
def parametros_substring(modulo, irradiancia, temp_celda):
    """Calcula el circuito equivalente (IL, I0, Rs, Rsh, a) y la Isc de un
    substring a la irradiancia [W/m2] y temp_celda [°C] dadas, a partir
    del modelo ya ajustado del panel completo. Devuelve la tupla
    (IL, I0, Rs, Rsh, a, isc_sub).
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
    """Calcula la curva IV del panel completo bajo irradiancia no uniforme
    por substring (longitud modulo["n_substrings"]), combinando los
    substrings en serie con sus diodos de bypass. v_bypass es la tensión
    de conducción del diodo de bypass; si es None, se usa la de modulo.
    Devuelve (v_total, i_array, p_total, MPP, v_substrings): la curva
    combinada, el MPP (dict con Pmax, Vmp, Imp e índice) y la matriz de
    tensiones por substring.
    """
    n_substrings = modulo["n_substrings"]
    if v_bypass is None:
        v_bypass = modulo["v_bypass"]

    assert len(irradiancias_substrings) == n_substrings

    # Barrido en corriente (común a todos los substrings en serie).
    irr_max_idx = int(np.argmax(irradiancias_substrings))
    _, _, _, _, _, isc_max = parametros_substring(
        modulo, irradiancias_substrings[irr_max_idx], temp_celda
    )
    i_max_ref = isc_max * 1.05  # Margen de seguridad

    i_array = np.linspace(0, i_max_ref, n_puntos)

    v_substrings = np.zeros((n_substrings, n_puntos))

    for idx, irr in enumerate(irradiancias_substrings):
        IL, I0, Rs, Rsh, a, isc_sub = parametros_substring(
            modulo, irr, temp_celda
        )

        # True donde el bypass conduce (corriente pedida > Isc del substring).
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


# GENERACIÓN DE SIMULACIONES ALEATORIAS
def generar_simulaciones_aleatorias(modulo, n_simulaciones, irr_min=150,
                                     irr_max=1000, temp_celda=35,
                                     semilla=None, n_puntos=500):
    """Genera n_simulaciones curvas IV del panel, con irradiancia aleatoria
    uniforme e independiente por substring en [irr_min, irr_max] W/m2 y
    temp_celda fija para todas. semilla fija la reproducibilidad de
    `random`; None no la fija. Devuelve una lista de dicts, uno por
    simulación, con id, irradiancias_substrings_Wm2, temp_celda_C, mpp y
    curva (V, I, P como listas de Python).
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
    """Serializa simulaciones (salida de generar_simulaciones_aleatorias) a
    JSON en ruta_salida, junto con metadatos del panel y de la generación
    (rango de irradiancia, temperatura, semilla). Devuelve el dict
    serializado (metadata + simulaciones).
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
    """Carga un archivo de simulaciones generado por guardar_simulaciones.
    Devuelve un dict con las claves "metadata" y "simulaciones".
    """
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        return json.load(f)


# VISUALIZACIÓN: curvas I-V y P-V con MPP
def graficar_curvas_iv_pv(simulaciones, id_simulacion,
                           carpeta_graficos=CARPETA_GRAFICOS,
                           color_curvas="#cfd8e3", color_mpp="#ff1744",
                           dpi=150):
    """Genera la figura I-V / P-V (con MPP) de simulaciones y la guarda
    como PNG en carpeta_graficos (se crea si no existe), con el nombre
    curvas_sim_<id_simulacion>.png. Devuelve la ruta relativa del PNG
    generado.
    """
    os.makedirs(carpeta_graficos, exist_ok=True)

    fig, (ax_iv, ax_pv) = plt.subplots(1, 2, figsize=(13, 5.5))

    vmp_list, imp_list, pmax_list = [], [], []

    for sim in simulaciones:
        v = np.array(sim["curva"]["V"])
        i = np.array(sim["curva"]["I"])
        p = np.array(sim["curva"]["P"])
        mpp = sim["mpp"]

        # Curvas de fondo
        ax_iv.plot(v, i, color=color_curvas, linewidth=1, zorder=1)
        ax_pv.plot(v, p, color=color_curvas, linewidth=1, zorder=1)

        # MPP
        ax_iv.scatter(mpp["Vmp [V]"], mpp["Imp [A]"], color=color_mpp, s=18,
                      zorder=3, alpha=0.85)
        ax_pv.scatter(mpp["Vmp [V]"], mpp["Pmax [W]"], color=color_mpp, s=18,
                      zorder=3, alpha=0.85)

        vmp_list.append(mpp["Vmp [V]"])
        imp_list.append(mpp["Imp [A]"])
        pmax_list.append(mpp["Pmax [W]"])

    # Curva I-V
    ax_iv.set_xlabel("Voltaje [V]")
    ax_iv.set_ylabel("Corriente [A]")
    ax_iv.set_title(f"Curvas I-V ({len(simulaciones)} simulaciones)")
    ax_iv.legend(loc="upper right", fontsize=8)
    ax_iv.grid(True, alpha=0.3)

    # Curva P-V
    ax_pv.set_xlabel("Voltaje [V]")
    ax_pv.set_ylabel("Potencia [W]")
    ax_pv.set_title(f"Curvas P-V ({len(simulaciones)} simulaciones)")
    ax_pv.legend(loc="upper right", fontsize=8)
    ax_pv.grid(True, alpha=0.3)

    plt.tight_layout()
    nombre_figura = f"curvas_sim_{id_simulacion}.png"
    ruta_figura = os.path.join(carpeta_graficos, nombre_figura)
    plt.savefig(ruta_figura, dpi=dpi)

    return ruta_figura
