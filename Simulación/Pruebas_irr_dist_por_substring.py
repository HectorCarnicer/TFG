#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:16:48 2026

@author: hectorc
"""

"""

Modelo personalizado de panel solar con pvlib
================================================

Este script construye un modelo eléctrico de panel solar a partir de los
datos típicos de un datasheet (Pmax, Voc, Isc, Vmp, Imp, coeficientes de
temperatura), usando el modelo de diodo único (Single Diode Model) ajustado
con el método CEC (California Energy Commission).

Requisitos:
    pip install pvlib matplotlib numpy pandas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pvlib
from pvlib.pvsystem import calcparams_cec, singlediode, v_from_i, i_from_v
from pvlib.ivtools.sdm import fit_cec_sam


# ---------------------------------------------------------------------------
# 1. DATOS DEL DATASHEET (edita estos valores con los de tu panel real)
# ---------------------------------------------------------------------------
# Condiciones estándar de medición (STC): 1000 W/m2, 25 °C, AM1.5
# Datos extraídos del datasheet: panel 65W, 12V nominal
datasheet = {
    "celltype": "multiSi",          # Ajusta a 'monoSi' si tu panel es monocristalino
    "v_mp": 17.6,                    # Voltaje en el punto de máxima potencia [V]
    "i_mp": 3.69,                    # Corriente en el punto de máxima potencia [A]
    "v_oc": 21.7,                    # Voltaje de circuito abierto [V]
    "i_sc": 3.99,                    # Corriente de cortocircuito [A]
    "alpha_sc": 0.00105 * 3.99,      # Coef. temp. Isc: +0.105%/°C * Isc -> A/°C
    "beta_voc": -0.00360 * 21.7,     # Coef. temp. Voc: -0.360%/°C * Voc -> V/°C
    "gamma_pmp": -0.45,              # Coef. temp. Pmax: -0.45%/°C (ya en %/°C)
    "cells_in_series": 36,           # Panel 12V nominal con Voc=21.7V -> típicamente 36 celdas
    "temp_ref": 25,                  # Temperatura de referencia [°C]
}


# ---------------------------------------------------------------------------
# 2. AJUSTE DEL MODELO DE DIODO ÚNICO (CEC)
# ---------------------------------------------------------------------------
# fit_cec_sam ajusta los 5 parámetros del circuito equivalente:
#   I_L_ref   -> corriente fotogenerada en STC
#   I_o_ref   -> corriente de saturación del diodo en STC
#   R_s       -> resistencia serie
#   R_sh_ref  -> resistencia paralelo (shunt) en STC
#   a_ref     -> producto n*Ns*k*T/q (factor de idealidad ajustado)
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

# Empaquetamos el modulo final como un diccionario reutilizable
modulo = {
    **datasheet,
    "I_L_ref": I_L_ref,
    "I_o_ref": I_o_ref,
    "R_s": R_s,
    "R_sh_ref": R_sh_ref,
    "a_ref": a_ref,
    "Adjust": Adjust,
}

print("=== Parámetros del modelo de diodo único (CEC) ===")
for k in ["I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref", "Adjust"]:
    print(f"  {k:10s} = {modulo[k]:.6g}")


# ---------------------------------------------------------------------------
# 3. FUNCIÓN PARA SIMULAR EL PANEL EN CONDICIONES ARBITRARIAS
# ---------------------------------------------------------------------------
def simular_panel(modulo, irradiancia, temp_celda):
    """
    Calcula la curva IV completa y el punto de máxima potencia del panel
    para una irradiancia [W/m2] y temperatura de celda [°C] dadas.

    Retorna un diccionario con los parámetros del circuito a esas
    condiciones y el resultado del punto de máxima potencia (IVCurveInfo).
    """
    # Recalcula los 5 parámetros del diodo a las condiciones reales
    IL, I0, Rs, Rsh, a = calcparams_cec(
        effective_irradiance=irradiancia,
        temp_cell=temp_celda,
        alpha_sc=modulo["alpha_sc"],
        a_ref=modulo["a_ref"],
        I_L_ref=modulo["I_L_ref"],
        I_o_ref=modulo["I_o_ref"],
        R_sh_ref=modulo["R_sh_ref"],
        R_s=modulo["R_s"],
        Adjust=modulo["Adjust"],
    )

    # Resuelve la ecuación del diodo único -> punto Pmax, Voc, Isc, etc.
    resultado = singlediode(IL, I0, Rs, Rsh, a, method="lambertw")

    return {
        "IL": IL, "I0": I0, "Rs": Rs, "Rsh": Rsh, "a": a,
        "resultado": resultado,
    }


def curva_iv(modulo, irradiancia, temp_celda, n_puntos=200):
    """Genera la curva IV completa (array de V y de I) para graficar."""
    sim = simular_panel(modulo, irradiancia, temp_celda)
    r = sim["resultado"]

    v_array = np.linspace(0, r["v_oc"], n_puntos)
    i_array = i_from_v(
        v_array, sim["IL"], sim["I0"], sim["Rs"], sim["Rsh"], sim["a"],
        method="lambertw",
    )
    i_array = np.clip(i_array, 0, None)  # evita corrientes negativas numéricas
    p_array = v_array * i_array
    return v_array, i_array, p_array, r


# ---------------------------------------------------------------------------
# 4. EJEMPLO DE USO: comparar el panel en distintas condiciones
# ---------------------------------------------------------------------------
condiciones = [
    {"irradiancia": 1000, "temp_celda": 25, "label": "STC (1000 W/m², 25°C)"},
    {"irradiancia": 800,  "temp_celda": 47, "label": "NOCT (800 W/m², 47°C)"},
    {"irradiancia": 200,  "temp_celda": 35, "label": "Baja irradiancia (200 W/m², 35°C)"},
]

resumen = []
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for c in condiciones:
    v, i, p, r = curva_iv(modulo, c["irradiancia"], c["temp_celda"])
    ax1.plot(v, i, label=c["label"])
    ax2.plot(v, p, label=c["label"])
    resumen.append({
        "Condición": c["label"],
        "Pmax [W]": round(float(r["p_mp"]), 2),
        "Vmp [V]": round(float(r["v_mp"]), 2),
        "Imp [A]": round(float(r["i_mp"]), 2),
        "Voc [V]": round(float(r["v_oc"]), 2),
        "Isc [A]": round(float(r["i_sc"]), 2),
    })

ax1.set_xlabel("Voltaje [V]")
ax1.set_ylabel("Corriente [A]")
ax1.set_title("Curva I-V")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Voltaje [V]")
ax2.set_ylabel("Potencia [W]")
ax2.set_title("Curva P-V")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("curvas_panel_solar.png", dpi=150)
print("\nGráfico guardado como 'curvas_panel_solar.png'")

print("\n=== Resumen de desempeño bajo distintas condiciones ===")
df_resumen = pd.DataFrame(resumen)
print(df_resumen.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. VALIDACIÓN: comparar el modelo ajustado contra los valores del datasheet
# ---------------------------------------------------------------------------
print("\n=== Validación del modelo vs datasheet ===")
validacion = pd.DataFrame({
    "Parámetro": ["Pmax [W]", "Vmp [V]", "Imp [A]", "Voc [V]", "Isc [A]"],
    "Datasheet (STC)": [65.0, 17.6, 3.69, 21.7, 3.99],
    "Modelo (STC)": [
        resumen[0]["Pmax [W]"], resumen[0]["Vmp [V]"], resumen[0]["Imp [A]"],
        resumen[0]["Voc [V]"], resumen[0]["Isc [A]"],
    ],
    "Datasheet (NOCT)": [46.8, 15.7, 2.95, 19.7, 3.23],
    "Modelo (NOCT)": [
        resumen[1]["Pmax [W]"], resumen[1]["Vmp [V]"], resumen[1]["Imp [A]"],
        resumen[1]["Voc [V]"], resumen[1]["Isc [A]"],
    ],
})
print(validacion.to_string(index=False))


# ---------------------------------------------------------------------------
# 6. SOMBREADO PARCIAL: irradiancia distinta por substring
# ---------------------------------------------------------------------------
# El panel tiene 36 celdas en serie repartidas en 3 substrings de 12 celdas,
# cada uno protegido por un diodo de bypass (configuración estándar en
# paneles de 12V nominal). Cada substring puede recibir una irradiancia
# distinta (p.ej. una fila de celdas tapada por una sombra parcial).
#
# Usamos pvlib.singlediode.bishop88_v_from_i / bishop88_i_from_v, que sí
# modelan correctamente la región de voltaje negativo donde actúa el diodo
# de bypass (a diferencia de i_from_v/v_from_i con método Lambert W, que
# asumen que no hay bypass).
from pvlib.singlediode import bishop88_v_from_i, bishop88_i_from_v

N_SUBSTRINGS = 3
CELLS_PER_SUBSTRING = datasheet["cells_in_series"] // N_SUBSTRINGS  # 12

def parametros_substring(modulo, irradiancia, temp_celda):
    """
    Recalcula IL, I0, Rs, Rsh, a para UN substring de CELLS_PER_SUBSTRING
    celdas, a partir de los parámetros ya ajustados para el panel completo.

    a_ref escala linealmente con el número de celdas en serie.
    Rs y Rsh (referidos al panel completo) escalan por la fracción de celdas.
    IL e I0 son por-celda en términos de la ecuación del diodo, así que no
    cambian con el número de celdas en serie (dependen del área/corriente).
    """
    fraccion = CELLS_PER_SUBSTRING / modulo["cells_in_series"]

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
    return IL, I0, Rs, Rsh, a


def curva_iv_substring(modulo, irradiancia, temp_celda, v_array):
    """Curva I(V) de un substring individual, incluyendo región de bypass.

    bishop88 evalúa directamente sobre un parámetro auxiliar; aquí lo usamos
    indirectamente generando la curva a partir de v_from_i / i_from_v con
    soporte para voltajes negativos mediante bishop88.
    """
    IL, I0, Rs, Rsh, a = parametros_substring(modulo, irradiancia, temp_celda)
    i_array = bishop88_i_from_v(v_array, IL, I0, Rs, Rsh, a)
    return i_array


def parametros_substring_y_isc(modulo, irradiancia, temp_celda):
    """Devuelve los 5 parámetros del diodo del substring + su propia Isc."""
    IL, I0, Rs, Rsh, a = parametros_substring(modulo, irradiancia, temp_celda)
    # Isc del substring: corriente cuando V=0. La calculamos resolviendo
    # bishop88 SOLO en el rango seguro (V=0), sin tocar voltajes negativos.
    isc_sub = float(bishop88_i_from_v(np.array([0.0]), IL, I0, Rs, Rsh, a)[0])
    return IL, I0, Rs, Rsh, a, isc_sub


def curva_iv_panel_sombreado(modulo, irradiancias_substrings, temp_celda,
                              v_bypass=-0.5, n_puntos=500):
    """
    Combina los substrings en serie para obtener la curva IV del panel
    completo bajo irradiancia no uniforme.

    irradiancias_substrings: lista con la irradiancia [W/m2] de cada uno
        de los N_SUBSTRINGS substrings, p.ej. [1000, 600, 1000].
    v_bypass: voltaje negativo al que conduce el diodo de bypass de cada
        substring (típico: -0.5 V, valor de un diodo de silicio en directa).

    Devuelve v_total, i_total, p_total (arrays) y el resumen de Pmax/Vmp/Imp.

    Nota técnica sobre el bypass:
    Cuando la corriente que circula por el panel (impuesta por los demás
    substrings, ya que todos comparten la misma corriente en serie) supera
    la Isc de ESTE substring, el substring por sí solo no puede sostener
    esa corriente en la región de polarización directa. En la realidad,
    el diodo de bypass en paralelo entra en conducción y fija el voltaje
    del substring en aprox. v_bypass (un valor negativo fijo), sin que el
    substring entre en breakdown inverso. Por eso, en vez de pedirle a
    bishop88_v_from_i que resuelva ese punto (lo que fuerza al solver hacia
    la rama de avalancha y genera NaN/warnings), evaluamos el bypass
    directamente como una condición lógica ANTES de llamar al solver.
    """
    assert len(irradiancias_substrings) == N_SUBSTRINGS

    # Rango de corriente común para "barrer" todas las curvas: usamos la
    # Isc más alta de los substrings como referencia. Es más robusto que
    # barrer voltaje, porque en serie la corriente es la misma para todos
    # los substrings (excepto cuando el bypass de alguno se activa).
    i_max_ref = max(irradiancias_substrings) / 1000 * modulo["i_sc"] * 1.05
    i_array = np.linspace(0, i_max_ref, n_puntos)

    v_substrings = np.zeros((N_SUBSTRINGS, n_puntos))

    for idx, irr in enumerate(irradiancias_substrings):
        IL, I0, Rs, Rsh, a, isc_sub = parametros_substring_y_isc(
            modulo, irr, temp_celda
        )

        # Máscara: puntos donde la corriente pedida SUPERA la Isc del
        # substring -> ahí el bypass conduce, fijamos v_bypass directamente
        # y NUNCA llamamos al solver de diodo en esa zona.
        activa_bypass = i_array > isc_sub

        v_sub = np.empty_like(i_array)
        # Zona normal (substring puede sostener la corriente él solo)
        if np.any(~activa_bypass):
            v_sub[~activa_bypass] = bishop88_v_from_i(
                i_array[~activa_bypass], IL, I0, Rs, Rsh, a
            )
        # Zona de bypass activo: voltaje fijo, sin resolver el diodo
        v_sub[activa_bypass] = v_bypass

        v_substrings[idx, :] = v_sub

    v_total = v_substrings.sum(axis=0)
    p_total = v_total * i_array

    idx_mpp = np.argmax(p_total)
    resumen_sombra = {
        "Pmax [W]": round(float(p_total[idx_mpp]), 2),
        "Vmp [V]": round(float(v_total[idx_mpp]), 2),
        "Imp [A]": round(float(i_array[idx_mpp]), 2),
    }

    return v_total, i_array, p_total, resumen_sombra, v_substrings


# --- Ejemplo: un substring (1 de 3) recibe sombra parcial ---
escenarios_sombra = [
    {"irr": [1000, 1000, 1000], "label": "Sin sombra (1000/1000/1000 W/m²)"},
    {"irr": [1000, 1000, 400],  "label": "1 substring sombreado (1000/1000/400 W/m²)"},
    {"irr": [1000, 600, 200],   "label": "Sombra progresiva (1000/600/200 W/m²)"},
]

TEMP_CELDA_SOMBRA = 35  # °C, asumido para este ejemplo

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
resumen_sombra_total = []

for esc in escenarios_sombra:
    v_t, i_t, p_t, res, v_subs = curva_iv_panel_sombreado(
        modulo, esc["irr"], TEMP_CELDA_SOMBRA
    )
    ax3.plot(v_t, i_t, label=esc["label"])
    ax4.plot(v_t, p_t, label=esc["label"])
    resumen_sombra_total.append({"Escenario": esc["label"], **res})

ax3.set_xlabel("Voltaje [V]")
ax3.set_ylabel("Corriente [A]")
ax3.set_title("Curva I-V del panel con sombreado parcial por substring")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.axvline(0, color="gray", linewidth=0.5)

ax4.set_xlabel("Voltaje [V]")
ax4.set_ylabel("Potencia [W]")
ax4.set_title("Curva P-V del panel con sombreado parcial por substring")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("curvas_panel_sombreado.png", dpi=150)
print("\nGráfico de sombreado parcial guardado como 'curvas_panel_sombreado.png'")

print("\n=== Resumen Pmax bajo sombreado parcial (irradiancia por substring) ===")
print(pd.DataFrame(resumen_sombra_total).to_string(index=False))
print(
    "\nNota: observa cómo en el escenario con sombra fuerte aparecen "
    "'escalones' en la curva P-V — son los puntos donde un diodo de bypass "
    "empieza a conducir y 'salta' el substring sombreado. Esto puede crear "
    "máximos locales de potencia, por lo que un MPPT real puede quedar "
    "atrapado en un óptimo local si no implementa un barrido global."
)