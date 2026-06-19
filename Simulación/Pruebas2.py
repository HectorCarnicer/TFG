#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:00:34 2026

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