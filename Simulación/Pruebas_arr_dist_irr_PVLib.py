#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:50:18 2026

@author: hectorc
"""

import numpy as np
import matplotlib.pyplot as plt
import pvlib

# ---------------------------------------------------------
# 1. Cargar módulo CEC
# ---------------------------------------------------------
cec = pvlib.pvsystem.retrieve_sam('CECMod')
module = cec['SunPower_SPR_X21_345']  # puedes cambiarlo

print("Módulo cargado:")
print(module)

# Sistema base (solo para usar calcparams_cec)
system = pvlib.pvsystem.PVSystem(
    module_parameters=module,
    temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
)

# ---------------------------------------------------------
# 2. Definir submódulos y condiciones
# ---------------------------------------------------------
n_sub = 3
irradiancias = np.array([1000, 600, 300])  # W/m² para cada submódulo
temp_cell = 25  # °C

# ---------------------------------------------------------
# 3. Obtener parámetros del modelo CEC para cada submódulo
# ---------------------------------------------------------
params_sub = []
for G in irradiancias:
    params = system.calcparams_cec(
        effective_irradiance=G,
        temp_cell=temp_cell
    )
    params_sub.append(params)

# params_sub[k] = (I_L, I_o, R_s, R_sh, nNsVth) para el submódulo k

# ---------------------------------------------------------
# 4. Barrido de corriente (serie → misma I en todos)
# ---------------------------------------------------------
# Estimamos Isc máximo a partir del submódulo más iluminado
Isc_est = params_sub[0][0]  # I_L del submódulo con más G
I_array = np.linspace(0, Isc_est * 1.05, 400)

V_total = []

for I in I_array:
    V_mod = 0.0
    for (I_L, I_o, R_s, R_sh, nNsVth) in params_sub:
        # Tensión del submódulo para esa corriente
        V_sub = pvlib.pvsystem.v_from_i(
            I,        # current
            I_L,      # photocurrent
            I_o,      # saturation_current
            R_s,      # resistance_series
            R_sh,     # resistance_shunt
            nNsVth    # nNsVth
        )

        # Modelo sencillo de bypass:
        # si el submódulo necesitaría tensión negativa → entra en bypass → V ≈ 0
        if np.isnan(V_sub) or V_sub < 0:
            V_sub = 0.0

        V_mod += V_sub

    V_total.append(V_mod)

V_total = np.array(V_total)
P_total = V_total * I_array

# ---------------------------------------------------------
# 5. Graficar curva I-V del módulo completo
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(V_total, I_array, label='Módulo con 3 submódulos y bypass')
plt.xlabel('Voltaje del módulo (V)')
plt.ylabel('Corriente (A)')
plt.title('Curva I-V de módulo con sombreado parcial y diodos bypass usando PVlib')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. Curva P-V
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(V_total, P_total, label='P-V del módulo')
plt.xlabel('Voltaje del módulo (V)')
plt.ylabel('Potencia (W)')
plt.title('Curva P-V de módulo con sombreado parcial y diodos bypass (pvlib)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Gráfico combinado I-V + P-V en una sola figura
# ---------------------------------------------------------


fig, ax1 = plt.subplots(figsize=(10, 6))

# --- Curva I-V (eje izquierdo) ---
ax1.plot(V_total, I_array, 'b-', linewidth=2, label='Corriente (I-V)')
ax1.set_xlabel('Voltaje del módulo (V)')
ax1.set_ylabel('Corriente (A)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# --- Segundo eje Y para la potencia ---
ax2 = ax1.twinx()
ax2.plot(V_total, P_total, 'r--', linewidth=2, label='Potencia (P-V)')
ax2.set_ylabel('Potencia (W)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title(f'Curvas I-V y P-V combinadas (Irradiancia = {irradiancias} W/m²)')
fig.tight_layout()
plt.show()



