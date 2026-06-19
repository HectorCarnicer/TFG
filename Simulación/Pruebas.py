#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:20:04 2026

@author: hectorc
"""
import numpy as np
import matplotlib.pyplot as plt
import pvlib

# ---------------------------------------------------------
# 1. Cargar base de datos CEC
# ---------------------------------------------------------
cec = pvlib.pvsystem.retrieve_sam('CECMod')

# Módulo válido
module = cec['SunPower_SPR_X21_345']

print("Módulo cargado:")
print(module)

# ---------------------------------------------------------
# 2. Crear un sistema FV usando el modelo CEC
# ---------------------------------------------------------
system = pvlib.pvsystem.PVSystem(
    module_parameters=module,
    temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
)

# ---------------------------------------------------------
# 3. Condiciones de simulación
# ---------------------------------------------------------
irradiancias = [1000, 800, 600, 400, 200]
temperatura_celda = 25  # °C

# ---------------------------------------------------------
# 4. Función para simular curva I-V (método compatible)
# ---------------------------------------------------------
def simular_iv(irr, temp):
    params = system.calcparams_cec(
        effective_irradiance=irr,
        temp_cell=temp
    )

    # Generamos un vector de tensiones manualmente
    V = np.linspace(0, params[1] * 1.2, 200)  # Voc estimado * margen
    I = pvlib.pvsystem.i_from_v(
        V,
        params[0],  # I_L
        params[1],  # I_o
        params[2],  # R_s
        params[3],  # R_sh
        params[4]   # nNsVth
    )

    return V, I

# ---------------------------------------------------------
# 5. Graficar curvas I-V
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

for irr in irradiancias:
    V, I = simular_iv(irr, temperatura_celda)
    plt.plot(V, I, label=f'{irr} W/m²')

plt.title('Curvas I-V para distintos niveles de irradiancia')
plt.xlabel('Voltaje (V)')
plt.ylabel('Corriente (A)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. Graficar curvas P-V
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

for irr in irradiancias:
    V, I = simular_iv(irr, temperatura_celda)
    P = V * I
    plt.plot(V, P, label=f'{irr} W/m²')

plt.title('Curvas P-V para distintos niveles de irradiancia')
plt.xlabel('Voltaje (V)')
plt.ylabel('Potencia (W)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. Gráfico combinado I-V + P-V
# ---------------------------------------------------------
irr = 1000
V, I = simular_iv(irr, temperatura_celda)
P = V * I

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(V, I, 'b-', label='Corriente (I-V)')
ax1.set_xlabel('Voltaje (V)')
ax1.set_ylabel('Corriente (A)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(V, P, 'r--', label='Potencia (P-V)')
ax2.set_ylabel('Potencia (W)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Curva I-V y P-V combinadas (1000 W/m²)')
fig.tight_layout()
plt.show()
