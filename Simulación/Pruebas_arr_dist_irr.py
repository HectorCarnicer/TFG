#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 21:38:07 2026

@author: hectorc
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Parámetros del modelo (sencillos para entender el concepto)
# ---------------------------------------------------------
Ns = 20   # células por submódulo
n_sub = 3 # número de submódulos

# Irradiancia distinta en cada submódulo (W/m2)
G = np.array([1000, 600, 300])  

# Parámetros eléctricos simplificados
Isc_stc = 9.0      # A
Voc_stc = 0.6      # V por célula
Rs = 0.01          # resistencia serie
Rsh = 200          # resistencia shunt
n = 1.3            # factor idealidad
T = 298            # K
k = 1.380649e-23   # constante Boltzmann
q = 1.602e-19      # carga del electrón

# ---------------------------------------------------------
# Función: corriente de un submódulo con bypass
# ---------------------------------------------------------
def submodule_current(Vsub, Gsub):
    """
    Devuelve la corriente del submódulo para un voltaje Vsub.
    Si el submódulo entra en bypass, devuelve 0 A.
    """
    # Corriente fotogenerada proporcional a irradiancia
    Iph = Isc_stc * (Gsub / 1000)

    # Tensión térmica
    Vt = n * k * T / q

    # Voc proporcional a irradiancia
    Voc = Ns * Voc_stc * (Gsub / 1000)

    # Si el submódulo está en bypass (tensión negativa), corriente = 0
    if Vsub < 0:
        return 0.0

    # Ecuación del diodo simplificada
    I = Iph - (Vsub / Rsh) - np.exp((Vsub + Rs) / (Ns * Vt))

    return max(I, 0)  # no corriente negativa

# ---------------------------------------------------------
# Función: corriente total del módulo con bypass
# ---------------------------------------------------------
def module_current(V):
    """
    Suma las contribuciones de los 3 submódulos.
    Cada uno puede entrar en bypass si su tensión cae por debajo de 0.
    """
    Vsub = V / n_sub  # reparto simple de tensión
    Itot = 0
    for i in range(n_sub):
        Itot += submodule_current(Vsub, G[i])
    return Itot

# ---------------------------------------------------------
# Generar curva I-V del módulo completo
# ---------------------------------------------------------
V = np.linspace(0, 60, 400)
I = np.array([module_current(v) for v in V])
P = V * I

# ---------------------------------------------------------
# Graficar
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(V, I, label="I-V del módulo")
plt.xlabel("Voltaje (V)")
plt.ylabel("Corriente (A)")
plt.title("Curva I-V de un módulo con 3 submódulos y diodos bypass")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(V, P, label="P-V del módulo")
plt.xlabel("Voltaje (V)")
plt.ylabel("Potencia (W)")
plt.title("Curva P-V del módulo con sombreado parcial")
plt.grid(True)
plt.legend()
plt.show()
