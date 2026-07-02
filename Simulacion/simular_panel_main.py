"""
simular_panel_main.py
======================

@author: Héctor Carnicer Ull

Script de ejecución para el modelo de panel solar bajo sombreado parcial.

Flujo:
    1. Lee los datos del panel (panel_datos.json) y ajusta el modelo CEC.
       Define las constantes de configuración: ID de simulación y MODO
       (leer datos existentes vs. generar/regenerar y sobrescribir)
    2. Según MODO, genera N curvas IV aleatorias y las guarda en
       sim_<ID>.dat, o bien carga un sim_<ID>.dat ya existente
    3. Genera una figura con dos gráficas en paralelo (curvas I-V y P-V),
       todas las curvas y los puntos MPP de
       cada curva

Archivos relacionados:
    panel_datos.json    -> datos del datasheet (editable sin tocar código)
    panel_funciones.py  -> funciones de lectura y simulación (reutilizable)
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from panel_funciones import (
    cargar_datasheet,
    ajustar_modelo_cec,
    generar_simulaciones_aleatorias,
    guardar_simulaciones,
    cargar_simulaciones,
)

# ---------------------------------------------------------------------------
# CONSTANTES DE CONFIGURACIÓN
# ---------------------------------------------------------------------------
RUTA_DATASHEET = "panel_datos.json"

ID_SIMULACION = "1"            # Identificador de esta tanda de simulaciones.
                                 # El archivo de datos resultante será
                                 # sim_<ID_SIMULACION>.dat

# MODO controla qué hace el script con los datos:
#   "leer"      -> Carga sim_<ID>.dat si existe. Si no existe, lanza un
#                  error.
#   "generar"   -> Genera N simulaciones nuevas. Si sim_<ID>.dat ya existe,
#                  lo regenera sobreescribiéndolo sin pedir confirmación.

MODO = "generar"  # "leer" | "generar"

# Parámetros usados únicamente cuando MODO == "generar"
N_SIMULACIONES = 500
IRR_MIN, IRR_MAX = 150, 1000    # Rango de irradiancia por substring [W/m2]
TEMP_CELDA_SIM = 35              # Temperatura en °C, fija para las simulaciones aleatorias
SEMILLA_ALEATORIA = 43753          # Para reproducibilidad; None = sin fijar

RUTA_SIM = f"sim_{ID_SIMULACION}.dat"


# ---------------------------------------------------------------------------
# CARGA DE DATOS DEL PANEL Y AJUSTE DEL MODELO
# ---------------------------------------------------------------------------
datasheet = cargar_datasheet(RUTA_DATASHEET)
modulo = ajustar_modelo_cec(datasheet)

print("=== Parámetros del modelo de diodo único (CEC) ===")
for k in ["I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref", "Adjust"]:
    print(f"  {k:10s} = {modulo[k]:.6g}")


# ---------------------------------------------------------------------------
# GENERAR SIMULACIONES
# ---------------------------------------------------------------------------
if MODO == "generar":
    if os.path.exists(RUTA_SIM):
        print(f"\n'{RUTA_SIM}' ya existe: será SOBRESCRITO (MODO='generar').")

    simulaciones = generar_simulaciones_aleatorias(
        modulo,
        n_simulaciones=N_SIMULACIONES,
        irr_min=IRR_MIN,
        irr_max=IRR_MAX,
        temp_celda=TEMP_CELDA_SIM,
        semilla=SEMILLA_ALEATORIA,
    )

    guardar_simulaciones(
        simulaciones, modulo, RUTA_SIM,
        irr_min=IRR_MIN, irr_max=IRR_MAX, temp_celda=TEMP_CELDA_SIM,
        semilla=SEMILLA_ALEATORIA,
    )
    print(f"\n=== {N_SIMULACIONES} simulaciones generadas y guardadas en "
          f"'{RUTA_SIM}' ===")

# ---------------------------------------------------------------------------
# LECTURA SIMULACIONES
# ---------------------------------------------------------------------------  
elif MODO == "leer":
    if not os.path.exists(RUTA_SIM):
        raise FileNotFoundError(
            f"No se encontró '{RUTA_SIM}'. Este archivo debe generarse "
            f"primero ejecutando este mismo script con MODO = 'generar' "
            f"(y el mismo ID_SIMULACION = '{ID_SIMULACION}')."
        )

    datos = cargar_simulaciones(RUTA_SIM)
    simulaciones = datos["simulaciones"]
    print(f"\n=== {len(simulaciones)} simulaciones leídas desde "
          f"'{RUTA_SIM}' ===")

else:
    raise ValueError(f"MODO debe ser 'leer' o 'generar', recibido: {MODO!r}")


# ---------------------------------------------------------------------------
# VISUALIZACIÓN: curvas I-V y P-V con MPP
# ---------------------------------------------------------------------------

# Colores para la visualización

COLOR_CURVAS = "#cfd8e3"      
COLOR_MPP = "#ff1744"            

fig, (ax_iv, ax_pv) = plt.subplots(1, 2, figsize=(13, 5.5))

vmp_list, imp_list, pmax_list = [], [], []

for sim in simulaciones:
    v = np.array(sim["curva"]["V"])
    i = np.array(sim["curva"]["I"])
    p = np.array(sim["curva"]["P"])
    mpp = sim["mpp"]

    # Curvas de fondo
    ax_iv.plot(v, i, color=COLOR_CURVAS, linewidth=1, zorder=1)
    ax_pv.plot(v, p, color=COLOR_CURVAS, linewidth=1, zorder=1)

    # MPP
    ax_iv.scatter(mpp["Vmp [V]"], mpp["Imp [A]"], color=COLOR_MPP, s=18,
                  zorder=3, alpha=0.85)
    ax_pv.scatter(mpp["Vmp [V]"], mpp["Pmax [W]"], color=COLOR_MPP, s=18,
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
nombre_figura = f"curvas_sim_{ID_SIMULACION}.png"
plt.savefig(nombre_figura, dpi=150)
print(f"\nGráfico guardado como '{nombre_figura}'")
