"""
simular_panel_main.py
======================

Script de ejecución para el modelo de panel solar bajo sombreado parcial.

Flujo:
    1. Lee los datos del panel (panel_datos.json) y ajusta el modelo CEC.
       Define las constantes de configuración: ID de simulación y MODO
       (leer datos existentes vs. generar/regenerar y sobrescribir).
    2. Según MODO, genera N curvas IV aleatorias y las guarda en
       sim_<ID>.dat, o bien carga un sim_<ID>.dat ya existente.
    3. Genera una figura con dos gráficas en paralelo (curvas I-V y P-V),
       todas las curvas en un tono muy poco saturado, los puntos MPP de
       cada curva en un tono muy saturado, y el MPP promedio de todas las
       simulaciones en un tercer color, destacado.

Archivos relacionados:
    panel_datos.json     -> datos del datasheet (editable sin tocar código)
    panel_funciones.py   -> funciones de lectura y simulación (reutilizable)
    simular_panel_main.py -> este script

Requisitos:
    pip install pvlib numpy pandas matplotlib
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
# 1. CONSTANTES DE CONFIGURACIÓN
# ---------------------------------------------------------------------------
RUTA_DATASHEET = "panel_datos.json"

ID_SIMULACION = "1"            # Identificador de esta tanda de simulaciones.
                                 # El archivo de datos resultante será
                                 # sim_<ID_SIMULACION>.dat

# MODO controla qué hace el script con los datos:
#   "leer"      -> Carga sim_<ID>.dat si existe. Si NO existe, lanza un
#                  error explícito (no genera nada por su cuenta).
#   "generar"   -> Genera N simulaciones nuevas. Si sim_<ID>.dat ya existe,
#                  lo SOBRESCRIBE (regenera) sin pedir confirmación.
MODO = "generar"  # "leer" | "generar"

# Parámetros usados únicamente cuando MODO == "generar"
N_SIMULACIONES = 50
IRR_MIN, IRR_MAX = 150, 1000    # rango de irradiancia por substring [W/m2]
TEMP_CELDA_SIM = 35              # °C, fija para las simulaciones aleatorias
SEMILLA_ALEATORIA = 43753          # reproducibilidad; None = sin fijar

RUTA_SIM = f"sim_{ID_SIMULACION}.dat"


# ---------------------------------------------------------------------------
# 2. CARGA DE DATOS DEL PANEL Y AJUSTE DEL MODELO
# ---------------------------------------------------------------------------
datasheet = cargar_datasheet(RUTA_DATASHEET)
modulo = ajustar_modelo_cec(datasheet)

print("=== Parámetros del modelo de diodo único (CEC) ===")
for k in ["I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref", "Adjust"]:
    print(f"  {k:10s} = {modulo[k]:.6g}")


# ---------------------------------------------------------------------------
# 3. GENERAR O LEER LAS SIMULACIONES (según MODO)
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
# 4. VISUALIZACIÓN: curvas I-V y P-V con MPP individual y MPP promedio
# ---------------------------------------------------------------------------
# Colores: curvas en tono MUY poco saturado (gris azulado claro), puntos MPP
# individuales en tono MUY saturado, y el MPP promedio en un tercer color
# bien diferenciado para que resalte sobre el resto.
COLOR_CURVAS = "#cfd8e3"      # gris-azul muy claro, poco saturado
COLOR_MPP = "#ff1744"          # rojo muy saturado
COLOR_MPP_PROMEDIO = "#00c853"  # verde saturado, bien diferenciado del rojo

fig, (ax_iv, ax_pv) = plt.subplots(1, 2, figsize=(13, 5.5))

vmp_list, imp_list, pmax_list = [], [], []

for sim in simulaciones:
    v = np.array(sim["curva"]["V"])
    i = np.array(sim["curva"]["I"])
    p = np.array(sim["curva"]["P"])
    mpp = sim["mpp"]

    # Curvas de fondo, muy poco saturadas
    ax_iv.plot(v, i, color=COLOR_CURVAS, linewidth=1, zorder=1)
    ax_pv.plot(v, p, color=COLOR_CURVAS, linewidth=1, zorder=1)

    # Punto MPP individual, muy saturado
    ax_iv.scatter(mpp["Vmp [V]"], mpp["Imp [A]"], color=COLOR_MPP, s=18,
                  zorder=3, alpha=0.85)
    ax_pv.scatter(mpp["Vmp [V]"], mpp["Pmax [W]"], color=COLOR_MPP, s=18,
                  zorder=3, alpha=0.85)

    vmp_list.append(mpp["Vmp [V]"])
    imp_list.append(mpp["Imp [A]"])
    pmax_list.append(mpp["Pmax [W]"])

# --- MPP promedio de todas las simulaciones ---
vmp_prom = float(np.mean(vmp_list))
imp_prom = float(np.mean(imp_list))
pmax_prom = float(np.mean(pmax_list))

ax_iv.scatter(vmp_prom, imp_prom, color=COLOR_MPP_PROMEDIO, s=160,
              marker="*", edgecolor="black", linewidth=0.8, zorder=4,
              label=f"MPP promedio ({vmp_prom:.2f} V, {imp_prom:.2f} A)")
ax_pv.scatter(vmp_prom, pmax_prom, color=COLOR_MPP_PROMEDIO, s=160,
              marker="*", edgecolor="black", linewidth=0.8, zorder=4,
              label=f"MPP promedio ({vmp_prom:.2f} V, {pmax_prom:.2f} W)")

ax_iv.set_xlabel("Voltaje [V]")
ax_iv.set_ylabel("Corriente [A]")
ax_iv.set_title(f"Curvas I-V ({len(simulaciones)} simulaciones)")
ax_iv.legend(loc="upper right", fontsize=8)
ax_iv.grid(True, alpha=0.3)

ax_pv.set_xlabel("Voltaje [V]")
ax_pv.set_ylabel("Potencia [W]")
ax_pv.set_title(f"Curvas P-V ({len(simulaciones)} simulaciones)")
ax_pv.legend(loc="upper right", fontsize=8)
ax_pv.grid(True, alpha=0.3)

plt.tight_layout()
nombre_figura = f"curvas_sim_{ID_SIMULACION}.png"
plt.savefig(nombre_figura, dpi=150)
print(f"\nGráfico guardado como '{nombre_figura}'")

print("\n=== MPP promedio sobre todas las simulaciones ===")
print(f"  Vmp promedio  = {vmp_prom:.3f} V")
print(f"  Imp promedio  = {imp_prom:.3f} A")
print(f"  Pmax promedio = {pmax_prom:.3f} W")