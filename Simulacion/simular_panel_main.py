"""
simular_panel_main.py

@author: Héctor Carnicer Ull

Script de ejecución: simula (o carga) N curvas IV de un panel bajo
sombreado parcial y genera el gráfico I-V / P-V con los MPP.
"""

import os

from panel_funciones import (
    cargar_datasheet,
    ajustar_modelo_cec,
    generar_simulaciones_aleatorias,
    guardar_simulaciones,
    cargar_simulaciones,
    graficar_curvas_iv_pv,
)

# CONFIGURACIÓN
CARPETA_DATOS = "datos_panel"  # Carpeta con los ficheros datos_panel_<ID>.json
CARPETA_GRAFICOS = "Graficos"  # Carpeta de salida de los gráficos

ID_DATOS = 4       # ID del panel a simular (datos_panel/datos_panel_<ID_DATOS>.json)
ID_SIMULACION = "5"  # ID de esta tanda; nombra sim_<ID_SIMULACION>.dat

# MODO: "leer" carga sim_<ID>.dat existente (error si no existe);
# "generar" genera N simulaciones nuevas y sobrescribe si ya existe.
MODO = "generar"  # "leer" | "generar"

VISUAL = 1  # Si es distinto de 0, genera y guarda el gráfico

# Parámetros usados únicamente cuando MODO == "generar"
N_SIMULACIONES = 500
IRR_MIN, IRR_MAX = 150, 1000     # Rango de irradiancia por substring [W/m2]
TEMP_CELDA_SIM = 35              # Temperatura de celda [°C]
SEMILLA_ALEATORIA = None         # Semilla de reproducibilidad; None = sin fijar

RUTA_SIM = f"datos_simulaciones/sim_{ID_SIMULACION}.dat"


# CARGA DE DATOS DEL PANEL Y AJUSTE DEL MODELO
datasheet = cargar_datasheet(ID_DATOS, carpeta_datos=CARPETA_DATOS)
modulo = ajustar_modelo_cec(datasheet)

print("=== Parámetros del modelo de diodo único (CEC) ===")
for k in ["I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref", "Adjust"]:
    print(f"  {k:10s} = {modulo[k]:.6g}")


# GENERAR O LEER SIMULACIONES
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


# VISUALIZACIÓN: curvas I-V y P-V con MPP
if VISUAL:
    print("\nVisualización activada, se generará el gráfico de las curvas I-V\n")

    ruta_figura = graficar_curvas_iv_pv(
        simulaciones, ID_SIMULACION, carpeta_graficos=CARPETA_GRAFICOS
    )

    print(f"\nGráfico guardado como '{ruta_figura}'")
