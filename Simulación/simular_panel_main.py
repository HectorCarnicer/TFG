"""
simular_panel_main.py
======================

Script de ejecución: usa las funciones de `panel_funciones.py` y los datos
de `panel_datos.json` para:

    1. Ajustar el modelo de diodo único (CEC) del panel.
    2. Validar el modelo contra los valores STC/NOCT del datasheet.
    3. Simular el panel bajo distintos escenarios de sombreado parcial
       (irradiancia distinta por substring) y graficar las curvas I-V/P-V.
    4. Generar N simulaciones con irradiancia aleatoria por substring y
       guardarlas en sim1.dat (JSON).

Archivos relacionados:
    panel_datos.json     -> datos del datasheet (editable sin tocar código)
    panel_funciones.py   -> funciones de lectura y simulación (reutilizable)
    simular_panel_main.py -> este script

Requisitos:
    pip install pvlib numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from panel_funciones import (
    cargar_datasheet,
    ajustar_modelo_cec,
    curva_iv,
    curva_iv_panel_sombreado,
    generar_simulaciones_aleatorias,
    guardar_simulaciones,
)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------------
RUTA_DATASHEET = "panel_datos.json"
RUTA_SIM_OUTPUT = "sim1.dat"

N_SIMULACIONES = 50            # número de curvas aleatorias a generar
IRR_MIN, IRR_MAX = 150, 1000    # rango de irradiancia por substring [W/m2]
TEMP_CELDA_SIM = 35              # °C, fija para las simulaciones aleatorias
SEMILLA_ALEATORIA = 42           # reproducibilidad; None = sin fijar


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS Y AJUSTE DEL MODELO
# ---------------------------------------------------------------------------
datasheet = cargar_datasheet(RUTA_DATASHEET)
modulo = ajustar_modelo_cec(datasheet)

print("=== Parámetros del modelo de diodo único (CEC) ===")
for k in ["I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref", "Adjust"]:
    print(f"  {k:10s} = {modulo[k]:.6g}")


# ---------------------------------------------------------------------------
# 2. VALIDACIÓN: panel completo en STC y NOCT vs datasheet
# ---------------------------------------------------------------------------
condiciones = [
    {"irradiancia": 1000, "temp_celda": 25, "label": "STC (1000 W/m², 25°C)"},
    {"irradiancia": 800,  "temp_celda": 47, "label": "NOCT (800 W/m², 47°C)"},
    {"irradiancia": 200,  "temp_celda": 35, "label": "Baja irradiancia (200 W/m², 35°C)"},
]

resumen = []
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

ax1.set_xlabel("Voltaje [V]"); ax1.set_ylabel("Corriente [A]")
ax1.set_title("Curva I-V"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Voltaje [V]"); ax2.set_ylabel("Potencia [W]")
ax2.set_title("Curva P-V"); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("curvas_panel_solar.png", dpi=150)
print("\nGráfico guardado como 'curvas_panel_solar.png'")

print("\n=== Resumen de desempeño bajo distintas condiciones ===")
print(pd.DataFrame(resumen).to_string(index=False))

ref = datasheet["datasheet_referencia"]
print("\n=== Validación del modelo vs datasheet ===")
validacion = pd.DataFrame({
    "Parámetro": ["Pmax [W]", "Vmp [V]", "Imp [A]", "Voc [V]", "Isc [A]"],
    "Datasheet (STC)": [ref["p_max_stc_W"], ref["v_mp_stc_V"], ref["i_mp_stc_A"],
                         ref["v_oc_stc_V"], ref["i_sc_stc_A"]],
    "Modelo (STC)": [resumen[0]["Pmax [W]"], resumen[0]["Vmp [V]"], resumen[0]["Imp [A]"],
                      resumen[0]["Voc [V]"], resumen[0]["Isc [A]"]],
    "Datasheet (NOCT)": [ref["p_max_noct_W"], ref["v_mp_noct_V"], ref["i_mp_noct_A"],
                          ref["v_oc_noct_V"], ref["i_sc_noct_A"]],
    "Modelo (NOCT)": [resumen[1]["Pmax [W]"], resumen[1]["Vmp [V]"], resumen[1]["Imp [A]"],
                       resumen[1]["Voc [V]"], resumen[1]["Isc [A]"]],
})
print(validacion.to_string(index=False))


# ---------------------------------------------------------------------------
# 3. SOMBREADO PARCIAL: escenarios de ejemplo
# ---------------------------------------------------------------------------
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

ax3.set_xlabel("Voltaje [V]"); ax3.set_ylabel("Corriente [A]")
ax3.set_title("Curva I-V con sombreado parcial por substring")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3); ax3.axvline(0, color="gray", linewidth=0.5)

ax4.set_xlabel("Voltaje [V]"); ax4.set_ylabel("Potencia [W]")
ax4.set_title("Curva P-V con sombreado parcial por substring")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("curvas_panel_sombreado.png", dpi=150)
print("\nGráfico de sombreado parcial guardado como 'curvas_panel_sombreado.png'")

print("\n=== Resumen Pmax bajo sombreado parcial (irradiancia por substring) ===")
print(pd.DataFrame(resumen_sombra_total).to_string(index=False))
print(
    "\nNota: en el escenario con sombra fuerte aparecen 'escalones' en la "
    "curva P-V — son los puntos donde un diodo de bypass empieza a "
    "conducir y 'salta' el substring sombreado."
)


# ---------------------------------------------------------------------------
# 4. GENERACIÓN MASIVA DE SIMULACIONES ALEATORIAS -> sim1.dat
# ---------------------------------------------------------------------------
simulaciones = generar_simulaciones_aleatorias(
    modulo,
    n_simulaciones=N_SIMULACIONES,
    irr_min=IRR_MIN,
    irr_max=IRR_MAX,
    temp_celda=TEMP_CELDA_SIM,
    semilla=SEMILLA_ALEATORIA,
)

guardar_simulaciones(
    simulaciones, modulo, RUTA_SIM_OUTPUT,
    irr_min=IRR_MIN, irr_max=IRR_MAX, temp_celda=TEMP_CELDA_SIM,
    semilla=SEMILLA_ALEATORIA,
)

print(f"\n=== {N_SIMULACIONES} simulaciones guardadas en '{RUTA_SIM_OUTPUT}' (JSON) ===")

df_check = pd.DataFrame([
    {
        "id": s["id"],
        "Irr_1 [W/m²]": s["irradiancias_substrings_Wm2"][0],
        "Irr_2 [W/m²]": s["irradiancias_substrings_Wm2"][1],
        "Irr_3 [W/m²]": s["irradiancias_substrings_Wm2"][2],
        "Pmax [W]": s["resumen"]["Pmax [W]"],
        "Vmp [V]": s["resumen"]["Vmp [V]"],
        "Imp [A]": s["resumen"]["Imp [A]"],
    }
    for s in simulaciones[:10]
])
print(f"\nPrimeras 10 simulaciones (de un total de {N_SIMULACIONES}):")
print(df_check.to_string(index=False))

print(
    "\nPara recargar el archivo en otro script:\n"
    "    from panel_funciones import cargar_simulaciones\n"
    "    datos = cargar_simulaciones('sim1.dat')\n"
    "    primera_curva_V = datos['simulaciones'][0]['curva']['V']\n"
)
