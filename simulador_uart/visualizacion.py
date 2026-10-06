"""
visualizacion.py

Lee un log de test (logs/test_{ID}.log) y genera dos gráficos por muestra:
Vmp real vs. Vmp predicho (graficos/grafico_{ID}.png) y Pmax real vs.
potencia entregada (graficos/grafico_potencia_{ID}.png), cada uno con su
panel de error absoluto.

Uso:
    python visualizacion.py            # usa ID_TEST definido más abajo
    python visualizacion.py 003        # o bien pasa el ID como argumento
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla, solo para exportar a archivo
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ============================================================================
# PARÁMETROS
# ============================================================================

DIRECTORIO_LOGS = "logs"           # carpeta donde está test_{ID_TEST}.log
ID_TEST = "029"                    # test a visualizar si no se pasa por argumento
DIRECTORIO_GRAFICOS = "graficos"   # carpeta de salida de los gráficos


# ============================================================================
# Paleta (paleta categórica validada del sistema de diseño; ver dataviz skill)
# ============================================================================

COLOR_FONDO = "#fcfcfb"
COLOR_TEXTO_PRIMARIO = "#0b0b0b"
COLOR_TEXTO_SECUNDARIO = "#52514e"
COLOR_TEXTO_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_EJE = "#c3c2b7"

COLOR_VMP_REAL = "#2a78d6"
COLOR_VMP_PREDICHO = "#eb6834"
COLOR_ERROR = "#1baf7a"

# Potencia reutiliza los mismos roles de color que Vmp: azul para la
# magnitud real, naranja para la magnitud calculada a partir del
# seguimiento, verde para el error — mismo lenguaje visual en ambos
# gráficos del mismo test.
COLOR_PMAX_REAL = COLOR_VMP_REAL
COLOR_POTENCIA_ENTREGADA = COLOR_VMP_PREDICHO


# ============================================================================
# Parseo del log
# ============================================================================

# Línea de muestra generada por simulador_sensor.py:
# "[Curva 000] Punto   1/5 -> V= 20.52 V | I= 0.21 A | Prediccion=  20.9089 |
#  t_inferencia=   834 us | Vmp_real= 17.14 V | Pmax_real= 12.3456 W |
#  Error=+3.7689 V (+21.99 %)"
PATRON_MUESTRA = re.compile(
    r"\[Curva\s+(?P<curva>\d+)\]\s+Punto\s+(?P<punto>\d+)/(?P<total>\d+)\s+->\s+"
    r"V=\s*(?P<v>-?\d+(?:\.\d+)?)\s*V\s*\|\s*"
    r"I=\s*(?P<i>-?\d+(?:\.\d+)?)\s*A\s*\|\s*"
    r"Prediccion=\s*(?P<pred>-?\d+(?:\.\d+)?)\s*\|\s*"
    r"t_inferencia=\s*(?P<t_us>\d+)\s*us\s*\|\s*"
    r"Vmp_real=\s*(?P<vmp_real>-?\d+(?:\.\d+)?)\s*V\s*\|\s*"
    r"Pmax_real=\s*(?P<pmax_real>-?\d+(?:\.\d+)?)\s*W\s*\|\s*"
    r"Error=\s*(?P<error>[+-]?\d+(?:\.\d+)?)\s*V\s*"
    r"\(\s*(?P<error_pct>[+-]?\d+(?:\.\d+)?)\s*%\)"
)


class Muestra:
    """Una muestra con predicción extraída del log de test."""

    def __init__(
        self, indice, id_curva, v, i, vmp_real, pmax_real, prediccion,
        error_abs, error_pct, t_inferencia_us,
    ):
        self.indice = indice
        self.id_curva = id_curva
        self.v = v
        self.i = i
        self.vmp_real = vmp_real
        self.pmax_real = pmax_real
        self.prediccion = prediccion
        self.error_abs = error_abs
        self.error_pct = error_pct
        self.t_inferencia_us = t_inferencia_us
        self.potencia_entregada = v * i  # potencia del punto de trabajo enviado


def ruta_log(directorio_logs, id_test):
    """Devuelve la ruta del log correspondiente a un ID de test."""
    return os.path.join(directorio_logs, f"test_{id_test}.log")


def ruta_grafico(directorio_graficos, id_test):
    """Devuelve la ruta de salida del gráfico de tensión correspondiente a
    un ID de test.
    """
    return os.path.join(directorio_graficos, f"grafico_{id_test}.png")


def ruta_grafico_potencia(directorio_graficos, id_test):
    """Devuelve la ruta de salida del gráfico de potencia correspondiente
    a un ID de test.
    """
    return os.path.join(directorio_graficos, f"grafico_potencia_{id_test}.png")


def leer_muestras_log(ruta):
    """Lee un test_{ID}.log y devuelve las muestras con predicción válida
    (ignora cabecera, líneas "SIN RESPUESTA" y resúmenes).

    Lanza FileNotFoundError si el log no existe, o ValueError si no
    contiene ninguna muestra reconocible.
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No se encuentra el log: {ruta}")

    muestras = []
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            m = PATRON_MUESTRA.search(linea)
            if not m:
                continue
            muestras.append(
                Muestra(
                    indice=len(muestras),
                    id_curva=int(m.group("curva")),
                    v=float(m.group("v")),
                    i=float(m.group("i")),
                    vmp_real=float(m.group("vmp_real")),
                    pmax_real=float(m.group("pmax_real")),
                    prediccion=float(m.group("pred")),
                    error_abs=abs(float(m.group("error"))),
                    error_pct=float(m.group("error_pct")),
                    t_inferencia_us=int(m.group("t_us")),
                )
            )

    if not muestras:
        raise ValueError(
            f"El log {ruta} no contiene ninguna línea de muestra reconocible "
            "(¿está vacío o el test no llegó a recibir respuestas?)."
        )

    return muestras


# ============================================================================
# Gráfico
# ============================================================================

def generar_grafico(muestras, id_test, ruta_salida):
    """Genera y guarda el gráfico de un test: Vmp real vs. predicho (panel
    superior) y error absoluto (panel inferior), uno por muestra.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]

    x = [m.indice for m in muestras]
    vmp_real = [m.vmp_real for m in muestras]
    vmp_pred = [m.prediccion for m in muestras]
    error_abs = [m.error_abs for m in muestras]

    n = len(muestras)
    mae = sum(error_abs) / n

    # con muchas muestras, línea/área continua en vez de marcadores individuales
    usar_marcadores = n <= 60

    fig, (ax_vmp, ax_err) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.patch.set_facecolor(COLOR_FONDO)

    # --- Panel superior: Vmp real vs. Vmp predicho -------------------------
    ax_vmp.set_facecolor(COLOR_FONDO)
    if usar_marcadores:
        ax_vmp.plot(
            x, vmp_real, color=COLOR_VMP_REAL, linewidth=1.5, alpha=0.55,
            zorder=2,
        )
        ax_vmp.scatter(
            x, vmp_real, color=COLOR_VMP_REAL, s=26, zorder=3,
            label="Vmp real", edgecolors=COLOR_FONDO, linewidths=0.6,
        )
        ax_vmp.plot(
            x, vmp_pred, color=COLOR_VMP_PREDICHO, linewidth=1.5, alpha=0.55,
            zorder=2,
        )
        ax_vmp.scatter(
            x, vmp_pred, color=COLOR_VMP_PREDICHO, s=26, zorder=3,
            label="Vmp predicho", edgecolors=COLOR_FONDO, linewidths=0.6,
        )
    else:
        ax_vmp.plot(
            x, vmp_real, color=COLOR_VMP_REAL, linewidth=1.4,
            label="Vmp real", zorder=2,
        )
        ax_vmp.plot(
            x, vmp_pred, color=COLOR_VMP_PREDICHO, linewidth=1.2, alpha=0.85,
            label="Vmp predicho", zorder=3,
        )

    ax_vmp.set_ylabel("Tensión Vmp (V)", color=COLOR_TEXTO_SECUNDARIO)
    fig.suptitle(
        f"Test {id_test} — MPP real vs. MPP predicho por muestra",
        x=0.015, y=0.995, ha="left", va="top",
        color=COLOR_TEXTO_PRIMARIO, fontsize=13,
    )
    ax_vmp.set_title(
        f"{n} muestras · MAE = {mae:.4f} V",
        loc="left", fontsize=9.5, color=COLOR_TEXTO_MUTED, pad=10,
    )
    leyenda = ax_vmp.legend(
        loc="upper right", frameon=True, framealpha=0.9,
        facecolor=COLOR_FONDO, edgecolor="none",
        labelcolor=COLOR_TEXTO_SECUNDARIO,
    )

    # --- Panel inferior: error absoluto -------------------------------------
    ax_err.set_facecolor(COLOR_FONDO)
    if usar_marcadores:
        ax_err.bar(x, error_abs, color=COLOR_ERROR, width=0.6, zorder=2)
    else:
        ax_err.fill_between(x, 0, error_abs, color=COLOR_ERROR, alpha=0.35, zorder=1)
        ax_err.plot(x, error_abs, color=COLOR_ERROR, linewidth=1.0, zorder=2)

    # se etiqueta el punto de error máximo por el bajo contraste del aqua
    idx_max = max(range(n), key=lambda k: error_abs[k])
    ax_err.annotate(
        f"máx {error_abs[idx_max]:.3f} V",
        xy=(x[idx_max], error_abs[idx_max]),
        xytext=(0, 8), textcoords="offset points",
        ha="center", fontsize=8.5, color=COLOR_TEXTO_PRIMARIO,
    )

    ax_err.set_ylabel("Error absoluto (V)", color=COLOR_TEXTO_SECUNDARIO)
    ax_err.set_xlabel("Índice de muestra", color=COLOR_TEXTO_SECUNDARIO)

    for ax in (ax_vmp, ax_err):
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine_name, spine in ax.spines.items():
            if spine_name in ("top", "right"):
                spine.set_visible(False)
            else:
                spine.set_color(COLOR_EJE)
        ax.tick_params(colors=COLOR_TEXTO_MUTED, labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    fig.savefig(ruta_salida, dpi=170, facecolor=COLOR_FONDO)
    plt.close(fig)


def generar_grafico_potencia(muestras, id_test, ruta_salida):
    """Genera y guarda el gráfico de potencia de un test: Pmax real vs.
    potencia entregada (panel superior) y error absoluto (panel
    inferior), uno por muestra. Misma estructura que generar_grafico().
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]

    x = [m.indice for m in muestras]
    pmax_real = [m.pmax_real for m in muestras]
    potencia_entregada = [m.potencia_entregada for m in muestras]
    error_abs = [abs(m.pmax_real - m.potencia_entregada) for m in muestras]

    n = len(muestras)
    mae = sum(error_abs) / n

    # con muchas muestras, línea/área continua en vez de marcadores individuales
    usar_marcadores = n <= 60

    fig, (ax_pot, ax_err) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.patch.set_facecolor(COLOR_FONDO)

    # --- Panel superior: Pmax real vs. potencia entregada -------------------
    ax_pot.set_facecolor(COLOR_FONDO)
    if usar_marcadores:
        ax_pot.plot(
            x, pmax_real, color=COLOR_PMAX_REAL, linewidth=1.5, alpha=0.55,
            zorder=2,
        )
        ax_pot.scatter(
            x, pmax_real, color=COLOR_PMAX_REAL, s=26, zorder=3,
            label="Pmax real", edgecolors=COLOR_FONDO, linewidths=0.6,
        )
        ax_pot.plot(
            x, potencia_entregada, color=COLOR_POTENCIA_ENTREGADA, linewidth=1.5,
            alpha=0.55, zorder=2,
        )
        ax_pot.scatter(
            x, potencia_entregada, color=COLOR_POTENCIA_ENTREGADA, s=26, zorder=3,
            label="Potencia entregada", edgecolors=COLOR_FONDO, linewidths=0.6,
        )
    else:
        ax_pot.plot(
            x, pmax_real, color=COLOR_PMAX_REAL, linewidth=1.4,
            label="Pmax real", zorder=2,
        )
        ax_pot.plot(
            x, potencia_entregada, color=COLOR_POTENCIA_ENTREGADA, linewidth=1.2,
            alpha=0.85, label="Potencia entregada", zorder=3,
        )

    ax_pot.set_ylabel("Potencia (W)", color=COLOR_TEXTO_SECUNDARIO)
    fig.suptitle(
        f"Test {id_test} — Pmax real vs. potencia entregada por muestra",
        x=0.015, y=0.995, ha="left", va="top",
        color=COLOR_TEXTO_PRIMARIO, fontsize=13,
    )
    ax_pot.set_title(
        f"{n} muestras · MAE = {mae:.4f} W",
        loc="left", fontsize=9.5, color=COLOR_TEXTO_MUTED, pad=10,
    )
    leyenda = ax_pot.legend(
        loc="upper right", frameon=True, framealpha=0.9,
        facecolor=COLOR_FONDO, edgecolor="none",
        labelcolor=COLOR_TEXTO_SECUNDARIO,
    )

    # --- Panel inferior: error absoluto -------------------------------------
    ax_err.set_facecolor(COLOR_FONDO)
    if usar_marcadores:
        ax_err.bar(x, error_abs, color=COLOR_ERROR, width=0.6, zorder=2)
    else:
        ax_err.fill_between(x, 0, error_abs, color=COLOR_ERROR, alpha=0.35, zorder=1)
        ax_err.plot(x, error_abs, color=COLOR_ERROR, linewidth=1.0, zorder=2)

    # se etiqueta el punto de error máximo por el bajo contraste del aqua
    idx_max = max(range(n), key=lambda k: error_abs[k])
    ax_err.annotate(
        f"máx {error_abs[idx_max]:.3f} W",
        xy=(x[idx_max], error_abs[idx_max]),
        xytext=(0, 8), textcoords="offset points",
        ha="center", fontsize=8.5, color=COLOR_TEXTO_PRIMARIO,
    )

    ax_err.set_ylabel("Error absoluto (W)", color=COLOR_TEXTO_SECUNDARIO)
    ax_err.set_xlabel("Índice de muestra", color=COLOR_TEXTO_SECUNDARIO)

    for ax in (ax_pot, ax_err):
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine_name, spine in ax.spines.items():
            if spine_name in ("top", "right"):
                spine.set_visible(False)
            else:
                spine.set_color(COLOR_EJE)
        ax.tick_params(colors=COLOR_TEXTO_MUTED, labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    fig.savefig(ruta_salida, dpi=170, facecolor=COLOR_FONDO)
    plt.close(fig)


# ============================================================================
# Ejecución
# ============================================================================

def main():
    id_test = sys.argv[1] if len(sys.argv) > 1 else ID_TEST

    ruta_entrada = ruta_log(DIRECTORIO_LOGS, id_test)
    ruta_salida_vmp = ruta_grafico(DIRECTORIO_GRAFICOS, id_test)
    ruta_salida_potencia = ruta_grafico_potencia(DIRECTORIO_GRAFICOS, id_test)

    print(f"Leyendo '{ruta_entrada}'...")
    muestras = leer_muestras_log(ruta_entrada)
    print(f"  -> {len(muestras)} muestras con predicción encontradas.")

    generar_grafico(muestras, id_test, ruta_salida_vmp)
    print(f"Gráfico de tensión guardado en: {ruta_salida_vmp}")

    generar_grafico_potencia(muestras, id_test, ruta_salida_potencia)
    print(f"Gráfico de potencia guardado en: {ruta_salida_potencia}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
