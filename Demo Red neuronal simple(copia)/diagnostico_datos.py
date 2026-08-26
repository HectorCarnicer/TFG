"""
diagnostico_datos.py

@author: Héctor Carnicer Ull

Diagnostico de datos: techo de R2 alcanzable con entrada (V, I),
distribucion de Vmp y ejemplos de curvas solapadas. Ver README.md para el
metodo y la interpretacion de los resultados.

Uso:
    python diagnostico_datos.py            (usa RUTA_DATOS)
    python diagnostico_datos.py otro.json
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from lectura_escritura import leer_datos_entrenamiento

# --- Parametros ------------------------------------------------------------
RUTA_DATOS = "sim_1.dat"
N_PUNTOS_CURVA = 150        # puntos por curva usados en el analisis
N_VECINOS = 60              # vecinos para estimar la varianza condicional
N_CONSULTAS = 3000          # puntos de consulta (submuestreo, por velocidad)
N_REFERENCIA = 40000        # puntos de referencia del vecindario
N_EJEMPLOS = 2              # parejas de curvas solapadas que se dibujan
SEMILLA = 42
GUARDAR_FIGURAS = True
MOSTRAR_FIGURAS = True


def varianza_condicional(X, grupos, vmp_por_grupo, n_vecinos, n_consultas,
                         n_referencia, semilla=42):
    """Estima E[Var(Vmp | V, I)] por vecinos proximos en (V, I).

    Devuelve la varianza condicional media y las discrepancias por consulta
    (curva propia y curva vecina mas discrepante).
    """
    rng = np.random.default_rng(semilla)

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xn = (X - mu) / sd

    idx_ref = rng.choice(len(Xn), size=min(n_referencia, len(Xn)), replace=False)
    idx_con = rng.choice(len(Xn), size=min(n_consultas, len(Xn)), replace=False)
    ref, g_ref = Xn[idx_ref], grupos[idx_ref]

    varianzas, discrepancias = [], []
    for inicio in range(0, len(idx_con), 200):
        bloque = idx_con[inicio:inicio + 200]
        d = np.linalg.norm(Xn[bloque, None, :] - ref[None, :, :], axis=2)
        vecinos = np.argsort(d, axis=1)[:, :n_vecinos]

        for fila, punto in zip(vecinos, bloque):
            propio = grupos[punto]
            ajenos = fila[g_ref[fila] != propio]
            if ajenos.size < 10:
                continue
            vmp_vecinos = vmp_por_grupo[g_ref[ajenos]]
            varianzas.append(vmp_vecinos.var())
            peor = ajenos[np.argmax(np.abs(vmp_vecinos - vmp_por_grupo[propio]))]
            discrepancias.append((punto, propio, g_ref[peor],
                                  abs(vmp_por_grupo[g_ref[peor]] - vmp_por_grupo[propio])))

    return float(np.mean(varianzas)), discrepancias


def figura_distribucion(vmp, X, y, ruta=None):
    """Histograma de Vmp y nube de puntos (V, I) coloreada por Vmp.

    Devuelve la figura; si se indica ruta, la guarda tambien en disco.
    """
    figura, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(vmp, bins=40, edgecolor="black", linewidth=0.4)
    ax1.set_xlabel("Vmp real [V]")
    ax1.set_ylabel("Numero de curvas")
    ax1.set_title("Distribucion de Vmp en el dataset")
    ax1.grid(alpha=0.3)

    sub = np.random.default_rng(0).choice(len(X), size=min(20000, len(X)), replace=False)
    dispersion = ax2.scatter(X[sub, 0], X[sub, 1], c=y[sub, 0], s=3,
                             cmap="viridis", alpha=0.5)
    plt.colorbar(dispersion, ax=ax2, label="Vmp de la curva [V]")
    ax2.set_xlabel("Tension del punto [V]")
    ax2.set_ylabel("Corriente del punto [A]")
    ax2.set_title("Puntos (V, I) coloreados por su etiqueta\n"
                  "(colores mezclados = misma entrada, distinta salida)")
    ax2.grid(alpha=0.3)

    figura.tight_layout()
    if ruta:
        figura.savefig(ruta, dpi=140)
        print(f"[figura] Guardada: {ruta}")
    return figura


def figura_solapamiento(curvas, ids, discrepancias, X, n_ejemplos=2, ruta=None):
    """Dibuja hasta n_ejemplos parejas de curvas que comparten un punto (V, I)
    con Vmp muy distinto.

    Devuelve la figura, o None si no hay discrepancias; si se indica ruta,
    la guarda tambien en disco.
    """
    por_id = {c["id"]: c for c in curvas}

    peores, vistas = [], set()
    for punto, g_a, g_b, delta in sorted(discrepancias, key=lambda t: -t[3]):
        pareja = frozenset((g_a, g_b))
        if pareja in vistas:
            continue
        vistas.add(pareja)
        peores.append((punto, g_a, g_b, delta))
        if len(peores) == n_ejemplos:
            break
    if not peores:
        return None

    figura, ejes = plt.subplots(1, len(peores), figsize=(6.5 * len(peores), 5),
                                squeeze=False)
    for ax, (punto, g_a, g_b, delta) in zip(ejes[0], peores):
        for gi, color in ((g_a, "tab:blue"), (g_b, "tab:red")):
            curva = por_id[ids[gi]]
            ax.plot(curva["V"], np.asarray(curva["V"]) * np.asarray(curva["I"]),
                    color=color, lw=1.8,
                    label=f"curva {curva['id']}  (Vmp = {curva['vmp']:.2f} V)")
            ax.axvline(curva["vmp"], color=color, ls="--", lw=1.2, alpha=0.7)
        ax.plot(X[punto, 0], X[punto, 0] * X[punto, 1], "ko", ms=9,
                label=f"punto comun ({X[punto, 0]:.2f} V, {X[punto, 1]:.2f} A)")
        ax.set_xlabel("Tension [V]")
        ax.set_ylabel("Potencia [W]")
        ax.set_title(f"Mismo punto de operacion, Vmp separados {delta:.2f} V")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    figura.tight_layout()
    if ruta:
        figura.savefig(ruta, dpi=140)
        print(f"[figura] Guardada: {ruta}")
    return figura


def main(ruta_datos=RUTA_DATOS):
    X, y, grupos, info = leer_datos_entrenamiento(ruta_datos,
                                                  n_puntos=N_PUNTOS_CURVA,
                                                  semilla=SEMILLA)

    curvas = info["curvas"]
    ids = np.array([c["id"] for c in curvas])
    vmp = np.array([c["vmp"] for c in curvas], dtype=np.float32)

    # Reindexado de los ids de curva a 0..n-1
    mapa = {identificador: i for i, identificador in enumerate(ids)}
    grupos_idx = np.array([mapa[valor] for valor in grupos], dtype=np.int32)

    print("\n=== DISTRIBUCION DE Vmp ===")
    print(f"media = {vmp.mean():.2f} V | std = {vmp.std():.2f} V | "
          f"min = {vmp.min():.2f} V | max = {vmp.max():.2f} V")
    bordes = np.linspace(vmp.min(), vmp.max(), 11)
    cuenta, _ = np.histogram(vmp, bins=bordes)
    for i, n in enumerate(cuenta):
        barra = "#" * int(40 * n / max(cuenta.max(), 1))
        print(f"  {bordes[i]:5.1f} - {bordes[i+1]:5.1f} V | {n:4d} {barra}")

    print("\n=== TECHO DE R2 CON ENTRADA (V, I) ===")
    var_cond, discrepancias = varianza_condicional(
        X, grupos_idx, vmp, N_VECINOS, N_CONSULTAS, N_REFERENCIA, SEMILLA)
    var_total = float(vmp.var())
    r2_max = 1.0 - var_cond / var_total

    print(f"Var(Vmp) global            = {var_total:8.3f} V^2")
    print(f"E[Var(Vmp | V, I)]         = {var_cond:8.3f} V^2")
    print(f"Error minimo posible (RMSE)= {np.sqrt(var_cond):8.3f} V")
    print(f"R2 MAXIMO ALCANZABLE       = {r2_max:8.3f}")

    if r2_max > 0.9:
        print("\n-> Hay informacion suficiente en (V, I). Si el modelo falla, el\n"
              "   problema es de entrenamiento: revisa normalizacion, learning\n"
              "   rate, numero de epocas o la activacion sigmoide.")
    elif r2_max > 0.5:
        print("\n-> Informacion parcial. El modelo puede ser util pero tendra un\n"
              "   error de fondo que no se elimina ajustando hiperparametros.")
    else:
        print("\n-> El problema esta mal determinado con dos entradas: distintas\n"
              "   curvas comparten punto de operacion con Vmp muy distintos.\n"
              "   Ningun ajuste de la red va a mejorar esto de forma apreciable.")

    directorio = "modelos/figuras"
    os.makedirs(directorio, exist_ok=True)
    figura_distribucion(vmp, X, y,
                        f"{directorio}/diagnostico_distribucion.png" if GUARDAR_FIGURAS else None)
    figura_solapamiento(curvas, ids, discrepancias, X, N_EJEMPLOS,
                        f"{directorio}/diagnostico_solapamiento.png" if GUARDAR_FIGURAS else None)
    if MOSTRAR_FIGURAS:
        plt.show()

    return r2_max


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RUTA_DATOS)
