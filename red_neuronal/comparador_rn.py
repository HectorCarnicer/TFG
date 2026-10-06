"""
comparador_rn.py
================
Compara varios modelos ya entrenados por rn_main.py (carpetas
modelos/mpp_v{ID}/): tabla por consola y graficos comparativos. Cada modelo se
evalua con su propia configuracion (la de su resumen) mediante las funciones de
diagnostico_datos.py. Detalles: README.md.

Parametros principales:
  IDS       : lista de IDs enteros de los modelos a comparar.
  N_MODELOS : numero de modelos esperado; debe coincidir con len(IDS).

Resultados en modelos/comparaciones/cmp_{ID1}-{ID2}-.../ (se reescriben si se
repite la comparacion).

Uso:
    python comparador_rn.py                 (usa IDS)
    python comparador_rn.py 1 2 3           (compara los IDs 1, 2 y 3)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from diagnostico_datos import (cargar_configuracion_modelo, evaluar_modelo,
                               figura_evaluacion, figura_curvas_ejemplo,
                               DIR_DATOS, AGREGACION_CURVA, N_EJEMPLOS_CURVAS)

# --- Parametros --------------------------------------------------------------
IDS = [0, 1, 2, 3]                 # IDs de los modelos a comparar
N_MODELOS = len(IDS)               # numero de modelos esperado (debe coincidir con len(IDS))
DIR_MODELOS = "modelos"            # misma carpeta raiz que usan rn_main.py / diagnostico_datos.py
DIR_COMPARACIONES = "comparaciones"   # subcarpeta de DIR_MODELOS para los resultados
GUARDAR_FIGURAS = True
GUARDAR_FIGURAS_INDIVIDUALES = True   # ademas de las comparativas, una evaluacion/
                                      # curvas_ejemplo (estilo diagnostico_datos.py) por modelo
MOSTRAR_FIGURAS = True


# ---------------------------------------------------------------------------
# Diagnostico de un unico modelo (reutiliza diagnostico_datos.py)
# ---------------------------------------------------------------------------
def diagnosticar_modelo(id_modelo: int, dir_modelos: str, dir_datos: str = DIR_DATOS,
                        verbose: bool = True) -> dict:
    """Evalua un modelo sobre su test con la configuracion de su propio resumen.

    Entrada: id_modelo (entero); dir_modelos (carpeta raiz de los modelos);
        dir_datos (carpeta de los sim_{ID}.dat); verbose.
    Salida: dict de evaluar_modelo (diagnostico_datos.py) mas 'carpeta_modelo'.
        Lanza FileNotFoundError si el modelo no existe.
    Globales: lee AGREGACION_CURVA (de diagnostico_datos); no modifica ninguna.
    """
    carpeta_modelo, config = cargar_configuracion_modelo(dir_modelos, id_modelo, verbose=verbose)
    resultado = evaluar_modelo(carpeta_modelo, id_modelo, config, dir_datos,
                               AGREGACION_CURVA, verbose=verbose)
    resultado["carpeta_modelo"] = carpeta_modelo
    return resultado


# ---------------------------------------------------------------------------
# Salida por consola
# ---------------------------------------------------------------------------
def _etiqueta_ventana(r: dict) -> str:
    """Texto corto del muestreo de un resultado.

    Entrada: r (dict de resultado de diagnosticar_modelo).
    Salida: 'dV=..' (paso fijo) o 'vent=..' (ventana aleatoria).
    Globales: ninguna.
    """
    return f"dV={r['delta_v']:.2f}" if r["paso_fijo"] else f"vent={r['ventana']:.2f}"


def imprimir_tabla_comparativa(resultados: list[dict]) -> None:
    """Imprime una tabla con una fila por modelo (ventana, datos y metricas de
    test, punto a punto y por curva) y el modelo con mejor R2 por curva.

    Entrada: resultados (lista de dicts de diagnosticar_modelo).
    Salida: ninguna (solo imprime por consola).
    Globales: ninguna.
    """
    print("\n=== COMPARATIVA DE MODELOS ===")
    columnas = [
        ("ID", 4), ("n", 3), ("ventana", 10), ("sentido", 11), ("datos", 10),
        ("curvas test", 11), ("R2 punto", 9), ("RMSE punto", 11),
        ("R2 curva", 9), ("RMSE curva", 11),
    ]
    cabecera = " | ".join(f"{titulo:>{ancho}}" for titulo, ancho in columnas)
    print(cabecera)
    print("-" * len(cabecera))
    for r in resultados:
        sentido = "ascendente" if r["ascendente"] else "descendente"
        datos = ",".join(str(i) for i in r["ids_datos"])
        fila = [
            f"{r['id']:>4}", f"{r['n_muestras']:>3}", f"{_etiqueta_ventana(r):>10}",
            f"{sentido:>11}", f"{datos:>10}", f"{len(r['curvas_test']):>11}",
            f"{r['metricas_punto']['R2']:>9.4f}", f"{r['metricas_punto']['RMSE [V]']:>11.4f}",
            f"{r['metricas_curva']['R2']:>9.4f}", f"{r['metricas_curva']['RMSE [V]']:>11.4f}",
        ]
        print(" | ".join(fila))

    mejor = max(resultados, key=lambda r: r["metricas_curva"]["R2"])
    print(f"\nMejor R2 de test (por curva): ID={mejor['id']}  "
          f"(R2={mejor['metricas_curva']['R2']:.4f}, "
          f"n_muestras={mejor['n_muestras']}, {_etiqueta_ventana(mejor)}, "
          f"{'ascendente' if mejor['ascendente'] else 'descendente'})")


# ---------------------------------------------------------------------------
# Figuras comparativas
# ---------------------------------------------------------------------------
def figura_comparacion_metricas(resultados: list[dict], ruta: str | None = None):
    """Dibuja barras de R2 y RMSE de test (punto a punto y por curva) por modelo.

    Entrada: resultados (lista de dicts de diagnosticar_modelo); ruta (si se
        indica, guarda la figura).
    Salida: la figura de matplotlib.
    Globales: crea una figura en el estado global de pyplot.
    """
    etiquetas = [f"ID {r['id']}\nn={r['n_muestras']}  {_etiqueta_ventana(r)}"
                for r in resultados]
    r2_punto = [r["metricas_punto"]["R2"] for r in resultados]
    r2_curva = [r["metricas_curva"]["R2"] for r in resultados]
    rmse_punto = [r["metricas_punto"]["RMSE [V]"] for r in resultados]
    rmse_curva = [r["metricas_curva"]["RMSE [V]"] for r in resultados]
    x = np.arange(len(resultados))
    ancho = 0.35

    figura, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, 2.6 * len(resultados)), 5))

    ax1.bar(x - ancho / 2, r2_punto, ancho, label="punto a punto", color="tab:blue")
    ax1.bar(x + ancho / 2, r2_curva, ancho, label="por curva", color="tab:cyan")
    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas, fontsize=8)
    ax1.set_ylabel("R2 de test")
    ax1.set_title("R2 real de test por modelo")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(x - ancho / 2, rmse_punto, ancho, label="punto a punto", color="tab:red")
    ax2.bar(x + ancho / 2, rmse_curva, ancho, label="por curva", color="tab:orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels(etiquetas, fontsize=8)
    ax2.set_ylabel("RMSE de test [V]")
    ax2.set_title("Error real de test por modelo")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    figura.tight_layout()
    if ruta:
        figura.savefig(ruta, dpi=140)
        print(f"[figura] Guardada: {ruta}")
    return figura


def figura_comparacion_distribucion(resultados: list[dict], ruta: str | None = None):
    """Dibuja el histograma solapado de Vmp de las curvas de test y las curvas
    descartadas de cada modelo.

    Entrada: resultados (lista de dicts de diagnosticar_modelo); ruta (si se
        indica, guarda la figura).
    Salida: la figura de matplotlib.
    Globales: crea una figura en el estado global de pyplot.
    """
    figura, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for r in resultados:
        ax1.hist(r["vmp_real"], bins=30, alpha=0.5, edgecolor="black", linewidth=0.3,
                 label=f"ID {r['id']}")
    ax1.set_xlabel("Vmp real [V]")
    ax1.set_ylabel("Numero de curvas de test")
    ax1.set_title("Distribucion de Vmp en el conjunto de test, por modelo")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    etiquetas = [f"ID {r['id']}" for r in resultados]
    x = np.arange(len(resultados))
    ax2.bar(x, [r["n_curvas_descartadas"] for r in resultados], color="tab:gray")
    ax2.set_xticks(x)
    ax2.set_xticklabels(etiquetas, fontsize=8)
    ax2.set_ylabel("Curvas descartadas")
    ax2.set_title("Curvas sin rango suficiente para la ventana")
    ax2.grid(alpha=0.3, axis="y")

    figura.tight_layout()
    if ruta:
        figura.savefig(ruta, dpi=140)
        print(f"[figura] Guardada: {ruta}")
    return figura


# ---------------------------------------------------------------------------
def main(ids: list[int] | None = None):
    """Compara los modelos indicados: tabla por consola y figuras en disco.

    Entrada: ids (lista de IDs enteros; por defecto IDS, que debe tener N_MODELOS
        elementos). Los IDs sin modelo entrenado se avisan y se omiten.
    Salida: lista de dicts de diagnosticar_modelo (uno por modelo valido).
        Lanza ValueError si IDS no coincide con N_MODELOS o hay menos de 2
        modelos validos. Guarda las figuras en modelos/comparaciones/cmp_.../.
    Globales: lee IDS, N_MODELOS, DIR_MODELOS, DIR_COMPARACIONES,
        GUARDAR_FIGURAS, GUARDAR_FIGURAS_INDIVIDUALES y MOSTRAR_FIGURAS; no
        modifica ninguna.
    """
    if ids is None:
        ids = IDS
        if len(ids) != N_MODELOS:
            raise ValueError(
                f"IDS tiene {len(ids)} elementos pero N_MODELOS={N_MODELOS}; "
                f"revisa la lista IDS o actualiza N_MODELOS en comparador_rn.py.")
    ids = list(ids)

    resultados = []
    for id_modelo in ids:
        try:
            resultados.append(diagnosticar_modelo(id_modelo, DIR_MODELOS))
        except FileNotFoundError as error:
            print(f"[aviso] Se omite ID={id_modelo}: {error}")

    if len(resultados) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 modelos validos para comparar; "
            f"solo se pudieron cargar {len(resultados)} de {len(ids)} IDS.")

    imprimir_tabla_comparativa(resultados)

    nombre_comparacion = "cmp_" + "-".join(str(r["id"]) for r in resultados)
    directorio = os.path.join(DIR_MODELOS, DIR_COMPARACIONES, nombre_comparacion)
    os.makedirs(directorio, exist_ok=True)
    print(f"\n[comparador] Resultados en: {directorio}  (se reescribira si ya existia)")

    figura_comparacion_metricas(
        resultados,
        os.path.join(directorio, f"{nombre_comparacion}_metricas.png") if GUARDAR_FIGURAS else None)
    figura_comparacion_distribucion(
        resultados,
        os.path.join(directorio, f"{nombre_comparacion}_distribucion.png") if GUARDAR_FIGURAS else None)

    if GUARDAR_FIGURAS_INDIVIDUALES:
        for r in resultados:
            nombre_modelo = f"mpp_v{r['id']}"
            figura_evaluacion(
                r["y_test"], r["y_pred_test"], r["vmp_real"], r["vmp_pred"],
                os.path.join(directorio, f"{nombre_modelo}_diagnostico_evaluacion.png")
                if GUARDAR_FIGURAS else None)
            figura_curvas_ejemplo(
                r["curvas_test"], r["vmp_pred"], N_EJEMPLOS_CURVAS,
                os.path.join(directorio, f"{nombre_modelo}_diagnostico_curvas_ejemplo.png")
                if GUARDAR_FIGURAS else None)

    if MOSTRAR_FIGURAS:
        plt.show()

    return resultados


if __name__ == "__main__":
    _ids_cli = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(_ids_cli)
