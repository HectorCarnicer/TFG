"""
main_sensor.py

Punto de entrada: parámetros de configuración del test y llamada a
lectura_escritura.py y simulador_sensor.py para ejecutarlo.

Uso:
    python main_sensor.py
"""

import os
import sys

from lectura_escritura import (
    abrir_log,
    cargar_dataset,
    cerrar_log,
    escribir_cabecera_log,
    escribir_log,
)
from simulador_sensor import (
    abrir_puerto,
    cerrar_puerto,
    ejecutar_modo_simulacion,
    ejecutar_simulacion,
)


# ============================================================================
# PARÁMETROS DE FUNCIONAMIENTO
# ============================================================================

# --- Comunicación UART ---
PUERTO = "/dev/ttyUSB0"        # puerto serie del microcontrolador
BAUDIOS = 115200
TIMEOUT_RESPUESTA_S = 2.0      # tiempo máximo de espera de respuesta por punto
REINTENTOS_POR_TIMEOUT = 1     # reintentos si no llega respuesta a tiempo
RETARDO_ENTRE_PUNTOS_S = 0.0   # pausa entre envíos; 0 = sin pausa

# --- Datos de entrada ---
DIRECTORIO_DATOS = "datos"
ID_DATOS = "1"                 # selecciona datos/sim_{ID_DATOS}.dat
RUTA_DATOS = os.path.join(DIRECTORIO_DATOS, f"sim_{ID_DATOS}.dat")

# --- Modo de simulación ---
# "Consecutivo" : recorre curvas del dataset una a una, enviando de cada
#                 una varios puntos aleatorios.
# "Simulacion"  : el primer punto es aleatorio; cada siguiente sigue la
#                 tensión predicha por el micro, cambiando de curva de
#                 trabajo cada MUESTRAS_HASTA_CAMBIO muestras, hasta
#                 alcanzar MUESTRAS_TOTALES.
MODO_SIM = "Consecutivo"

# Parámetros del modo "Consecutivo"
N_CURVAS = None                # None = todas las curvas del dataset
N_PUNTOS_POR_CURVA = 20        # puntos aleatorios enviados por curva

# Parámetros del modo "Simulacion"
MUESTRAS_HASTA_CAMBIO = 15     # muestras de seguimiento antes de cambiar de curva
MUESTRAS_TOTALES = 300         # muestras totales del test

# Tipo de muestreo dentro del modo "Simulacion":
# "Instantaneo" : el siguiente punto es el más cercano a la predicción.
# "Relativo"    : la tensión objetivo se acerca gradualmente a la
#                 predicción, un paso de
#                 (V_predicho - V_actual) / MUESTRAS_HASTA_CAMBIO
#                 por muestra.
TIPO_MUESTREO = "Instantaneo"

# Común a ambos modos
SEMILLA_ALEATORIA = 43753      # None = no reproducible

# --- Logging ---
DIRECTORIO_LOGS = "logs"
ID_TEST = "001"                # identifica logs/test_{ID_TEST}.log


# ============================================================================
# EJECUCIÓN
# ============================================================================

def main():
    if MODO_SIM not in ("Consecutivo", "Simulacion"):
        raise ValueError(
            f"MODO_SIM no reconocido: {MODO_SIM!r} (debe ser 'Consecutivo' o 'Simulacion')"
        )
    if MODO_SIM == "Simulacion" and TIPO_MUESTREO not in ("Instantaneo", "Relativo"):
        raise ValueError(
            f"TIPO_MUESTREO no reconocido: {TIPO_MUESTREO!r} "
            "(debe ser 'Instantaneo' o 'Relativo')"
        )

    print(f"Cargando dataset de curvas I-V desde '{RUTA_DATOS}'...")
    dataset = cargar_dataset(RUTA_DATOS)
    n_disponibles = len(dataset["simulaciones"])
    print(f"  -> {n_disponibles} curvas disponibles en el dataset.")

    fh_log = abrir_log(DIRECTORIO_LOGS, ID_TEST)

    campos_cabecera = {
        "Modo de simulación": MODO_SIM,
        "Puerto serie": PUERTO,
        "Baudios": BAUDIOS,
        "Archivo de datos": RUTA_DATOS,
    }
    if MODO_SIM == "Consecutivo":
        campos_cabecera["Nº curvas a usar"] = N_CURVAS if N_CURVAS is not None else n_disponibles
        campos_cabecera["Nº puntos/curva"] = N_PUNTOS_POR_CURVA
    else:
        campos_cabecera["Muestras hasta cambio de curva"] = MUESTRAS_HASTA_CAMBIO
        campos_cabecera["Muestras totales"] = MUESTRAS_TOTALES
        campos_cabecera["Tipo de muestreo"] = TIPO_MUESTREO
    campos_cabecera["Semilla aleatoria"] = SEMILLA_ALEATORIA

    escribir_cabecera_log(fh_log, ID_TEST, campos_cabecera)

    ser = None
    try:
        print(f"Abriendo puerto {PUERTO} @ {BAUDIOS} baudios...")
        ser = abrir_puerto(PUERTO, BAUDIOS, TIMEOUT_RESPUESTA_S)

        if MODO_SIM == "Consecutivo":
            ejecutar_simulacion(
                ser=ser,
                fh_log=fh_log,
                dataset=dataset,
                n_curvas=N_CURVAS,
                n_puntos_por_curva=N_PUNTOS_POR_CURVA,
                semilla=SEMILLA_ALEATORIA,
                reintentos_timeout=REINTENTOS_POR_TIMEOUT,
                retardo_entre_puntos_s=RETARDO_ENTRE_PUNTOS_S,
            )
        else:
            ejecutar_modo_simulacion(
                ser=ser,
                fh_log=fh_log,
                dataset=dataset,
                muestras_hasta_cambio=MUESTRAS_HASTA_CAMBIO,
                muestras_totales=MUESTRAS_TOTALES,
                tipo_muestreo=TIPO_MUESTREO,
                semilla=SEMILLA_ALEATORIA,
                reintentos_timeout=REINTENTOS_POR_TIMEOUT,
                retardo_entre_puntos_s=RETARDO_ENTRE_PUNTOS_S,
            )

    except KeyboardInterrupt:
        escribir_log(fh_log, "\nTest interrumpido manualmente (Ctrl+C).")
        return 130

    except Exception as exc:  # noqa: BLE001
        escribir_log(fh_log, f"\nERROR durante el test: {exc!r}")
        raise

    finally:
        if ser is not None:
            cerrar_puerto(ser)
        cerrar_log(fh_log)

    print(f"\nTest finalizado. Log guardado en: logs/test_{ID_TEST}.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
