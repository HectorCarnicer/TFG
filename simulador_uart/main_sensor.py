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
from simulador_sensor import abrir_puerto, cerrar_puerto, ejecutar_modo_simulacion


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
# El primer punto es aleatorio; cada siguiente sigue la tensión predicha
# por el micro, cambiando de curva de trabajo cada MUESTRAS_HASTA_CAMBIO
# muestras, hasta alcanzar MUESTRAS_TOTALES.
MUESTRAS_HASTA_CAMBIO = 100     # muestras de seguimiento antes de cambiar de curva
MUESTRAS_TOTALES = 2000        # muestras totales del test

# Tipo de muestreo:
# "Instantaneo" : el siguiente punto es directamente la predicción.
# "Relativo"    : la tensión objetivo se acerca gradualmente a la
#                 predicción, un paso de
#                 (V_predicho - V_actual) / MUESTRAS_HASTA_CAMBIO
#                 por muestra.
TIPO_MUESTREO = "Instantaneo"

SEMILLA_ALEATORIA =  13547   # None = no reproducible

# --- Logging ---
DIRECTORIO_LOGS = "logs"
ID_TEST = "029"               # identifica logs/test_{ID_TEST}.log


# ============================================================================
# EJECUCIÓN
# ============================================================================

def main():
    if TIPO_MUESTREO not in ("Instantaneo", "Relativo"):
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
        "Puerto serie": PUERTO,
        "Baudios": BAUDIOS,
        "Archivo de datos": RUTA_DATOS,
        "Muestras hasta cambio de curva": MUESTRAS_HASTA_CAMBIO,
        "Muestras totales": MUESTRAS_TOTALES,
        "Tipo de muestreo": TIPO_MUESTREO,
        "Semilla aleatoria": SEMILLA_ALEATORIA,
    }

    escribir_cabecera_log(fh_log, ID_TEST, campos_cabecera)

    ser = None
    try:
        print(f"Abriendo puerto {PUERTO} @ {BAUDIOS} baudios...")
        ser = abrir_puerto(PUERTO, BAUDIOS, TIMEOUT_RESPUESTA_S)

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
