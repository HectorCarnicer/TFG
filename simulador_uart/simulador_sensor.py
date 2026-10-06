"""
simulador_sensor.py

Comunicación UART con el microcontrolador y los dos modos de ejecución:

  - ejecutar_simulacion       -> modo "Consecutivo"
  - ejecutar_modo_simulacion  -> modo "Simulacion"

Formato de respuesta del microcontrolador tras cada muestra:

    "V= %6.2f V | I= %5.2f A | Prediccion= %9.4f | t_inferencia= %6lu us\r\n"
"""

import random
import re
import struct
import time

try:
    import serial  # pyserial
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Falta la dependencia 'pyserial'. Instálala con: pip install pyserial"
    ) from exc

from lectura_escritura import escribir_log


VALOR_RELLENO = -1.0  # marca los puntos de relleno al final de cada curva

PATRON_RESPUESTA = re.compile(
    r"V=\s*(?P<V>-?\d+(?:\.\d+)?)\s*V\s*\|\s*"
    r"I=\s*(?P<I>-?\d+(?:\.\d+)?)\s*A\s*\|\s*"
    r"Prediccion=\s*(?P<pred>-?\d+(?:\.\d+)?)\s*\|\s*"
    r"t_inferencia=\s*(?P<t_us>\d+)\s*us"
)


class RespuestaMicro:
    """Respuesta parseada de una muestra enviada al microcontrolador."""

    def __init__(self, v_eco, i_eco, prediccion_vmp, t_inferencia_us, linea_cruda):
        self.v_eco = v_eco
        self.i_eco = i_eco
        self.prediccion_vmp = prediccion_vmp
        self.t_inferencia_us = t_inferencia_us
        self.linea_cruda = linea_cruda


class ResumenCurva:
    """Resumen de error y estadísticas de una curva o tramo procesado."""

    def __init__(
        self,
        id_curva,
        n_puntos,
        vmp_real,
        error_medio_abs,
        error_final_abs,
        error_final_rel_pct,
        n_respuestas_ok,
        n_timeouts,
    ):
        self.id_curva = id_curva
        self.n_puntos = n_puntos
        self.vmp_real = vmp_real
        self.error_medio_abs = error_medio_abs
        self.error_final_abs = error_final_abs
        self.error_final_rel_pct = error_final_rel_pct
        self.n_respuestas_ok = n_respuestas_ok
        self.n_timeouts = n_timeouts


# ---------------------------------------------------------------------------
# Puerto serie
# ---------------------------------------------------------------------------

def abrir_puerto(puerto, baudios, timeout_s):
    """Abre el puerto serie y limpia los buffers de entrada/salida.

    Espera 2s tras abrir para permitir el reset de la placa.
    """
    ser = serial.Serial(port=puerto, baudrate=baudios, timeout=timeout_s)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def cerrar_puerto(ser):
    """Cierra el puerto serie, limpiando antes el buffer de entrada."""
    try:
        ser.reset_input_buffer()
    finally:
        ser.close()


# ---------------------------------------------------------------------------
# Envío de puntos [V, I]
# ---------------------------------------------------------------------------

def empaquetar_punto(v, i):
    """Empaqueta (V, I) como dos floats little-endian, sin cabecera (8 bytes)."""
    return struct.pack("<ff", v, i)


def enviar_punto(ser, v, i):
    """Envía un punto (V, I) por UART."""
    ser.write(empaquetar_punto(v, i))
    ser.flush()


# ---------------------------------------------------------------------------
# Lectura/parseo de la respuesta del microcontrolador
# ---------------------------------------------------------------------------

def leer_linea_respuesta(ser):
    """Lee una línea de la UART terminada en '\\r\\n'.

    Devuelve None si expira el timeout del puerto sin recibir la línea.
    """
    bruta = ser.readline()
    if not bruta:
        return None
    try:
        return bruta.decode("ascii", errors="replace").strip("\r\n")
    except UnicodeDecodeError:
        return None


def parsear_respuesta(linea):
    """Parsea una línea de respuesta del microcontrolador.

    Devuelve None si la línea no encaja con el formato esperado.
    """
    m = PATRON_RESPUESTA.search(linea)
    if not m:
        return None
    return RespuestaMicro(
        v_eco=float(m.group("V")),
        i_eco=float(m.group("I")),
        prediccion_vmp=float(m.group("pred")),
        t_inferencia_us=int(m.group("t_us")),
        linea_cruda=linea,
    )


# ---------------------------------------------------------------------------
# Selección de puntos de una curva
# ---------------------------------------------------------------------------

def puntos_validos(curva):
    """Devuelve los puntos (V, I) de una curva, descartando el relleno."""
    V = curva["V"]
    I = curva["I"]
    return [(v, i) for v, i in zip(V, I) if v != VALOR_RELLENO]


def seleccionar_puntos_aleatorios(curva, n_puntos, rng):
    """Escoge n_puntos puntos (V, I) al azar de la curva, sin reemplazo,
    y los devuelve ordenados de mayor a menor tensión.
    """
    validos = puntos_validos(curva)
    n = min(n_puntos, len(validos))
    if n <= 0:
        return []

    muestra = rng.sample(validos, n)
    muestra.sort(key=lambda par: par[0], reverse=True)
    return muestra


# ---------------------------------------------------------------------------
# Cálculo de error
# ---------------------------------------------------------------------------

def calcular_error(prediccion_vmp, vmp_real):
    """Calcula el error absoluto (V) y relativo (%) de la predicción."""
    error_abs = prediccion_vmp - vmp_real
    error_rel_pct = (error_abs / vmp_real * 100.0) if vmp_real != 0 else float("nan")
    return error_abs, error_rel_pct


def punto_mas_cercano(puntos, v_objetivo):
    """Devuelve el punto (V, I) de la lista con la V más cercana a v_objetivo."""
    return min(puntos, key=lambda par: abs(par[0] - v_objetivo))


def calcular_v_objetivo(tipo_muestreo, v_actual, v_predicho, muestras_hasta_cambio):
    """Calcula la tensión objetivo del siguiente punto según tipo_muestreo:
    salto directo a v_predicho ("Instantaneo"), o acercamiento gradual con
    paso (v_predicho - v_actual) / muestras_hasta_cambio ("Relativo").
    """
    if tipo_muestreo == "Relativo":
        return v_actual + (v_predicho - v_actual) / muestras_hasta_cambio
    return v_predicho


# ---------------------------------------------------------------------------
# Envío + lectura + reintentos de una única muestra
# ---------------------------------------------------------------------------

def intentar_muestra(ser, v, i, reintentos_timeout):
    """Envía un punto (V, I) y espera respuesta, reintentando (mismo
    punto) hasta reintentos_timeout veces. Devuelve None si se agotan los
    intentos sin respuesta válida.
    """
    intentos = reintentos_timeout + 1
    for _intento in range(intentos):
        enviar_punto(ser, v, i)
        linea = leer_linea_respuesta(ser)
        if linea:
            respuesta = parsear_respuesta(linea)
            if respuesta is not None:
                return respuesta
    return None


# ---------------------------------------------------------------------------
# Procesado de una curva completa (modo "Consecutivo")
# ---------------------------------------------------------------------------

def procesar_curva(
    ser,
    fh_log,
    simulacion,
    n_puntos,
    rng,
    reintentos_timeout=1,
    retardo_entre_puntos_s=0.0,
):
    """Envía n_puntos puntos aleatorios de una curva, calcula el error de
    cada respuesta frente al Vmp real y loguea cada muestra.

    Devuelve el ResumenCurva de la curva procesada.
    """
    id_curva = simulacion["id"]
    vmp_real = simulacion["mpp"]["Vmp [V]"]
    curva = simulacion["curva"]

    puntos = seleccionar_puntos_aleatorios(curva, n_puntos, rng)

    errores_abs = []
    n_ok = 0
    n_timeouts = 0
    ultimo_error_abs = float("nan")
    ultimo_error_rel = float("nan")

    for k, (v, i) in enumerate(puntos, start=1):
        respuesta = intentar_muestra(ser, v, i, reintentos_timeout)

        prefijo = f"[Curva {id_curva:03d}] Punto {k:3d}/{len(puntos)}"

        if respuesta is None:
            n_timeouts += 1
            escribir_log(
                fh_log,
                f"{prefijo} -> V={v:6.2f} V | I={i:5.2f} A | "
                f"SIN RESPUESTA del microcontrolador (timeout)",
            )
            if retardo_entre_puntos_s > 0:
                time.sleep(retardo_entre_puntos_s)
            continue

        n_ok += 1
        error_abs, error_rel_pct = calcular_error(respuesta.prediccion_vmp, vmp_real)
        errores_abs.append(abs(error_abs))
        ultimo_error_abs = error_abs
        ultimo_error_rel = error_rel_pct

        linea_completa = (
            f"{prefijo} -> V={respuesta.v_eco:6.2f} V | I={respuesta.i_eco:5.2f} A | "
            f"Prediccion={respuesta.prediccion_vmp:9.4f} | "
            f"t_inferencia={respuesta.t_inferencia_us:6d} us | "
            f"Vmp_real={vmp_real:6.2f} V | "
            f"Error={error_abs:+7.4f} V ({error_rel_pct:+6.2f} %)"
        )
        escribir_log(fh_log, linea_completa)

        if retardo_entre_puntos_s > 0:
            time.sleep(retardo_entre_puntos_s)

    error_medio_abs = sum(errores_abs) / len(errores_abs) if errores_abs else float("nan")

    resumen = ResumenCurva(
        id_curva=id_curva,
        n_puntos=len(puntos),
        vmp_real=vmp_real,
        error_medio_abs=error_medio_abs,
        error_final_abs=ultimo_error_abs,
        error_final_rel_pct=ultimo_error_rel,
        n_respuestas_ok=n_ok,
        n_timeouts=n_timeouts,
    )

    escribir_log(
        fh_log,
        (
            f"--- Resumen curva {id_curva:03d}: Vmp_real={resumen.vmp_real:6.2f} V | "
            f"error medio |E|={resumen.error_medio_abs:7.4f} V | "
            f"error última muestra={resumen.error_final_abs:+7.4f} V "
            f"({resumen.error_final_rel_pct:+6.2f} %) | "
            f"OK={resumen.n_respuestas_ok} timeouts={resumen.n_timeouts} ---"
        ),
    )

    return resumen


# ---------------------------------------------------------------------------
# Simulación completa (todas las curvas configuradas)
# ---------------------------------------------------------------------------

def ejecutar_simulacion(
    ser,
    fh_log,
    dataset,
    n_curvas,
    n_puntos_por_curva,
    semilla=None,
    reintentos_timeout=1,
    retardo_entre_puntos_s=0.0,
):
    """Recorre n_curvas curvas del dataset (todas si es None), enviando
    n_puntos_por_curva puntos aleatorios de cada una.

    Devuelve la lista de ResumenCurva, una por curva procesada.
    """
    rng = random.Random(semilla)
    simulaciones = dataset["simulaciones"]
    if n_curvas is not None:
        simulaciones = simulaciones[:n_curvas]

    resumenes = []
    t_inicio = time.time()

    for simulacion in simulaciones:
        resumen = procesar_curva(
            ser=ser,
            fh_log=fh_log,
            simulacion=simulacion,
            n_puntos=n_puntos_por_curva,
            rng=rng,
            reintentos_timeout=reintentos_timeout,
            retardo_entre_puntos_s=retardo_entre_puntos_s,
        )
        resumenes.append(resumen)

    duracion_s = time.time() - t_inicio
    _escribir_resumen_global(fh_log, resumenes, duracion_s)

    return resumenes


def _elegir_curva_distinta(simulaciones, curva_actual, rng):
    """Escoge al azar otra curva del dataset con irradiancias distintas a
    curva_actual. Si no hay ninguna, escoge cualquiera del dataset.
    """
    irr_actual = curva_actual["irradiancias_substrings_Wm2"]
    candidatas = [
        s for s in simulaciones if s["irradiancias_substrings_Wm2"] != irr_actual
    ]
    if not candidatas:
        candidatas = simulaciones
    return rng.choice(candidatas)


def _procesar_segmento_simulacion(
    ser,
    fh_log,
    curva,
    puntos_curva,
    v_inicial,
    i_inicial,
    muestras_hasta_cambio,
    muestras_totales,
    muestras_enviadas,
    tipo_muestreo,
    reintentos_timeout,
    retardo_entre_puntos_s,
):
    """Procesa un tramo de seguimiento sobre una curva de trabajo: envía
    v_inicial/i_inicial y cada punto siguiente es el más cercano a una
    tensión objetivo continua, recalculada según tipo_muestreo. Termina a
    las muestras_hasta_cambio muestras del tramo, o antes si se alcanza
    muestras_totales del test completo.

    Devuelve (ResumenCurva del tramo, última tensión objetivo,
    muestras_enviadas actualizado).
    """
    id_curva = curva["id"]
    vmp_real = curva["mpp"]["Vmp [V]"]

    v, i = v_inicial, i_inicial
    v_objetivo = v_inicial  # tensión objetivo continua, no discretizada
    errores_abs = []
    n_ok = 0
    n_timeouts = 0
    ultimo_error_abs = float("nan")
    ultimo_error_rel = float("nan")

    k = 0
    while k < muestras_hasta_cambio and muestras_enviadas < muestras_totales:
        k += 1
        muestras_enviadas += 1

        respuesta = intentar_muestra(ser, v, i, reintentos_timeout)
        prefijo = f"[Curva {id_curva:03d}] Punto {k:3d}/{muestras_hasta_cambio}"

        if respuesta is None:
            n_timeouts += 1
            escribir_log(
                fh_log,
                f"{prefijo} -> V={v:6.2f} V | I={i:5.2f} A | "
                f"SIN RESPUESTA del microcontrolador (timeout)",
            )
            if retardo_entre_puntos_s > 0:
                time.sleep(retardo_entre_puntos_s)
            continue

        n_ok += 1
        error_abs, error_rel_pct = calcular_error(respuesta.prediccion_vmp, vmp_real)
        errores_abs.append(abs(error_abs))
        ultimo_error_abs = error_abs
        ultimo_error_rel = error_rel_pct

        linea_completa = (
            f"{prefijo} -> V={respuesta.v_eco:6.2f} V | I={respuesta.i_eco:5.2f} A | "
            f"Prediccion={respuesta.prediccion_vmp:9.4f} | "
            f"t_inferencia={respuesta.t_inferencia_us:6d} us | "
            f"Vmp_real={vmp_real:6.2f} V | "
            f"Error={error_abs:+7.4f} V ({error_rel_pct:+6.2f} %)"
        )
        escribir_log(fh_log, linea_completa)

        v_objetivo = calcular_v_objetivo(
            tipo_muestreo, v_objetivo, respuesta.prediccion_vmp, muestras_hasta_cambio
        )
        v, i = punto_mas_cercano(puntos_curva, v_objetivo)

        if retardo_entre_puntos_s > 0:
            time.sleep(retardo_entre_puntos_s)

    error_medio_abs = sum(errores_abs) / len(errores_abs) if errores_abs else float("nan")

    resumen = ResumenCurva(
        id_curva=id_curva,
        n_puntos=k,
        vmp_real=vmp_real,
        error_medio_abs=error_medio_abs,
        error_final_abs=ultimo_error_abs,
        error_final_rel_pct=ultimo_error_rel,
        n_respuestas_ok=n_ok,
        n_timeouts=n_timeouts,
    )

    escribir_log(
        fh_log,
        (
            f"--- Resumen curva {id_curva:03d}: Vmp_real={resumen.vmp_real:6.2f} V | "
            f"error medio |E|={resumen.error_medio_abs:7.4f} V | "
            f"error última muestra={resumen.error_final_abs:+7.4f} V "
            f"({resumen.error_final_rel_pct:+6.2f} %) | "
            f"OK={resumen.n_respuestas_ok} timeouts={resumen.n_timeouts} ---"
        ),
    )

    return resumen, v_objetivo, muestras_enviadas


# ---------------------------------------------------------------------------
# Simulación completa (modo "Simulacion": seguimiento con cambios de curva)
# ---------------------------------------------------------------------------

def ejecutar_modo_simulacion(
    ser,
    fh_log,
    dataset,
    muestras_hasta_cambio,
    muestras_totales,
    tipo_muestreo="Instantaneo",
    semilla=None,
    reintentos_timeout=1,
    retardo_entre_puntos_s=0.0,
):
    """Modo "Simulacion": el primer punto es aleatorio; cada siguiente
    sigue la tensión objetivo derivada de la predicción del micro (ver
    calcular_v_objetivo), cambiando de curva de trabajo cada
    muestras_hasta_cambio muestras hasta alcanzar muestras_totales.

    Devuelve la lista de ResumenCurva, uno por tramo de trabajo.
    """
    if tipo_muestreo not in ("Instantaneo", "Relativo"):
        raise ValueError(
            f"tipo_muestreo no reconocido: {tipo_muestreo!r} "
            "(debe ser 'Instantaneo' o 'Relativo')"
        )

    rng = random.Random(semilla)
    simulaciones = dataset["simulaciones"]
    if not simulaciones:
        escribir_log(fh_log, "El dataset no contiene curvas.")
        return []

    resumenes = []
    t_inicio = time.time()
    muestras_enviadas = 0

    curva_actual = rng.choice(simulaciones)
    puntos_curva = puntos_validos(curva_actual["curva"])
    v_actual, i_actual = rng.choice(puntos_curva)

    escribir_log(
        fh_log,
        f"=== Curva de trabajo inicial: id={curva_actual['id']:03d} "
        f"irradiancias={curva_actual['irradiancias_substrings_Wm2']} Vmp_real="
        f"{curva_actual['mpp']['Vmp [V]']:.2f} V ===",
    )

    while muestras_enviadas < muestras_totales:
        resumen, v_objetivo_final, muestras_enviadas = _procesar_segmento_simulacion(
            ser=ser,
            fh_log=fh_log,
            curva=curva_actual,
            puntos_curva=puntos_curva,
            v_inicial=v_actual,
            i_inicial=i_actual,
            muestras_hasta_cambio=muestras_hasta_cambio,
            muestras_totales=muestras_totales,
            muestras_enviadas=muestras_enviadas,
            tipo_muestreo=tipo_muestreo,
            reintentos_timeout=reintentos_timeout,
            retardo_entre_puntos_s=retardo_entre_puntos_s,
        )
        resumenes.append(resumen)

        if muestras_enviadas >= muestras_totales:
            break

        curva_actual = _elegir_curva_distinta(simulaciones, curva_actual, rng)
        puntos_curva = puntos_validos(curva_actual["curva"])
        v_actual, i_actual = punto_mas_cercano(puntos_curva, v_objetivo_final)

        escribir_log(
            fh_log,
            f"=== Cambio de curva de trabajo -> id={curva_actual['id']:03d} "
            f"irradiancias={curva_actual['irradiancias_substrings_Wm2']} Vmp_real="
            f"{curva_actual['mpp']['Vmp [V]']:.2f} V ===",
        )

    duracion_s = time.time() - t_inicio
    _escribir_resumen_global(fh_log, resumenes, duracion_s)

    return resumenes


def _escribir_resumen_global(fh_log, resumenes, duracion_s):
    """Escribe el resumen global del test: MAE, timeouts y peor curva."""
    if not resumenes:
        escribir_log(fh_log, "No se ha procesado ninguna curva.")
        return

    errores_medios = [r.error_medio_abs for r in resumenes if r.error_medio_abs == r.error_medio_abs]
    total_ok = sum(r.n_respuestas_ok for r in resumenes)
    total_timeouts = sum(r.n_timeouts for r in resumenes)

    mae_global = sum(errores_medios) / len(errores_medios) if errores_medios else float("nan")
    peor_curva = max(resumenes, key=lambda r: (r.error_medio_abs if r.error_medio_abs == r.error_medio_abs else -1))

    escribir_log(fh_log, "=" * 70)
    escribir_log(fh_log, "RESUMEN GLOBAL DEL TEST")
    escribir_log(fh_log, f"  Curvas procesadas      : {len(resumenes)}")
    escribir_log(fh_log, f"  Muestras OK / timeouts : {total_ok} / {total_timeouts}")
    escribir_log(fh_log, f"  MAE global (media de |error| por curva): {mae_global:.4f} V")
    escribir_log(
        fh_log,
        f"  Peor curva             : {peor_curva.id_curva:03d} "
        f"(error medio {peor_curva.error_medio_abs:.4f} V)",
    )
    escribir_log(fh_log, f"  Duración total         : {duracion_s:.1f} s")
    escribir_log(fh_log, "=" * 70)
