"""
muestreo_ventanas.py
====================
Construccion de las ventanas [[V_1,I_1],...,[V_n,I_n]] que recibe la red, a
partir de una curva I-V, simulando el barrido de un algoritmo Perturb & Observe.

Dos modos (parametro paso_fijo):
  - True : n muestras separadas un paso fijo delta_v desde un punto de arranque.
  - False: n muestras a tension aleatoria dentro de una ventana de anchura fija.

Las corrientes se obtienen por interpolacion lineal sobre la curva. Detalles
del muestreo y de la interpolacion: README.md.
"""

from __future__ import annotations

import numpy as np


def seleccionar_indices(n_disponibles: int, n_elegidos: int | None, modo: str,
                        rng: np.random.Generator) -> np.ndarray:
    """Elige cuales de los n_disponibles indices (0..n_disponibles-1) se usan.

    Entrada: n_disponibles (numero de candidatos); n_elegidos (cuantos usar,
        None o >= n_disponibles = todos); modo ('uniforme' reparte los elegidos
        a lo largo del rango, 'aleatorio' sortea sin reemplazo); rng.
    Salida: array de indices ordenados. Lanza ValueError si el modo no existe.
    Globales: ninguna (en modo 'aleatorio' avanza el estado de rng).
    """
    if n_disponibles == 0:
        return np.array([], dtype=int)
    if n_elegidos is None or n_elegidos >= n_disponibles:
        return np.arange(n_disponibles)

    if modo == "uniforme":
        return np.unique(np.linspace(0, n_disponibles - 1, n_elegidos).round().astype(int))
    if modo == "aleatorio":
        return np.sort(rng.choice(n_disponibles, size=n_elegidos, replace=False))
    raise ValueError(f"Modo de muestreo desconocido: {modo!r} (usar 'uniforme' o 'aleatorio')")


def indices_inicio_validos(V: np.ndarray, n_muestras: int, delta_v: float,
                           ascendente: bool) -> np.ndarray:
    """Indices de V que sirven como arranque de una ventana a paso fijo.

    Un indice es valido si desde su tension caben n_muestras puntos separados
    delta_v, en el sentido indicado, sin salirse de [V.min(), V.max()].

    Entrada: V (tensiones de la curva, orden ascendente); n_muestras; delta_v [V];
        ascendente (True = barrido creciente).
    Salida: array de indices de V (vacio si ninguno es valido).
    Globales: ninguna.
    """
    if V.size == 0:
        return np.array([], dtype=int)

    extremo = (n_muestras - 1) * delta_v
    v_min, v_max = float(V[0]), float(V[-1])

    if ascendente:
        limite_inf, limite_sup = v_min, v_max - extremo
    else:
        limite_inf, limite_sup = v_min + extremo, v_max

    if limite_sup < limite_inf:
        return np.array([], dtype=int)

    return np.flatnonzero((V >= limite_inf) & (V <= limite_sup))


def indices_inicio_validos_ventana(V: np.ndarray, ventana: float,
                                   ascendente: bool) -> np.ndarray:
    """Indices de V que sirven como arranque de una ventana de anchura fija.

    Un indice es valido si la ventana completa cabe en [V.min(), V.max()]
    desde su tension, en el sentido indicado.

    Entrada: V (tensiones de la curva, orden ascendente); ventana [V]
        (anchura); ascendente (True = barrido creciente).
    Salida: array de indices de V (vacio si ninguno es valido).
    Globales: ninguna.
    """
    if V.size == 0:
        return np.array([], dtype=int)

    v_min, v_max = float(V[0]), float(V[-1])

    if ascendente:
        limite_inf, limite_sup = v_min, v_max - ventana
    else:
        limite_inf, limite_sup = v_min + ventana, v_max

    if limite_sup < limite_inf:
        return np.array([], dtype=int)

    return np.flatnonzero((V >= limite_inf) & (V <= limite_sup))


def construir_ventana(V: np.ndarray, I: np.ndarray, v_inicio: float,
                      n_muestras: int, delta_v: float,
                      ascendente: bool) -> np.ndarray:
    """Construye una ventana de n_muestras puntos [V_i, I_i] a paso fijo delta_v.

    Entrada: V, I (curva, V ascendente); v_inicio [V] (primera muestra);
        n_muestras; delta_v [V]; ascendente (signo del paso). La corriente se
        interpola linealmente sobre (V, I).
    Salida: vector 1D float32 [V_1, I_1, ..., V_n, I_n].
    Globales: ninguna.
    """
    paso = delta_v if ascendente else -delta_v
    v_muestras = v_inicio + paso * np.arange(n_muestras, dtype=np.float32)
    i_muestras = np.interp(v_muestras, V, I).astype(np.float32)
    return np.column_stack((v_muestras, i_muestras)).reshape(-1).astype(np.float32)


def construir_ventana_aleatoria(V: np.ndarray, I: np.ndarray, v_inicio: float,
                                n_muestras: int, ventana: float, ascendente: bool,
                                rng: np.random.Generator) -> np.ndarray:
    """Construye una ventana de n_muestras puntos [V_i, I_i] a tension aleatoria.

    Las tensiones se sortean uniformes dentro de un rango de anchura 'ventana'
    desde v_inicio y se ordenan de menor a mayor.

    Entrada: V, I (curva, V ascendente); v_inicio [V]; n_muestras; ventana [V];
        ascendente (el rango se extiende hacia arriba o hacia abajo de
        v_inicio); rng.
    Salida: vector 1D float32 [V_1, I_1, ..., V_n, I_n] con V creciente.
    Globales: ninguna (avanza el estado de rng).
    """
    if ascendente:
        limite_inf, limite_sup = v_inicio, v_inicio + ventana
    else:
        limite_inf, limite_sup = v_inicio - ventana, v_inicio

    v_muestras = np.sort(rng.uniform(limite_inf, limite_sup, size=n_muestras)).astype(np.float32)
    i_muestras = np.interp(v_muestras, V, I).astype(np.float32)
    return np.column_stack((v_muestras, i_muestras)).reshape(-1).astype(np.float32)


def construir_ventanas_curva(V, I, n_muestras: int, delta_v: float,
                             ascendente: bool, n_ventanas: int | None = None,
                             modo_muestreo: str = "uniforme",
                             rng: np.random.Generator | None = None,
                             paso_fijo: bool = True,
                             ventana: float | None = None):
    """Construye las ventanas de una curva completa, una por punto de arranque.

    Entrada: V, I (curva, V ascendente); n_muestras; delta_v [V] (solo con
        paso_fijo=True); ascendente; n_ventanas (arranques a usar, None =
        todos los validos); modo_muestreo ('uniforme' | 'aleatorio', reparto de
        los arranques elegidos); rng (si es None se crea uno sin semilla);
        paso_fijo; ventana [V] (anchura, obligatoria con paso_fijo=False).
    Salida: (X_ventanas, indices_inicio). X_ventanas: array
        (n_seleccionadas, 2*n_muestras), con 0 filas si la curva no tiene rango
        para ninguna ventana. indices_inicio: indices de V usados como arranque.
        Lanza ValueError si n_muestras < 1, delta_v <= 0 (con n_muestras > 1) o
        ventana no valida.
    Globales: ninguna (avanza el estado de rng).
    """
    if n_muestras < 1:
        raise ValueError("n_muestras debe ser >= 1")

    V = np.asarray(V, dtype=np.float32)
    I = np.asarray(I, dtype=np.float32)

    if rng is None:
        rng = np.random.default_rng()

    if paso_fijo:
        if delta_v <= 0 and n_muestras > 1:
            raise ValueError("delta_v debe ser > 0 si n_muestras > 1")
        candidatos = indices_inicio_validos(V, n_muestras, delta_v, ascendente)
    else:
        if ventana is None or ventana <= 0:
            raise ValueError("ventana debe ser > 0 si paso_fijo=False")
        candidatos = indices_inicio_validos_ventana(V, ventana, ascendente)

    if candidatos.size == 0:
        return np.empty((0, 2 * n_muestras), dtype=np.float32), candidatos

    seleccion = seleccionar_indices(candidatos.size, n_ventanas, modo_muestreo, rng)
    indices_inicio = candidatos[seleccion]

    if paso_fijo:
        filas = [construir_ventana(V, I, V[idx], n_muestras, delta_v, ascendente)
                for idx in indices_inicio]
    else:
        filas = [construir_ventana_aleatoria(V, I, V[idx], n_muestras, ventana,
                                             ascendente, rng)
                for idx in indices_inicio]
    return np.array(filas, dtype=np.float32), indices_inicio
