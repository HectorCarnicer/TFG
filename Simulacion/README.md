# Simulación de panel solar bajo sombreado parcial

Simulador de la curva IV de un panel fotovoltaico bajo irradiancia no
uniforme (sombreado parcial), usando el modelo de diodo único de PVlib
ajustado por el método CEC. Genera curvas IV aleatorias por panel, calcula
el punto de máxima potencia (MPP) de cada una y produce gráficos I-V / P-V.

## Estructura del repositorio

```
datos_panel/                  Datasheets de los paneles disponibles
    datos_panel_<ID>.json     Ficha de un panel concreto
datos_simulaciones/           Tandas de simulaciones generadas
    sim_<ID>.dat              Curvas IV + MPP de una tanda (JSON)
Graficos/                     Gráficos I-V / P-V generados
    curvas_sim_<ID>.png
panel_funciones.py            Funciones de lectura, modelo, simulación y gráfico
simular_panel_main.py         Script de ejecución
```

## Modelo físico

Cada panel se modela como un circuito de diodo único (5 parámetros: I_L,
I_0, R_s, R_sh, a), ajustado a partir de los datos del datasheet
(`v_mp`, `i_mp`, `v_oc`, `i_sc`, coeficientes de temperatura, número de
celdas...) mediante el método CEC de PVlib (`fit_cec_sam`).

El panel está dividido en `n_substrings` substrings (grupos de celdas en
serie, cada uno con su propio diodo de bypass). Bajo sombreado parcial,
cada substring puede recibir una irradiancia distinta. Para modelar esto:

- Los parámetros del circuito equivalente de cada substring se derivan de
  los del panel completo, escalados por la fracción de celdas que contiene:
  `a_ref` escala linealmente con el número de celdas en serie (proporcional
  al número de uniones); `R_s` y `R_sh` escalan por la fracción de celdas;
  `I_L` e `I_0` no cambian, porque no dependen del número de celdas en
  serie sino de la celda individual.
- Los substrings están en serie, así que comparten la misma corriente
  salvo cuando el diodo de bypass de alguno conduce. Por eso la curva se
  calcula barriendo un rango de **corriente** común (0 hasta ~1.05× la Isc
  del substring más iluminado) y resolviendo la tensión de cada substring
  para esa corriente con `bishop88_v_from_i` (PVlib). Cuando la corriente
  pedida supera la Isc de un substring, ese substring ya no puede
  sostenerla: se satura y su tensión se fija directamente al valor de
  `v_bypass` (diodo de bypass modelado como interruptor ideal a tensión
  fija, no como diodo real).
- La tensión total del panel es la suma de las tensiones de los substrings
  para cada punto de corriente. El MPP se obtiene con un `argmax` sobre la
  potencia de la rejilla resultante (sin refinamiento posterior), así que
  tiene un error de cuantización acotado por el número de puntos (`n_puntos`,
  500 por defecto).

## Flujo del programa (`simular_panel_main.py`)

1. Lee el datasheet del panel indicado en `ID_DATOS` (desde
   `datos_panel/datos_panel_<ID_DATOS>.json`) y ajusta el modelo CEC.
2. Según `MODO`:
   - `"generar"`: genera `N_SIMULACIONES` curvas nuevas, cada una con
     irradiancia aleatoria uniforme e independiente por substring en
     `[IRR_MIN, IRR_MAX]` W/m² y temperatura fija `TEMP_CELDA_SIM`. Guarda
     el resultado en `datos_simulaciones/sim_<ID_SIMULACION>.dat`
     (sobrescribe si ya existe).
   - `"leer"`: carga un `sim_<ID_SIMULACION>.dat` ya existente; falla si no
     existe.
3. Si `VISUAL` está activo, genera la figura I-V / P-V con todas las curvas
   y sus MPP marcados, y la guarda en
   `Graficos/curvas_sim_<ID_SIMULACION>.png`.

Todos los parámetros de configuración (panel, ID de simulación, modo,
tamaño de la tanda, rango de irradiancia, temperatura, semilla) son
constantes al principio del script.

## Funciones (`panel_funciones.py`)

- **`cargar_datasheet(id_datos, carpeta_datos)`** — Lee
  `datos_panel_<id_datos>.json` y devuelve un dict con los datos del panel,
  convirtiendo los coeficientes de temperatura de %/°C a las unidades que
  usa PVlib (A/°C, V/°C).

- **`ajustar_modelo_cec(datasheet)`** — Ajusta los 5 parámetros del
  circuito equivalente de diodo único al datasheet, vía `fit_cec_sam`.
  Devuelve el datasheet extendido con esos parámetros (a esto se le llama
  "módulo" en el resto del código).

- **`parametros_substring(modulo, irradiancia, temp_celda)`** — Deriva los
  parámetros del circuito equivalente de un substring a partir del módulo
  completo (ver "Modelo físico" arriba) y calcula su Isc resolviendo el
  modelo bishop88 en V=0.

- **`curva_iv_panel_sombreado(modulo, irradiancias_substrings, temp_celda, v_bypass, n_puntos)`**
  — Función central de simulación. Calcula la curva IV completa del panel
  para una irradiancia dada por substring, combinando los substrings en
  serie con sus diodos de bypass (ver "Modelo físico"). Devuelve la curva
  (tensión, corriente, potencia), el MPP y la matriz de tensiones por
  substring.

- **`generar_simulaciones_aleatorias(modulo, n_simulaciones, ...)`** —
  Genera N curvas llamando repetidamente a `curva_iv_panel_sombreado` con
  irradiancias aleatorias por substring. Devuelve una lista de dicts, uno
  por simulación, con las irradiancias usadas, el MPP y la curva completa.

- **`guardar_simulaciones(simulaciones, modulo, ruta_salida, ...)`** —
  Serializa la lista de simulaciones a JSON, añadiendo metadatos (número de
  substrings, rango de irradiancia, temperatura, datos del panel en STC,
  semilla usada) para que el archivo sea autocontenido.

- **`cargar_simulaciones(ruta_archivo)`** — Recarga un archivo generado por
  `guardar_simulaciones`.

- **`graficar_curvas_iv_pv(simulaciones, id_simulacion, carpeta_graficos, ...)`**
  — Dibuja todas las curvas I-V y P-V de una tanda de simulaciones en una
  única figura de dos paneles, marca el MPP de cada curva, y guarda el
  resultado como PNG en `carpeta_graficos` (creándola si no existe).

## Paneles disponibles (`datos_panel/`)

Cada `datos_panel_<ID>.json` es la ficha de un panel comercial (o, en el
caso de ID 0, el panel original del proyecto). Todos son de celda completa
(no half-cut), para que el escalado por fracción de celdas en
`parametros_substring` siga siendo válido sin cambios. Añadir un panel
nuevo es tan simple como crear un JSON más con el mismo esquema.

## Notas para el usuario

- Los ficheros `sim_<ID>.dat` pueden ser grandes (varios MB por tanda de
  500 curvas), ya que guardan la curva completa (500 puntos) de cada
  simulación, no solo el MPP.
- Reejecutar el script con el mismo `ID_SIMULACION` sobrescribe tanto el
  `.dat` como el `.png` correspondientes.
