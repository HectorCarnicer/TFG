# Simulador de sensor UART

Simula un sensor de tensión/corriente de un panel fotovoltaico que envía
muestras `[V, I]` por UART a un microcontrolador ST, y recibe de vuelta la
tensión de máxima potencia (Vmp) predicha por la red neuronal embebida en
el micro, junto con el tiempo de inferencia. Compara esa predicción con el
Vmp real de la curva y calcula el error.

## Estructura de archivos

- **`lectura_escritura.py`** — lectura del dataset y escritura del log de test.
- **`simulador_sensor.py`** — comunicación UART y los dos modos de ejecución.
- **`main_sensor.py`** — parámetros de configuración y punto de entrada.
- **`visualizacion.py`** — genera un gráfico a partir de un log de test.

## Protocolo UART

**Envío (Python → microcontrolador):** por cada muestra, 8 bytes sin
cabecera: `struct.pack('<ff', V, I)` — V primero, I después, ambos float32
little-endian.

**Recepción (microcontrolador → Python):** una línea de texto por cada
muestra procesada:

```
V= %6.2f V | I= %5.2f A | Prediccion= %9.4f | t_inferencia= %6lu us\r\n
```

## Flujo del programa

`main_sensor.py` es el punto de entrada. Al ejecutarse:

1. Valida `MODO_SIM` (y `TIPO_MUESTREO` si `MODO_SIM == "Simulacion"`).
2. Carga el dataset con `cargar_dataset(RUTA_DATOS)`.
3. Abre el log de test con `abrir_log` y escribe la cabecera con los
   parámetros de la sesión (`escribir_cabecera_log`).
4. Abre el puerto serie con `abrir_puerto`.
5. Según `MODO_SIM`, llama a `ejecutar_simulacion` (modo "Consecutivo") o
   a `ejecutar_modo_simulacion` (modo "Simulacion").
6. Al terminar (o si hay un error, o se interrumpe con Ctrl+C), cierra el
   puerto y el log en el bloque `finally`.

Cada muestra enviada se imprime por consola y se guarda en el log; al
terminar cada curva o tramo de seguimiento se añade una línea de resumen,
y al final del test un resumen global.

## Modos de simulación (`MODO_SIM`)

### `"Consecutivo"`

Recorre curvas del dataset una a una. De cada curva se descartan los
puntos de relleno (`V == -1.0`, usados por el generador del dataset para
igualar la longitud de las curvas) y se escogen `N_PUNTOS_POR_CURVA`
puntos al azar, repartidos por todo el rango de tensión, que se envían en
orden decreciente de V — simulando un barrido Voc → Isc. `N_CURVAS` limita
cuántas curvas del dataset se procesan (`None` = todas).

Función principal: `ejecutar_simulacion`, que llama a `procesar_curva`
por cada curva.

### `"Simulacion"`

Simula el seguimiento que haría un MPPT real, incluyendo cambios bruscos
de irradiancia en mitad del seguimiento:

1. El primer punto enviado es un punto aleatorio de una curva aleatoria
   del dataset (la "curva de trabajo").
2. Cuando el micro devuelve su tensión predicha, se calcula una tensión
   objetivo según `TIPO_MUESTREO` (ver más abajo), y el siguiente punto
   enviado es el de la curva de trabajo con la V más próxima a ese
   objetivo (`punto_mas_cercano`).
3. Cada `MUESTRAS_HASTA_CAMBIO` muestras se escoge al azar otra curva del
   dataset con irradiancias distintas, que pasa a ser la nueva curva de
   trabajo — simulando un cambio repentino de irradiancia. El seguimiento
   no se pierde: el primer punto de la nueva curva es el más cercano a la
   tensión objetivo acumulada hasta ese momento.
4. El test termina al alcanzar `MUESTRAS_TOTALES` muestras en total
   (puede cortar un tramo a la mitad si no es múltiplo de
   `MUESTRAS_HASTA_CAMBIO`).

Si un punto no obtiene respuesta del micro, no hay predicción nueva que
seguir: se reintenta el mismo punto en la siguiente muestra.

Funciones principales: `ejecutar_modo_simulacion`, que orquesta los
cambios de curva, y `_procesar_segmento_simulacion`, que procesa cada
tramo de seguimiento sobre una única curva de trabajo.

#### `TIPO_MUESTREO`

Determina cómo se calcula la tensión objetivo del siguiente punto a
partir de la predicción del micro (`calcular_v_objetivo`):

- **`"Instantaneo"`** — el objetivo es directamente la predicción
  recibida; el siguiente punto salta de golpe a ella.
- **`"Relativo"`** — el objetivo se acerca gradualmente a la predicción,
  un paso de `(V_predicho − V_actual) / MUESTRAS_HASTA_CAMBIO` por
  muestra. Esto evita que una predicción que se pase del Vmp real lance
  de golpe al sensor a una zona sin información (por ejemplo, I≈0 cerca
  de Voc), donde en modo `"Instantaneo"` el seguimiento puede quedarse
  atascado en bucle al no recibir nunca una predicción distinta.

En modo `"Relativo"`, la tensión objetivo se acumula de forma **continua**
de una muestra a otra: `calcular_v_objetivo` recibe siempre el valor
objetivo anterior (no el punto ya discretizado que se acaba de enviar) y
devuelve el nuevo objetivo a partir de él. `punto_mas_cercano` se usa solo
para traducir ese objetivo continuo al punto real de la curva que hay que
enviar. Esta distinción importa porque algunas curvas tienen huecos
grandes entre puntos consecutivos (el "codo" que producen las
transiciones de substring por diodo de bypass en el sombreado parcial):
si el objetivo se recalculara cada vez a partir del punto ya redondeado,
el paso `(V_predicho − V_actual) / MUESTRAS_HASTA_CAMBIO` podría no ser
nunca suficiente para cruzar el hueco, y el seguimiento quedaría
permanentemente fijado en el mismo punto.

Común a ambos modos: `SEMILLA_ALEATORIA` fija la aleatoriedad (curva/punto
iniciales, cambios de curva, selección de puntos) para que el test sea
reproducible; `None` hace que cada ejecución sea distinta.

## Descripción de funciones por módulo

### `lectura_escritura.py`

- **`cargar_dataset(ruta_datos)`** — carga el dataset JSON de curvas I-V.
- **`listar_simulaciones(dataset)`** — devuelve la lista de curvas.
- **`metadata_dataset(dataset)`** — devuelve el bloque de metadata.
- **`ruta_log(directorio_logs, id_test)`** — construye la ruta de un log.
- **`abrir_log(directorio_logs, id_test)`** — crea el directorio de logs
  si hace falta y abre el archivo de log en modo escritura.
- **`escribir_log(fh, texto, tambien_consola=True)`** — escribe una línea
  en el log (con flush) y opcionalmente la imprime por consola.
- **`escribir_cabecera_log(fh, id_test, campos)`** — escribe la cabecera
  del log con los parámetros de la sesión.
- **`cerrar_log(fh)`** — cierra el archivo de log.

### `simulador_sensor.py`

Clases:

- **`RespuestaMicro`** — respuesta parseada de una muestra (`v_eco`,
  `i_eco`, `prediccion_vmp`, `t_inferencia_us`, `linea_cruda`).
- **`ResumenCurva`** — estadísticas de error de una curva o tramo
  procesado (Vmp real, error medio/final, nº de respuestas OK/timeouts).

Puerto serie:

- **`abrir_puerto(puerto, baudios, timeout_s)`** — abre el puerto,
  espera 2s (reset de la placa) y limpia los buffers.
- **`cerrar_puerto(ser)`** — limpia el buffer de entrada y cierra.

Envío y recepción:

- **`empaquetar_punto(v, i)`** / **`enviar_punto(ser, v, i)`** —
  serializan y envían un punto (V, I).
- **`leer_linea_respuesta(ser)`** — lee una línea de la UART.
- **`parsear_respuesta(linea)`** — extrae `RespuestaMicro` de una línea.
- **`intentar_muestra(ser, v, i, reintentos_timeout)`** — envía un punto
  y reintenta hasta obtener respuesta válida o agotar los reintentos.

Selección de puntos y cálculo de error:

- **`puntos_validos(curva)`** — puntos (V, I) de una curva sin relleno.
- **`seleccionar_puntos_aleatorios(curva, n_puntos, rng)`** — selección
  aleatoria de puntos para el modo "Consecutivo".
- **`calcular_error(prediccion_vmp, vmp_real)`** — error absoluto/relativo.
- **`punto_mas_cercano(puntos, v_objetivo)`** — punto con la V más
  próxima a un objetivo dado.
- **`calcular_v_objetivo(tipo_muestreo, v_actual, v_predicho, muestras_hasta_cambio)`**
  — siguiente tensión objetivo, según `TIPO_MUESTREO`.

Ejecución:

- **`procesar_curva(...)`** — procesa una curva completa en modo
  "Consecutivo".
- **`ejecutar_simulacion(...)`** — recorre todas las curvas configuradas
  en modo "Consecutivo".
- **`_elegir_curva_distinta(simulaciones, curva_actual, rng)`** — escoge
  la siguiente curva de trabajo con irradiancias distintas.
- **`_procesar_segmento_simulacion(...)`** — procesa un tramo de
  seguimiento sobre una curva de trabajo en modo "Simulacion".
- **`ejecutar_modo_simulacion(...)`** — orquesta el modo "Simulacion"
  completo, incluyendo los cambios de curva de trabajo.
- **`_escribir_resumen_global(fh_log, resumenes, duracion_s)`** — escribe
  el resumen global del test.

### `main_sensor.py`

- **`main()`** — valida la configuración, carga el dataset, abre log y
  puerto, ejecuta el modo configurado y cierra ambos al terminar (también
  si hay error o `Ctrl+C`). Devuelve el código de salida del proceso.

### `visualizacion.py`

- **`ruta_log` / `ruta_grafico`** — construyen las rutas de entrada/salida.
- **`leer_muestras_log(ruta)`** — parsea el log y devuelve la lista de
  `Muestra` (una por línea de muestra con predicción).
- **`generar_grafico(muestras, id_test, ruta_salida)`** — genera el PNG
  con dos paneles: Vmp real vs. predicho, y error absoluto, por muestra.
- **`main()`** — resuelve el ID de test (argumento de línea de comandos o
  `ID_TEST`), lee el log y genera el gráfico.

## Uso

1. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Edita los parámetros al principio de `main_sensor.py` (puerto serie,
   baudios, `ID_DATOS`, ID de test, `MODO_SIM` y los parámetros del modo
   elegido).

3. Ejecuta:

   ```bash
   python main_sensor.py
   ```

4. Tras un test, genera el gráfico:

   ```bash
   python visualizacion.py           # usa el ID_TEST definido en visualizacion.py
   python visualizacion.py 003       # o bien pasa el ID como argumento
   ```

## Notas

- Si no llega respuesta del microcontrolador dentro de
  `TIMEOUT_RESPUESTA_S`, el punto se reintenta hasta
  `REINTENTOS_POR_TIMEOUT` veces; si sigue sin haber respuesta, se marca
  como "SIN RESPUESTA" en el log y se continúa con el siguiente punto.
- `N_CURVAS = None` usa todas las curvas del dataset; un entero limita el
  test a las primeras N.
- El dataset se espera en `datos/sim_{ID_DATOS}.dat`, en formato JSON, con
  una clave `simulaciones` (lista de curvas con `id`,
  `irradiancias_substrings_Wm2`, `mpp` y `curva`).
