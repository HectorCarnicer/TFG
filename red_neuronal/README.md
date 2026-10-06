# Predictor de MPP con red neuronal (V2: ventana de muestras)

Predictor de la tensión de máxima potencia (Vmp) de un panel fotovoltaico
mediante una red neuronal densa de regresión. La entrada es una **ventana de
varias muestras (V, I)** de la curva I-V, como las que iría tomando un
algoritmo Perturb & Observe (P&O) en tiempo real. Pensado para ejecutarse en un
microcontrolador (ST) como parte de un controlador MPPT: el modelo se exporta a
TensorFlow Lite y acepta voltios y amperios en crudo.

## Origen de los datos

Los ficheros `datos/sim_{ID}.dat` contienen simulaciones de paneles
fotovoltaicos con substrings en serie y diodo de bypass, generadas con un
modelo de diodo único ajustado al datasheet del panel. Cada simulación aplica
una irradiancia distinta a cada substring (sombreado parcial independiente) y
guarda la curva I-V resultante junto con su punto de máxima potencia (Vmp,
Imp, Pmax). Cada fichero puede corresponder a un panel o a unas condiciones
distintas. El JSON tiene la forma:

```json
{
  "metadata": { "n_simulaciones": 500, "panel": {...}, ... },
  "simulaciones": [
    {
      "id": 0,
      "curva": {"V": [...], "I": [...]},
      "mpp": {"Vmp [V]": ..., "Imp [A]": ..., "Pmax [W]": ...},
      "irradiancias_substrings_Wm2": [...]
    },
    ...
  ]
}
```

Las claves se buscan por subcadena (`"vmp"`, `"pmax"`, `"imp"`), de modo que
no importan las unidades ni las mayúsculas del nombre.

## Idea del modelo

La red recibe una ventana de `n` muestras `[[V_1,I_1], ..., [V_n,I_n]]`
(aplanada a un vector de `2n` valores) y predice el Vmp de la curva de la que
proceden. Un único punto (V, I) no basta para determinarlo: con sombreado
parcial, cuando un substring se satura y su diodo de bypass entra en
conducción, la curva desarrolla un escalón y curvas con Vmp muy distintos
pueden pasar por el mismo punto antes o después de él. Una ventana de varios
puntos aporta la forma local de la curva, lo que permite distinguir esos casos.

### Construcción de las ventanas

`muestreo_ventanas.py` genera las ventanas a partir de una curva ordenada por
tensión. Hay dos modos (parámetro `PASO_FIJO`):

- **Paso fijo** (`PASO_FIJO=True`): desde un punto de arranque se toman `n`
  muestras separadas `DELTA_V` voltios, como un P&O de paso fijo.
- **Tensión aleatoria** (`PASO_FIJO=False`): las `n` muestras se sortean con
  tensión uniforme dentro de un rango de anchura `VENTANA` desde el arranque
  y se ordenan de menor a mayor (P&O de paso variable, o robustez frente a
  desviaciones del paso real). `DELTA_V` se ignora en este modo.

El sentido del barrido es creciente o decreciente en tensión (`ASCENDENTE`).
Un punto de la curva es un arranque válido solo si la ventana completa cabe
dentro del rango `[V.min(), V.max()]` de la curva, para no extrapolar.
`N_VENTANAS_CURVA` limita cuántos arranques se toman por curva (`None` = todos
los válidos); si es menor que los disponibles, `MODO_MUESTREO` decide cómo se
reparten (`'uniforme'` a lo largo de la curva o `'aleatorio'` sin reemplazo).
Las curvas sin rango suficiente para ninguna ventana se descartan y se
cuentan en el resumen.

**Interpolación.** Las corrientes de cada muestra se obtienen por
interpolación lineal sobre la curva simulada. Los puntos de las simulaciones
no están repartidos uniformemente en tensión (hay tramos con huecos de más de
1 V entre puntos consecutivos), de modo que la fidelidad de una muestra
depende de la densidad de la simulación en esa zona; en los tramos más
dispersos la ventana puede suavizar detalles finos que un P&O real sí mediría.

## Limitación conocida

Si una ventana es compatible con curvas de Vmp distintos, la regresión con
pérdida MSE converge a la media condicional de esos Vmp: predice una tensión
intermedia que no coincide con ninguno de los picos. Por eso la estimación
por curva (ver más abajo) agrega las predicciones de todas sus ventanas con la
mediana, y las métricas punto a punto son más pesimistas que las métricas por
curva.

## Estructura del proyecto

```
rn_main.py              script principal: entrena y exporta el modelo
muestreo_ventanas.py    construcción de las ventanas [V,I] de una curva
lectura_escritura.py    lectura de datos, partición del dataset, E/S de TFLite
funciones_modelo.py     definición de la red, entrenamiento, predicción, métricas
diagnostico_datos.py    evalúa un modelo ya exportado sobre su propio conjunto de test
comparador_rn.py        compara varios modelos entrenados (tabla y gráficas)
datos/                  simulaciones (sim_{ID}.dat)
modelos/                salida: modelos/mpp_v{ID}/ (.tflite, resumen JSON y figuras)
```

## Flujo de `rn_main.py`

1. **Lectura de datos.** `leer_datos_entrenamiento` lee los ficheros
   `datos/sim_{ID}.dat` de `IDS_DATOS` y los combina en un único dataset. Por
   cada curva construye sus ventanas y asigna a cada una como etiqueta el Vmp
   de esa curva. Devuelve `X` (ventanas), `y` (Vmp), `grupos` (curva de origen
   de cada ventana) e `info` (metadata, curvas completas y estadísticas).

   Cada fichero numera sus curvas de forma independiente (normalmente
   `0..N-1`), así que los ids no serían únicos entre ficheros. Por eso cada
   curva recibe un id global `indice_fichero * OFFSET_ID_DATOS + id_original`;
   el id original y el fichero de origen se conservan en `info['curvas']`
   (`id_original`, `id_dato`).

2. **Partición del dataset.** `dividir_datos` separa train/val/test
   repartiendo **curvas completas**, no ventanas sueltas. Las ventanas de una
   misma curva comparten etiqueta y están muy correlacionadas: si se
   repartieran al azar acabarían a ambos lados de la partición y las métricas
   de test saldrían optimistas.

3. **Creación del modelo.** `crear_modelo` construye la red densa:

   ```
   Entrada (V_1,I_1,...,V_n,I_n) -> Normalización -> 5 x Dense(64, sigmoid)
              -> Dense(1) -> Desnormalización -> Vmp [V]
   ```

   Las capas de normalización (entrada) y desnormalización (salida) se
   ajustan con la media y la desviación típica del conjunto de entrenamiento y
   van **dentro** del propio modelo. Así el `.tflite` acepta voltios y amperios
   en crudo y devuelve voltios, sin replicar ningún escalado en el
   microcontrolador. Además es necesario con activaciones sigmoide, que se
   saturan si reciben valores lejos de cero.

4. **Entrenamiento.** `entrenar_modelo` usa `EarlyStopping` (para cuando la
   pérdida de validación deja de mejorar y restaura los mejores pesos) y
   `ReduceLROnPlateau` (reduce el learning rate a la mitad si se estanca),
   con semilla fija para reproducibilidad. `PACIENCIA=0` desactiva la parada
   temprana.

5. **Evaluación.** Se calculan MAE, RMSE, MAPE, error máximo y R² de dos
   formas: punto a punto sobre las ventanas de test, y **por curva**
   (`predecir_vmp_curva` construye todas las ventanas válidas de cada curva de
   test, predice cada una y agrega las predicciones con la mediana o la media,
   según `AGREGACION_CURVA`). La mediana es más robusta que la media frente a
   ventanas poco informativas, como las de los extremos de la curva (cerca del
   cortocircuito o del circuito abierto). Con `PASO_FIJO=False`, la semilla fija
   las tensiones aleatorias de las ventanas, de modo que la estimación es
   reproducible entre ejecuciones.

6. **Exportación a TensorFlow Lite.** `exportar_tflite` convierte el modelo
   Keras a `.tflite`. Si la conversión directa falla (ocurre con Keras 3 en
   algunos casos), se repite pasando por un `SavedModel` temporal. Admite
   cuantización a enteros (`CUANTIZAR=True`), calibrada con una muestra de
   ventanas reales de entrenamiento. Después se recarga el `.tflite` y se
   compara su predicción con la del modelo Keras (`VERIFICAR_TFLITE`).

7. **Resumen y figuras.** Se guarda `mpp_v{ID}_resumen.json` con los datos
   usados (ficheros, curvas, ventana, metadata), la partición, la
   arquitectura, el entrenamiento (épocas ejecutadas, batch, paciencia,
   semilla) y las métricas (punto a punto, por curva y del `.tflite`). También
   se guardan dos figuras: la evolución del entrenamiento (pérdida, MAE,
   dispersión predicción-real e histograma del error) y una curva de test de
   ejemplo con el Vmp real y el predicho. Todo va a `modelos/mpp_v{ID}/`, que se
   reescribe si se repite el mismo `ID`.

## Herramientas de evaluación

- **`diagnostico_datos.py`** — evalúa un modelo ya exportado sobre su propio
  conjunto de test, usando solo el `.tflite` y el resumen. Lee del resumen la
  ventana, los ficheros de datos, la partición y la semilla, reconstruye
  exactamente el mismo test y ejecuta el `.tflite` sobre él, de modo que sirve
  también para comprobar que el modelo exportado reproduce las métricas de su
  resumen. Si el resumen no registra algún campo, usa el valor por defecto de
  las constantes del fichero y avisa por consola. Genera la figura de
  evaluación (dispersión e histograma del error, punto a punto y por curva) y
  las curvas de test con mayor error.
- **`comparador_rn.py`** — compara varios modelos (`IDS`; `N_MODELOS` debe
  coincidir con `len(IDS)`): tabla por consola con ventana, ficheros de datos,
  R² y RMSE punto a punto y por curva, y gráficas con R² y RMSE por modelo y la
  distribución de Vmp del test. Cada modelo se evalúa con su propia
  configuración, de modo que pueden compararse modelos entrenados con ventanas
  o datos distintos. Los IDs sin modelo se avisan y se omiten (se necesitan al
  menos dos válidos). Los resultados van a
  `modelos/comparaciones/cmp_{ID1}-{ID2}-…/` y se reescriben si se repite la
  comparación.

## Funciones por módulo

**`muestreo_ventanas.py`:** `seleccionar_indices`, `indices_inicio_validos`,
`indices_inicio_validos_ventana`, `construir_ventana`,
`construir_ventana_aleatoria`, `construir_ventanas_curva`.

**`lectura_escritura.py`:** `leer_datos_entrenamiento` (con los auxiliares
`_leer_fichero_simulaciones`, `_construir_ejemplos_curva`, `_construir_info` y
`_buscar_clave`), `dividir_datos`, `exportar_tflite`
(con `_configurar_cuantizacion`), `cargar_modelo_tflite`, `guardar_resumen`.

**`funciones_modelo.py`:** `crear_modelo`, `entrenar_modelo`, `predecir`,
`predecir_tflite` (aplica los factores de cuantización si el modelo está
cuantizado), `evaluar`, `predecir_vmp_curva`.

**`diagnostico_datos.py`:** `cargar_configuracion_modelo`, `evaluar_modelo`,
`figura_evaluacion`, `figura_curvas_ejemplo`, `main`.

**`comparador_rn.py`:** `diagnosticar_modelo`, `imprimir_tabla_comparativa`,
`figura_comparacion_metricas`, `figura_comparacion_distribucion`, `main`.

**`rn_main.py`:** `graficar_entrenamiento`, `graficar_curva_ejemplo`, `main`.

Cada función lleva un docstring breve con su propósito, entrada, salida y
variables globales que lee o modifica.

## Parámetros principales (`rn_main.py`)

Todos los parámetros de uso están al principio del fichero, en mayúsculas y
agrupados por bloque:

- **Datos:** `DIR_DATOS`, `IDS_DATOS` (varios ids = se combinan en un único
  dataset), `MODO_MUESTREO`, `DESCARTAR_I_NEGATIVA` (elimina la cola posterior
  a Voc).
- **Ventana:** `N_MUESTRAS`, `DELTA_V`, `ASCENDENTE`, `N_VENTANAS_CURVA`,
  `PASO_FIJO`, `VENTANA`.
- **Partición:** `FRACCION_TEST`, `FRACCION_VALIDACION` (fracciones de
  curvas), `SEMILLA`.
- **Arquitectura:** `N_CAPAS`, `N_NEURONAS`, `ACTIVACION`, `OPTIMIZADOR`
  (`nadam`, `adam`, `rmsprop`, `sgd`, `adamw`), `LEARNING_RATE`,
  `FUNCION_PERDIDA`. `N_ENTRADAS = 2 * N_MUESTRAS`.
- **Entrenamiento:** `EPOCAS`, `BATCH_SIZE`, `PACIENCIA`, `REDUCIR_LR`,
  `VERBOSE_ENTRENAMIENTO`.
- **Estimación por curva:** `AGREGACION_CURVA` (`'mediana'` | `'media'`).
- **Salida:** `ID` (crea o reescribe `modelos/mpp_v{ID}/`), `DIR_MODELOS`,
  `CUANTIZAR`, `VERIFICAR_TFLITE`, `GUARDAR_RESUMEN`.
- **Visualización:** `VISUALIZAR`, `MOSTRAR_FIGURAS`, `GUARDAR_FIGURAS`,
  `DIR_FIGURAS`, `ESCALA_LOG_PERDIDA`.

## Requisitos

Python 3.10+, TensorFlow (Keras), NumPy, Matplotlib. El intérprete de TFLite se
toma del paquete `ai_edge_litert` si está instalado, o de `tf.lite` como
alternativa.

## Ejecución

```bash
python rn_main.py                  # entrena y exporta modelos/mpp_v{ID}/
python diagnostico_datos.py 4      # evalúa el modelo ID=4 sobre su test
python comparador_rn.py 1 2 3      # compara los modelos 1, 2 y 3
```
