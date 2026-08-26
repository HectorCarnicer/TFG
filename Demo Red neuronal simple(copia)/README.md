# Predictor de MPP con red neuronal (v1)

Predictor de la tensión de máxima potencia (Vmp) de un panel fotovoltaico
mediante una red neuronal densa, a partir de un único punto de operación
(V, I) de su curva I-V. Pensado para ejecutarse en un microcontrolador (ST)
como parte de un controlador MPPT.

Esta es la primera versión del proyecto: entrada de un solo punto (V, I).
No incluye todavía el muestreo de varios puntos de la curva ni la gestión
de múltiples experimentos/modelos.

## Origen de los datos

El fichero `sim_1.dat` contiene simulaciones de un panel fotovoltaico de 2
substrings en serie (18 células cada uno, con diodo de bypass), generadas
con un modelo de diodo único ajustado al datasheet del panel. Cada
simulación aplica una irradiancia distinta a cada substring (sombreado
parcial independiente) y guarda la curva I-V resultante junto con su punto
de máxima potencia (Vmp, Imp, Pmax). El JSON tiene la forma:

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

## Idea del modelo y limitación conocida

La red recibe un único punto de operación (V, I) y predice el Vmp de la
curva a la que pertenece ese punto. Esto asume implícitamente que un punto
(V, I) contiene suficiente información para determinar el Vmp de su curva.

Con sombreado parcial esto no siempre es cierto: cuando un substring se
satura y su diodo de bypass entra en conducción, la curva I-V desarrolla un
escalón, y curvas con Vmp muy distintos pueden pasar por el mismo punto (V,
I) antes o después de ese escalón. En esos casos ningún modelo puede
distinguir a qué curva pertenece un punto aislado, y la predicción con
menor error posible es la media de los Vmp compatibles con ese punto.
`diagnostico_datos.py` cuantifica cuánto limita esto al modelo antes de
entrenar (ver más abajo).

## Estructura del proyecto

```
rn_main.py              script principal: entrena y exporta el modelo
lectura_escritura.py    lectura de datos, partición del dataset, E/S de TFLite
funciones_modelo.py     definición de la red, entrenamiento, predicción, métricas
diagnostico_datos.py    diagnóstico de cuánta información hay en (V, I) antes de entrenar
sim_1.dat               datos de simulación (curvas I-V)
modelos/                salida de rn_main.py: modelo .tflite, resumen JSON y figuras
```

## Flujo de `rn_main.py`

1. **Lectura de datos.** `leer_datos_entrenamiento` (en `lectura_escritura.py`)
   lee `sim_1.dat` y construye el dataset: por cada curva se toman
   `N_PUNTOS_CURVA` puntos (V, I) —o todos si es `None`— y a cada uno se le
   asigna como etiqueta el Vmp de esa curva. Devuelve `X` (puntos V,I), `y`
   (Vmp), `grupos` (a qué curva pertenece cada punto) e `info` (metadata,
   curvas completas y estadísticas del dataset).

2. **Partición del dataset.** `dividir_datos` separa train/val/test
   repartiendo **curvas completas**, no puntos sueltos: todos los puntos de
   una misma curva caen en el mismo conjunto. Si se repartieran puntos al
   azar, puntos de la misma curva (muy correlacionados, con la misma
   etiqueta) acabarían a ambos lados del split y las métricas de test
   saldrían artificialmente optimistas.

3. **Creación del modelo.** `crear_modelo` (en `funciones_modelo.py`)
   construye una red densa:

   ```
   (V, I) -> Normalización -> 5 x Dense(64, sigmoid) -> Dense(1)
          -> Desnormalización -> Vmp [V]
   ```

   Las capas de normalización (a la entrada) y desnormalización (a la
   salida) se ajustan con la media/desviación típica del conjunto de
   entrenamiento y se incluyen **dentro** del propio modelo. Así el
   `.tflite` exportado acepta voltios/amperios en crudo y devuelve voltios
   directamente, sin tener que replicar ningún escalado en el
   microcontrolador. Es además necesario porque las activaciones sigmoide
   se saturan si reciben entradas lejos de cero.

4. **Entrenamiento.** `entrenar_modelo` entrena con `EarlyStopping` (para
   cuando la pérdida de validación deja de mejorar) y `ReduceLROnPlateau`
   (baja el learning rate si se estanca). Semilla fija para reproducibilidad.

5. **Evaluación.** Se calculan métricas (MAE, RMSE, MAPE, error máximo, R²)
   de dos formas: punto a punto sobre el conjunto de test, y por curva
   completa (`predecir_vmp_curva` evalúa el modelo en todos los puntos
   disponibles de cada curva de test y agrega las predicciones con la
   mediana, más robusta que la media frente a puntos poco informativos
   como el cortocircuito o el circuito abierto).

6. **Exportación a TensorFlow Lite.** `exportar_tflite` convierte el modelo
   Keras a `.tflite`. Si la conversión directa falla (ocurre con Keras 3 en
   algunos casos), se reintenta pasando por un `SavedModel` intermedio.
   Admite cuantización opcional a enteros (`CUANTIZAR=True`), calibrada con
   una muestra de datos reales de entrenamiento. Después se recarga el
   `.tflite` y se compara su predicción contra el modelo Keras para
   verificar que la conversión no introduce error apreciable.

7. **Resumen y figuras.** Se guarda un JSON (`mpp_v1_resumen.json`) con la
   configuración de datos, arquitectura, entrenamiento y métricas del
   modelo, y dos figuras: evolución del entrenamiento (pérdida, MAE,
   dispersión predicción-real, histograma del error) y una curva de test
   de ejemplo con el Vmp real y el predicho superpuestos.

## `diagnostico_datos.py`

Herramienta de diagnóstico **independiente del modelo entrenado**: analiza
solo los datos para responder una pregunta previa al entrenamiento —¿cuánta
información hay en un punto (V, I) sobre el Vmp de su curva?

- **Distribución de Vmp.** Histograma por consola; permite detectar
  agrupaciones o bimodalidad en los datos (indicio de que hay varios
  "regímenes" de curvas, por ejemplo con y sin bypass activo).

- **Techo de R² alcanzable.** Se estima
  `R2_max = 1 - E[Var(Vmp | V, I)] / Var(Vmp)`
  mediante vecinos próximos en el plano (V, I) normalizado: para una
  muestra de puntos de consulta se buscan sus vecinos más cercanos
  pertenecientes a **otras** curvas (se excluyen siempre los vecinos de la
  propia curva, para no subestimar la varianza) y se calcula la varianza de
  sus Vmp. Promediando esa varianza condicional sobre todas las consultas
  se obtiene `E[Var(Vmp | V, I)]`, y de ahí el techo de R² que **cualquier**
  modelo —no solo esta red— podría alcanzar con esa entrada. Es un límite
  teórico, no depende de los pesos de ningún modelo entrenado:
  - R²_max > 0.9: hay información suficiente en (V, I); si el modelo
    entrenado falla, el problema es de entrenamiento (arquitectura,
    learning rate, épocas, normalización).
  - R²_max < 0.5: el problema está mal determinado con dos entradas;
    ajustar hiperparámetros no lo va a arreglar.

- **Ejemplos visuales de solapamiento.** A partir de las discrepancias
  encontradas en el cálculo anterior, dibuja las parejas de curvas con Vmp
  más distintos que comparten un punto (V, I) casi idéntico, superpuestas
  con su Vmp real marcado, para ilustrar visualmente la indeterminación.

## Módulo `lectura_escritura.py`: funciones

- `_buscar_clave(diccionario, patron)` — busca la primera clave que
  contiene `patron` (sin distinguir mayúsculas ni tildes exactas), porque
  las claves del JSON incluyen unidades (`"Vmp [V]"`, `"Índice MPP"`).
- `_indices_muestreo(n_disponibles, n_puntos, modo, rng)` — decide qué
  índices de una curva se usan como muestras (todos, reparto uniforme, o
  aleatorio sin reemplazo).
- `leer_datos_entrenamiento(...)` — lee el JSON y construye `X`, `y`,
  `grupos`, `info` como se describe arriba.
- `dividir_datos(...)` — separa train/val/test por curva completa.
- `exportar_tflite(...)` — convierte y guarda el modelo en `.tflite`.
- `_configurar_cuantizacion(...)` — aplica las opciones de cuantización al
  convertidor.
- `cargar_modelo_tflite(...)` — carga un `.tflite` y reserva sus tensores.
- `guardar_resumen(...)` — guarda un diccionario como JSON de resumen.

## Módulo `funciones_modelo.py`: funciones

- `crear_modelo(...)` — construye y compila la red (arquitectura descrita
  arriba).
- `entrenar_modelo(...)` — entrena con `EarlyStopping` y
  `ReduceLROnPlateau` opcionales.
- `predecir(...)` — predicción con el modelo Keras.
- `predecir_tflite(...)` — predicción con un intérprete `.tflite` ya
  cargado; aplica automáticamente los factores de cuantización si el
  modelo está cuantizado.
- `evaluar(...)` — calcula MAE, RMSE, MAPE, error máximo y R² entre
  predicción y valor real.
- `predecir_vmp_curva(...)` — agrega las predicciones de todos los puntos
  de una curva completa (mediana o media) para obtener una estimación más
  estable que con un único punto.

## Parámetros principales (`rn_main.py`)

Todos los parámetros de uso están agrupados al principio del fichero, en
mayúsculas, agrupados por bloque (datos, partición, arquitectura,
entrenamiento, salida, visualización). Los más relevantes:

- `RUTA_DATOS`: fichero de simulaciones a usar.
- `N_PUNTOS_CURVA`: puntos tomados de cada curva para entrenar; `None`
  usa todos los disponibles (más lento pero aprovecha toda la curva).
- `N_CAPAS` / `N_NEURONAS` / `ACTIVACION`: tamaño y activación de la red.
- `EPOCAS` / `PACIENCIA`: entrenamiento máximo y parada temprana
  (`PACIENCIA=0` la desactiva).
- `CUANTIZAR`: exporta el `.tflite` cuantizado a enteros en vez de
  float32 (más pequeño y rápido en el microcontrolador, algo menos preciso).
- `ID_MODELO`: nombre base de los ficheros generados en `modelos/`.

## Requisitos

Python 3.10+, TensorFlow (Keras), NumPy, Matplotlib. El intérprete de
TFLite se toma del paquete `ai_edge_litert` si está instalado, o de
`tf.lite` como alternativa.

## Ejecución

```bash
python rn_main.py              # entrena y exporta modelos/mpp_v1.tflite
python diagnostico_datos.py    # diagnóstico previo sobre sim_1.dat
python diagnostico_datos.py otro_fichero.dat
```
