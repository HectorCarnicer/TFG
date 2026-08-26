Universidad Politécnica de Madrid

Escuela técnica superior de ingeniería y diseño industrial

# Trabajo de fin de grado

# Algoritmo MPPT basado en red neuronal embebida

Autor: Héctor Carnicer Ull

Controlador MPPT (Maximum Power Point Tracking) para un panel fotovoltaico
bajo sombreado parcial, que usa una red neuronal embebida en un
microcontrolador para estimar la tensión de máxima potencia (Vmp) a partir
de un punto de operación (V, I). El proyecto cubre todo el pipeline: desde
la simulación física del panel hasta el firmware que corre la inferencia
en el microcontrolador, pasando por el entrenamiento del modelo y un banco
de pruebas que simula el sensor real para validar el conjunto sin
necesidad de un panel físico.
 
## Componentes
 
El proyecto se organiza en cuatro partes independientes, cada una con su
propio README detallado:
 
| # | Componente | Dónde corre | README |
|---|---|---|---|
| 1 | Simulador de panel solar | PC (Python) | `Simulación de panel solar bajo sombreado parcial` |
| 2 | Predictor de MPP (red neuronal) | PC (Python/TensorFlow) | `Predictor de MPP con red neuronal (v1)` |
| 3 | Firmware del microcontrolador | STM32F411E-DISCO | `MPPT multi-módulo con red neuronal (STM32F411)` |
| 4 | Simulador de sensor UART | PC (Python) | `Simulador de sensor UART` |
 
## Flujo del pipeline completo
 
```
 (1) Simulador de panel  ──►  datos_panel/*.json + sim_<ID>.dat
        │                          (curvas I-V + MPP simuladas,
        │                           varios paneles / sombreados)
        ▼
 (2) Predictor de MPP    ──►  entrena una red densa (V,I)→Vmp
        │                          sobre sim_<ID>.dat
        │                     exporta modelos/mpp_v1.tflite
        ▼
 (3) Firmware STM32      ──►  el .tflite se integra vía X-CUBE-AI /
        │                     STM32Cube.AI Studio (carpeta AI/) y corre
        │                     la inferencia embebida en el micro
        ▼
 (4) Simulador de sensor ──►  reproduce por UART las curvas de
                               sim_<ID>.dat hacia el micro, recoge las
                               predicciones y las compara con el Vmp
                               real (log + gráficos de error)
```
 
Los cuatro componentes se comunican solo a través de artefactos
concretos: ficheros `sim_<ID>.dat` (de 1 a 2, y de 1 a 4) y el modelo
`.tflite` exportado (de 2 a 3). No comparten código entre sí.
 
### 1. Simulador de panel solar
 
Simula la curva I-V de un panel fotovoltaico bajo irradiancia no uniforme,
con el modelo de diodo único de PVlib (ajuste CEC) y el panel dividido en
substrings con diodo de bypass. Genera tandas de curvas aleatorias
(`datos_simulaciones/sim_<ID>.dat`) con su MPP, a partir de fichas de
panel en `datos_panel/` (varios paneles comerciales disponibles, no solo
el original del proyecto). También produce gráficos I-V/P-V de cada tanda.
 
Es la fuente de datos de entrenamiento y de test de todo el resto del
proyecto.
 
### 2. Predictor de MPP (red neuronal)
 
Entrena una red densa que recibe un punto (V, I) y predice el Vmp de la
curva a la que pertenece, usando los datos generados por el simulador de
panel. Incluye normalización/desnormalización integradas en el propio
modelo, partición del dataset por curva completa (evita fuga de
información), exportación a TensorFlow Lite (con cuantización opcional) y
verificación Keras vs. TFLite. `diagnostico_datos.py` calcula, antes de
entrenar, el techo teórico de R² alcanzable con la entrada (V, I) —
relevante porque con sombreado parcial ese techo puede ser bajo por
indeterminación física, no por un mal ajuste del modelo.
 
Produce el `.tflite` que se integra en el firmware.
 
### 3. Firmware del microcontrolador (STM32F411)
 
Corre en un STM32F411E-DISCO. Recibe pares (V, I) por UART (8 bytes,
2 floats little-endian), ejecuta la inferencia con el modelo embebido
(generado con X-CUBE-AI / STM32Cube.AI Studio a partir del `.tflite` del
componente 2) y devuelve por UART la predicción de Vmp junto con el
tiempo de inferencia. Implementado como una máquina de estados
(`IDLE` → `WAITING` → `BUSY`) con dos modos de operación (manual/automático)
seleccionables por botón físico.
 
### 4. Simulador de sensor UART
 
Sustituye al sensor físico y al panel real para poder probar el conjunto
modelo + firmware de extremo a extremo desde un PC. Envía por UART
muestras (V, I) tomadas de un dataset `sim_<ID>.dat`, recibe la
predicción del micro y calcula su error frente al Vmp real. Dos modos:
`"Consecutivo"` (recorre curvas del dataset punto a punto, para validar
la precisión del modelo) y `"Simulacion"` (simula un seguimiento MPPT
real con cambios bruscos de irradiancia en mitad del seguimiento, para
validar el comportamiento dinámico). Genera logs y gráficos de error.


