Universidad Politécnica de Madrid

Escuela técnica superior de ingeniería y diseño industrial

# Trabajo de fin de grado

# Algoritmo MPPT basado en red neuronal embebida

Autor: Héctor Carnicer Ull

Controlador MPPT (Maximum Power Point Tracking) para un panel fotovoltaico
bajo sombreado parcial, que usa una red neuronal embebida en un
microcontrolador para estimar la tensión de máxima potencia (Vmp) a partir
de una ventana de muestras (V, I) tomadas durante un barrido de tensión. El proyecto cubre todo el pipeline: desde
la simulación física del panel hasta el firmware que corre la inferencia
en el microcontrolador, pasando por el entrenamiento del modelo y un banco
de pruebas que simula el sensor real para validar el conjunto sin
necesidad de un panel físico.
 
## Componentes
 
El proyecto se organiza en cuatro partes independientes, cada una con su
propio README detallado:
 
| # | Componente | Carpeta | Dónde corre | README |
|---|---|---|---|---|
| 1 | Simulador de panel solar | `simulacion/` | PC (Python) | `Simulación de panel solar bajo sombreado parcial` |
| 2 | Predictor de MPP (red neuronal) | `red_neuronal/` | PC (Python/TensorFlow) | `Predictor de MPP con red neuronal (V2: ventana de muestras)` |
| 3 | Firmware del microcontrolador | `firmware/` | STM32F411E-DISCO | `MPPT multi-módulo con red neuronal (STM32F411)` |
| 4 | Simulador de sensor UART | `simulador_uart/` | PC (Python) | `Simulador de sensor UART` |
 
## Flujo del pipeline completo
 
```
 (1) Simulador de panel  ──►  datos_panel/*.json + sim_<ID>.dat
        │                          (curvas I-V + MPP simuladas,
        │                           varios paneles / sombreados)
        ▼
 (2) Predictor de MPP    ──►  entrena una red densa que recibe una
        │                          ventana de N pares (V,I) y predice Vmp
        │                     exporta modelos/mpp_v<ID>/mpp_v<ID>.tflite
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
 
Entrena una red densa de regresión que recibe una ventana de N pares (V, I)
consecutivos de la curva, tomados con un paso de tensión fijo como los que
recogería un barrido en tiempo real, y predice el Vmp de esa curva. Con un
único punto la predicción está limitada por la indeterminación física del
sombreado parcial (curvas distintas pasan por el mismo punto); la ventana
añade la forma local de la curva y resuelve buena parte de esa ambigüedad.

Puede entrenar combinando varios `sim_<ID>.dat`. Cada modelo se guarda en
`modelos/mpp_v<ID>/` junto con un resumen JSON con la configuración de
datos, ventana, partición, arquitectura y métricas. `diagnostico_datos.py`
evalúa un modelo ya entrenado sobre su propio conjunto de test y
`comparador_rn.py` compara varios modelos entre sí.

El modelo integrado en el firmware de esta versión es `mpp_v3`: ventana de
20 muestras con paso de 0,125 V en sentido ascendente, entrenado con
`sim_1.dat`.
 
### 3. Firmware del microcontrolador (STM32F411)
 
Corre en un STM32F411E-DISCO. Recibe pares (V, I) por UART (8 bytes,
2 floats little-endian) y responde a cada uno con la siguiente orden de
tensión, calculada con un algoritmo Perturb & Observe (P&O) combinado con un
salto predicho por la red neuronal:

- Mientras la variación de potencia entre muestras es pequeña, el sistema
  se considera en el entorno del MPP y sigue con P&O de paso fijo
  (0,125 V), guardando las muestras en una ventana.
- Cuando la potencia varía bruscamente y la ventana tiene 20 muestras,
  ejecuta la inferencia con el modelo embebido (generado con X-CUBE-AI /
  STM32Cube.AI Studio a partir del `.tflite` del componente 2) y ordena
  directamente la tensión predicha.
- Si la variación de tensión y corriente apunta a un cambio de
  irradiancia, descarta la ventana acumulada y empieza una nueva.
- Cada 60 muestras sin salto fuerza una revalidación, para no quedarse
  atrapado en un máximo local.

Implementado como una máquina de estados (`IDLE` → `WAITING` → `BUSY`) con
dos modos de operación (manual/automático) seleccionables por botón físico.
 
### 4. Simulador de sensor UART
 
Sustituye al sensor físico y al panel real para poder probar el conjunto
modelo + firmware de extremo a extremo desde un PC. Envía por UART
muestras (V, I) tomadas de un dataset `sim_<ID>.dat`, recibe la
predicción del micro y calcula su error frente al Vmp real. Dos modos:
`"Consecutivo"` (recorre curvas del dataset punto a punto, para validar
la precisión del modelo) y `"Simulacion"` (simula un seguimiento MPPT
real con cambios bruscos de irradiancia en mitad del seguimiento, para
validar el comportamiento dinámico). Genera logs, gráficos de error y
gráficos de la potencia entregada frente a la máxima disponible.

## Datos de simulación

Los ficheros de curvas `sim_<ID>.dat` no se incluyen en el repositorio por
su tamaño (unos 15 MB cada uno). Se generan con el simulador de panel
(`simulacion/simular_panel_main.py`), que los guarda en
`simulacion/datos_simulaciones/`. Cada fichero incluye en su metadata el
panel, el rango de irradiancia, la temperatura y la semilla con que se
generó.

En esta versión se usan cinco datasets, todos con 500 curvas, irradiancia
por substring entre 150 y 1000 W/m² y 35 °C:

| Dataset | Panel | Semilla |
|---|---|---|
| `sim_1.dat` | `datos_panel_0.json` | 43753 |
| `sim_2.dat` … `sim_5.dat` | `datos_panel_1.json` … `datos_panel_4.json` | sin fijar |

Solo `sim_1.dat` se puede regenerar idéntico. Para usar los datasets en los
demás componentes hay que copiarlos a:

- `red_neuronal/datos/sim_<ID>.dat`
- `simulador_uart/datos/sim_1.dat`
