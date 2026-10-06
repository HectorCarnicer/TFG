# MPPT multi-módulo con red neuronal (STM32F411)

Firmware para el STM32F411E-DISCO (STM32F411VET6) que controla el punto de
trabajo de un panel solar combinando un algoritmo Perturba y Observa (P&O) con
saltos de tensión calculados por una red neuronal embebida (generada con
X-CUBE-AI / STM32Cube.AI Studio) que estima la tensión del punto de máxima
potencia (Vmpp) a partir de una ventana de muestras (V, I). El
microcontrolador recibe las muestras por UART desde un PC (que simula el
sensor conectado a un DC/DC y a un panel solar), decide la siguiente orden de
tensión y la devuelve junto con el tiempo de inferencia. El algoritmo se
describe en la sección "Algoritmo de control".

## Hardware

| Función                | Pin  | Notas                                   |
|-------------------------|------|------------------------------------------|
| Botón modo MANUAL       | PD11 | Entrada, interrupción EXTI, flanco de subida |
| Botón modo AUTO         | PD12 | Entrada, interrupción EXTI, flanco de subida |
| LED STATE_IDLE           | PA1  | Salida push-pull |
| LED STATE_WAITING         | PA4  | Salida push-pull (PA3 está ocupado por USART2_RX) |
| LED STATE_BUSY            | PA5  | Salida push-pull |
| USART2 TX                | PA2  | 115200 baudios, 8N1 |
| USART2 RX                | PA3  | 115200 baudios, 8N1 |

TIM1 se configura como base de tiempos libre a 1 MHz (prescaler 47 sobre reloj
de 48 MHz), usada junto con un contador de overflows por software
(`g_tim1_ovf_count`) para formar un temporizador extendido de resolución 1 µs
con el que se mide el tiempo de inferencia.

## Estructura del repositorio

```
Core/Inc, Core/Src        Lógica de la aplicación (HAL, máquina de estados, puente con la red)
AI/App                    Puente generado por X-CUBE-AI / STM32Cube.AI Studio con las funciones aiInit/aiRun/...
AI/Generated               Código y pesos de la red neuronal, generados a partir del modelo (.tflite)
AI/Middlewares              Runtime de inferencia de ST (stai) y librería estática precompilada
```

`AI/` reproduce la disposición clásica de X-CUBE-AI (`App/{Inc,Src}`,
`Generated/{Inc,Src}`, `Middlewares/{Inc,Lib}`) aunque el código se haya
generado con la herramienta más reciente (STM32Cube.AI Studio), para que
sustituir la carpeta entera al regenerar el modelo sea un simple
arrastrar-y-soltar sin tocar la configuración del proyecto.

## Flujo del programa

`main()`:

1. `HAL_Init()`, `STM32CubeAI_Studio_AI_Init()` (inicializa la red neuronal),
   `SystemClock_Config()`.
2. Inicializa los periféricos: GPIO (botones y LEDs), TIM1, USART2.
3. `StateMachine_Init()`: estado inicial `STATE_IDLE`.
4. Bucle infinito llamando a `StateMachine_Run()`.

## Máquina de estados

Implementada en `state_m.c`. Tres estados:

- **STATE_IDLE** — esperando a que se pulse un botón para elegir modo. No
  arranca ni el temporizador ni la recepción UART.
- **STATE_WAITING** — esperando el siguiente par (V, I). En modo `MODE_AUTO`
  basta con que llegue por UART; en `MODE_MANUAL` además hace falta pulsar
  PD11.
- **STATE_BUSY** — procesa el último par recibido con el algoritmo de control
  (`StateMachine_ProcessSample()`), envía el resultado por UART y vuelve a
  `STATE_WAITING`.

Transiciones:

- Pulsar PD11 o PD12 en `STATE_IDLE` → `HAL_GPIO_EXTI_Callback()` →
  `StateMachine_EnterMode()`: fija el modo, arranca TIM1, limpia errores
  pendientes de USART2 y arma la primera recepción (`HAL_UART_Receive_IT`) →
  pasa a `STATE_WAITING`.
- Pulsar PD11 estando ya en marcha y en `MODE_MANUAL` → marca `manual_flag`,
  consumida en el siguiente paso de `STATE_WAITING`.
- Llega un paquete completo por UART → `HAL_UART_RxCpltCallback()`: marca
  `pair_ready_flag` y rearma la recepción para el siguiente paquete.
- Error de USART2 (overrun, trama, ruido, paridad) → `HAL_UART_ErrorCallback()`:
  el HAL aborta la recepción en curso al detectar un error; esta callback la
  limpia y la rearma, para que un error puntual no deje la recepción muerta
  de forma permanente.
- `STATE_WAITING` con condición cumplida → `STATE_BUSY`:
  `StateMachine_ProcessSample()` calcula la orden de tensión (si toca saltar,
  ejecuta `AI_SetInputs()` → `AI_RunInference()` midiendo el tiempo con
  `Get_Timer1_Ticks_us()` → `AI_GetOutput()`), la envía por
  `UART2_SendString()` y vuelve a `STATE_WAITING`.

## Protocolo UART

El PC envía cada muestra como 8 bytes sin cabecera: dos `float` de 32 bits en
little-endian, `[tensión][corriente]` (mismo layout que `VI_Pair_t` en
`data_types.h`). El microcontrolador responde con una línea de texto por
muestra:

```
V=  12.34 V | I=  2.10 A | Prediccion=  12.4750 | t_inferencia=      0 us
```

`Prediccion` es la **orden de tensión** para la siguiente muestra (V): la
predicción de la red en un salto, o el resultado del paso de P&O o del barrido
en el resto de casos. `t_inferencia` es la duración de la inferencia (µs) y
vale 0 cuando en esa muestra no se ha ejecutado la red.

## Algoritmo de control

Implementado en `state_m.c`; los parámetros están en `main_config.h`. En cada
muestra recibida se calcula la orden de tensión siguiendo estos pasos.

**Magnitudes.** A partir de la muestra actual y la anterior se calculan
`ΔP`, `ΔV` y `ΔI`. La primera muestra tras entrar en modo AUTO/MANUAL no tiene
anterior: solo se guarda en la ventana.

**Ventana.** Se acumulan hasta `N_MUESTRAS` pares [V, I]. Cuando está llena,
la red puede consumirla.

**Entorno de MPP.** Si `|ΔP| < DP_LIM` se considera que el sistema está en el
entorno de un máximo: se aplica P&O clásico (se mantiene el sentido de
perturbación si la potencia subió, se invierte si bajó; paso `DV`) y se guarda
la muestra en la ventana.

**Salto con red neuronal.** Si `|ΔP| ≥ DP_LIM` (fuera del entorno de MPP) y la
ventana está llena, se ejecuta la red y la orden de tensión pasa a ser
directamente su predicción (el sistema "salta" a la tensión predicha). Tras el
salto se vacía la ventana. Si la ventana aún no está llena, se sigue
recogiendo muestras con P&O.

**Ventana siempre fresca.** Al abandonar el entorno de MPP se vacía la
ventana: lo acumulado mientras el sistema estaba convergido ya no representa
la situación actual (p. ej. tras un cambio de curva) y saltar con esos datos
daría una predicción obsoleta.

**Detección de cambio de irradiancia.** Fuera del entorno de MPP, si
`|ΔV| < DV_LIM` o `|ΔI| > DI_LIM`, la variación de potencia no se explica por
el paso de tensión ordenado (V casi no se movió, o I cambió bruscamente), de
modo que se interpreta como un cambio real de irradiancia y se vacía la
ventana. `DI_LIM` depende del panel: debe quedar por encima de lo que cambia
la corriente en un paso normal de P&O y por debajo de los saltos de corriente
de un cambio de curva real.

**Revalidación periódica.** Si pasan `N_REVALIDACION` muestras seguidas sin
saltar con la red, se trata la muestra actual como si estuviera fuera del
entorno de MPP: se descarta la ventana y se recoge una nueva para un salto de
comprobación. Sin ella, un P&O que converge a un máximo local se queda
atrapado ahí, porque nunca vuelve a considerarse "lejos del MPP". Es relevante
en paneles con varios substrings con diodos de bypass, cuya curva P-V puede
tener más de un máximo.

**Barrido forzado de revalidación (`BARRIDO_FORZADO`).** Con `BARRIDO_FORZADO`
a 0, la ventana de revalidación se rellena con el P&O normal. Con 1, durante
la revalidación la orden de tensión es un barrido ascendente de `+DV` por
muestra:

- Motivo: el P&O reactivo, rebotando sobre un máximo ya convergido, produce
  ventanas con oscilación alrededor de un punto, un tipo de entrada que puede
  no estar en el conjunto de entrenamiento de la red (barridos monótonos con
  paso `DV`). Un barrido ascendente reproduce esa forma de ventana.
- Diente de sierra: el barrido está acotado por `V_MAX_BARRIDO`. Si el
  siguiente paso lo superara, la orden pasa directamente a `V_MIN_BARRIDO` y se
  descarta la ventana en curso, de modo que toda ventana usada para inferencia
  es puramente ascendente (no se invierte el sentido del barrido, para no
  generar ventanas descendentes o con inflexión).
- Cota superior: sin `V_MAX_BARRIDO`, un barrido que parte cerca del pico alto
  seguiría subiendo hacia la tensión de circuito abierto, donde la corriente
  cae tan bruscamente que `|ΔI| > DI_LIM` vaciaría la ventana casi en cada
  muestra y el salto no llegaría a ejecutarse.
- Cota inferior: `V_MIN_BARRIDO` debe quedar en una zona donde el convertidor
  siga fielmente las órdenes de tensión (la zona cercana a 0 V puede no
  hacerlo) y por debajo del pico de menor tensión del panel, para poder
  explorarlo.
- Salvaguarda contra bloqueos: si la placa no sigue las órdenes del barrido
  (`|ΔV| ≈ 0`), la detección de cambio de irradiancia vaciaría la ventana en
  cada muestra y esta nunca se completaría. Por ello, durante un barrido
  forzado se cuentan los vaciados consecutivos (`g_resets_sin_avance`); al
  llegar a `N_SIN_AVANCE_BARRIDO` se deja de vaciar la ventana para que se
  complete y el salto con la red ordene una tensión distinta. El contador se
  pone a cero cuando la placa responde, tras un salto con la red y al dar la
  vuelta el diente de sierra.

Con `BARRIDO_FORZADO` a 0, `V_MAX_BARRIDO`, `V_MIN_BARRIDO` y
`N_SIN_AVANCE_BARRIDO` no se usan. El log no registra el valor de
`BARRIDO_FORZADO`, conviene anotar con cuál se compiló cada prueba.

### Parámetros (`main_config.h`)

| Macro                  | Valor  | Significado |
|------------------------|--------|-------------|
| `N_MUESTRAS`           | 20     | Pares [V, I] de la ventana que consume la red |
| `DV`                   | 0,125 V | Paso de P&O y del barrido forzado |
| `DP_LIM`               | 0,05 W | Umbral de `|ΔP|` para el entorno de MPP |
| `DV_LIM`               | 0,03 V | Umbral inferior de `|ΔV|` para sospechar cambio de irradiancia |
| `DI_LIM`               | 0,15 A | Umbral superior de `|ΔI|` para sospechar cambio de irradiancia |
| `N_REVALIDACION`       | 60     | Muestras sin salto tras las que se fuerza una revalidación |
| `BARRIDO_FORZADO`      | 0      | 1 = barrido forzado en la revalidación; 0 = P&O normal |
| `V_MAX_BARRIDO`        | 19 V   | Techo del barrido forzado |
| `V_MIN_BARRIDO`        | 6 V    | Tensión de reinicio del barrido forzado |
| `N_SIN_AVANCE_BARRIDO` | `N_MUESTRAS` | Vaciados seguidos que se interpretan como bloqueo |

`DP_LIM`, `DV_LIM` y `DI_LIM` dependen del panel y del ruido de medida y se
ajustan experimentalmente.

## Integración de la red neuronal

El modelo (ventana de `N_MUESTRAS` pares [V, I] → Vmpp; `2·N_MUESTRAS`
entradas intercaladas `V0, I0, V1, I1, …` y 1 salida, en `float32`) se genera en
`AI/` mediante X-CUBE-AI / STM32Cube.AI Studio, que expone la API `stai`
(`AI/Middlewares/Inc/stai.h`, `AI/Generated/Inc/network.h`) y un fichero de
arranque (`AI/App/Src/app_x-cube-ai.c`) con `STM32CubeAI_Studio_AI_Init()`
(llamada una vez desde `main()`).

`Core/Src/ai_bridge.c` implementa las tres funciones que usa la máquina de
estados (`AI_SetInputs`, `AI_RunInference`, `AI_GetOutput`) apoyándose
únicamente en la API pública de `stai` (`stai_network_get_inputs`,
`stai_network_get_outputs`, `stai_network_run`) y en el símbolo
`network_context`, que el generador declara con enlace externo en
`app_x-cube-ai.c`. Al no depender de ninguna variable interna ni de ningún
marcador `USER CODE` dentro de `AI/App`, la carpeta `AI/` completa puede
sustituirse por una nueva generación (otro modelo, otra versión del
generador) sin reintegrar nada a mano: basta con actualizar el nombre de la
librería estática (`-lNetworkRuntimeXXXX_CM4_GCC`) en la configuración del
proyecto si cambia de versión.

`app_x-cube-ai.c` conserva una única modificación respecto al código
generado: `aiRun()` no incluye el arnés de perfilado (`aiTestUtility.h`,
contadores de ciclos, impresión por UART y un `HAL_Delay` de 5 segundos) que
trae la plantilla por defecto, ya que ese código no está entre marcadores
`USER CODE` y bloquearía la máquina de estados en cada inferencia. Al
regenerar el modelo, si la compilación falla señalando `aiTestUtility.h`,
basta con volver a quitar ese `#include` y limpiar el cuerpo de `aiRun()` (ya
no se usa de todas formas: `AI_RunInference()` invoca `stai_network_run()`
directamente a través de `ai_bridge.c`).

## Compilación

Proyecto STM32CubeIDE (build system administrado). Requiere el paquete
STM32Cube FW_F4 configurado por el propio IDE. Tras cualquier cambio en `AI/`
se recomienda un Clean + Build completo.
