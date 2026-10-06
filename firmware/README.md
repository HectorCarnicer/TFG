# MPPT multi-módulo con red neuronal (STM32F411)

Firmware para el STM32F411E-DISCO (STM32F411VET6) que estima el punto de máxima
potencia (Vmpp) de un módulo solar a partir de un par tensión/corriente (V, I),
usando una red neuronal embebida generada con X-CUBE-AI / STM32Cube.AI Studio.
El microcontrolador recibe los pares (V, I) por UART desde un PC (que simula el
comportamiento de un sensor conectado a un DC/DC y a un panel solar), calcula la
inferencia y devuelve la predicción junto con el tiempo de inferencia.

## Hardware

| Función                | Pin  | Notas                                   |
|-------------------------|------|------------------------------------------|
| Botón modo MANUAL       | PD11 | Entrada, interrupción EXTI, flanco de subida |
| Botón modo AUTO         | PD12 | Entrada, interrupción EXTI, flanco de subida |
| LED STATE_IDLE           | PA1  | Salida push-pull |
| LED STATE_WAITING         | PA4  | Reubicado desde PA3 (en uso por USART2_RX) |
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
- **STATE_BUSY** — ejecuta la inferencia sobre el último par recibido, mide el
  tiempo y envía el resultado por UART. Vuelve a `STATE_WAITING`.

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
- `STATE_WAITING` con condición cumplida → `STATE_BUSY`: copia local del par
  recibido, `AI_SetInputs()` → `AI_RunInference()` (con medición de tiempo
  mediante `Get_Timer1_Ticks_us()`) → `AI_GetOutput()` → formatea y envía el
  resultado por `UART2_SendString()` → vuelve a `STATE_WAITING`.

## Protocolo UART

El PC envía cada muestra como 8 bytes sin cabecera: dos `float` de 32 bits en
little-endian, `[tensión][corriente]` (mismo layout que `VI_Pair_t` en
`data_types.h`). El microcontrolador responde con una línea de texto:

```
V=  12.34 V | I=  2.10 A | Prediccion=   18.7423 | t_inferencia=    842 us
```

## Integración de la red neuronal

El modelo (`[V, I] → [Vmpp]`, 2 entradas y 1 salida en `float32`) se genera en
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
