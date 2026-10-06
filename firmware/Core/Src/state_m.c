/**
  ******************************************************************************
  * @file    state_m.c
  * @brief   Implementación de la máquina de estados (ver state_m.h).
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "state_m.h"
#include "main.h"
#include "data_types.h"
#include "ai_bridge.h"
#include <math.h>
#include <stdio.h>

/* Private variables -----------------------------------------------------------*/
static SystemState_t   g_state = STATE_IDLE;
static OperatingMode_t g_mode  = MODE_NONE;

static volatile VI_Pair_t rx_pair;
static volatile uint8_t   pair_ready_flag = 0U;
static volatile uint8_t   manual_flag = 0U;

/* Ventana de N_MUESTRAS pares [V,I] que consume la red neuronal */
static VI_Pair_t g_samples[N_MUESTRAS];
static uint32_t  g_sample_count = 0U;

/* Estado del algoritmo P&O + salto con RN (ver README, "Algoritmo de control") */
static float   g_po_step   = DV;  /*!< Paso de perturbación P&O actual (V); el signo es el sentido */
static float   g_prev_voltage = 0.0f;
static float   g_prev_current = 0.0f;
static float   g_prev_power   = 0.0f;
static uint8_t g_has_prev     = 0U; /*!< 1 cuando existe una muestra anterior con la que comparar */
static uint8_t g_was_entorno_mpp = 0U; /*!< 1 si la muestra anterior estaba en entorno de MPP */
static uint32_t g_samples_since_jump = 0U; /*!< Muestras consecutivas sin salto con RN */
static uint32_t g_resets_sin_avance = 0U; /*!< Vaciados consecutivos de la ventana durante un barrido forzado */

static char tx_buf[128];

/* Private function prototypes --------------------------------------------------*/
static void StateMachine_SetState(SystemState_t new_state);
static void StateMachine_EnterMode(OperatingMode_t mode);
static void StateMachine_ProcessSample(void);

/**
  * @brief  Cambia de estado y enciende el LED correspondiente.
  * @param  new_state Estado destino.
  * @retval None
  * @note   Modifica: g_state.
  */
static void StateMachine_SetState(SystemState_t new_state)
{
  HAL_GPIO_WritePin(LED_IDLE_GPIO_Port,    LED_IDLE_Pin,    GPIO_PIN_RESET);
  HAL_GPIO_WritePin(LED_WAITING_GPIO_Port, LED_WAITING_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(LED_BUSY_GPIO_Port,    LED_BUSY_Pin,    GPIO_PIN_RESET);

  switch (new_state)
  {
    case STATE_IDLE:
      HAL_GPIO_WritePin(LED_IDLE_GPIO_Port, LED_IDLE_Pin, GPIO_PIN_SET);
      break;

    case STATE_WAITING:
      HAL_GPIO_WritePin(LED_WAITING_GPIO_Port, LED_WAITING_Pin, GPIO_PIN_SET);
      break;

    case STATE_BUSY:
      HAL_GPIO_WritePin(LED_BUSY_GPIO_Port, LED_BUSY_Pin, GPIO_PIN_SET);
      break;

    default:
      break;
  }

  g_state = new_state;
}

/**
  * @brief  Pone la máquina de estados y el algoritmo en su estado inicial (STATE_IDLE).
  * @retval None
  * @note   Modifica: g_mode, pair_ready_flag, manual_flag, g_sample_count, g_po_step,
  *         g_has_prev, g_was_entorno_mpp, g_samples_since_jump, g_resets_sin_avance,
  *         g_prev_voltage, g_prev_current, g_prev_power, g_state.
  */
void StateMachine_Init(void)
{
  g_mode = MODE_NONE;
  pair_ready_flag = 0U;
  manual_flag = 0U;
  g_sample_count = 0U;
  g_po_step = DV;
  g_has_prev = 0U;
  g_was_entorno_mpp = 0U;
  g_samples_since_jump = 0U;
  g_resets_sin_avance = 0U;
  g_prev_voltage = 0.0f;
  g_prev_current = 0.0f;
  g_prev_power   = 0.0f;
  StateMachine_SetState(STATE_IDLE);
}

/**
  * @brief  Arranca TIM1 y la recepción UART, reinicia el algoritmo y pasa a STATE_WAITING.
  * @param  mode MODE_AUTO o MODE_MANUAL.
  * @retval None
  * @note   Modifica: g_mode, g_sample_count, g_po_step, g_has_prev, g_was_entorno_mpp,
  *         g_samples_since_jump, g_resets_sin_avance, g_prev_voltage, g_prev_current,
  *         g_prev_power, rx_pair, g_state.
  */
static void StateMachine_EnterMode(OperatingMode_t mode)
{
  g_mode = mode;
  g_sample_count = 0U;
  g_po_step = DV;
  g_has_prev = 0U;
  g_was_entorno_mpp = 0U;
  g_samples_since_jump = 0U;
  g_resets_sin_avance = 0U;
  g_prev_voltage = 0.0f;
  g_prev_current = 0.0f;
  g_prev_power   = 0.0f;

  HAL_TIM_Base_Start_IT(&htim1);

  /* Descarta un error pendiente (p. ej. overrun) antes de armar la recepción. */
  __HAL_UART_CLEAR_PEFLAG(&huart2);

  HAL_UART_Receive_IT(&huart2, (uint8_t *)&rx_pair, sizeof(VI_Pair_t));

  StateMachine_SetState(STATE_WAITING);
}

/**
  * @brief  Añade un par [V,I] a la ventana si no está llena.
  * @param  voltage Tensión medida (V).
  * @param  current Corriente medida (A).
  * @retval None
  * @note   Modifica: g_samples, g_sample_count.
  */
static void StateMachine_StoreSample(float voltage, float current)
{
  if (g_sample_count < N_MUESTRAS)
  {
    g_samples[g_sample_count].voltage = voltage;
    g_samples[g_sample_count].current = current;
    g_sample_count++;
  }
}

/**
  * @brief  Ejecuta la red neuronal sobre la ventana y la vacía.
  * @param  inference_time_us [out] Duración de la inferencia (us).
  * @retval Tensión a la que saltar, predicha por la red (V).
  * @note   Modifica: g_sample_count, g_resets_sin_avance.
  */
static float StateMachine_RunJump(uint32_t *inference_time_us)
{
  AI_SetInputs(g_samples, N_MUESTRAS);

  uint64_t t_start_us = Get_Timer1_Ticks_us();
  AI_RunInference();
  uint64_t t_end_us = Get_Timer1_Ticks_us();

  *inference_time_us = (uint32_t)(t_end_us - t_start_us);

  g_sample_count = 0U;
  g_resets_sin_avance = 0U;

  return AI_GetOutput();
}

/**
  * @brief  Detecta un posible cambio de irradiancia y, si lo hay, vacía la ventana.
  *         Durante un barrido forzado, deja de vaciarla tras N_SIN_AVANCE_BARRIDO
  *         vaciados seguidos (placa sin responder a las órdenes de tensión).
  * @param  entorno_mpp    1 si la muestra está en entorno de MPP.
  * @param  barrido_activo 1 si hay un barrido forzado en curso.
  * @param  delta_v        Variación de tensión respecto a la muestra anterior (V).
  * @param  delta_i        Variación de corriente respecto a la muestra anterior (A).
  * @retval None
  * @note   Modifica: g_sample_count, g_resets_sin_avance.
  */
static void StateMachine_CheckIrradianceChange(uint8_t entorno_mpp, uint8_t barrido_activo,
                                               float delta_v, float delta_i)
{
  uint8_t posible_cambio = (!entorno_mpp && ((fabsf(delta_v) < DV_LIM) || (fabsf(delta_i) > DI_LIM))) ? 1U : 0U;

  if (posible_cambio)
  {
    uint8_t bloqueo_detectado = (barrido_activo && (g_resets_sin_avance >= N_SIN_AVANCE_BARRIDO)) ? 1U : 0U;

    if (!bloqueo_detectado)
    {
      g_sample_count = 0U;
      if (barrido_activo)
      {
        g_resets_sin_avance++;
      }
    }
  }
  else if (barrido_activo)
  {
    /* La placa responde a las órdenes: no hay bloqueo. */
    g_resets_sin_avance = 0U;
  }
}

/**
  * @brief  Orden de tensión del barrido forzado: +DV por muestra; si se superaría
  *         V_MAX_BARRIDO, salta a V_MIN_BARRIDO y descarta la ventana (diente de sierra).
  * @param  voltage      Tensión medida (V).
  * @param  barrido_wrap [out] 1 si esta muestra provoca el salto a V_MIN_BARRIDO.
  * @retval Tensión a ordenar (V).
  * @note   Modifica: g_sample_count y g_resets_sin_avance (solo al dar la vuelta).
  */
static float StateMachine_NextSweepOrder(float voltage, uint8_t *barrido_wrap)
{
  if ((voltage + fabsf(DV)) > V_MAX_BARRIDO)
  {
    g_sample_count = 0U;
    g_resets_sin_avance = 0U;
    *barrido_wrap = 1U;
    return V_MIN_BARRIDO;
  }

  return voltage + fabsf(DV);
}

/**
  * @brief  Paso de P&O: invierte el sentido si la potencia ha bajado.
  * @param  voltage Tensión medida (V).
  * @param  delta_p Variación de potencia respecto a la muestra anterior (W).
  * @retval Tensión a ordenar (V).
  * @note   Modifica: g_po_step.
  */
static float StateMachine_NextPoOrder(float voltage, float delta_p)
{
  if (delta_p < 0.0f)
  {
    g_po_step = -g_po_step;
  }

  return voltage + g_po_step;
}

/**
  * @brief  Paso de recogida de ventana: detecta cambio de irradiancia, calcula la
  *         siguiente orden (barrido forzado o P&O) y guarda la muestra.
  * @param  voltage        Tensión medida (V).
  * @param  current        Corriente medida (A).
  * @param  delta_p        Variación de potencia (W).
  * @param  delta_v        Variación de tensión (V).
  * @param  delta_i        Variación de corriente (A).
  * @param  entorno_mpp    1 si la muestra está en entorno de MPP.
  * @param  barrido_activo 1 si hay un barrido forzado en curso.
  * @retval Tensión a ordenar (V).
  * @note   Modifica: g_samples, g_sample_count, g_po_step, g_resets_sin_avance.
  */
static float StateMachine_CollectStep(float voltage, float current,
                                      float delta_p, float delta_v, float delta_i,
                                      uint8_t entorno_mpp, uint8_t barrido_activo)
{
  float   next_voltage_order;
  uint8_t barrido_wrap = 0U;

  StateMachine_CheckIrradianceChange(entorno_mpp, barrido_activo, delta_v, delta_i);

  if (barrido_activo)
  {
    next_voltage_order = StateMachine_NextSweepOrder(voltage, &barrido_wrap);
  }
  else
  {
    next_voltage_order = StateMachine_NextPoOrder(voltage, delta_p);
  }

  if (!barrido_wrap)
  {
    StateMachine_StoreSample(voltage, current);
  }

  return next_voltage_order;
}

/**
  * @brief  Decide la orden de tensión de una muestra que ya tiene muestra anterior:
  *         P&O en entorno de MPP, salto con RN si la ventana está completa, o
  *         recogida de ventana en otro caso. Fuerza una revalidación tras
  *         N_REVALIDACION muestras sin salto.
  * @param  voltage           Tensión medida (V).
  * @param  current           Corriente medida (A).
  * @param  power             Potencia V·I (W).
  * @param  jumped            [out] 1 si se ha ejecutado un salto con RN.
  * @param  inference_time_us [out] Duración de la inferencia (us); solo válido si hubo salto.
  * @retval Tensión a ordenar (V).
  * @note   Modifica: g_sample_count, g_samples, g_po_step, g_was_entorno_mpp,
  *         g_resets_sin_avance.
  */
static float StateMachine_ProcessWithHistory(float voltage, float current, float power,
                                             uint8_t *jumped, uint32_t *inference_time_us)
{
  float delta_p = power   - g_prev_power;
  float delta_v = voltage - g_prev_voltage;
  float delta_i = current - g_prev_current;
  uint8_t forzar_revalidacion = (g_samples_since_jump >= N_REVALIDACION) ? 1U : 0U;
  uint8_t entorno_mpp = (!forzar_revalidacion && (fabsf(delta_p) < DP_LIM)) ? 1U : 0U;
  uint8_t barrido_activo = (BARRIDO_FORZADO && forzar_revalidacion) ? 1U : 0U;
  float   next_voltage_order;

  if (!entorno_mpp && g_was_entorno_mpp)
  {
    /* Se sale del entorno de MPP: la ventana acumulada ya no es válida. */
    g_sample_count = 0U;
  }

  if (!entorno_mpp && (g_sample_count >= N_MUESTRAS))
  {
    next_voltage_order = StateMachine_RunJump(inference_time_us);
    *jumped = 1U;
  }
  else
  {
    next_voltage_order = StateMachine_CollectStep(voltage, current, delta_p, delta_v, delta_i,
                                                  entorno_mpp, barrido_activo);
  }

  g_was_entorno_mpp = entorno_mpp;

  return next_voltage_order;
}

/**
  * @brief  Guarda la muestra como referencia de la siguiente y actualiza el
  *         contador de muestras sin salto.
  * @param  voltage Tensión medida (V).
  * @param  current Corriente medida (A).
  * @param  power   Potencia V·I (W).
  * @param  jumped  1 si en esta muestra hubo salto con RN.
  * @retval None
  * @note   Modifica: g_prev_voltage, g_prev_current, g_prev_power, g_has_prev,
  *         g_samples_since_jump.
  */
static void StateMachine_RememberSample(float voltage, float current, float power, uint8_t jumped)
{
  g_prev_voltage = voltage;
  g_prev_current = current;
  g_prev_power   = power;
  g_has_prev = 1U;

  if (jumped)
  {
    g_samples_since_jump = 0U;
  }
  else
  {
    g_samples_since_jump++;
  }
}

/**
  * @brief  Envía por UART la línea de resultado de la muestra.
  * @param  voltage           Tensión medida (V).
  * @param  current           Corriente medida (A).
  * @param  order             Orden de tensión (V), enviada en el campo "Prediccion".
  * @param  inference_time_us Duración de la inferencia (us); 0 si no hubo salto.
  * @retval None
  * @note   Modifica: tx_buf.
  */
static void StateMachine_SendLog(float voltage, float current, float order, uint32_t inference_time_us)
{
  snprintf(tx_buf, sizeof(tx_buf),
           "V= %6.2f V | I= %5.2f A | Prediccion= %9.4f | t_inferencia= %6lu us\r\n",
           (double)voltage, (double)current,
           (double)order,
           (unsigned long)inference_time_us);

  UART2_SendString(tx_buf);
}

/**
  * @brief  Procesa el par [V,I] recibido: calcula la orden de tensión (P&O + salto
  *         con RN), la envía por UART y vuelve a STATE_WAITING.
  * @retval None
  * @note   Modifica: g_samples, g_sample_count, g_po_step, g_prev_*, g_has_prev,
  *         g_was_entorno_mpp, g_samples_since_jump, g_resets_sin_avance, tx_buf, g_state.
  */
static void StateMachine_ProcessSample(void)
{
  float voltage = rx_pair.voltage;
  float current = rx_pair.current;
  float power   = voltage * current;

  float    next_voltage_order = voltage + g_po_step;
  uint32_t inference_time_us = 0U;
  uint8_t  jumped = 0U;

  if (g_has_prev)
  {
    next_voltage_order = StateMachine_ProcessWithHistory(voltage, current, power,
                                                         &jumped, &inference_time_us);
  }
  else
  {
    /* Primera muestra: no hay referencia anterior, solo se guarda. */
    StateMachine_StoreSample(voltage, current);
    g_was_entorno_mpp = 0U;
  }

  StateMachine_RememberSample(voltage, current, power, jumped);
  StateMachine_SendLog(voltage, current, next_voltage_order, inference_time_us);

  StateMachine_SetState(STATE_WAITING);
}

/**
  * @brief  Ejecuta un paso de la máquina de estados (no bloqueante).
  * @retval None
  * @note   Modifica: pair_ready_flag, manual_flag, g_state (y lo que modifique
  *         StateMachine_ProcessSample).
  */
void StateMachine_Run(void)
{
  switch (g_state)
  {
    case STATE_IDLE:
      break;

    case STATE_WAITING:
    {
      uint8_t ready_to_process;

      if (g_mode == MODE_AUTO)
      {
        ready_to_process = pair_ready_flag;
      }
      else
      {
        ready_to_process = pair_ready_flag && manual_flag;
      }

      if (ready_to_process)
      {
        pair_ready_flag = 0U;
        manual_flag = 0U;
        StateMachine_SetState(STATE_BUSY);
      }
      break;
    }

    case STATE_BUSY:
      StateMachine_ProcessSample();
      break;

    default:
      StateMachine_SetState(STATE_IDLE);
      break;
  }
}

/**
  * @brief  Callback de los botones: en STATE_IDLE elige el modo; en MODE_MANUAL
  *         el botón manual autoriza procesar el siguiente par.
  * @param  GPIO_Pin Pin que generó la interrupción.
  * @retval None
  * @note   Modifica: manual_flag y lo que modifique StateMachine_EnterMode.
  */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
  if (GPIO_Pin == BUTTON_MANUAL_Pin)
  {
    if (g_state == STATE_IDLE)
    {
      StateMachine_EnterMode(MODE_MANUAL);
    }
    else if (g_mode == MODE_MANUAL)
    {
      manual_flag = 1U;
    }
  }
  else if (GPIO_Pin == BUTTON_AUTO_Pin)
  {
    if (g_state == STATE_IDLE)
    {
      StateMachine_EnterMode(MODE_AUTO);
    }
  }
}

/**
  * @brief  Callback de fin de recepción UART: marca el par como disponible y
  *         rearma la recepción.
  * @param  huart Handle de la UART.
  * @retval None
  * @note   Modifica: pair_ready_flag, rx_pair.
  */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    pair_ready_flag = 1U;
    HAL_UART_Receive_IT(&huart2, (uint8_t *)&rx_pair, sizeof(VI_Pair_t));
  }
}

/**
  * @brief  Callback de error de USART2: limpia el error y rearma la recepción.
  * @param  huart Handle de la UART.
  * @retval None
  * @note   Modifica: rx_pair.
  */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    __HAL_UART_CLEAR_PEFLAG(huart);
    HAL_UART_Receive_IT(&huart2, (uint8_t *)&rx_pair, sizeof(VI_Pair_t));
  }
}
