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
#include <stdio.h>

/* Private variables -----------------------------------------------------------*/
static SystemState_t   g_state = STATE_IDLE;
static OperatingMode_t g_mode  = MODE_NONE;

static volatile VI_Pair_t rx_pair;
static volatile uint8_t   pair_ready_flag = 0U;
static volatile uint8_t   manual_flag = 0U;

static char tx_buf[128];

/* Private function prototypes --------------------------------------------------*/
static void StateMachine_SetState(SystemState_t new_state);
static void StateMachine_EnterMode(OperatingMode_t mode);

/**
  * @brief  Actualiza g_state y el LED correspondiente.
  * @param  new_state Estado al que se transiciona
  * @retval None
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
  * @brief  Inicializa la máquina de estados en STATE_IDLE.
  * @retval None
  */
void StateMachine_Init(void)
{
  g_mode = MODE_NONE;
  pair_ready_flag = 0U;
  manual_flag = 0U;
  StateMachine_SetState(STATE_IDLE);
}

/**
  * @brief  Arranca TIM1 y la recepción UART, fija el modo y pasa a
  *         STATE_WAITING.
  * @param  mode MODE_AUTO o MODE_MANUAL
  * @retval None
  */
static void StateMachine_EnterMode(OperatingMode_t mode)
{
  g_mode = mode;

  HAL_TIM_Base_Start_IT(&htim1);

  /* Descarta un posible error pendiente (p. ej. overrun) antes de armar la
   * primera recepción; ver HAL_UART_ErrorCallback(). */
  __HAL_UART_CLEAR_PEFLAG(&huart2);

  HAL_UART_Receive_IT(&huart2, (uint8_t *)&rx_pair, sizeof(VI_Pair_t));

  StateMachine_SetState(STATE_WAITING);
}

/**
  * @brief  Ejecuta un paso de la máquina de estados. No bloqueante.
  * @retval None
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
    {
      float voltage = rx_pair.voltage;
      float current = rx_pair.current;

      AI_SetInputs(voltage, current);

      uint64_t t_start_us = Get_Timer1_Ticks_us();
      AI_RunInference();
      uint64_t t_end_us = Get_Timer1_Ticks_us();

      uint32_t inference_time_us = (uint32_t)(t_end_us - t_start_us);
      float prediction = AI_GetOutput();

      snprintf(tx_buf, sizeof(tx_buf),
               "V= %6.2f V | I= %5.2f A | Prediccion= %9.4f | t_inferencia= %6lu us\r\n",
               (double)voltage, (double)current, (double)prediction,
               (unsigned long)inference_time_us);

      UART2_SendString(tx_buf);

      StateMachine_SetState(STATE_WAITING);
      break;
    }

    default:
      StateMachine_SetState(STATE_IDLE);
      break;
  }
}

/**
  * @brief  Callback HAL de interrupción externa (botones).
  * @param  GPIO_Pin Pin que ha generado la interrupción
  * @retval None
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
  * @brief  Callback HAL de fin de recepción UART.
  * @param  huart Handle de la UART
  * @retval None
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
  * @brief  Callback HAL de error de USART2. Limpia el error y rearma la
  *         recepción.
  * @param  huart Handle de la UART
  * @retval None
  */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    __HAL_UART_CLEAR_PEFLAG(huart);
    HAL_UART_Receive_IT(&huart2, (uint8_t *)&rx_pair, sizeof(VI_Pair_t));
  }
}
