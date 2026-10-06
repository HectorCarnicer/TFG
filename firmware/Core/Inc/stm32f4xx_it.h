/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    stm32f4xx_it.h
  * @brief   This file contains the headers of the interrupt handlers.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __STM32F4xx_IT_H
#define __STM32F4xx_IT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void NMI_Handler(void);
void HardFault_Handler(void);
void MemManage_Handler(void);
void BusFault_Handler(void);
void UsageFault_Handler(void);
void SVC_Handler(void);
void DebugMon_Handler(void);
void PendSV_Handler(void);
void SysTick_Handler(void);
/* USER CODE BEGIN EFP */

/* TIM1: interrupción de actualización (overflow), usada para medir tiempos
 *       -> delega en HAL_TIM_IRQHandler().
 * USART2: interrupción de transmisión/recepción, usada para el envío de
 *       resultados y la recepción de pares tensión/corriente por UART sin
 *       bloquear la CPU -> delega en HAL_UART_IRQHandler().
 * EXTI15_10: línea compartida por los botones PD11 (modo MANUAL / paso
 *       manual) y PD12 (modo AUTO) -> delega en HAL_GPIO_EXTI_IRQHandler()
 *       una vez por cada pin, ver stm32f4xx_it.c. */
void TIM1_UP_TIM10_IRQHandler(void);
void USART2_IRQHandler(void);
void EXTI15_10_IRQHandler(void);

/* USER CODE END EFP */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_IT_H */
