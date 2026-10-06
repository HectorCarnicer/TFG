/**
  ******************************************************************************
  * @file    main_config.h
  * @brief   Asignación de pines y configuración propia del proyecto.
  ******************************************************************************
  */

#ifndef MAIN_CONFIG_H
#define MAIN_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"

/* Exported variables ---------------------------------------------------------*/
extern volatile uint32_t g_tim1_ovf_count;

/* Exported functions prototypes -----------------------------------------------*/
uint64_t Get_Timer1_Ticks_us(void);
void UART2_SendString(const char *str);

/* Pin definitions ---------------------------------------------------------------*/
#define UART2_TX_Pin GPIO_PIN_2
#define UART2_TX_Port GPIOA

#define UART2_RX_Pin GPIO_PIN_3
#define UART2_RX_Port GPIOA

#define BUTTON_MANUAL_Pin GPIO_PIN_11
#define BUTTON_MANUAL_GPIO_Port GPIOD

#define BUTTON_AUTO_Pin GPIO_PIN_12
#define BUTTON_AUTO_GPIO_Port GPIOD

#define BUTTONS_EXTI_IRQn EXTI15_10_IRQn

#define LED_IDLE_Pin GPIO_PIN_1
#define LED_IDLE_GPIO_Port GPIOA

#define LED_WAITING_Pin GPIO_PIN_4
#define LED_WAITING_GPIO_Port GPIOA

#define LED_BUSY_Pin GPIO_PIN_5
#define LED_BUSY_GPIO_Port GPIOA

#ifdef __cplusplus
}
#endif

#endif /* MAIN_CONFIG_H */
