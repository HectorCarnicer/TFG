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

/* Algoritmo P&O + salto con red neuronal (ver README, "Algoritmo de control") --*/
#define N_MUESTRAS  20U     /*!< Pares [V,I] de la ventana que consume la red */
#define DV          0.125f  /*!< Paso de perturbación P&O y del barrido forzado (V) */
#define DP_LIM      0.05f   /*!< |DeltaP| bajo el cual se considera "entorno de MPP" (W) */
#define DV_LIM      0.03f   /*!< |DeltaV| bajo el cual se sospecha cambio de irradiancia (V) */
#define DI_LIM      0.15f   /*!< |DeltaI| sobre el cual se sospecha cambio de irradiancia (A) */
#define N_REVALIDACION 60U  /*!< Muestras sin salto con RN tras las que se fuerza una revalidación */

#define BARRIDO_FORZADO 0U  /*!< 1 = la revalidación usa un barrido ascendente forzado; 0 = P&O normal */

#define V_MAX_BARRIDO 19.0f /*!< Techo del barrido forzado (V); al alcanzarlo salta a V_MIN_BARRIDO */
#define V_MIN_BARRIDO 6.0f  /*!< Tensión de reinicio del barrido forzado (V) */
#define N_SIN_AVANCE_BARRIDO N_MUESTRAS /*!< Vaciados seguidos de ventana que se interpretan como bloqueo del barrido */

#ifdef __cplusplus
}
#endif

#endif /* MAIN_CONFIG_H */
