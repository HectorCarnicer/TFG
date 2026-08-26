/**
  ******************************************************************************
  * @file    state_m.h
  * @brief   Interfaz pública de la máquina de estados.
  ******************************************************************************
  */

#ifndef STATE_M_H
#define STATE_M_H

#ifdef __cplusplus
extern "C" {
#endif

/* Exported functions prototypes ----------------------------------------------*/

/**
  * @brief  Inicializa la máquina de estados en STATE_IDLE.
  * @retval None
  */
void StateMachine_Init(void);

/**
  * @brief  Ejecuta un paso de la máquina de estados. No bloqueante.
  * @retval None
  */
void StateMachine_Run(void);

#ifdef __cplusplus
}
#endif

#endif /* STATE_M_H */
