/**
  ******************************************************************************
  * @file    ai_bridge.h
  * @brief   Interfaz entre la máquina de estados y la red neuronal (AI/).
  ******************************************************************************
  */

#ifndef AI_BRIDGE_H
#define AI_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "data_types.h"

/**
  * @brief  Inicializa el runtime de inferencia y el contexto de la red.
  * @retval None
  */
void AI_Init(void);

/**
  * @brief  Carga un barrido de pares tensión/corriente en el tensor de
  *         entrada de la red.
  * @param  samples Vector de pares tensión/corriente
  * @param  n_samples Número de pares en samples
  * @retval None
  */
void AI_SetInputs(const VI_Pair_t *samples, uint32_t n_samples);

/**
  * @brief  Ejecuta la inferencia de la red neuronal.
  * @retval None
  */
void AI_RunInference(void);

/**
  * @brief  Lee la salida de la red tras una inferencia.
  * @retval Vmpp predicho (V)
  */
float AI_GetOutput(void);

#ifdef __cplusplus
}
#endif

#endif /* AI_BRIDGE_H */
