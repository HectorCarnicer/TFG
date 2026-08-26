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

/**
  * @brief  Carga un par tensión/corriente en el tensor de entrada de la red.
  * @param  voltage Tensión (V)
  * @param  current Corriente (A)
  * @retval None
  */
void AI_SetInputs(float voltage, float current);

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
