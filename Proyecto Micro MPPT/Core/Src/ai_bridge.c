/**
  ******************************************************************************
  * @file    ai_bridge.c
  * @brief   Implementación del puente con la red neuronal (ver ai_bridge.h).
  ******************************************************************************
  */

#include "ai_bridge.h"
#include "network.h"

/* Definido en AI/App/Src/app_x-cube-ai.c (STAI_NETWORK_CONTEXT_DECLARE). */
extern stai_network network_context[];

void AI_SetInputs(float voltage, float current)
{
  stai_ptr  inputs[STAI_NETWORK_IN_NUM];
  stai_size n_in;

  if (stai_network_get_inputs(network_context, inputs, &n_in) != STAI_SUCCESS)
  {
    return;
  }

  float *ai_in = (float *)inputs[0];
  ai_in[0] = voltage;
  ai_in[1] = current;
}

void AI_RunInference(void)
{
  stai_network_run(network_context, STAI_MODE_SYNC);
}

float AI_GetOutput(void)
{
  stai_ptr  outputs[STAI_NETWORK_OUT_NUM];
  stai_size n_out;

  if (stai_network_get_outputs(network_context, outputs, &n_out) != STAI_SUCCESS)
  {
    return 0.0f;
  }

  float *ai_out = (float *)outputs[0];
  return ai_out[0];
}
