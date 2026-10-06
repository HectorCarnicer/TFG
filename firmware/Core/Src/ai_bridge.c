/**
  ******************************************************************************
  * @file    ai_bridge.c
  * @brief   Implementación del puente con la red neuronal (ver ai_bridge.h).
  ******************************************************************************
  */

#include "ai_bridge.h"
#include "network.h"
#include "user_init.h"

/* Contexto de la red. Antes declarado en AI/App/Src/app_x-cube-ai.c
 * (interfaz generada, no usada en este proyecto); se declara aquí para que
 * ai_bridge.c no dependa de ningún fichero de AI/App. */
STAI_NETWORK_CONTEXT_DECLARE(network_context, STAI_NETWORK_CONTEXT_SIZE)

STAI_ALIGNED(32)
static uint8_t activations_pool[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];
static stai_ptr data_activations[] = { activations_pool };

void AI_Init(void)
{
  stai_runtime_init();
  user_stai_network_init(network_context);
  stai_network_set_activations(network_context, data_activations, STAI_NETWORK_ACTIVATIONS_NUM);
}

void AI_SetInputs(const VI_Pair_t *samples, uint32_t n_samples)
{
  stai_ptr  inputs[STAI_NETWORK_IN_NUM];
  stai_size n_in;

  if (stai_network_get_inputs(network_context, inputs, &n_in) != STAI_SUCCESS)
  {
    return;
  }

  float *ai_in = (float *)inputs[0];

  for (uint32_t i = 0U; i < n_samples; i++)
  {
    ai_in[2U * i]      = samples[i].voltage;
    ai_in[2U * i + 1U] = samples[i].current;
  }
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
