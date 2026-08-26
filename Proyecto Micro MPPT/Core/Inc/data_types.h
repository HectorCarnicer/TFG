/**
  ******************************************************************************
  * @file    data_types.h
  * @brief   Tipos de datos compartidos entre main.c y la máquina de estados.
  ******************************************************************************
  */

#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
  * @brief Estados de la máquina de estados principal.
  */
typedef enum
{
  STATE_IDLE = 0,
  STATE_WAITING,
  STATE_BUSY
} SystemState_t;

/**
  * @brief Modo de funcionamiento del bucle de adquisición/inferencia.
  */
typedef enum
{
  MODE_NONE = 0,
  MODE_AUTO,
  MODE_MANUAL
} OperatingMode_t;

/**
  * @brief Par tensión/corriente de un módulo solar.
  */
typedef struct
{
  float voltage;    /*!< Tensión (V) */
  float current;    /*!< Corriente (A) */
} VI_Pair_t;

#ifdef __cplusplus
}
#endif

#endif /* DATA_TYPES_H */
