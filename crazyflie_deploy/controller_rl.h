/**
 * RL Controller Interface
 * Header for INT8 quantized policy controller
 */

#ifndef __CONTROLLER_RL_H__
#define __CONTROLLER_RL_H__

#include "stabilizer_types.h"

/**
 * Initialize the RL controller
 * Must be called before using controllerRL()
 */
void controllerRLInit(void);

/**
 * Test the RL controller inference
 * Returns true if controller passes self-test
 */
bool controllerRLTest(void);

/**
 * RL control loop (called at 100Hz from stabilizer)
 * Runs INT8 policy inference and outputs motor commands
 * Falls back to PID if safety checks fail
 */
void controllerRL(control_t *control,
                  const setpoint_t *setpoint,
                  const sensorData_t *sensors,
                  const state_t *state,
                  const uint32_t tick);

#endif /* __CONTROLLER_RL_H__ */
