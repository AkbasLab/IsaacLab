"""
Crazyflie RL Controller Module
Integrates INT8 quantized policy for real-time control
"""

#include "controller_rl.h"
#include "policy_int8.h"
#include "stabilizer_types.h"
#include "log.h"
#include "param.h"
#include <math.h>
#include <string.h>

/* Configuration */
#define MAX_ROLL_PITCH_DEG 45.0f
#define MAX_YAW_RATE_DEG_S 200.0f
#define MIN_THRUST_PWM 10000
#define MAX_THRUST_PWM 60000
#define MAX_ACTION_CHANGE 0.3f
#define DEG_TO_RAD 0.017453292519943295f

/* State */
static uint8_t rl_enabled = 0;
static float prev_actions[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static uint32_t inference_count = 0;
static uint32_t failsafe_count = 0;

/* Performance monitoring */
static uint32_t last_inference_time_us = 0;

void controllerRLInit(void) {
    policy_init();
    memset(prev_actions, 0, sizeof(prev_actions));
    inference_count = 0;
    failsafe_count = 0;
}

bool controllerRLTest(void) {
    /* Test inference with zero inputs */
    float obs[POLICY_INPUT_DIM] = {0};
    float actions[POLICY_OUTPUT_DIM];
    
    policy_inference_int8(obs, actions);
    
    /* Check outputs are finite and in range */
    for (int i = 0; i < POLICY_OUTPUT_DIM; i++) {
        if (!isfinite(actions[i])) {
            return false;
        }
        if (actions[i] < -1.5f || actions[i] > 1.5f) {
            return false;  /* Suspicious values */
        }
    }
    
    return true;
}

static inline float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static bool safety_check(const state_t *state) {
    /* Check attitude limits */
    if (fabsf(state->attitude.roll) > MAX_ROLL_PITCH_DEG) return false;
    if (fabsf(state->attitude.pitch) > MAX_ROLL_PITCH_DEG) return false;
    
    /* Check position (if available) */
    if (state->position.z < 0.05f || state->position.z > 2.0f) return false;
    
    return true;
}

void controllerRL(control_t *control,
                  const setpoint_t *setpoint,
                  const sensorData_t *sensors,
                  const state_t *state,
                  const uint32_t tick) {
    
    /* Check if RL controller is enabled */
    if (!rl_enabled) {
        /* Use default PID controller */
        controllerPid(control, setpoint, sensors, state, tick);
        return;
    }
    
    /* Safety checks */
    if (!safety_check(state)) {
        rl_enabled = 0;
        failsafe_count++;
        controllerPid(control, setpoint, sensors, state, tick);
        return;
    }
    
    /* Construct observation vector (9D) */
    float obs[POLICY_INPUT_DIM];
    
    /* Linear acceleration (body frame, m/s²) */
    obs[0] = sensors->acc.x * 9.81f;
    obs[1] = sensors->acc.y * 9.81f;
    obs[2] = sensors->acc.z * 9.81f;
    
    /* Angular velocity (body frame, rad/s) */
    obs[3] = sensors->gyro.x * DEG_TO_RAD;
    obs[4] = sensors->gyro.y * DEG_TO_RAD;
    obs[5] = sensors->gyro.z * DEG_TO_RAD;
    
    /* Euler angles (rad) */
    obs[6] = state->attitude.roll * DEG_TO_RAD;
    obs[7] = state->attitude.pitch * DEG_TO_RAD;
    obs[8] = state->attitude.yaw * DEG_TO_RAD;
    
    /* Run inference */
    uint32_t start_time = usecTimestamp();
    float actions[POLICY_OUTPUT_DIM];
    policy_inference_int8(obs, actions);
    last_inference_time_us = usecTimestamp() - start_time;
    
    inference_count++;
    
    /* Validate outputs */
    for (int i = 0; i < POLICY_OUTPUT_DIM; i++) {
        if (!isfinite(actions[i])) {
            rl_enabled = 0;
            failsafe_count++;
            controllerPid(control, setpoint, sensors, state, tick);
            return;
        }
    }
    
    /* Rate limit actions (prevent jitter) */
    for (int i = 0; i < POLICY_OUTPUT_DIM; i++) {
        float delta = actions[i] - prev_actions[i];
        delta = clamp(delta, -MAX_ACTION_CHANGE, MAX_ACTION_CHANGE);
        actions[i] = prev_actions[i] + delta;
        prev_actions[i] = actions[i];
    }
    
    /* Convert actions to motor commands */
    /* Actions: [thrust, roll_moment, pitch_moment, yaw_moment] */
    /* All normalized to [-1, 1] */
    
    float thrust_norm = clamp((actions[0] + 1.0f) / 2.0f, 0.0f, 1.0f);
    uint16_t thrust_pwm = (uint16_t)(MIN_THRUST_PWM + 
                                     thrust_norm * (MAX_THRUST_PWM - MIN_THRUST_PWM));
    
    /* Map moments to rates (simple proportional) */
    float roll_rate = actions[1] * 200.0f;   /* deg/s */
    float pitch_rate = actions[2] * 200.0f;
    float yaw_rate = actions[3] * 100.0f;
    
    /* Clamp rates */
    roll_rate = clamp(roll_rate, -MAX_YAW_RATE_DEG_S, MAX_YAW_RATE_DEG_S);
    pitch_rate = clamp(pitch_rate, -MAX_YAW_RATE_DEG_S, MAX_YAW_RATE_DEG_S);
    yaw_rate = clamp(yaw_rate, -MAX_YAW_RATE_DEG_S, MAX_YAW_RATE_DEG_S);
    
    /* Set control outputs */
    control->thrust = thrust_pwm;
    control->roll = roll_rate;
    control->pitch = pitch_rate;
    control->yaw = yaw_rate;
    
    /* Final safety clamps */
    if (control->thrust > MAX_THRUST_PWM) control->thrust = MAX_THRUST_PWM;
    if (control->thrust < MIN_THRUST_PWM) control->thrust = MIN_THRUST_PWM;
}

/* Parameters for runtime control */
PARAM_GROUP_START(rl)
PARAM_ADD(PARAM_UINT8, enabled, &rl_enabled)
PARAM_GROUP_STOP(rl)

/* Logging */
LOG_GROUP_START(rl)
LOG_ADD(LOG_FLOAT, action0, &prev_actions[0])
LOG_ADD(LOG_FLOAT, action1, &prev_actions[1])
LOG_ADD(LOG_FLOAT, action2, &prev_actions[2])
LOG_ADD(LOG_FLOAT, action3, &prev_actions[3])
LOG_ADD(LOG_UINT32, infer_count, &inference_count)
LOG_ADD(LOG_UINT32, failsafe_count, &failsafe_count)
LOG_ADD(LOG_UINT32, infer_time_us, &last_inference_time_us)
LOG_GROUP_STOP(rl)
