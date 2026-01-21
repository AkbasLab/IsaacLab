/**
 * rl_tools_adapter.h - RL Tools Adapter Header
 * 
 * C interface for the RL policy controller.
 * Copied from learning_to_fly_controller for compatibility.
 */

#ifdef __cplusplus
extern "C"
#endif
void rl_tools_init();
#ifdef __cplusplus
extern "C"
#endif
float rl_tools_test(float*);
#ifdef __cplusplus
extern "C"
#endif
void rl_tools_control(float* state, float* actions);
#ifdef __cplusplus
extern "C"
#endif
char* rl_tools_get_checkpoint_name();
