# Makefile for Crazyflie RL Controller
#
# This Makefile integrates the RL controller with the Crazyflie firmware
# using the out-of-tree (OOT) build system.
#
# Copied from learning_to_fly_controller for compatibility.

CRAZYFLIE_BASE := external/crazyflie_firmware

OOT_CONFIG := $(PWD)/config 
EXTRA_CFLAGS += -I$(PWD) -I$(PWD)/external/rl_tools/include -std=c++17 -O3 -DRL_TOOLS_CONTROLLER

include $(CRAZYFLIE_BASE)/tools/make/oot.mk
