# Dockerfile for Crazyflie RL Firmware Builder
#
# EXACT replica of learning-to-fly/Dockerfile_build_firmware
# This builds Crazyflie firmware with an RL policy checkpoint.
#
# The Docker image contains:
# - gcc-arm-none-eabi cross-compiler
# - crazyflie-firmware (Bitcraze)
# - rl_tools headers
# - Controller source files (copied at build time)
#
# Usage:
#   1. Build image (first time only):
#      docker build -t crazyflie-l2f-firmware .
#
#   2. Build firmware with your checkpoint:
#      docker run --rm \
#        -v /path/to/your/actor.h:/controller/data/actor.h:ro \
#        -v /path/to/output:/output \
#        crazyflie-l2f-firmware
#
# Output: /output/cf2.bin - flash with: cfloader flash cf2.bin stm32-fw

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies (same as L2F)
RUN apt-get update && apt-get install -y git python3 build-essential gcc-arm-none-eabi

# Create controller directory structure
WORKDIR /controller

# Clone crazyflie-firmware as submodule (recursive for STM32 libs)
RUN git clone --depth 1 --recursive https://github.com/bitcraze/crazyflie-firmware.git external/crazyflie_firmware

# Clone rl_tools (headers only, non-recursive)
RUN git clone --depth 1 https://github.com/rl-tools/rl-tools.git external/rl_tools

# Configure crazyflie firmware for CF2 platform
WORKDIR /controller/external/crazyflie_firmware
RUN make cf2_defconfig

# Return to controller root
WORKDIR /controller

# Copy controller source files into the image
# These are the files from learning_to_fly_controller
COPY controller/rl_tools_adapter.cpp /controller/
COPY controller/rl_tools_adapter.h /controller/
COPY controller/rl_tools_controller.c /controller/
COPY controller/rl_tools_controller.h /controller/
COPY controller/Makefile /controller/
COPY controller/Kbuild /controller/
COPY controller/config /controller/

# Create data directory for actor.h (will be mounted at runtime)
RUN mkdir -p /controller/data

# Default: build firmware and copy to /output
# The actor.h must be mounted at /controller/data/actor.h
CMD ["bash", "-c", "make -j$(nproc) && cp build/cf2.bin /output/cf2.bin"]
