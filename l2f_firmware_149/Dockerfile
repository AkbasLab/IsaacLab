FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git python3 build-essential gcc-arm-none-eabi

# Clone fresh repo
RUN git clone https://github.com/arplaboratory/learning_to_fly_controller.git /controller
WORKDIR /controller
RUN git submodule update --init --recursive -- external/crazyflie_firmware
RUN git submodule update --init -- external/rl_tools

# Configure the crazyflie firmware
WORKDIR /controller/external/crazyflie_firmware
RUN make cf2_defconfig

# Copy modified files (overwrite defaults)
WORKDIR /controller
COPY rl_tools_adapter.cpp /controller/rl_tools_adapter.cpp
COPY rl_tools_adapter.h /controller/rl_tools_adapter.h
COPY rl_tools_controller.c /controller/rl_tools_controller.c
COPY data/actor.h /controller/data/actor.h

CMD ["bash", "-c", "make -j$(nproc) && cp build/cf2.bin /output/cf2.bin"]
