# Crazyflie Firmware Deployment Guide

This guide covers deploying trained Isaac Lab Crazyflie policies to a physical Crazyflie 2.1 drone.

## Pipeline Overview

Train checkpoint (`.pt`) -> Export `actor.h` -> Build firmware (`cf2.bin`) -> Flash with `cfloader`

## Standard Task-Local Firmware Flow

This is the self-contained firmware path under `crazyflie_l2f/firmware`.

### Build with pre-exported `firmware/actor_isaac_lab.h`

From PowerShell on Windows:

```powershell
wsl -e bash -lc "docker run --rm \
  -v /mnt/d/coding/Capstone/learning-to-fly/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/firmware/actor_isaac_lab.h:/controller/data/actor.h:ro \
  -v /mnt/d/coding/Capstone/learning-to-fly/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/build_firmware:/output \
  arpllab/learning_to_fly_build_firmware"
```

Expected output: `D:\coding\Capstone\learning-to-fly\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\build_firmware\cf2.bin`

## Custom 149-Dim Firmware Flow via `l2f_firmware_149`

Use this path for the custom 149-dimensional pointnav / hold firmware variant in `D:\coding\Capstone\learning-to-fly\IsaacLab\l2f_firmware_149`.

This flow uses:

- the exported `actor.h`
- the custom `rl_tools_adapter.cpp/.h`
- the custom `rl_tools_controller.c`
- the custom Dockerfile in `l2f_firmware_149`

### Exact commands in sequence

Run these commands from PowerShell:

```powershell
cd D:\coding\Capstone\learning-to-fly\IsaacLab
```

```powershell
python source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\export_policy_standalone.py `
  "D:\coding\Capstone\learning-to-fly\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\checkpoints_hold_finetune_unified_motion_curriculum_export149_preservewarmstart_20260326_130156\best_model.pt" `
  "D:\coding\Capstone\learning-to-fly\IsaacLab\l2f_firmware_149\data" `
  --name hold_finetune_export149
```

That writes `D:\coding\Capstone\learning-to-fly\IsaacLab\l2f_firmware_149\data\actor.h`.

Then build the Docker image inside WSL:

```powershell
wsl -e bash -lc "cd /mnt/d/coding/Capstone/learning-to-fly/IsaacLab/l2f_firmware_149 && docker build -t l2f-firmware-149 ."
```

Then run the image to produce `cf2.bin`:

```powershell
wsl -e bash -lc "mkdir -p /mnt/d/coding/Capstone/learning-to-fly/IsaacLab/l2f_firmware_149/build_firmware && cd /mnt/d/coding/Capstone/learning-to-fly/IsaacLab/l2f_firmware_149 && docker run --rm -v /mnt/d/coding/Capstone/learning-to-fly/IsaacLab/l2f_firmware_149/build_firmware:/output l2f-firmware-149"
```

Expected output: `D:\coding\Capstone\learning-to-fly\IsaacLab\l2f_firmware_149\build_firmware\cf2.bin`

### What the two Docker commands do

`docker build`:
- reads `l2f_firmware_149/Dockerfile`
- clones the controller repo and submodules into the image
- copies in your custom firmware files
- copies `data/actor.h` into the image
- creates the image named `l2f-firmware-149`

`docker run`:
- starts a container from `l2f-firmware-149`
- mounts `build_firmware` from your host into `/output`
- runs `make`
- copies `build/cf2.bin` to `/output/cf2.bin`

Important note:

- `l2f_firmware_149/Dockerfile` copies `data/actor.h` during `docker build`, not `docker run`
- if you export a new checkpoint, you must run `docker build` again before `docker run`

## Flash to Crazyflie

Put Crazyflie into bootloader mode first:

1. Turn the Crazyflie off.
2. Hold the power button for 3+ seconds.
3. Wait for the blue LEDs to blink.

### Flash standard task-local firmware

```powershell
cfloader flash D:\coding\Capstone\learning-to-fly\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\build_firmware\cf2.bin stm32-fw -w radio://0/80/2M
```

### Flash custom `l2f_firmware_149` firmware

```powershell
cfloader flash D:\coding\Capstone\learning-to-fly\IsaacLab\l2f_firmware_149\build_firmware\cf2.bin stm32-fw -w radio://0/80/2M
```

## Notes

- `cfloader` typically comes from `cfclient` / `crazyflie-clients-python`
- default radio URI here is `radio://0/80/2M`
- the custom `l2f_firmware_149` path expects `actor.h`, not `agent.h`
