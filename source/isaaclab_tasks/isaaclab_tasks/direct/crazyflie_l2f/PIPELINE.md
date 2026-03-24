# Crazyflie L2F Training Pipeline

Current training and evaluation flow for the Crazyflie 2.1 point-navigation and precision-hold policies in Isaac Lab.

**Last Updated:** March 23, 2026  
**Status:** Active workflow

---

## Overview

The current policy lineage is:

1. Train a base point-navigation policy with [train_pointnav.py](/d:/coding/Capstone/learning-to-fly/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/train_pointnav.py)
2. Fine-tune it for fixed-goal hold with [train_hold_finetune.py](/d:/coding/Capstone/learning-to-fly/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/train_hold_finetune.py)
3. Fine-tune again for tighter precision hold from the best phase-3 hold checkpoint
4. Evaluate fixed-goal behavior with [eval_pointnav_goal.py](/d:/coding/Capstone/learning-to-fly/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/eval_pointnav_goal.py)

Current checkpoint lineage:

`checkpoints_pointnav/best_model.pt`  
-> hold fine-tune phases  
-> `checkpoints_hold_finetune_phase3_20260323_161844/best_model.pt`  
-> precision hold fine-tune  
-> `checkpoints_hold_finetune_tight_20260323_212826/best_model.pt`

---

## Checkpoint Layout

Each training directory can contain:

- `best_model.pt`: best mean reward checkpoint
- `best_hold_model.pt`: best hold-event checkpoint
- `best_proxy_hold_model.pt`: best proxy-hold checkpoint
- `checkpoint_<N>.pt`: periodic checkpoint
- `stage_start_<tag>.pt`: curriculum stage start snapshot
- `stage_complete_<tag>.pt`: curriculum stage completion snapshot
- `final_model.pt`: final save after training

Important directories currently in use:

- `checkpoints_pointnav`
- `checkpoints_hold_finetune`
- `checkpoints_hold_finetune_phase2_20260323_024001`
- `checkpoints_hold_finetune_phase3_20260323_161251`
- `checkpoints_hold_finetune_phase3_20260323_161844`
- `checkpoints_hold_finetune_tight_20260323_212716`
- `checkpoints_hold_finetune_tight_20260323_212826`

---

## Step 1: Base PointNav Training

Train the base 149-dimensional point-navigation policy:

```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\train_pointnav.py --headless
```

Output:

- `source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\checkpoints_pointnav\best_model.pt`

Notes:

- Observation space is 149 dims: hover state plus 3D goal-relative state
- This stage learns to travel toward goals, not precision hold

---

## Step 2: Hold Fine-Tune

Warm-start from the base pointnav checkpoint and fine-tune for fixed-goal hold behavior:

```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\train_hold_finetune.py --checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\checkpoints_pointnav\best_model.pt --target_x 0.0 --target_y 0.0 --target_z 0.3 --z_reference_offset 1.0 --timestamp_checkpoint_dir --checkpoint_dir_name checkpoints_hold_finetune_phase3
```

Representative successful output directory:

- `checkpoints_hold_finetune_phase3_20260323_161844`

Key outputs from that directory:

- `best_model.pt`
- `best_hold_model.pt`
- `best_proxy_hold_model.pt`

The most commonly used hold checkpoint in the recent workflow was:

- `source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\checkpoints_hold_finetune_phase3_20260323_161844\best_model.pt`

---

## Step 3: Precision Hold Fine-Tune

Continue training from the validated phase-3 hold checkpoint, but tighten the 3D hold radius and emphasize staying exactly on the point.

Current command:

```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\train_hold_finetune.py --checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\checkpoints_hold_finetune_phase3_20260323_161844\best_model.pt --target_x 0.0 --target_y 0.0 --target_z 0.3 --z_reference_offset 1.0 --goal_radius 0.03 --spawn_z_min 0.25 --spawn_z_max 0.35 --hold_time 10 --episode_length_s 15 --num_envs 2048 --max_iterations 300 --timestamp_checkpoint_dir --checkpoint_dir_name checkpoints_hold_finetune_tight --headless
```

What changed in this phase:

- True 3D goal distance is used for hold behavior
- Goal radius is tightened to `0.03 m`
- Hold bonus is much stronger than simple touch reward
- Spawn height is sampled near the target height
- The objective is "stay very close to the exact point"

Latest known output directory:

- `checkpoints_hold_finetune_tight_20260323_212826`

Current latest checkpoint:

- `source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\checkpoints_hold_finetune_tight_20260323_212826\best_model.pt`

---

## Step 4: Fixed-Goal Evaluation

Evaluate a checkpoint against a fixed goal with:

```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\eval_pointnav_goal.py --checkpoint source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\checkpoints_hold_finetune_tight_20260323_212826\best_model.pt --target_x 0.0 --target_y 0.0 --target_z 0.3 --z_reference_offset 1.0 --deterministic --duration 10 --show_goal_marker --output_dir source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\play_eval_results\goal_eval_hold_finetune_tight_best_ground_0p3_vis --kit_args "--/app/vulkan=false --/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=0"
```

Notes on the current eval behavior:

- The script uses the explicit checkpoint path you pass
- `target_z` is interpreted with `z_reference_offset`
- For height-hold-style eval, the script can start the drone at the commanded `z` height while still allowing lateral XY motion

Outputs:

- `goal_eval_data.csv`
- `summary.json`

under the selected `play_eval_results/...` directory

---

## Current Recommended Models

For broad fixed-goal hold:

- `checkpoints_hold_finetune_phase3_20260323_161844\best_model.pt`

For tighter precision hold:

- `checkpoints_hold_finetune_tight_20260323_212826\best_model.pt`

If comparing checkpoints, keep the eval command identical and change only `--checkpoint`.

---

## Key Script Roles

| File | Purpose |
|------|---------|
| `train_pointnav.py` | Base point-navigation PPO training |
| `train_hold_finetune.py` | Fixed-goal hold and precision-hold fine-tuning |
| `eval_pointnav_goal.py` | Fixed-goal evaluation with CSV logging |
| `play_eval.py` | Older interactive/analysis evaluation path |
| `export_policy_standalone.py` | Export policy weights for deployment |
| `DEPLOYMENT.md` | Firmware/export/deployment flow |

---

## Common Workflow

Quick practical loop:

1. Train or fine-tune a checkpoint
2. Evaluate it with `eval_pointnav_goal.py`
3. Inspect `goal_eval_data.csv`
4. If the drone holds too loosely, continue precision fine-tuning from the best prior checkpoint
5. Promote the new `best_model.pt` only after fixed-goal eval improves

---

## Notes

- High training hold rate does not automatically mean clean eval transients; train/eval start-state mismatch matters
- The precision-hold stage is intended to reduce vertical bias and keep the drone inside a much tighter radius around the exact 3D point
- Checkpoint folders are gitignored in this task directory
