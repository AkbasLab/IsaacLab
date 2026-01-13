"""
Simulation-only validation script
Tests trained policy without quantization for baseline performance
"""

import argparse
import torch
from pathlib import Path

# IsaacLab imports
from isaaclab_tasks.direct.quadcopter import QuadcopterEnvCfg
from isaaclab.envs import DirectRLEnv


def validate_sim_only(checkpoint_path: str, num_episodes: int = 100, render: bool = True):
    """Run validation in simulation with FP32 policy
    
    Args:
        checkpoint_path: Path to trained policy checkpoint (.pt or .pth)
        num_episodes: Number of episodes to evaluate
        render: Whether to render the simulation
    """
    print("=" * 80)
    print("Simulation-Only Validation")
    print("=" * 80)
    
    # Load environment
    print("\nInitializing environment...")
    cfg = QuadcopterEnvCfg()
    cfg.scene.num_envs = 16  # Multiple envs for faster evaluation
    cfg.viewer.eye = (7.5, 7.5, 7.5)
    cfg.sim.render_interval = 1 if render else cfg.decimation
    
    env = DirectRLEnv(cfg)
    
    # Load policy
    print(f"Loading policy from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    
    # Extract policy network
    if 'model_state_dict' in checkpoint:
        policy_state = checkpoint['model_state_dict']
    elif 'policy' in checkpoint:
        policy_state = checkpoint['policy']
    else:
        policy_state = checkpoint
    
    # Build policy network (32x32x2 MLP)
    from torch import nn
    policy = nn.Sequential(
        nn.Linear(9, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 4),
    ).to(env.device)
    
    # Load weights
    policy.load_state_dict(policy_state)
    policy.eval()
    
    print(f"Policy architecture: {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Evaluation loop
    print(f"\nRunning {num_episodes} episodes...")
    episode_returns = []
    episode_lengths = []
    success_count = 0
    
    obs_dict, _ = env.reset()
    obs = obs_dict['policy']
    
    episode_count = 0
    current_returns = torch.zeros(env.num_envs, device=env.device)
    current_lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)
    
    with torch.no_grad():
        while episode_count < num_episodes:
            # Policy inference
            actions = policy(obs)
            
            # Step environment
            obs_dict, rewards, dones, _ = env.step(actions)
            obs = obs_dict['policy']
            
            current_returns += rewards
            current_lengths += 1
            
            # Check for episode completion
            if dones.any():
                for idx in torch.where(dones)[0]:
                    if episode_count >= num_episodes:
                        break
                    
                    episode_returns.append(current_returns[idx].item())
                    episode_lengths.append(current_lengths[idx].item())
                    
                    # Check success (reached within 0.3m of goal)
                    final_pos = env._robot.data.root_pos_w[idx]
                    goal_pos = env._desired_pos_w[idx]
                    distance = torch.linalg.norm(final_pos - goal_pos)
                    if distance < 0.3:
                        success_count += 1
                    
                    current_returns[idx] = 0
                    current_lengths[idx] = 0
                    episode_count += 1
                    
                    if episode_count % 10 == 0:
                        print(f"  Episodes completed: {episode_count}/{num_episodes}")
    
    # Report results
    print("\n" + "=" * 80)
    print("Validation Results")
    print("=" * 80)
    
    mean_return = sum(episode_returns) / len(episode_returns)
    mean_length = sum(episode_lengths) / len(episode_lengths)
    success_rate = success_count / len(episode_returns)
    
    print(f"Mean Episode Return:  {mean_return:.2f}")
    print(f"Mean Episode Length:  {mean_length:.1f} steps")
    print(f"Success Rate:         {success_rate * 100:.1f}%")
    print(f"Min Return:           {min(episode_returns):.2f}")
    print(f"Max Return:           {max(episode_returns):.2f}")
    
    # Pass criteria
    print("\n" + "=" * 80)
    print("Validation Criteria")
    print("=" * 80)
    
    return_threshold = 50.0  # Adjust based on reward scale
    success_threshold = 0.5
    
    return_pass = mean_return > return_threshold
    success_pass = success_rate > success_threshold
    
    print(f"Mean Return > {return_threshold}: {'PASS' if return_pass else 'FAIL'}")
    print(f"Success Rate > {success_threshold}: {'PASS' if success_pass else 'FAIL'}")
    
    if return_pass and success_pass:
        print("\n✓ Simulation validation PASSED")
        return 0
    else:
        print("\n✗ Simulation validation FAILED - policy needs more training")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate policy in simulation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    
    args = parser.parse_args()
    
    exit_code = validate_sim_only(args.checkpoint, args.episodes, not args.no_render)
    exit(exit_code)
