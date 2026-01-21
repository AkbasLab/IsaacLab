#!/usr/bin/env python3
"""
Standalone export script that converts Isaac Lab trained policy to rl_tools C header format.
This matches the EXACT format used by the learning-to-fly firmware build system.

The firmware expects:
- Network: 146 → 64 → 64 → 4 with FAST_TANH activation
- Weights stored as unsigned char (byte) arrays representing raw float32 bytes
- Test observation and expected action for validation
"""

import struct
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import math


# ============================================================================
# L2F-Compatible Actor Network Definition (same as in train_hover.py)
# ============================================================================
class L2FActorNetwork(nn.Module):
    """Actor network matching L2F's architecture exactly: 146 → 64 → 64 → 4"""
    
    # L2F constants
    MASS = 0.027  # kg
    GRAVITY = 9.81  # m/s^2
    MAX_RPM = 21702.0
    THRUST_CONSTANT = 3.16e-10
    
    # Hover action in [-1, 1] space
    HOVER_ACTION = 2 * math.sqrt(MASS * GRAVITY / (4 * THRUST_CONSTANT)) / MAX_RPM - 1
    
    def __init__(self, obs_dim: int = 146, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize output layer bias to atanh(HOVER_ACTION) so network outputs hover by default
        hover_bias = math.atanh(self.HOVER_ACTION)
        nn.init.zeros_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, hover_bias)
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


# ============================================================================
# Float to Bytes Conversion
# ============================================================================
def float_array_to_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy array of float32 to raw bytes"""
    return arr.astype(np.float32).tobytes()


def bytes_to_c_array(data: bytes, indent: str = "                ") -> str:
    """Convert bytes to C unsigned char array string"""
    values = [str(b) for b in data]
    # Join all values with ", "
    return ", ".join(values)


# ============================================================================
# C Header Generation (matching L2F format exactly)
# ============================================================================
def generate_layer_code(
    layer_idx: int,
    weights: np.ndarray,  # shape: (out_features, in_features)
    biases: np.ndarray,   # shape: (out_features,)
    in_features: int,
    out_features: int,
    is_output: bool = False
) -> str:
    """Generate C++ code for a single layer matching L2F format"""
    
    group = "Output" if is_output else "Normal"
    
    weights_bytes = float_array_to_bytes(weights.flatten())
    biases_bytes = float_array_to_bytes(biases.flatten())
    
    code = f'''    namespace layer_{layer_idx} {{
        namespace weights {{
            namespace parameters_memory {{
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {{{bytes_to_c_array(weights_bytes)}}};
                using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned long, {out_features}, {in_features}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned long, 1>>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
                const CONTAINER_TYPE container = {{(float*)memory}};
            }}
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::spec<parameters_memory::CONTAINER_TYPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::{group}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
        }}
        namespace biases {{
            namespace parameters_memory {{
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {{{bytes_to_c_array(biases_bytes)}}};
                using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned long, 1, {out_features}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned long, 1>>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
                const CONTAINER_TYPE container = {{(float*)memory}};
            }}
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::spec<parameters_memory::CONTAINER_TYPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::{group}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
        }}
        using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Specification<float, unsigned long, {in_features}, {out_features}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::FAST_TANH, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain, 1, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::{group}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamicTag, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned long, 1>>;
        using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<SPEC>;
        const TYPE layer = {{weights::parameters, biases::parameters}};
    }}'''
    
    return code


def create_test_observation() -> np.ndarray:
    """Create a test observation representing hover state"""
    obs = np.zeros(146, dtype=np.float32)
    
    # Position error (already 0 for hover)
    # obs[0:3] = 0
    
    # Rotation matrix (identity = level orientation)
    # Row-major: [[1,0,0], [0,1,0], [0,0,1]]
    obs[3] = 1.0   # R[0,0]
    obs[4] = 0.0   # R[0,1]
    obs[5] = 0.0   # R[0,2]
    obs[6] = 0.0   # R[1,0]
    obs[7] = 1.0   # R[1,1]
    obs[8] = 0.0   # R[1,2]
    obs[9] = 0.0   # R[2,0]
    obs[10] = 0.0  # R[2,1]
    obs[11] = 1.0  # R[2,2]
    
    # Linear velocity (0 for hover)
    # obs[12:15] = 0
    
    # Angular velocity (0 for hover)
    # obs[15:18] = 0
    
    # Action history: 32 timesteps × 4 motors = 128 values
    # Fill with hover action (~0.334)
    hover_action = L2FActorNetwork.HOVER_ACTION
    obs[18:146] = hover_action
    
    return obs


def export_to_firmware(
    checkpoint_path: Path,
    output_path: Path,
    model_name: str = "isaaclab_policy"
) -> bool:
    """
    Export trained Isaac Lab policy to rl_tools C header format.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_path: Directory to write actor.h
        model_name: Name for metadata
        
    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print("Exporting Isaac Lab Policy to rl_tools Format")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output:     {output_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle the RSL-RL checkpoint format
    # The checkpoint has 'actor' key containing the OrderedDict directly
    if isinstance(checkpoint, dict) and 'actor' in checkpoint:
        actor_state = checkpoint['actor']
    elif isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # Filter actor weights from state dict
        actor_state = {}
        for k, v in state_dict.items():
            if k.startswith('actor.'):
                new_key = k.replace('actor.', '')
                actor_state[new_key] = v
            elif not k.startswith('critic.') and not k.startswith('std'):
                actor_state[k] = v
    else:
        raise ValueError("Unsupported checkpoint format")
    
    # Create actor network and load weights
    actor = L2FActorNetwork()
    
    # Load state dict - need to skip log_std if it's in actor_state but not in our network
    # Filter out any keys not in the network
    load_state = {k: v for k, v in actor_state.items() if k in actor.state_dict()}
    actor.load_state_dict(load_state, strict=False)
    actor.eval()
    
    # Print what we loaded
    print(f"\n  Loaded weights from checkpoint:")
    print(f"    fc1.weight: mean={actor.fc1.weight.mean():.4f}, std={actor.fc1.weight.std():.4f}")
    print(f"    fc2.weight: mean={actor.fc2.weight.mean():.4f}, std={actor.fc2.weight.std():.4f}")
    print(f"    fc3.weight: mean={actor.fc3.weight.mean():.4f}, std={actor.fc3.weight.std():.4f}")
    print(f"    fc3.bias: {actor.fc3.bias.data.numpy()}")
    
    # Extract weights
    w1 = actor.fc1.weight.detach().numpy()  # (64, 146)
    b1 = actor.fc1.bias.detach().numpy()    # (64,)
    w2 = actor.fc2.weight.detach().numpy()  # (64, 64)
    b2 = actor.fc2.bias.detach().numpy()    # (64,)
    w3 = actor.fc3.weight.detach().numpy()  # (4, 64)
    b3 = actor.fc3.bias.detach().numpy()    # (4,)
    
    print(f"\n  Layer 0: {w1.shape[1]} → {w1.shape[0]}")
    print(f"  Layer 1: {w2.shape[1]} → {w2.shape[0]}")
    print(f"  Layer 2: {w3.shape[1]} → {w3.shape[0]}")
    
    # Create test observation and compute expected action
    test_obs = create_test_observation()
    with torch.no_grad():
        test_obs_tensor = torch.from_numpy(test_obs).unsqueeze(0)
        expected_action = actor(test_obs_tensor).squeeze(0).numpy()
    
    print(f"\n  Test observation shape: {test_obs.shape}")
    print(f"  Expected action: {expected_action}")
    print(f"  Hover action target: {L2FActorNetwork.HOVER_ACTION:.4f}")
    
    # Check if output is close to hover
    mean_action = np.mean(expected_action)
    print(f"  Mean action output: {mean_action:.4f}")
    
    # Generate C++ header
    header = '''// rl_tools checkpoint generated from Isaac Lab training
// Architecture: 146 → 64 → 64 → 4 with FAST_TANH activation

namespace rl_tools::checkpoint::actor {
'''
    
    # Add layers
    header += generate_layer_code(0, w1, b1, 146, 64, is_output=False) + "\n"
    header += generate_layer_code(1, w2, b2, 64, 64, is_output=False) + "\n"
    header += generate_layer_code(2, w3, b3, 64, 4, is_output=True) + "\n"
    
    # Add model definition
    header += '''    namespace model_definition {
        using namespace RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn_models::sequential::interface;
        using MODEL = Module<layer_0::TYPE, Module<layer_1::TYPE, Module<layer_2::TYPE>>>;
    }
    using MODEL = model_definition::MODEL;
    const MODEL model = {layer_0::layer, {layer_1::layer, {layer_2::layer}}};
}
'''
    
    # Add test observation
    obs_bytes = float_array_to_bytes(test_obs)
    header += '''#include <rl_tools/containers.h>
namespace rl_tools::checkpoint::observation {
    static_assert(sizeof(unsigned char) == 1);
    alignas(float) const unsigned char memory[] = {''' + bytes_to_c_array(obs_bytes) + '''};
    using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned long, 1, 146, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned long, 1>>;
    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
    const CONTAINER_TYPE container = {(float*)memory};
}

'''
    
    # Add expected action
    action_bytes = float_array_to_bytes(expected_action)
    header += '''#include <rl_tools/containers.h>
namespace rl_tools::checkpoint::action {
    static_assert(sizeof(unsigned char) == 1);
    alignas(float) const unsigned char memory[] = {''' + bytes_to_c_array(action_bytes) + '''};
    using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned long, 1, 4, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned long, 1>>;
    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
    const CONTAINER_TYPE container = {(float*)memory};
}

'''
    
    # Add metadata
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    header += f'''namespace rl_tools::checkpoint::meta{{
   char name[] = "{timestamp}_{model_name}";
   char commit_hash[] = "isaaclab_export";
}}
'''
    
    # Write file
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "actor.h"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
    
    print(f"\n  [OK] Exported to: {output_file}")
    print(f"  [OK] File size: {output_file.stat().st_size:,} bytes")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export Isaac Lab policy to Crazyflie firmware format"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to PyTorch checkpoint file (.pt)"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for actor.h"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="isaaclab_policy",
        help="Model name for metadata"
    )
    
    args = parser.parse_args()
    
    success = export_to_firmware(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_name=args.name,
    )
    
    exit(0 if success else 1)
