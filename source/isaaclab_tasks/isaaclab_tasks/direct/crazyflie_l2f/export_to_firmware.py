#!/usr/bin/env python3
"""
Firmware Export Pipeline for Isaac Lab Crazyflie L2F

This module exports trained policies from Isaac Lab to the rl_tools checkpoint
format used by the Crazyflie firmware. The exported checkpoint follows the
EXACT format used by learning-to-fly to ensure compatibility.

The checkpoint is in the Sequential module format with:
- rl_tools::checkpoint::actor namespace with layer_0, layer_1, layer_2
- rl_tools::checkpoint::observation/action for testing
- rl_tools::checkpoint::meta for model name and commit hash

Architecture: 146 inputs → 64 hidden → 64 hidden → 4 outputs (FAST_TANH activation)

Usage:
    from crazyflie_l2f.export_to_firmware import export_policy_to_firmware
    
    export_policy_to_firmware(
        policy=actor,
        output_path="actor.h",
        model_name="isaac_lab_policy"
    )

References:
- learning-to-fly/scripts/convert_checkpoint_to_firmware.py
- learning-to-fly/training/ppo_gpu/export.py
"""

import struct
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def float_to_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy float32 array to bytes for C++ checkpoint."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr.tobytes()


def bytes_to_c_array_inline(data: bytes) -> str:
    """
    Convert bytes to C array initializer string (single line).
    
    This matches the format used by learning-to-fly convert script.
    """
    return ", ".join(str(b) for b in data)


def generate_layer_code(
    layer_name: str,
    weights: np.ndarray,
    biases: np.ndarray,
    input_dim: int,
    output_dim: int,
    group_type: str,
) -> str:
    """
    Generate C++ code for a single dense layer in Sequential format.
    
    This EXACTLY matches the format from learning-to-fly/scripts/convert_checkpoint_to_firmware.py
    """
    # Convert to bytes (row-major order)
    weights_bytes = float_to_bytes(weights.flatten())
    biases_bytes = float_to_bytes(biases.flatten())
    
    # Format as inline C arrays (no newlines in the array)
    weights_c = bytes_to_c_array_inline(weights_bytes)
    biases_c = bytes_to_c_array_inline(biases_bytes)
    
    code = f'''    namespace {layer_name} {{
        namespace weights {{
            namespace parameters_memory {{
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {{{weights_c}}};
                using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned int, {output_dim}, {input_dim}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned int, 1>>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
                const CONTAINER_TYPE container = {{(float*)memory}}; 
            }}
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::spec<parameters_memory::CONTAINER_TYPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::{group_type}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
        }}
        namespace biases {{
            namespace parameters_memory {{
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {{{biases_c}}};
                using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned int, 1, {output_dim}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned int, 1>>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
                const CONTAINER_TYPE container = {{(float*)memory}}; 
            }}
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::spec<parameters_memory::CONTAINER_TYPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::{group_type}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
        }}
        using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Specification<float, unsigned int, {input_dim}, {output_dim}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::FAST_TANH, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain, 1, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::{group_type}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamicTag, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned int, 1>>;
        using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<SPEC>;
        const TYPE layer = {{weights::parameters, biases::parameters}};
    }}
'''
    return code


def export_policy_to_firmware(
    policy: torch.nn.Module,
    output_path: Union[str, Path],
    model_name: str = "isaac_lab_policy",
    obs_mean: Optional[np.ndarray] = None,
    obs_std: Optional[np.ndarray] = None,
) -> str:
    """
    Export a trained policy to rl_tools checkpoint format for Crazyflie firmware.
    
    This generates the EXACT format used by learning-to-fly/scripts/convert_checkpoint_to_firmware.py
    
    IMPORTANT: For PPO policies trained with observation normalization, you must either:
    1. Bake normalization into the first layer weights (recommended)
    2. Use a modified firmware adapter that applies normalization
    
    If obs_mean and obs_std are provided, normalization is baked into layer_0.
    
    Args:
        policy: PyTorch actor network with 3 linear layers (146→64→64→4)
        output_path: Path to save the checkpoint file
        model_name: Name for the model metadata
        obs_mean: Optional observation mean for normalization (shape: 146)
        obs_std: Optional observation std for normalization (shape: 146)
        
    Returns:
        Path to the saved checkpoint file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract weights from the policy
    # Expected architecture: 3 linear layers with tanh activation
    linear_layers = []
    for name, module in policy.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(module)
    
    if len(linear_layers) != 3:
        raise ValueError(f"Expected 3 linear layers, found {len(linear_layers)}")
    
    # Verify dimensions
    expected_dims = [(146, 64), (64, 64), (64, 4)]
    for i, (layer, (in_dim, out_dim)) in enumerate(zip(linear_layers, expected_dims)):
        if layer.in_features != in_dim or layer.out_features != out_dim:
            raise ValueError(
                f"Layer {i} dimension mismatch: "
                f"expected ({in_dim}, {out_dim}), "
                f"got ({layer.in_features}, {layer.out_features})"
            )
    
    # Extract weights and biases
    layers_data = []
    for i, layer in enumerate(linear_layers):
        weights = layer.weight.detach().cpu().numpy()  # Shape: (out_dim, in_dim)
        biases = layer.bias.detach().cpu().numpy()     # Shape: (out_dim,)
        layers_data.append((weights, biases))
    
    # Optionally bake in observation normalization to layer_0
    # Normalized obs = (obs - mean) / std
    # layer_0 output = W @ normalized_obs + b
    #                = W @ ((obs - mean) / std) + b
    #                = (W / std) @ obs - (W @ mean / std) + b
    #                = W_new @ obs + b_new
    # where W_new = W / std (broadcast) and b_new = b - W @ mean / std
    if obs_mean is not None and obs_std is not None:
        print("Baking observation normalization into layer_0 weights...")
        obs_mean = np.asarray(obs_mean, dtype=np.float32)
        obs_std = np.asarray(obs_std, dtype=np.float32)
        
        # Prevent division by zero
        obs_std = np.maximum(obs_std, 1e-8)
        obs_std_inv = 1.0 / obs_std
        
        weights_0, biases_0 = layers_data[0]
        # W_new[i,j] = W[i,j] / std[j]
        weights_new = weights_0 * obs_std_inv[np.newaxis, :]
        # b_new = b - W @ mean / std = b - W_new @ mean
        biases_new = biases_0 - weights_new @ obs_mean
        
        layers_data[0] = (weights_new.astype(np.float32), biases_new.astype(np.float32))
    
    # Layer specifications matching learning-to-fly format
    layer_specs = [
        ('layer_0', 146, 64, 'Input'),
        ('layer_1', 64, 64, 'Normal'),
        ('layer_2', 64, 4, 'Output'),
    ]
    
    # Generate checkpoint content
    content = []
    
    # Header
    content.append(f'''// Auto-generated checkpoint file for Learning-to-Fly firmware
// Generated by Isaac Lab Crazyflie L2F Pipeline
// Model: {model_name}
// Date: {datetime.now().isoformat()}

#include <rl_tools/nn_models/sequential/model.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/nn/parameters/parameters.h>

namespace rl_tools::checkpoint::actor {{
''')
    
    # Generate each layer
    for (layer_name, input_dim, output_dim, group_type), (weights, biases) in zip(layer_specs, layers_data):
        content.append(generate_layer_code(
            layer_name=layer_name,
            weights=weights,
            biases=biases,
            input_dim=input_dim,
            output_dim=output_dim,
            group_type=group_type,
        ))
    
    # Model definition using Sequential Module
    content.append('''    namespace model_definition {
        using namespace rl_tools::nn_models::sequential::interface;
        using MODEL = Module<layer_0::TYPE, Module<layer_1::TYPE, Module<layer_2::TYPE>>>;
    }
    using MODEL = model_definition::MODEL;
    const MODEL model = {layer_0::layer, {layer_1::layer, {layer_2::layer}}};
}
''')
    
    # Observation namespace for testing
    # CRITICAL: Create a proper test observation that represents hover state
    # The observation at hover is:
    # - Position error: [0, 0, 0] (at target)
    # - Rotation matrix: identity [1,0,0,0,1,0,0,0,1] 
    # - Linear velocity: [0, 0, 0]
    # - Angular velocity: [0, 0, 0]
    # - Action history: all hover actions (32 * 4 = 128 values)
    
    # Build test observation array
    test_obs = np.zeros(146, dtype=np.float32)
    # Position error: already 0
    # Rotation matrix (identity): [1,0,0, 0,1,0, 0,0,1]
    test_obs[3] = 1.0   # r00
    test_obs[7] = 1.0   # r11
    test_obs[11] = 1.0  # r22
    # Linear velocity: already 0
    # Angular velocity: already 0
    # Action history: fill with hover action (~0.334)
    hover_action = 2.0 * np.sqrt(0.027 * 9.81 / (4 * 3.16e-10)) / 21702.0 - 1.0
    test_obs[18:146] = hover_action
    
    # If normalization is baked in, we need to apply it to test_obs
    # Because the network weights already include normalization,
    # but the test runs on raw observation
    # Actually, the test in firmware does: evaluate(model, test_obs, output)
    # With normalization baked in, test_obs should be raw (not normalized)
    # So we use test_obs as-is
    
    obs_bytes = float_to_bytes(test_obs)
    obs_c_array = bytes_to_c_array_inline(obs_bytes)
    
    # Compute expected action by running the policy with normalization baked in
    # Use the layers_data which already has normalization baked into layer 0
    with torch.no_grad():
        test_obs_tensor = torch.from_numpy(test_obs).unsqueeze(0).float()
        x = test_obs_tensor
        for i, (weights, biases) in enumerate(layers_data):
            weights_t = torch.from_numpy(weights).float()
            biases_t = torch.from_numpy(biases).float()
            x = x @ weights_t.T + biases_t
            x = torch.tanh(x)
        expected_action = x.squeeze(0).numpy().astype(np.float32)
    
    action_bytes = float_to_bytes(expected_action)
    action_c_array = bytes_to_c_array_inline(action_bytes)
    
    print(f"  Test observation: pos_err=[0,0,0], rot=I, vel=[0,0,0], ang_vel=[0,0,0]")
    print(f"  Test expected action: {expected_action}")

    content.append(f'''#include <rl_tools/containers.h>
namespace rl_tools::checkpoint::observation {{
    static_assert(sizeof(unsigned char) == 1);
    alignas(float) const unsigned char memory[] = {{{obs_c_array}}};
    using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned long, 1, 146, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned long, 1>>;
    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
    const CONTAINER_TYPE container = {{(float*)memory}}; 
}}

#include <rl_tools/containers.h>
namespace rl_tools::checkpoint::action {{
    static_assert(sizeof(unsigned char) == 1);
    alignas(float) const unsigned char memory[] = {{{action_c_array}}};
    using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned long, 1, 4, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned long, 1>>;
    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
    const CONTAINER_TYPE container = {{(float*)memory}}; 
}}

namespace rl_tools::checkpoint::meta {{
    char name[] = "{model_name}";
    char commit_hash[] = "isaac_lab_export";
}}
''')
    
    # Write to file
    checkpoint_content = ''.join(content)
    output_path.write_text(checkpoint_content)
    
    print(f"Exported policy to: {output_path}")
    print(f"  Model name: {model_name}")
    print(f"  Normalization baked in: {obs_mean is not None}")
    
    return str(output_path)


def export_from_checkpoint_file(
    checkpoint_path: Union[str, Path],
    output_path: Union[str, Path],
    model_name: str = "isaac_lab_policy",
    include_normalizer: bool = True,
) -> str:
    """
    Export a policy from a saved PyTorch checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pt or .pth checkpoint file
        output_path: Path to save the rl_tools checkpoint (.h file)
        model_name: Name for the model metadata
        include_normalizer: Whether to bake observation normalizer into weights
        
    Returns:
        Path to the saved checkpoint file
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # RSL-RL style checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'actor' in checkpoint:
            state_dict = checkpoint['actor']
        else:
            state_dict = checkpoint
        
        # Extract observation normalizer if present
        obs_mean = None
        obs_std = None
        if include_normalizer:
            if 'obs_mean' in checkpoint:
                obs_mean = checkpoint['obs_mean'].numpy()
                # Handle both obs_std and obs_var formats
                if 'obs_std' in checkpoint:
                    obs_std = checkpoint['obs_std'].numpy()
                elif 'obs_var' in checkpoint:
                    obs_std = np.sqrt(checkpoint['obs_var'].numpy())
            elif 'running_mean_std' in checkpoint:
                rms = checkpoint['running_mean_std']
                obs_mean = rms['mean'].numpy()
                obs_std = np.sqrt(rms['var'].numpy())
        
        # Create a dummy network and load weights
        # Handle import for both module and script execution
        try:
            from .networks import L2FActorNetwork
        except ImportError:
            from networks import L2FActorNetwork
        actor = L2FActorNetwork()
        
        # Filter actor weights from state dict
        actor_state = {}
        for k, v in state_dict.items():
            if k.startswith('actor.'):
                actor_state[k.replace('actor.', '')] = v
            elif not k.startswith('critic.'):
                # Include log_std and fc layers
                actor_state[k] = v
        
        actor.load_state_dict(actor_state, strict=False)
    else:
        # Direct model
        actor = checkpoint
        obs_mean = None
        obs_std = None
    
    return export_policy_to_firmware(
        policy=actor,
        output_path=output_path,
        model_name=model_name,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export Isaac Lab policy to Crazyflie firmware checkpoint"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to PyTorch checkpoint file (.pt or .pth)"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output path for rl_tools checkpoint (.h)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="isaac_lab_policy",
        help="Model name for metadata"
    )
    parser.add_argument(
        "--no-normalizer",
        action="store_true",
        help="Don't bake observation normalizer into weights"
    )
    
    args = parser.parse_args()
    
    export_from_checkpoint_file(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_name=args.name,
        include_normalizer=not args.no_normalizer,
    )
