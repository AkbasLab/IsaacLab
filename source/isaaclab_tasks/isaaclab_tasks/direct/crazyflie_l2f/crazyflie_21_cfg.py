# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Crazyflie 2.1 Robot Configuration for Isaac Lab

This defines our own Crazyflie 2.1 asset using a URDF with accurate
physical parameters from the Learning to Fly (L2F) project:
- Mass: 27g (0.027 kg)
- Arm length: 28mm
- Inertia: Ixx=Iyy=3.85e-6, Izz=5.9675e-6 kg·m²
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Get the path to the URDF file relative to this config file
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CRAZYFLIE_21_URDF_PATH = os.path.join(_CURRENT_DIR, "assets", "crazyflie_21.urdf")
CRAZYFLIE_21_USD_DIR = os.path.join(_CURRENT_DIR, "assets", "usd_cache", "crazyflie_21_visual_v2")

# Crazyflie 2.1 Configuration - L2F accurate parameters
CRAZYFLIE_21_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=CRAZYFLIE_21_URDF_PATH,
        usd_dir=CRAZYFLIE_21_USD_DIR,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        # URDF converter settings - floating base quadcopter
        fix_base=False,
        # Joint drive - use None for stiffness/damping since we don't drive the props
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="none",  # We don't use position/velocity targets
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            # Spin propellers to visualize rotation direction
            "m1_joint": 200.0,   # CCW
            "m2_joint": -200.0,  # CW
            "m3_joint": 200.0,   # CCW
            "m4_joint": -200.0,  # CW
        },
    ),
    actuators={
        # Dummy actuators - we apply forces directly
        "motors": ImplicitActuatorCfg(
            joint_names_expr=["m.*_joint"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Crazyflie 2.1 quadcopter with L2F-accurate parameters."""

# Also provide physics constants from L2F for use in environments
class CrazyflieL2FParams:
    """Physical parameters from Learning to Fly for Crazyflie 2.1."""
    
    # Mass
    MASS = 0.027  # kg (27g)
    
    # Rotor arm length (distance from center to motor)
    ARM_LENGTH = 0.028  # m (28mm)
    
    # Inertia tensor (diagonal, in body frame)
    IXX = 3.85e-6    # kg·m²
    IYY = 3.85e-6    # kg·m²
    IZZ = 5.9675e-6  # kg·m²
    
    # Thrust coefficient: F = k_f * rpm^2
    # L2F uses polynomial: thrust = c0 + c1*rpm + c2*rpm^2
    # with c2 = 3.16e-10 N/RPM²
    K_THRUST = 3.16e-10  # N/RPM²
    
    # Torque constant: tau_yaw = K_TORQUE * thrust
    K_TORQUE = 0.005964552
    
    # RPM limits
    RPM_MIN = 0.0
    RPM_MAX = 21702.0
    
    # Motor time constant
    TAU_MOTOR = 0.15  # seconds
    
    # Gravity
    GRAVITY = 9.81  # m/s²
    
    # Computed hover RPM (each motor)
    # At hover: 4 * k_f * rpm^2 = m * g
    # rpm = sqrt(m * g / (4 * k_f))
    @classmethod
    def hover_rpm(cls) -> float:
        import math
        return math.sqrt(cls.MASS * cls.GRAVITY / (4 * cls.K_THRUST))
    
    # Computed max thrust per motor
    @classmethod
    def max_thrust_per_motor(cls) -> float:
        return cls.K_THRUST * cls.RPM_MAX ** 2
    
    # Computed thrust-to-weight ratio
    @classmethod
    def thrust_to_weight_ratio(cls) -> float:
        max_total_thrust = 4 * cls.max_thrust_per_motor()
        weight = cls.MASS * cls.GRAVITY
        return max_total_thrust / weight
