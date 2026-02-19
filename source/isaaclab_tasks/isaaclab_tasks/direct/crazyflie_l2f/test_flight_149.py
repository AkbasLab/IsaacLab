#!/usr/bin/env python3
"""
RL navigation dead-man test.

The firmware runs the policy.
This script ONLY provides a goal trigger.

Modes:
  - hover:    Hold position at specified height (default)
  - navigate: Fly to specified (x, y, z) position
"""

import time
import signal
import sys
import threading
import argparse
import cflib.crtp
from cflib.crazyflie import Crazyflie

# ---------- Keyboard handling ----------
try:
    import msvcrt
    def key_pressed():
        return msvcrt.kbhit()
    def get_key():
        return msvcrt.getch().decode(errors="ignore")
    PLATFORM = "windows"
except ImportError:
    import select
    import tty
    import termios
    PLATFORM = "linux"
    def key_pressed():
        return select.select([sys.stdin], [], [], 0)[0]
    def get_key():
        return sys.stdin.read(1)

# ---------- Config ----------
DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"
SEND_HZ = 50
SEND_DT = 1.0 / SEND_HZ
TAKEOFF_DURATION = 3.0  # seconds to hover before navigating

# ---------- Global state ----------
_running = True
_hovering = False
_connected = False

# Navigation targets
_mode = "hover"
_goal_x = 0.0
_goal_y = 0.0
_goal_z = 0.3
_goal_yaw = 0.0

_cf = None


# ---------- Safety ----------
def signal_handler(sig, frame):
    global _running
    print("\nEMERGENCY STOP")
    _running = False
    try:
        if _cf:
            _cf.commander.send_stop_setpoint()
            _cf.close_link()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def test_rl_goal(uri: str, mode: str, x: float, y: float, z: float, yaw: float):
    global _hovering, _connected, _mode, _goal_x, _goal_y, _goal_z, _goal_yaw, _cf

    _mode = mode
    _goal_x = x
    _goal_y = y
    _goal_z = z
    _goal_yaw = yaw

    print("Initializing drivers...")
    cflib.crtp.init_drivers()

    if PLATFORM == "linux":
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

    cf = Crazyflie(rw_cache="./cache")
    _cf = cf

    connected_evt = threading.Event()

    def on_connected(uri):
        global _connected
        print("Connected.")
        _connected = True
        connected_evt.set()

        # HARD IDLE: ensure firmware does nothing on connect
        cf.commander.send_stop_setpoint()

    def on_disconnected(uri):
        print("Disconnected.")
        sys.exit(0)

    cf.connected.add_callback(on_connected)
    cf.disconnected.add_callback(on_disconnected)

    print(f"Connecting to {uri} ...")
    cf.open_link(uri)
    connected_evt.wait()

    if _mode == "hover":
        mode_info = f"HOVER at height {_goal_z}m"
    else:
        mode_info = f"NAVIGATE to ({_goal_x}, {_goal_y}, {_goal_z}) yaw={_goal_yaw}°"

    print(f"""
==================================================
 DEAD-MAN RL TEST - {_mode.upper()} MODE
--------------------------------------------------
 Target: {mode_info}
{'--------------------------------------------------' if _mode == 'navigate' else ''}
{f' Takeoff: Hover at {_goal_z}m for {TAKEOFF_DURATION}s first' if _mode == 'navigate' else ''}
--------------------------------------------------
 HOLD SPACE  → enable RL controller
 RELEASE     → motors stop
 Q / ESC     → quit
==================================================""")

    # Navigation phase tracking
    nav_phase = "takeoff"  # "takeoff" or "navigate"
    takeoff_start_time = None

    last_key_time = 0.0

    try:
        while _running and _connected:

            # ---- keyboard ----
            if key_pressed():
                k = get_key()

                if k.lower() == "q" or k == "\x1b":
                    break

                if k in (" ", "\r", "\n"):
                    if not _hovering:
                        print("RL ENABLED")
                        # Reset navigation phase on fresh enable
                        nav_phase = "takeoff"
                        takeoff_start_time = time.time()
                    _hovering = True
                    last_key_time = time.time()

            # ---- dead-man timeout ----
            if _hovering and (time.time() - last_key_time) > 0.2:
                print("RL DISABLED")
                _hovering = False
                nav_phase = "takeoff"  # Reset phase for next enable
                takeoff_start_time = None
                cf.commander.send_stop_setpoint()

            # ---- command loop ----
            if _hovering:
                if _mode == "hover":
                    # Hover mode: hold position at specified height
                    # vx, vy, yawrate MUST be zero
                    # firmware computes position internally
                    cf.commander.send_hover_setpoint(
                        0.0,    # vx
                        0.0,    # vy
                        0.0,    # yaw rate
                        _goal_z # target height
                    )
                else:
                    # Navigate mode: two-phase approach
                    # Phase 1: Takeoff - hover at target height for safety
                    # Phase 2: Navigate - fly to (x, y, z) position
                    
                    if nav_phase == "takeoff":
                        # Hover straight up to target height
                        cf.commander.send_hover_setpoint(
                            0.0,    # vx
                            0.0,    # vy
                            0.0,    # yaw rate
                            _goal_z # target height
                        )
                        
                        # Check if takeoff duration elapsed
                        elapsed = time.time() - takeoff_start_time
                        if elapsed >= TAKEOFF_DURATION:
                            print(f"TAKEOFF COMPLETE - NOW NAVIGATING to ({_goal_x}, {_goal_y}, {_goal_z})")
                            nav_phase = "navigate"
                    else:
                        # Navigate to target position
                        cf.commander.send_position_setpoint(
                            _goal_x,  # x position
                            _goal_y,  # y position
                            _goal_z,  # z position (height)
                            _goal_yaw # yaw angle in degrees
                        )

            time.sleep(SEND_DT)

    finally:
        if PLATFORM == "linux":
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        try:
            cf.commander.send_stop_setpoint()
            cf.close_link()
        except:
            pass


# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RL dead-man test for Crazyflie with hover and navigation modes"
    )
    parser.add_argument("--uri", default=DEFAULT_URI,
                        help="Crazyflie URI (default: %(default)s)")
    parser.add_argument("--mode", choices=["hover", "navigate"], default="hover",
                        help="Flight mode: 'hover' or 'navigate' (default: hover)")
    parser.add_argument("--x", type=float, default=0.0,
                        help="Target X position in meters (navigate mode)")
    parser.add_argument("--y", type=float, default=0.0,
                        help="Target Y position in meters (navigate mode)")
    parser.add_argument("--z", "--height", type=float, default=0.3, dest="z",
                        help="Target Z position/height in meters (default: 0.3)")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="Target yaw angle in degrees (navigate mode, default: 0)")
    args = parser.parse_args()

    if args.mode == "hover":
        print(f"Mode: HOVER at height {args.z}m")
    else:
        print(f"Mode: NAVIGATE to ({args.x}, {args.y}, {args.z}) yaw={args.yaw}°")

    input("Press ENTER to connect (motors will NOT spin)...")
    test_rl_goal(args.uri, args.mode, args.x, args.y, args.z, args.yaw)
