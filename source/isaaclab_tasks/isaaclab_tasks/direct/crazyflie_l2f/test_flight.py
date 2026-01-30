#!/usr/bin/env python3
"""
Real-world flight test for trained RL controllers.

Supports two modes:
  - hover: Hold position at a fixed height (dead-man's switch)
  - nav:   Navigate to random goal positions (dead-man's switch)

Dead-man's switch: Only flies while you hold SPACE.
The onboard RL controller uses IMU/Flow Deck data to generate motor commands.

Usage:
    # Hover mode (default)
    python test_flight.py --mode hover --height 0.5
    
    # Navigation mode with random goals within 0.5m
    python test_flight.py --mode nav --height 0.5 --nav_range 0.5
    
    # Scan for drones
    python test_flight.py --scan
"""

import time
import signal
import sys
import threading
import argparse
import random
import math
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Platform-specific key detection
try:
    import msvcrt  # Windows
    def key_pressed():
        return msvcrt.kbhit()
    def get_key():
        return msvcrt.getch()
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

# Default URI - change if needed
DEFAULT_URI = 'radio://0/80/2M/E7E7E7E7E7'

# Connection settings
CONNECTION_TIMEOUT = 5.0  # seconds to wait for connection
PACKET_LOSS_THRESHOLD = 10  # consecutive lost packets before disconnect

# Navigation settings
NAV_GOAL_TIMEOUT = 5.0  # seconds before sampling new goal
NAV_GOAL_THRESHOLD = 0.15  # meters - considered "reached" when this close
TAKEOFF_TIME = 2.0  # seconds to hover at origin before navigating

# Global state
_scf = None
_cf = None
_running = True
_hovering = False
_connected = False
_consecutive_failures = 0


def signal_handler(sig, frame):
    """Handle Ctrl+C immediately"""
    global _running
    print("\n\n--- EMERGENCY STOP ---")
    _running = False
    safe_disconnect()
    sys.exit(0)


def safe_disconnect():
    """Safely disconnect from the Crazyflie"""
    global _cf, _scf, _connected
    try:
        if _cf and _connected:
            try:
                _cf.commander.send_stop_setpoint()
            except:
                pass
        if _cf and _cf.link:
            try:
                _cf.close_link()
            except:
                pass
    except:
        pass
    _connected = False


signal.signal(signal.SIGINT, signal_handler)


def sample_random_goal(nav_range: float, height: float, height_range: float = 0.3):
    """
    Sample a random goal position within the navigation range.
    
    Args:
        nav_range: Maximum XY distance from origin (meters)
        height: Base hover height (meters)
        height_range: Height variation range (±meters)
    
    Returns:
        (x, y, z) goal position
    """
    # Random angle and distance for XY
    angle = random.uniform(0, 2 * math.pi)
    distance = random.uniform(0.2, nav_range)  # At least 0.2m away
    
    x = distance * math.cos(angle)
    y = distance * math.sin(angle)
    
    # Random height within range
    z = height + random.uniform(-height_range, height_range)
    z = max(0.2, min(z, height + height_range))  # Clamp to safe range
    
    return x, y, z


def test_flight_deadman(uri: str, mode: str = "hover", height: float = 0.3, 
                        nav_range: float = 0.5, nav_height_range: float = 0.2):
    """
    Dead-man's switch flight test.
    
    Args:
        uri: Crazyflie radio URI
        mode: "hover" or "nav"
        height: Target hover height (meters)
        nav_range: For nav mode, max XY distance for random goals (meters)
        nav_height_range: For nav mode, height variation (±meters)
    """
    global _scf, _cf, _running, _hovering, _connected, _consecutive_failures
    
    print(f"Initializing drivers...")
    cflib.crtp.init_drivers()
    
    print(f"Connecting to {uri} (timeout: {CONNECTION_TIMEOUT}s)...")
    
    # Set up terminal for raw input on Linux
    if PLATFORM == "linux":
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
    
    try:
        # Create Crazyflie instance with callbacks
        cf = Crazyflie(rw_cache='./cache')
        _cf = cf
        
        # Connection state tracking
        connection_complete = threading.Event()
        connection_failed = threading.Event()
        
        def on_connected(link_uri):
            global _connected
            _connected = True
            connection_complete.set()
        
        def on_connection_failed(link_uri, msg):
            print(f"\nConnection failed: {msg}")
            connection_failed.set()
        
        def on_disconnected(link_uri):
            global _connected, _running
            if _connected:
                print("\n--- DRONE DISCONNECTED ---")
            _connected = False
            _running = False
        
        def on_link_quality(quality):
            global _consecutive_failures
            if quality < 10:  # Very low link quality
                _consecutive_failures += 1
            else:
                _consecutive_failures = 0
        
        # Register callbacks
        cf.connected.add_callback(on_connected)
        cf.connection_failed.add_callback(on_connection_failed)
        cf.disconnected.add_callback(on_disconnected)
        cf.link_quality_updated.add_callback(on_link_quality)
        
        # Open link
        cf.open_link(uri)
        
        # Wait for connection with timeout
        start_time = time.time()
        while not connection_complete.is_set() and not connection_failed.is_set():
            if time.time() - start_time > CONNECTION_TIMEOUT:
                print(f"\n--- CONNECTION TIMEOUT ({CONNECTION_TIMEOUT}s) ---")
                cf.close_link()
                return
            time.sleep(0.1)
        
        if connection_failed.is_set():
            cf.close_link()
            return
        
        print("Connected!")
        print("")
        print("=" * 50)
        if mode == "hover":
            print("  HOVER MODE - DEAD-MAN'S SWITCH")
            print("=" * 50)
            print(f"  Target height: {height}m")
        else:
            print("  NAVIGATION MODE - DEAD-MAN'S SWITCH")
            print("=" * 50)
            print(f"  Base height:   {height}m (±{nav_height_range}m)")
            print(f"  Nav range:     ±{nav_range}m XY")
            print(f"  Goal timeout:  {NAV_GOAL_TIMEOUT}s")
        print("=" * 50)
        print("  HOLD SPACE = Fly")
        print("  RELEASE    = Stop motors")
        print("  N          = New random goal (nav mode)")
        print("  Q or ESC   = Quit")
        print("=" * 50)
        print("")
        
        # Unlock commander
        cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.1)
        
        last_key_time = 0
        KEY_TIMEOUT = 0.15  # Stop if no key for 150ms
        
        # Navigation state
        current_goal = (0.0, 0.0, height)  # Start at origin (takeoff position)
        goal_start_time = time.time()
        goals_reached = 0
        takeoff_complete = False
        takeoff_start_time = None
        
        if mode == "nav":
            print(f"[NAV] Will takeoff to height {height:.2f}m first, then navigate")
        
        while _running and _connected:
            # Check for packet loss / drone power off
            if _consecutive_failures >= PACKET_LOSS_THRESHOLD:
                print("\n--- LINK LOST (packet loss) ---")
                break
            
            new_goal_requested = False
            
            if key_pressed():
                key = get_key()
                
                # Decode if bytes (Windows)
                if isinstance(key, bytes):
                    key = key.decode('utf-8', errors='ignore')
                
                # Quit on Q or ESC
                if key.lower() == 'q' or key == '\x1b':
                    print("\nQuitting...")
                    break
                
                # N = new goal (nav mode)
                if key.lower() == 'n' and mode == "nav":
                    new_goal_requested = True
                
                # SPACE or ENTER = fly
                if key == ' ' or key == '\r' or key == '\n':
                    last_key_time = time.time()
                    if not _hovering:
                        if mode == "hover":
                            print("--- HOVERING... ---")
                        else:
                            print(f"--- FLYING to ({current_goal[0]:.2f}, {current_goal[1]:.2f}, {current_goal[2]:.2f}) ---")
                        _hovering = True
            
            # Check if key is still being "held" (received recently)
            if _hovering and (time.time() - last_key_time) > KEY_TIMEOUT:
                print("--- STOPPED ---")
                _hovering = False
                # Reset takeoff state so we start fresh if user resumes
                takeoff_complete = False
                takeoff_start_time = None
                current_goal = (0.0, 0.0, height)  # Reset to origin
                try:
                    cf.commander.send_stop_setpoint()
                except:
                    break
            
            # Navigation mode: handle takeoff phase first, then goals
            if mode == "nav" and _hovering:
                # Start takeoff timer when we first start hovering
                if takeoff_start_time is None:
                    takeoff_start_time = time.time()
                    print(f"[NAV] Taking off to {height:.2f}m...")
                
                # Check if takeoff is complete
                if not takeoff_complete:
                    takeoff_elapsed = time.time() - takeoff_start_time
                    if takeoff_elapsed >= TAKEOFF_TIME:
                        takeoff_complete = True
                        # Now sample the first navigation goal
                        current_goal = sample_random_goal(nav_range, height, nav_height_range)
                        goal_start_time = time.time()
                        print(f"[NAV] Takeoff complete! First goal: ({current_goal[0]:.2f}, {current_goal[1]:.2f}, {current_goal[2]:.2f})")
                else:
                    # Normal navigation - check for goal timeout
                    elapsed = time.time() - goal_start_time
                    if elapsed > NAV_GOAL_TIMEOUT or new_goal_requested:
                        goals_reached += 1
                        current_goal = sample_random_goal(nav_range, height, nav_height_range)
                        goal_start_time = time.time()
                        reason = "timeout" if not new_goal_requested else "manual"
                        print(f"[NAV] Goal #{goals_reached} ({reason}): ({current_goal[0]:.2f}, {current_goal[1]:.2f}, {current_goal[2]:.2f})")
            
            # Send commands based on state and mode
            if _hovering:
                try:
                    if mode == "hover":
                        # Simple hover at fixed position
                        cf.commander.send_hover_setpoint(0, 0, 0, height)
                    elif mode == "nav" and not takeoff_complete:
                        # During takeoff: rise straight up using hover command
                        cf.commander.send_hover_setpoint(0, 0, 0, height)
                    else:
                        # Navigate to goal position (after takeoff is complete)
                        # send_position_setpoint(x, y, z, yaw) - absolute position
                        cf.commander.send_position_setpoint(
                            current_goal[0], 
                            current_goal[1], 
                            current_goal[2], 
                            0.0  # yaw = 0
                        )
                except Exception as e:
                    print(f"\n--- SEND FAILED: {e} ---")
                    break
            
            time.sleep(0.02)  # 50Hz loop
        
        # Final stop
        try:
            cf.commander.send_stop_setpoint()
        except:
            pass
        
        if mode == "nav":
            print(f"\n--- Test ended: {goals_reached} goals reached ---")
        else:
            print("--- Test ended safely ---")
        
        # Close link gracefully
        cf.close_link()
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore terminal on Linux
        if PLATFORM == "linux":
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        safe_disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dead-man switch flight test for trained RL controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  hover  - Hold position at fixed height (default)
  nav    - Navigate to random goal positions

Examples:
  # Hover at 0.5m height
  python test_flight.py --mode hover --height 0.5
  
  # Navigate with random goals within 0.3m
  python test_flight.py --mode nav --height 0.5 --nav_range 0.3
  
  # Larger navigation range (0.8m) with more height variation
  python test_flight.py --mode nav --nav_range 0.8 --nav_height_range 0.3
"""
    )
    
    parser.add_argument('--uri', default=DEFAULT_URI, 
                        help=f'Crazyflie URI (default: {DEFAULT_URI})')
    parser.add_argument('--mode', choices=['hover', 'nav'], default='hover',
                        help='Flight mode: hover or nav (default: hover)')
    parser.add_argument('--height', type=float, default=0.5, 
                        help='Base hover/flight height in meters (default: 0.5)')
    parser.add_argument('--nav_range', type=float, default=0.5,
                        help='Navigation mode: max XY distance for random goals in meters (default: 0.5)')
    parser.add_argument('--nav_height_range', type=float, default=0.2,
                        help='Navigation mode: height variation ± meters (default: 0.2)')
    parser.add_argument('--scan', action='store_true', 
                        help='Scan for Crazyflies instead of testing')
    
    args = parser.parse_args()
    
    if args.scan:
        print("Scanning for Crazyflies...")
        cflib.crtp.init_drivers()
        available = cflib.crtp.scan_interfaces()
        if available:
            print("Found Crazyflies:")
            for i, uri in enumerate(available):
                print(f"  [{i}] {uri[0]}")
            print(f"\nTo test hover, run:")
            print(f"  python test_flight.py --uri {available[0][0]} --mode hover")
            print(f"\nTo test navigation, run:")
            print(f"  python test_flight.py --uri {available[0][0]} --mode nav --nav_range 0.5")
        else:
            print("No Crazyflies found.")
            print("\nTroubleshooting:")
            print("  - Is the Crazyradio PA plugged in?")
            print("  - Is the Crazyflie powered on?")
            print("  - Try a different radio channel")
    else:
        print("=" * 60)
        if args.mode == "hover":
            print("  RL CONTROLLER TEST - HOVER MODE")
        else:
            print("  RL CONTROLLER TEST - NAVIGATION MODE")
        print("=" * 60)
        print(f"  URI:            {args.uri}")
        print(f"  Mode:           {args.mode}")
        print(f"  Height:         {args.height}m")
        if args.mode == "nav":
            print(f"  Nav range:      ±{args.nav_range}m XY")
            print(f"  Height range:   ±{args.nav_height_range}m")
        print("=" * 60)
        print()
        print("--- SAFETY WARNING ---")
        print("  The onboard RL controller will generate motor commands")
        print("  based on IMU/Flow Deck sensor data to reach the target.")
        print()
        print("  HOLD SPACE   = Fly (sends position commands)")
        print("  RELEASE      = Immediately stop motors")
        print("  N            = Sample new random goal (nav mode)")
        print("  Q or ESC     = Quit")
        print()
        
        input("Press ENTER to connect and begin...")
        test_flight_deadman(
            args.uri, 
            mode=args.mode,
            height=args.height,
            nav_range=args.nav_range,
            nav_height_range=args.nav_height_range
        )
