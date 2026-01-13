#!/usr/bin/env python3
"""
Automated Crazyflie firmware integration script
Copies RL controller files and modifies build system
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def check_prerequisites(firmware_path: Path) -> bool:
    """Verify firmware directory structure"""
    required_dirs = [
        "src/modules/src",
        "src/utils/interface",
        "vendor"
    ]
    
    for dir_path in required_dirs:
        if not (firmware_path / dir_path).exists():
            print(f"❌ Missing directory: {dir_path}")
            return False
    
    print("✓ Firmware directory structure valid")
    return True


def copy_controller_files(isaac_path: Path, firmware_path: Path):
    """Copy controller files to firmware"""
    src_dir = isaac_path / "crazyflie_deploy"
    dst_dir = firmware_path / "src/modules/src"
    
    files_to_copy = [
        "controller_rl.c",
        "controller_rl.h",
        "policy_int8.c",
        "policy_int8.h",
        "policy_int8_weights.c"
    ]
    
    for filename in files_to_copy:
        src = src_dir / filename
        dst = dst_dir / filename
        
        if not src.exists():
            print(f"❌ Source file not found: {src}")
            continue
        
        shutil.copy2(src, dst)
        print(f"✓ Copied {filename}")


def patch_makefile(firmware_path: Path):
    """Add controller objects to Makefile"""
    makefile = firmware_path / "src/modules/src/Makefile"
    
    if not makefile.exists():
        print("❌ Makefile not found")
        return False
    
    with open(makefile, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "controller_rl.o" in content:
        print("✓ Makefile already patched")
        return True
    
    # Find PROJ_OBJ section and add our objects
    lines = content.split('\n')
    proj_obj_idx = -1
    
    for i, line in enumerate(lines):
        if 'PROJ_OBJ' in line and not line.strip().startswith('#'):
            proj_obj_idx = i
            break
    
    if proj_obj_idx == -1:
        print("❌ Could not find PROJ_OBJ in Makefile")
        return False
    
    # Insert after last PROJ_OBJ line
    insert_idx = proj_obj_idx
    while insert_idx < len(lines) and (lines[insert_idx].strip().endswith('\\') or 'PROJ_OBJ' in lines[insert_idx]):
        insert_idx += 1
    
    lines.insert(insert_idx, "PROJ_OBJ += controller_rl.o")
    lines.insert(insert_idx + 1, "PROJ_OBJ += policy_int8.o")
    lines.insert(insert_idx + 2, "PROJ_OBJ += policy_int8_weights.o")
    
    with open(makefile, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✓ Patched Makefile")
    return True


def patch_controller_header(firmware_path: Path):
    """Add RL controller type to controller.h"""
    header = firmware_path / "src/utils/interface/controller.h"
    
    if not header.exists():
        print("❌ controller.h not found")
        return False
    
    with open(header, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "ControllerTypeRL" in content:
        print("✓ controller.h already patched")
        return True
    
    # Find ControllerType enum
    lines = content.split('\n')
    enum_end = -1
    
    for i, line in enumerate(lines):
        if 'ControllerTypeAny' in line:
            enum_end = i
            break
    
    if enum_end == -1:
        print("❌ Could not find ControllerType enum")
        return False
    
    # Insert ControllerTypeRL before ControllerTypeAny
    lines.insert(enum_end, "  ControllerTypeRL,")
    
    with open(header, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✓ Patched controller.h")
    return True


def patch_controller_impl(firmware_path: Path):
    """Add RL controller registration to controller.c"""
    impl = firmware_path / "src/modules/src/controller.c"
    
    if not impl.exists():
        print("❌ controller.c not found")
        return False
    
    with open(impl, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "controllerRLInit" in content:
        print("✓ controller.c already patched")
        return True
    
    lines = content.split('\n')
    
    # Add include at top
    include_idx = -1
    for i, line in enumerate(lines):
        if '#include "controller' in line:
            include_idx = i
    
    if include_idx != -1:
        lines.insert(include_idx + 1, '#include "controller_rl.h"')
    
    # Add init call
    for i, line in enumerate(lines):
        if 'controllerMellingerInit' in line:
            lines.insert(i + 1, "  controllerRLInit();")
            break
    
    # Add controller dispatch
    for i, line in enumerate(lines):
        if 'void controller(' in line:
            # Find the if-else chain
            for j in range(i, min(i + 30, len(lines))):
                if 'controllerType == ControllerTypeMellinger' in lines[j]:
                    lines.insert(j, "  if (controllerType == ControllerTypeRL) {")
                    lines.insert(j + 1, "    controllerRL(control, setpoint, sensors, state, tick);")
                    lines.insert(j + 2, "  } else if (controllerType == ControllerTypeMellinger) {")
                    # Remove old "if" and replace with "} else if"
                    lines[j + 3] = lines[j + 3].replace("if", "} else if", 1)
                    break
            break
    
    with open(impl, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✓ Patched controller.c")
    return True


def setup_cmsis_nn(firmware_path: Path):
    """Clone and setup CMSIS-NN library"""
    cmsis_path = firmware_path / "vendor/CMSIS-NN"
    
    if cmsis_path.exists():
        print("✓ CMSIS-NN already exists")
        return True
    
    print("Cloning CMSIS-NN...")
    try:
        subprocess.run([
            "git", "clone",
            "https://github.com/ARM-software/CMSIS-NN.git",
            str(cmsis_path)
        ], check=True, cwd=firmware_path / "vendor")
        
        subprocess.run([
            "git", "checkout", "v4.0.0"
        ], check=True, cwd=cmsis_path)
        
        print("✓ CMSIS-NN cloned")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to clone CMSIS-NN")
        return False


def patch_root_makefile(firmware_path: Path):
    """Add CMSIS-NN includes to root Makefile"""
    makefile = firmware_path / "Makefile"
    
    if not makefile.exists():
        print("❌ Root Makefile not found")
        return False
    
    with open(makefile, 'r') as f:
        content = f.read()
    
    if "CMSIS-NN" in content:
        print("✓ Root Makefile already patched")
        return True
    
    lines = content.split('\n')
    
    # Find INCLUDES section
    for i, line in enumerate(lines):
        if 'INCLUDES +=' in line:
            lines.insert(i + 1, "INCLUDES += -Ivendor/CMSIS-NN/Include")
            break
    
    # Find CFLAGS section
    for i, line in enumerate(lines):
        if 'CFLAGS +=' in line and '-mfpu' not in line:
            lines.insert(i + 1, "CFLAGS += -DARM_MATH_CM4 -D__FPU_PRESENT=1")
            break
    
    with open(makefile, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✓ Patched root Makefile")
    return True


def main():
    parser = argparse.ArgumentParser(description="Integrate RL controller into Crazyflie firmware")
    parser.add_argument("--isaac-path", type=Path, required=True, help="Path to IsaacLab root")
    parser.add_argument("--firmware-path", type=Path, required=True, help="Path to crazyflie-firmware")
    parser.add_argument("--skip-cmsis", action="store_true", help="Skip CMSIS-NN setup")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Crazyflie RL Controller Integration")
    print("=" * 80)
    
    # Verify paths
    if not args.isaac_path.exists():
        print(f"❌ IsaacLab path not found: {args.isaac_path}")
        return 1
    
    if not args.firmware_path.exists():
        print(f"❌ Firmware path not found: {args.firmware_path}")
        return 1
    
    # Check prerequisites
    if not check_prerequisites(args.firmware_path):
        return 1
    
    # Copy files
    print("\n1. Copying controller files...")
    copy_controller_files(args.isaac_path, args.firmware_path)
    
    # Patch build system
    print("\n2. Patching build system...")
    if not patch_makefile(args.firmware_path):
        return 1
    
    # Patch controller registration
    print("\n3. Registering controller...")
    if not patch_controller_header(args.firmware_path):
        return 1
    
    if not patch_controller_impl(args.firmware_path):
        return 1
    
    # Setup CMSIS-NN
    if not args.skip_cmsis:
        print("\n4. Setting up CMSIS-NN...")
        if not setup_cmsis_nn(args.firmware_path):
            return 1
        
        if not patch_root_makefile(args.firmware_path):
            return 1
    
    print("\n" + "=" * 80)
    print("✓ Integration complete!")
    print("=" * 80)
    print("\nNext steps:")
    print(f"  cd {args.firmware_path}")
    print("  make clean")
    print("  make -j8")
    print("  make cload  # or 'make flash' via USB")
    print("\nFor testing, see DEPLOYMENT_GUIDE.md")
    
    return 0


if __name__ == "__main__":
    exit(main())
