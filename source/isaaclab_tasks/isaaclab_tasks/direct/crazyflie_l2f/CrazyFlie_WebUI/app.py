#!/usr/bin/env python3
"""
Crazyflie Web UI Controller (FastAPI + WebSocket)

Features:
- Serves ui.html
- WebSocket control channel: connect / disconnect / stop / deadman / hover / navigate / plan
- Telemetry stream:
    stateEstimate (x,y,z), stabilizer (roll,pitch,yaw), acc (x,y,z), gyro (x,y,z), pm.vbat
- Telemetry broadcast via async "telemetry pump"
- Plan follower with LOOP option
- Dead-man: if client stops sending keep-alives, motors stop
- Anti-lunge plan start:
    * Hold current position ONCE at plan start
    * Loop restarts do NOT return to initial hold point
- Server log lines broadcast to UI (type="log")

Now also attempts to log (best-effort; depends on your CF firmware/log TOC):
- Per-motor outputs (m1..m4) via one of: motor.m*, motorPower.m*, pwm.m*
- "IMU sensors" raw via one of: imu.acc_*/imu.gyro_*/imu.mag_* (or similar)
- Magnetometer (mag.x/y/z) if available

Requirements:
  pip install fastapi uvicorn "uvicorn[standard]" cflib

Run:
  py -m uvicorn app:app --host 0.0.0.0 --port 8000

Files:
  - ui.html next to app.py
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, Response

# ----------------------------
# Config
# ----------------------------
DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"

SEND_HZ = 50.0
SEND_DT = 1.0 / SEND_HZ

DEADMAN_TIMEOUT_S = 0.25  # if no deadman keepalive within this, stop

# Hold current pose ONCE before starting a plan (anti-lunge)
PLAN_HOLD_BEFORE_S = 1.0
PLAN_HOLD_USE_PLAN_Z = True

# Crazyflie log update rate
LOG_PERIOD_MS = 10  # ~100 Hz-ish (depends on radio conditions)

# Telemetry broadcast rate (server -> UI)
TELEM_PUMP_HZ = 20.0
TELEM_PUMP_DT = 1.0 / TELEM_PUMP_HZ

HERE = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(HERE, "ui.html")
FLIGHTPLAN_DIR = os.path.join(HERE, "flightplans")

# ----------------------------
# State / Data Structures
# ----------------------------
@dataclass
class Telemetry:
    # Position
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None

    # Velocity
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None

    # Attitude
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None

    # Accel (filtered)
    ax: Optional[float] = None
    ay: Optional[float] = None
    az: Optional[float] = None

    # Gyro (filtered)
    wx: Optional[float] = None
    wy: Optional[float] = None
    wz: Optional[float] = None

    # Battery
    battery: Optional[float] = None

    last_update_s: float = 0.0


@dataclass
class Target:
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    yaw: Optional[float] = None


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float
    t: float  # seconds from plan start


@dataclass
class ControlState:
    connected: bool = False
    uri: str = DEFAULT_URI

    # Modes
    mode: str = "idle"  # "idle" | "hover" | "navigate" | "plan"
    phase: str = "idle"

    # Deadman
    deadman_enabled: bool = False
    deadman_last_seen: float = 0.0

    # Simple modes
    hover_z: float = 0.30
    nav_x: float = 0.0
    nav_y: float = 0.0
    nav_z: float = 0.30
    nav_yaw: float = 0.0

    # Plan
    plan: List[Waypoint] = field(default_factory=list)
    plan_loop: bool = True
    plan_active: bool = False

    # Plan timing & initial hold (ONE-SHOT)
    plan_started_at: float = 0.0
    plan_hold_until: float = 0.0
    plan_hold_target: Optional[Waypoint] = None

    # Latest values
    telemetry: Telemetry = field(default_factory=Telemetry)
    target: Target = field(default_factory=Target)

    # Internal / objects
    cf: Optional[Crazyflie] = None
    logcfgs: List[LogConfig] = field(default_factory=list)
    send_thread: Optional[threading.Thread] = None
    send_thread_stop: threading.Event = field(default_factory=threading.Event)

    # Latest raw log values (only for vars we configure)
    latest_log: Dict[str, float] = field(default_factory=dict)
    latest_lock: threading.Lock = field(default_factory=threading.Lock)

    # Human-readable notes about optional logs that actually started
    log_notes: List[str] = field(default_factory=list)


def _clear_telemetry_state() -> None:
    STATE.telemetry = Telemetry()
    STATE.target = Target()
    with STATE.latest_lock:
        STATE.latest_log.clear()
    STATE.log_notes = []


def _mark_link_down(reason: str = "") -> None:
    STATE.connected = False
    STATE.cf = None
    STATE.mode = "idle"
    STATE.phase = "idle"
    STATE.plan_active = False
    STATE.deadman_enabled = False
    STATE.plan_started_at = 0.0
    STATE.plan_hold_until = 0.0
    STATE.plan_hold_target = None
    _clear_telemetry_state()


STATE = ControlState()

# WebSocket clients (support multiple dashboards)
CLIENTS: set[WebSocket] = set()
CLIENTS_LOCK = asyncio.Lock()

# Telemetry pump task handle
TELEM_TASK: Optional[asyncio.Task] = None

# ----------------------------
# Utilities
# ----------------------------
def now_s() -> float:
    return time.time()


def ts_iso() -> str:
    return time.strftime("%H:%M:%S")


async def broadcast(msg: Dict[str, Any]) -> None:
    data = json.dumps(msg)
    async with CLIENTS_LOCK:
        dead: List[WebSocket] = []
        for ws in CLIENTS:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            CLIENTS.discard(ws)


async def ws_log(text: str, level: str = "info") -> None:
    await broadcast(
        {
            "type": "log",
            "t": now_s(),
            "ts": ts_iso(),
            "level": level,
            "text": text,
        }
    )


async def ws_status(msg: str = "") -> None:
    await broadcast(
        {
            "type": "status",
            "connected": STATE.connected,
            "uri": STATE.uri,
            "msg": msg,
        }
    )
    if msg:
        lvl = "info"
        if "error" in msg.lower() or "failed" in msg.lower():
            lvl = "error"
        await ws_log(f"STATUS: {msg}", lvl)


def _get_first_present(latest: Dict[str, float], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in latest:
            try:
                return float(latest[k])
            except Exception:
                return None
    return None


def _canonical_raw(latest: Dict[str, float]) -> Dict[str, Optional[float]]:
    """
    Build a stable 'raw' dict with canonical key names, even if your CF firmware
    uses different variable names. This is what the UI logs to CSV.
    """
    raw: Dict[str, Optional[float]] = {}

    # --- Keep flight_logger.py canonical fields ---
    raw["stateEstimate.x"] = _get_first_present(latest, ["stateEstimate.x"])
    raw["stateEstimate.y"] = _get_first_present(latest, ["stateEstimate.y"])
    raw["stateEstimate.z"] = _get_first_present(latest, ["stateEstimate.z"])

    raw["acc.x"] = _get_first_present(latest, ["acc.x"])
    raw["acc.y"] = _get_first_present(latest, ["acc.y"])
    raw["acc.z"] = _get_first_present(latest, ["acc.z"])

    raw["gyro.x"] = _get_first_present(latest, ["gyro.x"])
    raw["gyro.y"] = _get_first_present(latest, ["gyro.y"])
    raw["gyro.z"] = _get_first_present(latest, ["gyro.z"])

    raw["stabilizer.roll"] = _get_first_present(latest, ["stabilizer.roll"])
    raw["stabilizer.pitch"] = _get_first_present(latest, ["stabilizer.pitch"])
    raw["stabilizer.yaw"] = _get_first_present(latest, ["stabilizer.yaw"])

    # Battery (use canonical CF name)
    raw["pm.vbat"] = _get_first_present(latest, ["pm.vbat"])

    # --- Per-motor outputs (canonical names: motor.m1..motor.m4) ---
    motor_candidates = [
        ("motor.m1", ["motor.m1", "motorPower.m1", "pwm.m1"]),
        ("motor.m2", ["motor.m2", "motorPower.m2", "pwm.m2"]),
        ("motor.m3", ["motor.m3", "motorPower.m3", "pwm.m3"]),
        ("motor.m4", ["motor.m4", "motorPower.m4", "pwm.m4"]),
    ]
    for canon, cands in motor_candidates:
        raw[canon] = _get_first_present(latest, cands)

    # --- IMU raw (canonical: imu.acc_*, imu.gyro_*, imu.mag_*) ---
    imu_acc = [
        ("imu.acc_x", ["imu.acc_x", "imu.accX", "imu.acc.x", "imu.accx", "imu.accRaw.x"]),
        ("imu.acc_y", ["imu.acc_y", "imu.accY", "imu.acc.y", "imu.accy", "imu.accRaw.y"]),
        ("imu.acc_z", ["imu.acc_z", "imu.accZ", "imu.acc.z", "imu.accz", "imu.accRaw.z"]),
    ]
    imu_gyro = [
        ("imu.gyro_x", ["imu.gyro_x", "imu.gyroX", "imu.gyro.x", "imu.gyrox", "imu.gyroRaw.x"]),
        ("imu.gyro_y", ["imu.gyro_y", "imu.gyroY", "imu.gyro.y", "imu.gyroy", "imu.gyroRaw.y"]),
        ("imu.gyro_z", ["imu.gyro_z", "imu.gyroZ", "imu.gyro.z", "imu.gyroz", "imu.gyroRaw.z"]),
    ]
    imu_mag = [
        ("imu.mag_x", ["imu.mag_x", "imu.magX", "imu.mag.x", "imu.magx", "mag.x"]),
        ("imu.mag_y", ["imu.mag_y", "imu.magY", "imu.mag.y", "imu.magy", "mag.y"]),
        ("imu.mag_z", ["imu.mag_z", "imu.magZ", "imu.mag.z", "imu.magz", "mag.z"]),
    ]
    for canon, cands in (imu_acc + imu_gyro + imu_mag):
        raw[canon] = _get_first_present(latest, cands)

    # --- Velocity (stateEstimate.vx/vy/vz if available) ---
    raw["velocity.x"] = _get_first_present(latest, ["stateEstimate.vx"])
    raw["velocity.y"] = _get_first_present(latest, ["stateEstimate.vy"])
    raw["velocity.z"] = _get_first_present(latest, ["stateEstimate.vz"])

    return raw


async def ws_telemetry() -> None:
    a = STATE.telemetry
    t = STATE.target

    with STATE.latest_lock:
        latest = dict(STATE.latest_log)

    raw = _canonical_raw(latest)

    await broadcast(
        {
            "type": "telemetry",
            "t": now_s(),
            "mode": STATE.mode,
            "phase": STATE.phase,
            "target": {"x": t.x, "y": t.y, "z": t.z, "yaw": t.yaw},
            "actual": {
                "x": a.x,
                "y": a.y,
                "z": a.z,
                "yaw": a.yaw,
                "battery": a.battery,
                "roll": a.roll,
                "pitch": a.pitch,
                "ax": a.ax,
                "ay": a.ay,
                "az": a.az,
                "wx": a.wx,
                "wy": a.wy,
                "wz": a.wz,
                "vx": a.vx,
                "vy": a.vy,
                "vz": a.vz,
            },
            # IMPORTANT: UI CSV logger prefers msg.raw with canonical CF names
            "raw": raw,
        }
    )


async def telemetry_pump() -> None:
    while True:
        try:
            async with CLIENTS_LOCK:
                has_clients = len(CLIENTS) > 0
            if has_clients:
                await ws_telemetry()
            await asyncio.sleep(TELEM_PUMP_DT)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.2)


def _safe_send_stop(cf: Crazyflie) -> None:
    try:
        cf.commander.send_stop_setpoint()
    except Exception:
        pass


def _parse_waypoints(raw: Any) -> List[Waypoint]:
    out: List[Waypoint] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        try:
            out.append(
                Waypoint(
                    x=float(item["x"]),
                    y=float(item["y"]),
                    z=float(item["z"]),
                    yaw=float(item.get("yaw", 0.0)),
                    t=float(item["t"]),
                )
            )
        except Exception:
            continue
    out.sort(key=lambda w: w.t)
    return out


def _plan_sample_at(plan: List[Waypoint], t_now: float) -> Waypoint:
    if not plan:
        raise ValueError("empty plan")

    if t_now <= plan[0].t:
        return plan[0]
    if t_now >= plan[-1].t:
        return plan[-1]

    for i in range(len(plan) - 1):
        a = plan[i]
        b = plan[i + 1]
        if a.t <= t_now <= b.t:
            dt = max(1e-6, b.t - a.t)
            u = (t_now - a.t) / dt
            x = a.x + (b.x - a.x) * u
            y = a.y + (b.y - a.y) * u
            z = a.z + (b.z - a.z) * u
            yaw = a.yaw + (b.yaw - a.yaw) * u
            return Waypoint(x=x, y=y, z=z, yaw=yaw, t=t_now)

    return plan[-1]


# ----------------------------
# Crazyflie connection & telemetry
# ----------------------------
def _stop_logcfgs(cf: Crazyflie) -> None:
    for lc in list(STATE.logcfgs):
        try:
            lc.stop()
        except Exception:
            pass
        try:
            cf.log.remove_config(lc)
        except Exception:
            pass
    STATE.logcfgs = []


def _try_start_logcfg(cf: Crazyflie, lc: LogConfig, note: str) -> bool:
    """
    Best-effort start: if firmware doesn't have any variable, this can fail.
    We skip that config without killing the whole connection.
    """
    try:
        cf.log.add_config(lc)
        lc.start()
        STATE.logcfgs.append(lc)
        STATE.log_notes.append(note)
        return True
    except Exception:
        try:
            cf.log.remove_config(lc)
        except Exception:
            pass
        return False


def _setup_logging(cf: Crazyflie) -> None:
    """
    Configure Crazyflie logging. Split configs to avoid packet size limits.

    Always logs the "flight_logger.py" set:
      stateEstimate.*, stabilizer.roll/pitch/yaw, acc.*, gyro.*, pm.vbat

    Then tries to log:
      motor outputs (m1..m4)
      IMU raw sensors (imu.*)
      mag.* (if present)
    """
    _stop_logcfgs(cf)
    STATE.log_notes = []

    def _log_cb(_timestamp, data, _logconf):
        # Runs in CF logging thread; do NOT call asyncio here.
        try:
            # Store raw snapshot
            with STATE.latest_lock:
                for k, v in data.items():
                    try:
                        STATE.latest_log[k] = float(v)
                    except Exception:
                        pass

            # Update "friendly" telemetry fields
            if "stateEstimate.x" in data:
                STATE.telemetry.x = float(data["stateEstimate.x"])
            if "stateEstimate.y" in data:
                STATE.telemetry.y = float(data["stateEstimate.y"])
            if "stateEstimate.z" in data:
                STATE.telemetry.z = float(data["stateEstimate.z"])

            if "stabilizer.roll" in data:
                STATE.telemetry.roll = float(data["stabilizer.roll"])
            if "stabilizer.pitch" in data:
                STATE.telemetry.pitch = float(data["stabilizer.pitch"])
            if "stabilizer.yaw" in data:
                STATE.telemetry.yaw = float(data["stabilizer.yaw"])

            if "acc.x" in data:
                STATE.telemetry.ax = float(data["acc.x"])
            if "acc.y" in data:
                STATE.telemetry.ay = float(data["acc.y"])
            if "acc.z" in data:
                STATE.telemetry.az = float(data["acc.z"])

            if "gyro.x" in data:
                STATE.telemetry.wx = float(data["gyro.x"])
            if "gyro.y" in data:
                STATE.telemetry.wy = float(data["gyro.y"])
            if "gyro.z" in data:
                STATE.telemetry.wz = float(data["gyro.z"])

            if "pm.vbat" in data:
                STATE.telemetry.battery = float(data["pm.vbat"])

            # Velocity (optional - depends on firmware TOC)
            if "stateEstimate.vx" in data:
                STATE.telemetry.vx = float(data["stateEstimate.vx"])
            if "stateEstimate.vy" in data:
                STATE.telemetry.vy = float(data["stateEstimate.vy"])
            if "stateEstimate.vz" in data:
                STATE.telemetry.vz = float(data["stateEstimate.vz"])

            STATE.telemetry.last_update_s = now_s()
        except Exception:
            pass

    # --- Required / base logs (match flight_logger.py) ---
    lc_pos_att = LogConfig(name="pos_att", period_in_ms=LOG_PERIOD_MS)
    for v in [
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.z",
        "stabilizer.roll",
        "stabilizer.pitch",
        "stabilizer.yaw",
    ]:
        lc_pos_att.add_variable(v, "float")
    lc_pos_att.data_received_cb.add_callback(_log_cb)

    lc_acc = LogConfig(name="acc", period_in_ms=LOG_PERIOD_MS)
    for v in ["acc.x", "acc.y", "acc.z"]:
        lc_acc.add_variable(v, "float")
    lc_acc.data_received_cb.add_callback(_log_cb)

    lc_gyro = LogConfig(name="gyro", period_in_ms=LOG_PERIOD_MS)
    for v in ["gyro.x", "gyro.y", "gyro.z"]:
        lc_gyro.add_variable(v, "float")
    lc_gyro.data_received_cb.add_callback(_log_cb)

    lc_bat = LogConfig(name="bat", period_in_ms=LOG_PERIOD_MS)
    lc_bat.add_variable("pm.vbat", "float")
    lc_bat.data_received_cb.add_callback(_log_cb)

    for lc, note in [
        (lc_pos_att, "logging: pos+attitude"),
        (lc_acc, "logging: acc"),
        (lc_gyro, "logging: gyro"),
        (lc_bat, "logging: battery"),
    ]:
        if not _try_start_logcfg(cf, lc, note):
            # If base logging fails, something is very wrong; raise.
            raise RuntimeError(f"Failed to start required log config: {lc.name}")

    # --- Optional: per-motor outputs (try multiple common name sets) ---
    motor_sets: List[Tuple[str, List[str]]] = [
        ("motor.m1..m4", ["motor.m1", "motor.m2", "motor.m3", "motor.m4"]),
        ("motorPower.m1..m4", ["motorPower.m1", "motorPower.m2", "motorPower.m3", "motorPower.m4"]),
        ("pwm.m1..m4", ["pwm.m1", "pwm.m2", "pwm.m3", "pwm.m4"]),
    ]
    for label, vars_ in motor_sets:
        lc = LogConfig(name=f"motors_{label}", period_in_ms=LOG_PERIOD_MS)
        for v in vars_:
            lc.add_variable(v, "float")
        lc.data_received_cb.add_callback(_log_cb)
        if _try_start_logcfg(cf, lc, f"logging: motors ({label})"):
            break  # keep the first that works

    # --- Optional: IMU raw sensors (best-effort) ---
    imu_sets: List[Tuple[str, List[str]]] = [
        ("imu.acc_* + imu.gyro_* + imu.mag_*",
         ["imu.acc_x", "imu.acc_y", "imu.acc_z",
          "imu.gyro_x", "imu.gyro_y", "imu.gyro_z",
          "imu.mag_x", "imu.mag_y", "imu.mag_z"]),
        ("imu.accX/accY/accZ + imu.gyroX/gyroY/gyroZ + imu.magX/magY/magZ",
         ["imu.accX", "imu.accY", "imu.accZ",
          "imu.gyroX", "imu.gyroY", "imu.gyroZ",
          "imu.magX", "imu.magY", "imu.magZ"]),
        ("mag.x/y/z",
         ["mag.x", "mag.y", "mag.z"]),
    ]
    for label, vars_ in imu_sets:
        lc = LogConfig(name=f"imu_{label}", period_in_ms=LOG_PERIOD_MS)
        for v in vars_:
            lc.add_variable(v, "float")
        lc.data_received_cb.add_callback(_log_cb)
        if _try_start_logcfg(cf, lc, f"logging: imu ({label})"):
            break

    # --- Optional: velocity (stateEstimate.vx/vy/vz) ---
    lc_vel = LogConfig(name="velocity", period_in_ms=LOG_PERIOD_MS)
    for v in ["stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"]:
        lc_vel.add_variable(v, "float")
    lc_vel.data_received_cb.add_callback(_log_cb)
    _try_start_logcfg(cf, lc_vel, "logging: velocity (stateEstimate.vx/vy/vz)")


def _connect_blocking(uri: str) -> None:
    cflib.crtp.init_drivers(enable_debug_driver=False)

    cf = Crazyflie(rw_cache="./cache")
    evt = threading.Event()
    err: Dict[str, str] = {"msg": ""}

    def _on_connected(_uri):
        evt.set()

    def _on_failed(_uri, msg):
        err["msg"] = f"connect failed: {msg}"
        evt.set()

    def _on_lost(_uri, msg):
        err["msg"] = f"link lost: {msg}"
        _mark_link_down(err["msg"])

    def _on_disconnected(_uri):
        _mark_link_down("disconnected")

    cf.connected.add_callback(_on_connected)
    cf.connection_failed.add_callback(_on_failed)
    cf.connection_lost.add_callback(_on_lost)
    try:
        cf.disconnected.add_callback(_on_disconnected)
    except Exception:
        pass

    cf.open_link(uri)

    if not evt.wait(timeout=10.0):
        try:
            cf.close_link()
        except Exception:
            pass
        raise RuntimeError("connect timeout")

    if err["msg"]:
        try:
            cf.close_link()
        except Exception:
            pass
        raise RuntimeError(err["msg"])

    _safe_send_stop(cf)

    # Clear previous runtime state
    _clear_telemetry_state()

    _setup_logging(cf)

    STATE.cf = cf
    STATE.connected = True
    STATE.uri = uri


def _disconnect_blocking() -> None:
    cf = STATE.cf
    if not cf:
        return

    try:
        _safe_send_stop(cf)
    except Exception:
        pass

    try:
        _stop_logcfgs(cf)
    except Exception:
        pass

    try:
        cf.close_link()
    except Exception:
        pass

    _mark_link_down("manual disconnect")


# ----------------------------
# Send loop (50 Hz setpoints)
# ----------------------------
def _send_loop() -> None:
    while not STATE.send_thread_stop.is_set():
        cf = STATE.cf
        if not cf or not STATE.connected:
            time.sleep(0.05)
            continue

        if not STATE.deadman_enabled or (now_s() - STATE.deadman_last_seen) > DEADMAN_TIMEOUT_S:
            STATE.phase = "idle"
            STATE.target = Target()
            try:
                cf.commander.send_stop_setpoint()
            except Exception:
                pass
            time.sleep(SEND_DT)
            continue

        try:
            if STATE.mode == "hover":
                STATE.phase = "hover"
                # Hover holds current XY; log actual position as target for sim2real comparison
                a = STATE.telemetry
                STATE.target.x = a.x if a.x is not None else 0.0
                STATE.target.y = a.y if a.y is not None else 0.0
                STATE.target.z = float(STATE.hover_z)
                STATE.target.yaw = 0.0

                cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, float(STATE.hover_z))

            elif STATE.mode == "navigate":
                STATE.phase = "navigate"
                STATE.target.x = float(STATE.nav_x)
                STATE.target.y = float(STATE.nav_y)
                STATE.target.z = float(STATE.nav_z)
                STATE.target.yaw = float(STATE.nav_yaw)

                cf.commander.send_position_setpoint(
                    float(STATE.nav_x),
                    float(STATE.nav_y),
                    float(STATE.nav_z),
                    float(STATE.nav_yaw),
                )

            elif STATE.mode == "plan":
                if not STATE.plan:
                    STATE.phase = "plan(empty)"
                    cf.commander.send_stop_setpoint()
                else:
                    tnow = now_s()

                    # One-shot anti-lunge hold at start
                    if not STATE.plan_active:
                        STATE.plan_active = True

                        a = STATE.telemetry
                        first = STATE.plan[0]

                        if a.x is not None and a.y is not None and a.z is not None:
                            hold_z = float(first.z) if PLAN_HOLD_USE_PLAN_Z else float(a.z)
                            hold_yaw = float(first.yaw)

                            STATE.plan_hold_target = Waypoint(
                                x=float(a.x), y=float(a.y), z=hold_z, yaw=hold_yaw, t=0.0
                            )
                            STATE.plan_hold_until = tnow + float(PLAN_HOLD_BEFORE_S)
                            STATE.plan_started_at = STATE.plan_hold_until
                            STATE.phase = "plan(hold_start)"
                        else:
                            STATE.plan_hold_target = None
                            STATE.plan_hold_until = 0.0
                            STATE.plan_started_at = tnow
                            STATE.phase = "plan(start)"

                    if STATE.plan_hold_target is not None and tnow < STATE.plan_hold_until:
                        wp = STATE.plan_hold_target
                        STATE.phase = "plan(hold)"
                    else:
                        tplan = tnow - STATE.plan_started_at
                        plan_end = STATE.plan[-1].t

                        if tplan > plan_end:
                            if STATE.plan_loop:
                                STATE.plan_started_at = tnow
                                tplan = 0.0
                                STATE.phase = "plan(loop)"
                            else:
                                STATE.phase = "plan(hold_end)"
                                wp = STATE.plan[-1]

                        if STATE.phase != "plan(hold_end)":
                            STATE.phase = "plan(run)"
                            wp = _plan_sample_at(STATE.plan, tplan)

                    STATE.target.x = float(wp.x)
                    STATE.target.y = float(wp.y)
                    STATE.target.z = float(wp.z)
                    STATE.target.yaw = float(wp.yaw)

                    cf.commander.send_position_setpoint(float(wp.x), float(wp.y), float(wp.z), float(wp.yaw))

            else:
                STATE.phase = "idle"
                cf.commander.send_stop_setpoint()

        except Exception:
            try:
                cf.commander.send_stop_setpoint()
            except Exception:
                pass

        time.sleep(SEND_DT)


def _ensure_send_thread() -> None:
    if STATE.send_thread and STATE.send_thread.is_alive():
        return
    STATE.send_thread_stop.clear()
    t = threading.Thread(target=_send_loop, daemon=True)
    STATE.send_thread = t
    t.start()


def _build_plots_zip_from_csv_text(csv_text: str) -> bytes:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(io.StringIO(csv_text))
    if "t" not in df.columns:
        raise ValueError("CSV is missing required time column: t")

    t = df["t"]
    buffer = io.BytesIO()

    def _save_plot_to_zip(zf: zipfile.ZipFile, name: str, draw_fn) -> None:
        plt.figure()
        draw_fn()
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format="png", dpi=200)
        plt.close()
        img.seek(0)
        zf.writestr(f"{name}.png", img.read())

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        _save_plot_to_zip(
            zf,
            "position",
            lambda: (
                plt.plot(t, df["stateEstimate.x"], label="x (m)"),
                plt.plot(t, df["stateEstimate.y"], label="y (m)"),
                plt.plot(t, df["stateEstimate.z"], label="z (m)"),
                plt.title("Drone Position over Time"),
                plt.xlabel("Time [s]"),
                plt.ylabel("Position [m]"),
                plt.legend(),
                plt.grid(True),
            ),
        )
        _save_plot_to_zip(
            zf,
            "attitude",
            lambda: (
                plt.plot(t, df["stabilizer.roll"], label="Roll (°)"),
                plt.plot(t, df["stabilizer.pitch"], label="Pitch (°)"),
                plt.plot(t, df["stabilizer.yaw"], label="Yaw (°)"),
                plt.title("Attitude (Orientation) over Time"),
                plt.xlabel("Time [s]"),
                plt.ylabel("Angle [°]"),
                plt.legend(),
                plt.grid(True),
            ),
        )
        _save_plot_to_zip(
            zf,
            "acceleration",
            lambda: (
                plt.plot(t, df["acc.x"], label="ax"),
                plt.plot(t, df["acc.y"], label="ay"),
                plt.plot(t, df["acc.z"], label="az"),
                plt.title("Accelerometer Data"),
                plt.xlabel("Time [s]"),
                plt.ylabel("Acceleration [g]"),
                plt.legend(),
                plt.grid(True),
            ),
        )
        _save_plot_to_zip(
            zf,
            "gyro",
            lambda: (
                plt.plot(t, df["gyro.x"], label="wx"),
                plt.plot(t, df["gyro.y"], label="wy"),
                plt.plot(t, df["gyro.z"], label="wz"),
                plt.title("Gyroscope Data"),
                plt.xlabel("Time [s]"),
                plt.ylabel("Angular velocity [°/s]"),
                plt.legend(),
                plt.grid(True),
            ),
        )

        numeric_cols = df.select_dtypes(include="number").columns
        skip = {
            "t",
            "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
            "stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw",
            "acc.x", "acc.y", "acc.z",
            "gyro.x", "gyro.y", "gyro.z",
        }
        for col in numeric_cols:
            if col in skip:
                continue
            safe = col.replace(".", "_").replace("/", "_")
            _save_plot_to_zip(
                zf,
                safe,
                lambda col=col: (
                    plt.plot(t, df[col], label=col),
                    plt.title(f"{col} over Time"),
                    plt.xlabel("Time [s]"),
                    plt.ylabel(col),
                    plt.legend(),
                    plt.grid(True),
                ),
            )

        zf.writestr("source.csv", csv_text.encode("utf-8"))

    buffer.seek(0)
    return buffer.read()


def _load_plan_waypoints_from_csv(filename: str) -> List[Dict[str, float]]:
    path = os.path.join(FLIGHTPLAN_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Flight plan not found: {filename}")

    out: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "yaw": float(row.get("yaw", 0.0) or 0.0),
                    "t": float(row["t"]),
                }
            )
    return out


# ----------------------------
# FastAPI lifespan
# ----------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global TELEM_TASK
    _ensure_send_thread()
    if TELEM_TASK is None or TELEM_TASK.done():
        TELEM_TASK = asyncio.create_task(telemetry_pump())

    try:
        yield
    finally:
        try:
            STATE.send_thread_stop.set()
        except Exception:
            pass

        if TELEM_TASK is not None:
            TELEM_TASK.cancel()
            try:
                await TELEM_TASK
            except Exception:
                pass
            TELEM_TASK = None

        try:
            await asyncio.to_thread(_disconnect_blocking)
        except Exception:
            pass


app = FastAPI(title="Crazyflie Web UI Controller", lifespan=lifespan)

# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not os.path.exists(UI_PATH):
        return HTMLResponse("<h1>ui.html not found</h1><p>Put ui.html next to app.py</p>", status_code=404)
    with open(UI_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/plots/export")
async def export_plots(request: Request):
    payload = await request.json()
    csv_text = str(payload.get("csv") or "")
    stamp = str(payload.get("stamp") or int(time.time()))
    if not csv_text.strip():
        return Response(content=json.dumps({"error": "Missing csv payload"}), status_code=400, media_type="application/json")

    try:
        zip_bytes = await asyncio.to_thread(_build_plots_zip_from_csv_text, csv_text)
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), status_code=400, media_type="application/json")

    headers = {"Content-Disposition": f'attachment; filename="flight_plots_{stamp}.zip"'}
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)


@app.get("/api/flightplans/{name}")
async def get_flightplan(name: str):
    plans = {
        "figure8_3d_eval": "figure8_eval_origin_3x_webui.csv",
    }
    filename = plans.get(name)
    if filename is None:
        return Response(
            content=json.dumps({"error": f"Unknown flight plan preset: {name}"}),
            status_code=404,
            media_type="application/json",
        )

    try:
        waypoints = await asyncio.to_thread(_load_plan_waypoints_from_csv, filename)
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), status_code=400, media_type="application/json")

    return {"name": name, "loop": True, "waypoints": waypoints}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with CLIENTS_LOCK:
        CLIENTS.add(websocket)

    await ws_status("welcome")
    await ws_telemetry()
    await ws_log("WebSocket client connected")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            mtype = msg.get("type")

            if mtype == "connect":
                uri = str(msg.get("uri") or DEFAULT_URI)
                if STATE.connected:
                    await ws_status("already connected")
                    await ws_log("Connect requested but already connected", "warn")
                    continue
                try:
                    await ws_status("connecting...")
                    await ws_log(f"Connecting to {uri}...")
                    await asyncio.to_thread(_connect_blocking, uri)
                    _ensure_send_thread()
                    await ws_status("connected")
                    await ws_log(f"Connected to {uri}", "ok")
                    # Report which optional logs started
                    if STATE.log_notes:
                        await ws_log("Log configs active: " + " | ".join(STATE.log_notes), "info")
                except Exception as e:
                    await ws_status(f"connect error: {e}")
                    await ws_log(f"Connect error: {e}", "error")

            elif mtype == "disconnect":
                if not STATE.connected:
                    await ws_status("already disconnected")
                    await ws_log("Disconnect requested but already disconnected", "warn")
                    continue
                await ws_log("Disconnecting...")
                await asyncio.to_thread(_disconnect_blocking)
                await ws_status("disconnected")
                await ws_log("Disconnected", "warn")

            elif mtype == "stop":
                STATE.deadman_enabled = False
                STATE.mode = "idle"
                STATE.phase = "idle"
                STATE.plan_active = False
                STATE.plan_hold_target = None
                STATE.plan_hold_until = 0.0
                STATE.plan_started_at = 0.0
                cf = STATE.cf
                if cf:
                    await asyncio.to_thread(_safe_send_stop, cf)
                await ws_status("STOP")
                await ws_log("STOP issued (deadman disabled, motors stop)", "warn")

            elif mtype == "deadman":
                enabled = bool(msg.get("enabled", False))
                STATE.deadman_enabled = enabled
                STATE.deadman_last_seen = now_s()

            elif mtype == "mode":
                mode = str(msg.get("mode") or "idle").lower()
                if mode not in ("idle", "hover", "navigate"):
                    continue

                if mode == "hover":
                    STATE.mode = "hover"
                    STATE.hover_z = float(msg.get("z", STATE.hover_z))
                    STATE.phase = "hover(set)"
                    await ws_log(f"Mode set: hover z={STATE.hover_z:.2f}")

                elif mode == "navigate":
                    STATE.mode = "navigate"
                    STATE.nav_x = float(msg.get("x", STATE.nav_x))
                    STATE.nav_y = float(msg.get("y", STATE.nav_y))
                    STATE.nav_z = float(msg.get("z", STATE.nav_z))
                    STATE.nav_yaw = float(msg.get("yaw", STATE.nav_yaw))
                    STATE.phase = "navigate(set)"
                    await ws_log(
                        f"Mode set: navigate x={STATE.nav_x:.2f} y={STATE.nav_y:.2f} z={STATE.nav_z:.2f} yaw={STATE.nav_yaw:.1f}"
                    )

                else:
                    STATE.mode = "idle"
                    STATE.phase = "idle"
                    await ws_log("Mode set: idle", "warn")

                if STATE.mode != "plan":
                    STATE.plan_active = False
                    STATE.plan_hold_target = None
                    STATE.plan_hold_until = 0.0
                    STATE.plan_started_at = 0.0

                await ws_status(f"mode={STATE.mode}")

            elif mtype == "plan":
                wps_raw = msg.get("waypoints", [])
                loop = bool(msg.get("loop", True))
                wps = _parse_waypoints(wps_raw)

                if not wps:
                    STATE.plan = []
                    STATE.plan_active = False
                    STATE.mode = "idle"
                    STATE.phase = "plan(empty)"
                    STATE.plan_hold_target = None
                    STATE.plan_hold_until = 0.0
                    STATE.plan_started_at = 0.0
                    await ws_status("plan cleared/empty")
                    await ws_log("Plan cleared/empty", "warn")
                    continue

                STATE.plan = wps
                STATE.plan_loop = loop
                STATE.plan_active = False
                STATE.plan_started_at = 0.0
                STATE.plan_hold_until = 0.0
                STATE.plan_hold_target = None

                STATE.mode = "plan"
                STATE.phase = "plan(loaded)"

                await ws_status(f"plan loaded ({len(STATE.plan)} wps), loop={STATE.plan_loop}")
                await ws_log(f"Plan loaded: {len(STATE.plan)} waypoints, loop={STATE.plan_loop}", "ok")

    except WebSocketDisconnect:
        pass
    finally:
        async with CLIENTS_LOCK:
            CLIENTS.discard(websocket)
        await ws_log("WebSocket client disconnected", "warn")
