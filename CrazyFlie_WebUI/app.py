#!/usr/bin/env python3
"""
Crazyflie Web UI Controller (FastAPI + WebSocket)

Features implemented:
- Web UI hosted from Python (serves ui.html)
- WebSocket control channel: connect / disconnect / stop / deadman / hover / navigate / plan
- Telemetry stream (stateEstimate + battery) to UI
- Plan follower with LOOP option (time-parametrized waypoints)
- **Anti-lunge plan start**: when a plan is received, the server prepends a short hold
  at the drone's current position (or current x/y with safe z) before beginning the plan

Requirements:
  pip install fastapi uvicorn "uvicorn[standard]" cflib

Run:
  py -m uvicorn app:app --host 0.0.0.0 --port 8000

Files:
  - ui.html in the same directory as app.py (or adjust UI_PATH)

Notes:
- This uses the Crazyflie *high-level commander style* setpoints:
    send_hover_setpoint() and send_position_setpoint()
- Dead-man requires the client to send keep-alive pings; if not, motors stop.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ----------------------------
# Config
# ----------------------------
DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"

SEND_HZ = 50.0
SEND_DT = 1.0 / SEND_HZ

DEADMAN_TIMEOUT_S = 0.25          # if no deadman keepalive within this, stop
PLAN_HOLD_BEFORE_S = 1.0          # anti-lunge: hold at current pose before plan starts
PLAN_HOLD_USE_PLAN_Z = True       # True: hold at (current x,y) but z -> first waypoint z
PLAN_MIN_SPEED_MPS = 0.05

# Telemetry update rate (log subsystem)
LOG_PERIOD_MS = 100  # 10 Hz

HERE = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(HERE, "ui.html")

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Crazyflie Web UI Controller")

# If you want to serve a favicon/static folder, create ./static and uncomment:
# app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


# ----------------------------
# State / Data Structures
# ----------------------------
@dataclass
class Telemetry:
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    yaw: Optional[float] = None
    battery: Optional[float] = None


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
    mode: str = "idle"   # "idle" | "hover" | "navigate" | "plan"
    phase: str = "idle"  # optional detail

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
    plan_start_time: float = 0.0
    plan_active: bool = False

    # Latest values
    telemetry: Telemetry = field(default_factory=Telemetry)
    target: Target = field(default_factory=Target)

    # Internal / objects
    cf: Optional[Crazyflie] = None
    logcfg: Optional[LogConfig] = None
    send_thread: Optional[threading.Thread] = None
    send_thread_stop: threading.Event = field(default_factory=threading.Event)


STATE = ControlState()

# WebSocket clients (support multiple dashboards)
CLIENTS: set[WebSocket] = set()
CLIENTS_LOCK = asyncio.Lock()


# ----------------------------
# Utilities
# ----------------------------
def now_s() -> float:
    return time.time()


async def broadcast(msg: Dict[str, Any]) -> None:
    data = json.dumps(msg)
    async with CLIENTS_LOCK:
        dead = []
        for ws in CLIENTS:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            CLIENTS.discard(ws)


async def ws_status(msg: str = "") -> None:
    await broadcast({
        "type": "status",
        "connected": STATE.connected,
        "uri": STATE.uri,
        "msg": msg,
    })


async def ws_telemetry() -> None:
    a = STATE.telemetry
    t = STATE.target
    await broadcast({
        "type": "telemetry",
        "t": now_s(),  # wall-clock seconds (client can display)
        "mode": STATE.mode,
        "phase": STATE.phase,
        "target": {"x": t.x, "y": t.y, "z": t.z, "yaw": t.yaw},
        "actual": {"x": a.x, "y": a.y, "z": a.z, "yaw": a.yaw, "battery": a.battery},
    })


def _safe_send_stop(cf: Crazyflie) -> None:
    try:
        cf.commander.send_stop_setpoint()
    except Exception:
        pass


def _prepend_hold_at_current(waypoints: List[Waypoint]) -> List[Waypoint]:
    """
    Anti-lunge: add two hold waypoints at current position.
    - t=0: hold current
    - t=PLAN_HOLD_BEFORE_S: hold current
    Then shift the incoming plan times by PLAN_HOLD_BEFORE_S.
    """
    if not waypoints:
        return waypoints

    a = STATE.telemetry
    if a.x is None or a.y is None or a.z is None:
        # If no telemetry yet, we cannot reliably hold "current". Return unchanged.
        return waypoints

    first = waypoints[0]
    hold_z = float(first.z) if PLAN_HOLD_USE_PLAN_Z else float(a.z)
    hold_yaw = float(first.yaw) if (a.yaw is None) else float(a.yaw)

    hold0 = Waypoint(x=float(a.x), y=float(a.y), z=hold_z, yaw=hold_yaw, t=0.0)
    hold1 = Waypoint(x=float(a.x), y=float(a.y), z=hold_z, yaw=hold_yaw, t=float(PLAN_HOLD_BEFORE_S))

    shifted: List[Waypoint] = []
    for wp in waypoints:
        shifted.append(Waypoint(x=wp.x, y=wp.y, z=wp.z, yaw=wp.yaw, t=float(wp.t) + float(PLAN_HOLD_BEFORE_S)))

    return [hold0, hold1] + shifted


def _parse_waypoints(raw: Any) -> List[Waypoint]:
    out: List[Waypoint] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        try:
            out.append(Waypoint(
                x=float(item["x"]),
                y=float(item["y"]),
                z=float(item["z"]),
                yaw=float(item.get("yaw", 0.0)),
                t=float(item["t"]),
            ))
        except Exception:
            continue
    # Ensure sorted by time
    out.sort(key=lambda w: w.t)
    return out


def _plan_sample_at(plan: List[Waypoint], t_now: float) -> Waypoint:
    """
    Piecewise-linear interpolation across time-indexed waypoints.
    Assumes plan sorted by t.
    """
    if not plan:
        raise ValueError("empty plan")

    # clamp ends
    if t_now <= plan[0].t:
        return plan[0]
    if t_now >= plan[-1].t:
        return plan[-1]

    # find segment
    # simple linear scan (plans are small). Replace with binary search if needed.
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
def _setup_logging(cf: Crazyflie) -> None:
    """
    Configure Crazyflie logging for pose and battery.
    """
    # Remove old logcfg if exists
    if STATE.logcfg is not None:
        try:
            cf.log.remove_config(STATE.logcfg)
        except Exception:
            pass
        STATE.logcfg = None

    logcfg = LogConfig(name="telemetry", period_in_ms=LOG_PERIOD_MS)

    # Common variables available when state estimator is enabled
    # (These are standard in CF firmware with stateEstimate)
    logcfg.add_variable("stateEstimate.x", "float")
    logcfg.add_variable("stateEstimate.y", "float")
    logcfg.add_variable("stateEstimate.z", "float")

    # Yaw is sometimes available as "stateEstimate.yaw" depending on firmware;
    # If not available, you can remove it or switch to "stabilizer.yaw"
    # We'll attempt stabilizer.yaw as it exists widely.
    logcfg.add_variable("stabilizer.yaw", "float")

    # Battery voltage:
    logcfg.add_variable("pm.vbat", "float")

    def _log_cb(timestamp, data, logconf):
        # Runs in CF logging thread; keep it quick.
        try:
            STATE.telemetry.x = float(data.get("stateEstimate.x"))
            STATE.telemetry.y = float(data.get("stateEstimate.y"))
            STATE.telemetry.z = float(data.get("stateEstimate.z"))
            STATE.telemetry.yaw = float(data.get("stabilizer.yaw"))
            STATE.telemetry.battery = float(data.get("pm.vbat"))
        except Exception:
            pass

        # Push to websocket clients via asyncio loop thread-safely
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(ws_telemetry())
        except RuntimeError:
            # if no running loop in this thread, use the main loop reference if set later
            pass

    logcfg.data_received_cb.add_callback(_log_cb)
    cf.log.add_config(logcfg)
    logcfg.start()
    STATE.logcfg = logcfg


def _connect_blocking(uri: str) -> None:
    """
    Blocking connect; called in thread via asyncio.to_thread
    """
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

    cf.connected.add_callback(_on_connected)
    cf.connection_failed.add_callback(_on_failed)
    cf.connection_lost.add_callback(_on_lost)

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

    # On connect: ensure idle
    _safe_send_stop(cf)

    # Start telemetry logging
    try:
        _setup_logging(cf)
    except Exception as e:
        # Logging not strictly required; continue
        print("WARN: telemetry logging setup failed:", e)

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
        if STATE.logcfg is not None:
            try:
                STATE.logcfg.stop()
            except Exception:
                pass
            STATE.logcfg = None
    except Exception:
        pass
    try:
        cf.close_link()
    except Exception:
        pass
    STATE.cf = None
    STATE.connected = False
    STATE.mode = "idle"
    STATE.phase = "idle"
    STATE.plan_active = False
    STATE.deadman_enabled = False


# ----------------------------
# Send loop (50 Hz setpoints)
# ----------------------------
def _send_loop() -> None:
    """
    Runs in a dedicated thread and streams setpoints while connected.
    Enforces dead-man.
    """
    while not STATE.send_thread_stop.is_set():
        cf = STATE.cf
        if not cf or not STATE.connected:
            time.sleep(0.05)
            continue

        # Deadman enforcement
        if not STATE.deadman_enabled or (now_s() - STATE.deadman_last_seen) > DEADMAN_TIMEOUT_S:
            # Stop motors (idle)
            STATE.phase = "idle"
            STATE.target = Target()
            try:
                cf.commander.send_stop_setpoint()
            except Exception:
                pass
            time.sleep(SEND_DT)
            continue

        # If deadman held, send according to mode
        try:
            if STATE.mode == "hover":
                STATE.phase = "hover"
                STATE.target.x = None
                STATE.target.y = None
                STATE.target.z = float(STATE.hover_z)
                STATE.target.yaw = 0.0

                cf.commander.send_hover_setpoint(
                    0.0,  # vx
                    0.0,  # vy
                    0.0,  # yawrate
                    float(STATE.hover_z)
                )

            elif STATE.mode == "navigate":
                # Optional: a short "takeoff/hover" phase could be implemented here;
                # For now, we just command the setpoint.
                STATE.phase = "navigate"
                STATE.target.x = float(STATE.nav_x)
                STATE.target.y = float(STATE.nav_y)
                STATE.target.z = float(STATE.nav_z)
                STATE.target.yaw = float(STATE.nav_yaw)

                cf.commander.send_position_setpoint(
                    float(STATE.nav_x),
                    float(STATE.nav_y),
                    float(STATE.nav_z),
                    float(STATE.nav_yaw)
                )

            elif STATE.mode == "plan":
                if not STATE.plan:
                    STATE.phase = "plan(empty)"
                    cf.commander.send_stop_setpoint()
                else:
                    if not STATE.plan_active:
                        STATE.plan_active = True
                        STATE.plan_start_time = now_s()
                        STATE.phase = "plan(start)"

                    tplan = now_s() - STATE.plan_start_time
                    plan_end = STATE.plan[-1].t

                    if tplan > plan_end:
                        if STATE.plan_loop:
                            STATE.plan_start_time = now_s()
                            tplan = 0.0
                            STATE.phase = "plan(loop)"
                        else:
                            # end: hold last point (still requires deadman)
                            STATE.phase = "plan(hold_end)"
                            wp = STATE.plan[-1]
                    else:
                        STATE.phase = "plan(run)"
                        wp = _plan_sample_at(STATE.plan, tplan)

                    STATE.target.x = float(wp.x)
                    STATE.target.y = float(wp.y)
                    STATE.target.z = float(wp.z)
                    STATE.target.yaw = float(wp.yaw)

                    cf.commander.send_position_setpoint(
                        float(wp.x), float(wp.y), float(wp.z), float(wp.yaw)
                    )

            else:
                STATE.phase = "idle"
                cf.commander.send_stop_setpoint()

        except Exception:
            # If anything goes wrong, stop
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


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not os.path.exists(UI_PATH):
        return HTMLResponse(
            "<h1>ui.html not found</h1><p>Put ui.html next to app.py</p>",
            status_code=404
        )
    with open(UI_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with CLIENTS_LOCK:
        CLIENTS.add(websocket)

    # Provide immediate status
    await ws_status("welcome")

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
                    continue
                try:
                    await ws_status("connecting...")
                    await asyncio.to_thread(_connect_blocking, uri)
                    _ensure_send_thread()
                    await ws_status("connected")
                except Exception as e:
                    await ws_status(f"connect error: {e}")

            elif mtype == "disconnect":
                if not STATE.connected:
                    await ws_status("already disconnected")
                    continue
                await asyncio.to_thread(_disconnect_blocking)
                await ws_status("disconnected")

            elif mtype == "stop":
                # immediate stop setpoint + disable deadman
                STATE.deadman_enabled = False
                STATE.mode = "idle"
                STATE.phase = "idle"
                cf = STATE.cf
                if cf:
                    await asyncio.to_thread(_safe_send_stop, cf)
                await ws_status("STOP")

            elif mtype == "deadman":
                enabled = bool(msg.get("enabled", False))
                STATE.deadman_enabled = enabled
                STATE.deadman_last_seen = now_s()
                # do not change mode here; just authorizes sending
                # (if disabled, send loop will stop)
                await ws_status("deadman on" if enabled else "deadman off")

            elif mtype == "mode":
                mode = str(msg.get("mode") or "idle").lower()
                if mode not in ("idle", "hover", "navigate"):
                    continue

                if mode == "hover":
                    STATE.mode = "hover"
                    STATE.hover_z = float(msg.get("z", STATE.hover_z))
                    STATE.phase = "hover(set)"

                elif mode == "navigate":
                    STATE.mode = "navigate"
                    STATE.nav_x = float(msg.get("x", STATE.nav_x))
                    STATE.nav_y = float(msg.get("y", STATE.nav_y))
                    STATE.nav_z = float(msg.get("z", STATE.nav_z))
                    STATE.nav_yaw = float(msg.get("yaw", STATE.nav_yaw))
                    STATE.phase = "navigate(set)"

                else:
                    STATE.mode = "idle"
                    STATE.phase = "idle"

                await ws_status(f"mode={STATE.mode}")

            elif mtype == "plan":
                # Receive plan waypoints from UI
                wps_raw = msg.get("waypoints", [])
                loop = bool(msg.get("loop", True))
                wps = _parse_waypoints(wps_raw)

                if not wps:
                    STATE.plan = []
                    STATE.plan_active = False
                    STATE.mode = "idle"
                    STATE.phase = "plan(empty)"
                    await ws_status("plan cleared/empty")
                    continue

                # Anti-lunge: prepend hold at current position
                safe_wps = _prepend_hold_at_current(wps)

                STATE.plan = safe_wps
                STATE.plan_loop = loop
                STATE.plan_active = False
                STATE.plan_start_time = 0.0
                STATE.mode = "plan"
                STATE.phase = "plan(loaded)"

                await ws_status(f"plan loaded ({len(STATE.plan)} wps), loop={STATE.plan_loop}")

            else:
                # ignore unknown messages
                pass

    except WebSocketDisconnect:
        pass
    finally:
        async with CLIENTS_LOCK:
            CLIENTS.discard(websocket)


@app.on_event("startup")
async def _startup():
    # ensure send loop thread is running (it will idle until connected)
    _ensure_send_thread()


@app.on_event("shutdown")
async def _shutdown():
    # stop thread + disconnect
    try:
        STATE.send_thread_stop.set()
    except Exception:
        pass
    try:
        await asyncio.to_thread(_disconnect_blocking)
    except Exception:
        pass
