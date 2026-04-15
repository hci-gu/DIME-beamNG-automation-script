"""BeamNG scenario driver for the Driver Impairment Test route."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pyttsx3
from PIL import Image
from beamngpy import BeamNGpy, Scenario, StaticObject, Vehicle

try:
    import obsws_python as obs

    OBS_WEBSOCKET_AVAILABLE = True
except ImportError:
    obs = None
    OBS_WEBSOCKET_AVAILABLE = False

try:
    import keyboard

    KEYBOARD_AVAILABLE = True
except ImportError:
    keyboard = None
    KEYBOARD_AVAILABLE = False
    logging.warning(
        "'keyboard' module not found; respawn hotkeys disabled. "
        "Install with: pip install keyboard"
    )


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
Color = tuple[float, float, float, float]

MAP_NAME = "east_coast_usa"
SCENARIO_NAME = "Driver Impairment Test"
SCENARIO_DESCRIPTION = "You may continue when ready."
SCENARIO_AUTHORS = "University of Gothenburg"
SCENARIO_SETTINGS = {"Start Time": 0}

BNG_HOME = r"C:\BeamNG.tech.v0.38.3.0\BeamNG.tech.v0.38.3.0"
GRAPHICS_BACKEND = "vk"

PLAYER_SPAWN: Vec3 = (501, -877, 41.0)
PLAYER_ROT_QUAT: Quat = (0.0087653, 0.00096168, 0.93361151, 0.35821542)
PLAYER_VEHICLE_NAME = "ego_vehicle"
PLAYER_VEHICLE_MODEL = "etk800"
PLAYER_LICENSE = "UVT 6xY"
PLAYER_CAMERA_FOV = 60

RESPAWN_HOTKEY = "s"
RESET_HOTKEY = "home"
SCENARIO_POLL_SECONDS = 1.0

# How often to refresh BeamNG state and evaluate prompts/NPC triggers.
# Try 0.2, 0.5, or 1.0 to reduce script-side polling frequency.
MAIN_LOOP_SLEEP_SECONDS = 0.1

CONE_SHAPE = "/levels/east_coast_usa/art/shapes/misc/cone.dae"
SPEED_SIGN_50_SHAPE = (
    "/levels/east_coast_usa/art/shapes/signs_usa/sign_speedlimit_50.dae"
)
SPEED_SIGN_40_SHAPE = (
    "/levels/east_coast_usa/art/shapes/signs_usa/sign_speedlimit_40.dae"
)

WELCOME_PROMPT_TEXT = (
    "Welcome. Please continue to drive along the route until the experimenter "
    "stops the simulation. You are required to drive safely and obey driving "
    "rules consistent with Swedish law. If you damage the vehicle, your "
    "vehicle will be repositioned at an earlier point on the route. You may "
    "continue when ready."
)
OVERTAKE_MESSAGE = "Overtake the vehicle in front of you"

BEAMNG_BINARY_CANDIDATES = (
    "Bin64/BeamNG.tech.x64.exe",
    "Bin64/BeamNG.x64.exe",
    "Bin64/BeamNG.drive.x64.exe",
)


def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        logging.warning("Invalid integer for %s=%r; using %s", name, raw_value, default)
        return default


OBS_ENABLE_RECORDING = env_flag("OBS_ENABLE_RECORDING", False)
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = env_int("OBS_PORT", 4455)
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "")
OBS_SCENE_NAME = os.getenv("OBS_SCENE_NAME", "BeamNG Single Window")
OBS_SOURCE_NAME = os.getenv("OBS_SOURCE_NAME", "BeamNG Window")
OBS_SOURCE_KIND = os.getenv("OBS_SOURCE_KIND", "window_capture")
OBS_WINDOW = os.getenv("OBS_WINDOW", "")
OBS_RECORD_DIRECTORY = os.getenv("OBS_RECORD_DIRECTORY", "")
OBS_CAPTURE_CURSOR = env_flag("OBS_CAPTURE_CURSOR", False)
OBS_WINDOW_PRIORITY = env_int("OBS_WINDOW_PRIORITY", 1)

BNG_USER_PATH = os.getenv("BNG_USER_PATH")
BNG_FORCE_GRAPHICS = env_flag("BNG_FORCE_GRAPHICS", False)
BNG_DISPLAY_MODE = os.getenv("BNG_DISPLAY_MODE", "Window")
BNG_RESOLUTION = os.getenv("BNG_RESOLUTION", "")
BNG_WINDOW_X = env_int("BNG_WINDOW_X", 0)
BNG_WINDOW_Y = env_int("BNG_WINDOW_Y", 0)
BNG_WINDOW_WIDTH = env_int("BNG_WINDOW_WIDTH", 0)
BNG_WINDOW_HEIGHT = env_int("BNG_WINDOW_HEIGHT", 0)
BNG_WINDOW_PLACEMENT = os.getenv("BNG_WINDOW_PLACEMENT", "")
BNG_SHOW_FPS = env_flag("BNG_SHOW_FPS", True)


@dataclass(frozen=True)
class Range2D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, pos: Vec3) -> bool:
        x, y, _ = pos
        return self.x_min < x < self.x_max and self.y_min < y < self.y_max


@dataclass(frozen=True)
class RespawnCheckpoint:
    name: str
    bounds: Range2D
    pos: Vec3
    rot: Quat


@dataclass(frozen=True)
class PromptTrigger:
    name: str
    pos: Vec3
    radius: float
    text: str
    gui_text: Optional[str] = None


@dataclass(frozen=True)
class NPCConfig:
    name: str
    model: str
    license_text: str
    pos: Vec3
    rot: Quat
    color: Optional[Color] = None
    trigger_distance: float = 100.0
    speed_limit_mps: float = 12.0
    start_message: Optional[str] = OVERTAKE_MESSAGE


@dataclass(frozen=True)
class SignConfig:
    name: str
    anchor_pos: Vec3
    forward_offset: float
    right_offset: float
    z_offset: float
    yaw_offset_deg: float
    shape: str


class OBSController:
    def __init__(self) -> None:
        self.client: Any = None
        self.recording_started = False

    def connect(self) -> None:
        if not OBS_ENABLE_RECORDING:
            return

        if not OBS_WEBSOCKET_AVAILABLE:
            logging.warning(
                "OBS recording requested but 'obsws-python' is not installed. "
                "Install with: pip install obsws-python"
            )
            return

        try:
            self.client = obs.ReqClient(
                host=OBS_HOST,
                port=OBS_PORT,
                password=OBS_PASSWORD,
                timeout=3,
            )
            version = self.client.get_version()
            logging.info(
                "Connected to OBS Studio %s via obs-websocket %s",
                getattr(version, "obs_version", "unknown"),
                getattr(version, "obs_web_socket_version", "unknown"),
            )
        except Exception as exc:
            logging.warning("Failed to connect to OBS: %s", exc)
            self.client = None

    def setup_single_window_capture(self) -> None:
        if self.client is None:
            return

        try:
            self._ensure_scene()
            self._ensure_window_source()
            self.client.set_current_program_scene(OBS_SCENE_NAME)

            if OBS_RECORD_DIRECTORY:
                self.client.set_record_directory(OBS_RECORD_DIRECTORY)

            logging.info(
                "OBS single-window capture ready in scene '%s' with source '%s'",
                OBS_SCENE_NAME,
                OBS_SOURCE_NAME,
            )
        except Exception as exc:
            logging.warning("Failed to set up OBS single-window capture: %s", exc)

    def start_recording(self) -> None:
        if self.client is None or self.recording_started:
            return

        try:
            self.client.start_record()
            self.recording_started = True
            logging.info("OBS recording started")
        except Exception as exc:
            logging.warning("Failed to start OBS recording: %s", exc)

    def stop_recording(self) -> None:
        if self.client is None or not self.recording_started:
            return

        try:
            self.client.stop_record()
            logging.info("OBS recording stopped")
        except Exception as exc:
            logging.warning("Failed to stop OBS recording: %s", exc)
        finally:
            self.recording_started = False

    def _ensure_scene(self) -> None:
        scene_list = self.client.get_scene_list()
        scene_names = {
            self._field(scene, "scene_name", "sceneName")
            for scene in getattr(scene_list, "scenes", [])
        }

        if OBS_SCENE_NAME not in scene_names:
            self.client.create_scene(OBS_SCENE_NAME)

    def _ensure_window_source(self) -> None:
        input_names = {
            self._field(source, "input_name", "inputName")
            for source in getattr(self.client.get_input_list(), "inputs", [])
        }
        settings = self._build_window_capture_settings()

        if OBS_SOURCE_NAME in input_names:
            if settings:
                self.client.set_input_settings(OBS_SOURCE_NAME, settings, True)
            return

        self.client.create_input(
            OBS_SCENE_NAME,
            OBS_SOURCE_NAME,
            OBS_SOURCE_KIND,
            settings,
            True,
        )

    def _build_window_capture_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {"capture_cursor": OBS_CAPTURE_CURSOR}

        if OBS_SOURCE_KIND == "window_capture":
            settings["priority"] = OBS_WINDOW_PRIORITY

        if OBS_WINDOW:
            settings["window"] = OBS_WINDOW
        else:
            logging.warning(
                "OBS_WINDOW is not set. OBS will create/update the source, but you may "
                "need to select the BeamNG window once in OBS and then reuse that exact "
                "window identifier via OBS_WINDOW for fully unattended startup."
            )

        return settings

    @staticmethod
    def _field(item: Any, attr_name: str, dict_key: str) -> Any:
        if isinstance(item, dict):
            return item.get(dict_key)
        return getattr(item, attr_name, None)


def build_window_placement(x: int, y: int, width: int, height: int) -> str:
    return f"0 1 -1 -1 -1 -1 {x} {y} {x + width} {y + height}"


def get_desired_display_settings() -> dict[str, str]:
    settings: dict[str, str] = {}

    if BNG_DISPLAY_MODE:
        settings["GraphicDisplayModes"] = BNG_DISPLAY_MODE

    if BNG_RESOLUTION:
        settings["GraphicDisplayResolutions"] = BNG_RESOLUTION

    window_placement = BNG_WINDOW_PLACEMENT
    if (
        not window_placement
        and BNG_WINDOW_WIDTH > 0
        and BNG_WINDOW_HEIGHT > 0
    ):
        window_placement = build_window_placement(
            BNG_WINDOW_X,
            BNG_WINDOW_Y,
            BNG_WINDOW_WIDTH,
            BNG_WINDOW_HEIGHT,
        )

    if window_placement:
        settings["WindowPlacement"] = window_placement

    return settings


def write_beamng_json_settings(path: Path, settings: dict[str, str]) -> bool:
    try:
        data: dict[str, Any] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}

        data.update(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return True
    except Exception as exc:
        logging.warning("Failed writing BeamNG JSON settings %s: %s", path, exc)
        return False


def write_beamng_ini_settings(path: Path, settings: dict[str, str]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            original_lines = path.read_text(encoding="utf-8").splitlines()
        else:
            original_lines = []

        pending = dict(settings)
        output_lines: list[str] = []

        for line in original_lines:
            stripped = line.strip()
            if (
                stripped
                and "=" in line
                and not stripped.startswith(("#", ";", "["))
            ):
                key, _ = line.split("=", 1)
                normalized_key = key.strip()
                if normalized_key in pending:
                    output_lines.append(f"{normalized_key} = {pending.pop(normalized_key)}")
                    continue

            output_lines.append(line)

        for key, value in pending.items():
            output_lines.append(f"{key} = {value}")

        path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        return True
    except Exception as exc:
        logging.warning("Failed writing BeamNG INI settings %s: %s", path, exc)
        return False


def persist_beamng_display_settings(user_path: Optional[str]) -> bool:
    if not BNG_FORCE_GRAPHICS:
        return False

    if not user_path:
        logging.warning(
            "BNG_FORCE_GRAPHICS is enabled but BNG_USER_PATH is not set, so "
            "pre-launch settings persistence is unavailable for this run."
        )
        return False

    settings = get_desired_display_settings()
    if not settings:
        return False

    settings_dir = Path(user_path) / "settings"
    targets = [
        settings_dir / "game-settings.ini",
        settings_dir / "cloud" / "game-settings-cloud.ini",
        settings_dir / "settings.json",
    ]

    wrote_any = False
    for target in targets:
        if target.suffix == ".json":
            wrote_any = write_beamng_json_settings(target, settings) or wrote_any
        else:
            wrote_any = write_beamng_ini_settings(target, settings) or wrote_any

    if wrote_any:
        logging.info(
            "Persisted BeamNG display settings under %s: %s",
            settings_dir,
            settings,
        )

    return wrote_any


def apply_beamng_display_settings(bng: BeamNGpy) -> None:
    if not BNG_FORCE_GRAPHICS:
        return

    settings = get_desired_display_settings()
    changed = False

    for key, value in settings.items():
        bng.settings.change(key, value)
        changed = True

    if changed:
        bng.settings.apply_graphics()
        logging.info(
            "Applied BeamNG display settings via API: %s",
            settings,
        )


def enable_beamng_fps_overlay(bng: BeamNGpy) -> None:
    if not BNG_SHOW_FPS:
        return

    last_error: Optional[Exception] = None
    command_targets = [getattr(bng, "control", None), bng]

    for target in command_targets:
        if target is None or not hasattr(target, "queue_lua_command"):
            continue

        try:
            target.queue_lua_command('metrics("fps")')
            logging.info("Enabled BeamNG FPS overlay")
            return
        except Exception as exc:
            last_error = exc

    logging.warning("Failed to enable BeamNG FPS overlay: %s", last_error)


RESPAWN_CHECKPOINTS = [
    RespawnCheckpoint(
        name="cp4_event_1",
        bounds=Range2D(-488, 501, -877, 249),
        pos=(500, -875, 41.0),
        rot=(0.0087653, 0.00096168, 0.93361151, 0.35821542),
    ),
    RespawnCheckpoint(
        name="cp1_event_1",
        bounds=Range2D(-876, -488, -877, 608),
        pos=(-722, 482, 22),
        rot=(0.007477, -0.0027, 0.883, 0.4677),
    ),
    RespawnCheckpoint(
        name="cp2_event_1",
        bounds=Range2D(-876, -570, float("-inf"), float("inf")),
        pos=(-800, 950, 35),
        rot=(0.0022, 0.0166, -0.736, 0.676),
    ),
    RespawnCheckpoint(
        name="cp3_event_1",
        bounds=Range2D(-570, -175, 910, 980),
        pos=(-571, 941, 35),
        rot=(0.004, -0.0054, -0.694, 0.719),
    ),
    RespawnCheckpoint(
        name="cp3_event_2",
        bounds=Range2D(-175, -70, 860, 910),
        pos=(-175, 910, 35),
        rot=(0.004, -0.0054, -0.694, 0.719),
    ),
    RespawnCheckpoint(
        name="cp6_event_1",
        bounds=Range2D(float("-inf"), float("inf"), 608, 950),
        pos=(-920.711, 662.963, 24.557),
        rot=(0.0208, 0.0216, 0.952, 0.302),
    ),
    RespawnCheckpoint(
        name="cp6_event_2",
        bounds=Range2D(-70, 78, 860, 880),
        pos=(-70, 873, 25),
        rot=(0.026, 0.025, -0.06, 0.78),
    ),
    RespawnCheckpoint(
        name="cp4_event_2",
        bounds=Range2D(812, 902, -420, 581),
        pos=(811.88, 581.11, 65.38),
        rot=(-0.0072, -0.000579, -0.147, 0.989),
    ),
    RespawnCheckpoint(
        name="cp1_event_2",
        bounds=Range2D(892, 902, -611, 581),
        pos=(892, -611, 40),
        rot=(0.005, 0.0007, 0.1946, 0.981),
    ),
]

VOICE_PROMPTS = [
    PromptTrigger(
        name="intro",
        pos=(500, -875, 41.0),
        radius=20.0,
        text=WELCOME_PROMPT_TEXT,
        gui_text=WELCOME_PROMPT_TEXT,
    ),
    PromptTrigger(
        name="continue_straight_1",
        pos=(470, -831, 43),
        radius=10.0,
        text="Continue Straight",
    ),
    PromptTrigger(
        name="spell_norway",
        pos=(430, -800, 41),
        radius=20.0,
        text="Attention! Please spell 'Norway'",
    ),
    PromptTrigger(
        name="continue_straight_2",
        pos=(-450, 226, 28),
        radius=20.0,
        text="Continue Straight",
    ),
    PromptTrigger(
        name="continue_straight_3",
        pos=(-605, 395, 21),
        radius=20.0,
        text="Continue Straight",
    ),
    PromptTrigger(
        name="continue_straight_4",
        pos=(-655, 446, 23),
        radius=20.0,
        text="Continue Straight",
    ),
    PromptTrigger(
        name="spell_alone",
        pos=(-731.8, 481.8, 22.4),
        radius=20.0,
        text="Attention! Please spell 'alone'",
    ),
    PromptTrigger(
        name="overtake_1",
        pos=(-920.711, 662.963, 24.557),
        radius=10.0,
        text=(
            "Attention! Please read the license plate of the car in front "
            "of you and then overtake when possible"
        ),
    ),
    PromptTrigger(
        name="continue_straight_5",
        pos=(-852.8610016107559, 586.0869419574738, 26.510813504457474),
        radius=10.0,
        text="Continue Straight",
    ),
    PromptTrigger(
        name="red_lorry",
        pos=(-571, 941, 35),
        radius=20.0,
        text="Attention! Say red lorry, yellow lorry twice",
    ),
    PromptTrigger(
        name="spell_dollar",
        pos=(-175, 910, 35),
        radius=30.0,
        text="Please spell 'Dollar'",
    ),
    PromptTrigger(
        name="continue_straight_6",
        pos=(-14.27, 866.40, 26.56),
        radius=20.0,
        text="Continue Straight",
    ),
    PromptTrigger(
        name="continue_straight_7",
        pos=(225, 880, 37),
        radius=20.0,
        text="Continue Straight",
    ),
    PromptTrigger(
        name="overtake_2",
        pos=(492.3, 914.3, 55.69),
        radius=10.0,
        text=(
            "Attention! Please read the license plate of the car in front "
            "of you and then overtake when possible"
        ),
    ),
    PromptTrigger(
        name="continue_straight_8",
        pos=(902, -420, 44),
        radius=20.0,
        text="Continue Straight",
    ),
    PromptTrigger(
        name="unique_new_york",
        pos=(909, -258, 40),
        radius=20.0,
        text="Attention! Please say 'Unique New York' twice",
    ),
    PromptTrigger(
        name="turn_right",
        pos=(830, -759, 42),
        radius=20.0,
        text="Please turn right.",
    ),
]

NPC_CONFIGS = [
    NPCConfig(
        name="npc6",
        model="scintilla",
        license_text="RNO LR8",
        pos=(-879.305, 612.342, 24.706),
        rot=(-0.0006, -0.00053, 0.92952, 0.36877),
        color=(0.0, 0.0, 1.0, 0.2),
    ),
    NPCConfig(
        name="npc7",
        model="autobello",
        license_text="GNX 566",
        pos=(-852.8610016107559, 586.0869419574738, 26.510813504457474),
        rot=(0, 0, -0.662620, 0.748956),
    ),
    NPCConfig(
        name="npc8",
        model="etk800",
        license_text="JYT 7T1",
        pos=(-176.642288, 918.723938, 23.2221851),
        rot=(0, 0, -0.662620, 0.748956),
        color=(1.0, 1.0, 0.0, 1.0),
    ),
    NPCConfig(
        name="npc9",
        model="etk800",
        license_text="PII 377",
        pos=(455.650055, 902.720032, 51.4617844),
        rot=(0, 0, -0.750111, 0.661312),
        color=(1.0, 1.0, 1.0, 0.0),
    ),
    NPCConfig(
        name="npc10",
        model="autobello",
        license_text="HJG 89U",
        pos=(-748, 488, 23),
        rot=(-0.026407821103930473, -0.01194380410015583, -0.8407955765724182, 0.5405763387680054),
        color=(0.5, 0.5, 0.5, 0.0),
    ),
    NPCConfig(
        name="npc11",
        model="etki",
        license_text="YRU 593",
        pos=(-667.2760881692193, 950.6957388893661, 37.89792846151886),
        rot=(0.0018872566288337111, 0.005335998721420765, 0.7313352823257446, 0.681994616985321),
        color=(0.5, 0.5, 0.0, 0.6),
    ),
    NPCConfig(
        name="npc12",
        model="etki",
        license_text="NRL 002",
        pos=(529.8497040759812, 931.7092222183737, 61.2222659783788),
        rot=(0.03721029683947563, 0.06410301476716995, -0.810825526714325, 0.5805758237838745),
        color=(1.0, 0.0, 0.0, 0.6),
    ),
]

SIGN_CONFIGS = [
    SignConfig(
        name="speed_sign_70",
        anchor_pos=(-721.0817284743873, 481.14044092518566, 22.124332515832975),
        forward_offset=-2.5,
        right_offset=0.0,
        z_offset=0.2,
        yaw_offset_deg=180.0,
        shape=SPEED_SIGN_50_SHAPE,
    ),
    SignConfig(
        name="speed_sign_40",
        anchor_pos=(-559.9609605189526, 941.1932155510767, 34.83220948548933),
        forward_offset=-2.0,
        right_offset=-6.0,
        z_offset=0.2,
        yaw_offset_deg=0.0,
        shape=SPEED_SIGN_40_SHAPE,
    ),
    SignConfig(
        name="speed_sign_50",
        anchor_pos=(-386.5015354813413, 918.9126735679774, 30.40501675377709),
        forward_offset=0.0,
        right_offset=-7.0,
        z_offset=0.2,
        yaw_offset_deg=0.0,
        shape=SPEED_SIGN_50_SHAPE,
    ),
]

def dist3_squared(a: Vec3, b: Vec3) -> float:
    return (
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def within_radius(a: Vec3, b: Vec3, radius: float) -> bool:
    return dist3_squared(a, b) < radius * radius


def quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conj(q: Quat) -> Quat:
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    vx, vy, vz = v
    rotated = quat_mul(quat_mul(q, (vx, vy, vz, 0.0)), quat_conj(q))
    return rotated[:3]


def vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vscale(v: Vec3, scale: float) -> Vec3:
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def quat_from_z_rotation_degrees(degrees: float) -> Quat:
    radians = math.radians(degrees)
    half = radians / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def offset_position(
    anchor_pos: Vec3,
    base_rot: Quat,
    forward_offset: float,
    right_offset: float,
    z_offset: float,
) -> Vec3:
    forward = quat_rotate(base_rot, (0.0, 1.0, 0.0))
    right = quat_rotate(base_rot, (1.0, 0.0, 0.0))
    pos = vadd(anchor_pos, vscale(forward, forward_offset))
    pos = vadd(pos, vscale(right, right_offset))
    return vadd(pos, (0.0, 0.0, z_offset))


def add_cone_line(
    scenario: Scenario,
    start_pos: Vec3,
    spacing_m: float = 5.0,
    count: int = 10,
    yaw_deg: float = 0.0,
) -> None:
    yaw_rad = math.radians(yaw_deg)
    dx = math.cos(yaw_rad) * spacing_m
    dy = math.sin(yaw_rad) * spacing_m
    half = yaw_rad / 2.0
    rot_quat = (0.0, 0.0, math.sin(half), math.cos(half))

    for index in range(count):
        cone_pos = (
            start_pos[0] + index * dx,
            start_pos[1] + index * dy,
            start_pos[2],
        )
        scenario.add_object(
            StaticObject(
                name=f"cone_{index}",
                pos=cone_pos,
                rot_quat=rot_quat,
                scale=(1.0, 1.0, 1.0),
                shape=CONE_SHAPE,
            )
        )


def save_birdseye_image(
    bng: BeamNGpy,
    center_pos: Vec3,
    filename: str = "birdseye_route_view.png",
    height: float = 200,
) -> None:
    camera_pos = (center_pos[0], center_pos[1], center_pos[2] + height)
    rendered = bng.camera.render(
        pos=camera_pos,
        dir=(0, 0, -1),
        fov=70,
        resolution=(1920, 1080),
        render_annotations=False,
    )
    color_key = "colour" if "colour" in rendered else "color"
    Image.fromarray(np.asarray(rendered[color_key])).save(filename)


def resolve_beamng_home(configured_home: str) -> str:
    """Return a BeamNG home folder that directly contains Bin64 executables."""
    candidate_roots: list[Path] = []

    if configured_home:
        configured_path = Path(configured_home)
        candidate_roots.extend(
            [
                configured_path,
                configured_path.parent,
                configured_path / "BeamNG.tech.v0.35.5.0",
                configured_path / "BeamNG.tech.v0.38.3.0",
            ]
        )

    seen: set[Path] = set()
    for root in candidate_roots:
        root = root.expanduser()
        if root in seen:
            continue
        seen.add(root)

        for binary in BEAMNG_BINARY_CANDIDATES:
            if (root / binary).exists():
                return str(root)

    return configured_home


class SpeechController:
    def __init__(self) -> None:
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def say(self, text: str) -> None:
        if self._running:
            self._queue.put(text)

    def _worker(self) -> None:
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.setProperty("volume", 1.0)
        except Exception as exc:
            logging.warning("TTS init failed in worker: %s", exc)

        try:
            while True:
                text = self._queue.get()
                try:
                    if text is None:
                        return

                    logging.info("Speaking prompt: %s", text)
                    if engine is None:
                        logging.warning("TTS engine unavailable. Prompt was: %s", text)
                        continue

                    engine.say(text)
                    engine.runAndWait()
                except Exception as exc:
                    logging.warning("TTS worker failed: %s", exc)
                finally:
                    self._queue.task_done()
        finally:
            try:
                if engine is not None:
                    engine.stop()
            except Exception:
                pass


class RespawnController:
    def __init__(
        self,
        spawn_pos: Vec3,
        spawn_rot: Quat,
        checkpoints: list[RespawnCheckpoint],
    ) -> None:
        self._spawn_pos = spawn_pos
        self._spawn_rot = spawn_rot
        self._checkpoints = checkpoints
        self._checkpoint_respawn_requested = False
        self._reset_requested = False

        if KEYBOARD_AVAILABLE:
            keyboard.add_hotkey(RESPAWN_HOTKEY, self._request_checkpoint_respawn, suppress=False)
            keyboard.add_hotkey(RESET_HOTKEY, self._request_reset, suppress=False)
            logging.info(
                "Hotkeys active: %s = checkpoint respawn, %s = full reset",
                RESPAWN_HOTKEY.upper(),
                RESET_HOTKEY,
            )

    def stop(self) -> None:
        if KEYBOARD_AVAILABLE:
            keyboard.unhook_all()

    def process(self, vehicle: Vehicle, current_pos: Vec3) -> bool:
        if self._checkpoint_respawn_requested:
            self._checkpoint_respawn_requested = False
            logging.info(
                "Checkpoint respawn requested at (%.3f, %.3f, %.3f)",
                current_pos[0],
                current_pos[1],
                current_pos[2],
            )
            for checkpoint in self._checkpoints:
                if checkpoint.bounds.contains(current_pos):
                    vehicle.teleport(checkpoint.pos, checkpoint.rot)
                    logging.info("Respawned at checkpoint %s", checkpoint.name)
                    return True

            logging.info("No respawn checkpoint matched current position")

        if self._reset_requested:
            self._reset_requested = False
            vehicle.teleport(self._spawn_pos, self._spawn_rot)
            logging.info("Vehicle reset to initial spawn")
            return True

        return False

    def _request_checkpoint_respawn(self) -> None:
        self._checkpoint_respawn_requested = True

    def _request_reset(self) -> None:
        self._reset_requested = True


def build_scenario() -> tuple[Scenario, Vehicle, dict[str, Vehicle]]:
    scenario = Scenario(
        MAP_NAME,
        SCENARIO_NAME,
        description=SCENARIO_DESCRIPTION,
        authors=SCENARIO_AUTHORS,
        settings=SCENARIO_SETTINGS,
    )

    player_vehicle = Vehicle(
        PLAYER_VEHICLE_NAME,
        model=PLAYER_VEHICLE_MODEL,
        license=PLAYER_LICENSE,
    )
    scenario.add_vehicle(player_vehicle, pos=PLAYER_SPAWN, rot_quat=PLAYER_ROT_QUAT)

    npc_vehicle_map: dict[str, Vehicle] = {}
    for config in NPC_CONFIGS:
        npc_vehicle = Vehicle(config.name, model=config.model, license=config.license_text)
        scenario.add_vehicle(npc_vehicle, pos=config.pos, rot_quat=config.rot)
        npc_vehicle_map[config.name] = npc_vehicle

    add_speed_signs(scenario)
    return scenario, player_vehicle, npc_vehicle_map


def add_speed_signs(scenario: Scenario) -> None:
    for config in SIGN_CONFIGS:
        scenario.add_object(
            StaticObject(
                name=config.name,
                pos=offset_position(
                    config.anchor_pos,
                    PLAYER_ROT_QUAT,
                    config.forward_offset,
                    config.right_offset,
                    config.z_offset,
                ),
                rot_quat=quat_mul(
                    PLAYER_ROT_QUAT,
                    quat_from_z_rotation_degrees(config.yaw_offset_deg),
                ),
                scale=(1, 1, 1),
                shape=config.shape,
            )
        )


def configure_vehicles(
    player_vehicle: Vehicle,
    npc_vehicle_map: dict[str, Vehicle],
) -> None:
    player_vehicle.ai.set_mode("manual")

    for config in NPC_CONFIGS:
        npc_vehicle = npc_vehicle_map[config.name]
        npc_vehicle.ai.set_mode("disabled")
        if config.color is not None:
            npc_vehicle.set_color(config.color)


def wait_for_scenario_start(bng: BeamNGpy) -> None:
    logging.info("Waiting for the user to start the scenario in BeamNG.tech...")
    while True:
        try:
            gamestate = bng.get_gamestate()
            if (
                gamestate.get("state") == "scenario"
                and gamestate.get("scenario_state") == "running"
            ):
                logging.info("Scenario started by user.")
                return
        except Exception as exc:
            logging.warning("Error checking game state: %s", exc)

        time.sleep(SCENARIO_POLL_SECONDS)


def process_voice_prompts(
    bng: BeamNGpy,
    speech: SpeechController,
    player_pos: Vec3,
    played_prompts: set[str],
) -> None:
    for prompt in VOICE_PROMPTS:
        if prompt.name in played_prompts:
            continue
        if not within_radius(player_pos, prompt.pos, prompt.radius):
            continue

        if prompt.gui_text:
            bng.display_gui_message(prompt.gui_text)
        speech.say(prompt.text)
        played_prompts.add(prompt.name)
        logging.info("Prompt '%s' triggered at %s", prompt.name, player_pos)


def process_npc_triggers(
    bng: BeamNGpy,
    player_pos: Vec3,
    npc_vehicle_map: dict[str, Vehicle],
    triggered_npcs: set[str],
) -> None:
    for config in NPC_CONFIGS:
        if config.name in triggered_npcs:
            continue

        npc_pos = npc_vehicle_map[config.name].state["pos"]
        if not within_radius(player_pos, npc_pos, config.trigger_distance):
            continue

        npc_vehicle = npc_vehicle_map[config.name]
        npc_vehicle.ai.set_mode("traffic")
        npc_vehicle.ai.drive_in_lane(True)
        npc_vehicle.ai.set_speed(config.speed_limit_mps, mode="limit")
        triggered_npcs.add(config.name)

        logging.info("%s triggered and switched to traffic mode", config.name)
        if config.start_message:
            bng.ui.display_message(config.start_message)


def run_main_loop(
    bng: BeamNGpy,
    scenario: Scenario,
    player_vehicle: Vehicle,
    npc_vehicle_map: dict[str, Vehicle],
    respawn_controller: RespawnController,
    speech: SpeechController,
) -> None:
    played_prompts: set[str] = set()
    triggered_npcs: set[str] = set()

    while True:
        scenario.update()
        player_pos = player_vehicle.state["pos"]

        if respawn_controller.process(player_vehicle, player_pos):
            time.sleep(MAIN_LOOP_SLEEP_SECONDS)
            continue

        process_voice_prompts(bng, speech, player_pos, played_prompts)
        process_npc_triggers(bng, player_pos, npc_vehicle_map, triggered_npcs)

        time.sleep(MAIN_LOOP_SLEEP_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    bng: Optional[BeamNGpy] = None
    obs_controller = OBSController()
    speech = SpeechController()
    resolved_bng_home = resolve_beamng_home(BNG_HOME)
    respawn_controller = RespawnController(
        spawn_pos=PLAYER_SPAWN,
        spawn_rot=PLAYER_ROT_QUAT,
        checkpoints=RESPAWN_CHECKPOINTS,
    )

    try:
        obs_controller.connect()
        logging.info("Using BeamNG home: %s", resolved_bng_home)
        if BNG_USER_PATH:
            logging.info("Using BeamNG user path override: %s", BNG_USER_PATH)
            persist_beamng_display_settings(BNG_USER_PATH)
        bng = BeamNGpy("localhost", 64256, home=resolved_bng_home, user=BNG_USER_PATH)
        bng.open(None, "-gfx", GRAPHICS_BACKEND)
        logging.info("Connected to BeamNG.tech")
        try:
            actual_user_path = bng.system.get_environment_paths().get("user")
        except Exception as exc:
            logging.warning("Failed to query BeamNG user path: %s", exc)
            actual_user_path = None

        if actual_user_path and actual_user_path != BNG_USER_PATH:
            persist_beamng_display_settings(actual_user_path)
        apply_beamng_display_settings(bng)

        scenario, player_vehicle, npc_vehicle_map = build_scenario()
        scenario.make(bng)
        logging.info("Scenario created")

        bng.load_scenario(scenario)
        logging.info("Scenario loaded")

        configure_vehicles(player_vehicle, npc_vehicle_map)
        logging.info("Vehicle setup complete")

        bng.camera.set_player_mode(player_vehicle, "driver", {"fov": PLAYER_CAMERA_FOV})
        logging.info("Camera set to interior view")
        obs_controller.setup_single_window_capture()

        wait_for_scenario_start(bng)
        scenario.update()
        enable_beamng_fps_overlay(bng)
        obs_controller.start_recording()

        speech.start()
        run_main_loop(
            bng=bng,
            scenario=scenario,
            player_vehicle=player_vehicle,
            npc_vehicle_map=npc_vehicle_map,
            respawn_controller=respawn_controller,
            speech=speech,
        )
    except Exception as exc:
        logging.error("An unexpected error occurred: %s", exc, exc_info=True)
    finally:
        obs_controller.stop_recording()
        speech.stop()
        respawn_controller.stop()
        if bng is not None:
            bng.close()


if __name__ == "__main__":
    main()
