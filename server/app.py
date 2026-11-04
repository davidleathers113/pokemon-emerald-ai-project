#!/usr/bin/env python3
"""
Fixed Pokemon Emerald server - headless FastAPI server
"""

# Standard library imports
import asyncio
import base64
import contextlib
import datetime
import glob
import io
import json
import logging
import math
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from urllib.parse import quote_plus

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import hashlib

# Add parent directory to path for local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local application imports — emulator imported conditionally in setup_environment()
from utils.anticheat import AntiCheatTracker
from utils.json_utils import normalize_replan_edits
from utils.llm_provider_ui import infer_llm_provider_family

# Set up logging - reduced verbosity for multiprocess mode
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Game type: "emerald" (GBA, default) or "red" (Game Boy)
game_type = os.environ.get("GAME_TYPE", "emerald").lower()

# Global state
env = None
anticheat_tracker = None  # AntiCheat tracker for submission logging
step_counter = 0  # Track steps for submission logging
last_action_time = None  # Track time of last action for decision time calculation
running = True
step_count = 0
direct_objectives_sequence = None
direct_objectives_start_index = 0
direct_objectives_battling_start_index = 0
direct_objectives_manager = None
current_run_dir = None  # Timestamped directory for this execution run
agent_step_count = 0  # Track agent steps separately from frame steps
debug_state_enabled = False  # Save game state JSON on each get_game_state call (--debug-state)
_debug_state_counter = 0  # Step index for debug state entries
_debug_state_log = []  # Accumulates all debug state snapshots (written to debug_states.json)
current_obs = None
fps = 80
# Local addition: game speed. GAME_FPS while nothing is queued (the GBA runs at ~59.7,
# so 60 looks normal to a human watching); TURBO_FPS while the harness is executing a
# button sequence, walk or dialogue advance, since a model call (tens of seconds) waits
# on it. TURBO_FPS=0 means as fast as the CPU allows. Game logic is frame-based, so
# speed doesn't change what happens in the game; the RTC follows wall time either way.
GAME_FPS = float(os.environ.get("GAME_FPS", "60"))
TURBO_FPS = float(os.environ.get("TURBO_FPS", "480"))
# The viewer frame (PNG for /stream) is encoded at most this often, not per frame:
# encoding costs ~3x a frame of emulation, which would otherwise eat most of turbo.
VIEWER_FRAME_INTERVAL_S = 1.0 / 40
_last_viewer_frame_time = 0.0


def _is_simplest_scaffold() -> bool:
    """Return True when the run is using the simplest scaffold."""
    return os.environ.get("EXCLUDE_BUILTIN_SUBAGENTS") == "1"


def _build_story_planning_objective():
    """Create scaffold-aware planning guidance when story objectives run out."""
    from agents.objectives import DirectObjective

    if _is_simplest_scaffold():
        return DirectObjective(
            id="autonomous_01_plan_more_objectives",
            description=(
                "You have reached the end of the story objectives currently available. "
                "Create more objectives to keep progressing through the game."
            ),
            action_type="create_new_objectives",
            category="story",
            target_location=None,
            navigation_hint=(
                "Review your current progress and create new objectives."
            ),
            completion_condition="objectives_created",
            priority=1,
        )

    return DirectObjective(
        id="autonomous_01_plan_more_objectives",
        description=(
            "The current story sequence is complete. Use subagent_plan_objectives "
            "to create more story objectives based on your progress and the walkthrough."
        ),
        action_type="create_new_objectives",
        category="story",
        target_location=None,
        navigation_hint=(
            "Call subagent_plan_objectives with a brief reason explaining that the "
            "current story objectives are exhausted and more objectives are needed."
        ),
        completion_condition="objectives_created",
        priority=1,
    )


def _format_interaction_type_for_ui(interaction_type: str) -> str:
    """Normalize orchestrator labels for UI readability.

    Backends log interaction_type as `<backend>_<module_name>`. For orchestrator
    entries we prefer `<scaffold>_<backend>_orchestrator` in the UI.
    """
    if not interaction_type:
        return "unknown"

    backend_prefixes = ("gemini_", "openai_", "openrouter_", "anthropic_", "vertex_")
    for prefix in backend_prefixes:
        if interaction_type.startswith(prefix) and interaction_type.endswith("_orchestrator"):
            backend = prefix[:-1]
            module_name = interaction_type[len(prefix):]
            if module_name:
                base = module_name[:-13]  # len("_orchestrator") == 13
                return f"{base}_{backend}_orchestrator"
    return interaction_type


def _provider_family_for_llm_log_entry(entry: dict) -> str:
    """UI color bucket from raw log line (handles OpenRouter via model slug)."""
    raw_type = entry.get("interaction_type") or ""
    mi = entry.get("model_info") or {}
    model_name = mi.get("model") or ""
    meta = entry.get("metadata") or {}
    backend = meta.get("backend") or mi.get("backend")
    return infer_llm_provider_family(raw_type, model_name, backend)


# Performance monitoring
last_fps_log = time.time()
frame_count_since_log = 0
action_queue = []  # Queue for multi-action sequences (now stores dicts with timing info)
current_action = None  # Current action being held
action_frames_remaining = 0  # Frames left to hold current action
release_frames_remaining = 0  # Frames left to wait after release
current_action_release_delay = 0  # Release delay for current action

### ACTION TIMING SYSTEM ###
# LLM-controlled speed presets for flexible action timing
SPEED_PRESETS = {
    "fast": {"hold": 6, "release": 3},  # 9 frames total - dialogue, menus
    "normal": {"hold": 10, "release": 8},  # 18 frames total - movement (2x faster than old default!)
    "slow": {"hold": 16, "release": 16},  # 32 frames total - careful inputs
}
DEFAULT_SPEED = "normal"

# Legacy constants for backward compatibility
ACTION_HOLD_FRAMES = SPEED_PRESETS["normal"]["hold"]
ACTION_RELEASE_DELAY = SPEED_PRESETS["normal"]["release"]

### MOVEMENT FEEDBACK (local addition) ###
# A small model kept pressing UP into trees for nine decisions because nothing told it
# the presses did nothing. Every press_buttons / navigate_to / go_to_exit call becomes a
# "sequence"; the game loop compares the player's tile and map around each direction
# press, drops the rest of the sequence when a press is BLOCKED (or the walk is
# interrupted by a battle or a text box), and leaves a report the endpoint returns.
action_queue_lock = threading.Lock()  # guards rebuilding action_queue against appends
_sequence_counter = 0
sequence_reports: Dict[int, Dict[str, Any]] = {}
SEQUENCE_REPORTS_MAX = 64
_active_action_data = None  # queue entry whose hold/release is running (for bookkeeping)
_move_check = None  # in-flight checked direction press: button, before snapshot, seq
_move_check_settle_frames = 0
_start_wait_frames = 0
_pending_before_snapshot = None
# The last BLOCKED press; an identical press from the same tile is refused, not run.
last_blocked_press: Optional[Dict[str, Any]] = None
# Macro waits in progress (battle play-outs, menu waits, train). The game loop runs at
# TURBO_FPS while any is open: between presses nothing is queued, so battle animations
# and text played at GAME_FPS while a macro (and the model behind it) only waited on
# them. Every such wait is a frame budget, so turbo doesn't shorten it in game time.
_turbo_macros = 0


@contextlib.contextmanager
def _turbo():
    global _turbo_macros
    _turbo_macros += 1
    try:
        yield
    finally:
        _turbo_macros -= 1


# Battle messages as the game loop saw them, one entry per message: (frame number, text).
# The play-out used to sample the message window from the event loop every 30 ms; at
# turbo speed a message that doesn't wait for A ("Go! TORCHIC!") prints and is replaced
# between two samples (seen in test_battle_flow's auto-advance test once macro waits ran
# at turbo), so the game loop records each one as it prints.
battle_message_log: "deque" = None  # created on first use (collections.deque, maxlen 64)
_frames_run = 0
_last_battle_message_raw = b""


def _record_battle_message() -> None:
    """Called under memory_lock after every emulated frame; cheap outside battle text."""
    global battle_message_log, _frames_run, _last_battle_message_raw
    _frames_run += 1
    if game_type != "emerald" or env is None or getattr(env, "memory_reader", None) is None:
        return
    try:
        from server import menu_state as ms

        reader = env.memory_reader
        if not reader._read_u8(ms.S_TEXT_PRINTERS + ms.B_WIN_MSG * ms.TEXT_PRINTER_SIZE + 0x1B):
            return  # the battle message window isn't printing
        from server.field_control import G_MAIN_IN_BATTLE_BYTE, IN_BATTLE_MASK

        if not reader._read_u8(G_MAIN_IN_BATTLE_BYTE) & IN_BATTLE_MASK:
            return
        raw = bytes(reader._read_bytes(ms.G_DISPLAYED_STRING_BATTLE, ms.DISPLAYED_STRING_BATTLE_SIZE))
        if raw == _last_battle_message_raw:
            return
        _last_battle_message_raw = raw
        if battle_message_log is None:
            from collections import deque

            battle_message_log = deque(maxlen=64)
        battle_message_log.append((_frames_run, ms.battle_text(reader)))
    except Exception as e:  # a missed quote must never stop the game loop
        logger.debug(f"battle message record failed: {e}")


def _battle_messages_since(frame: int) -> List[Tuple[int, str]]:
    return [(at, text) for at, text in list(battle_message_log or ()) if at > frame and text]

# Video recording state
video_writer = None
video_recording = False
video_filename = ""
video_frame_counter = 0
video_frame_skip = 4  # Record every 4th frame (120/4 = 30 FPS)

# Playwright WebUI recording state
playwright_recording = False       # signal flag: set False to stop the recording thread
playwright_thread = None           # background thread running the async recording loop
playwright_video_path = None       # final .mp4 path set when recording finishes

# Frame cache for separate frame server
# Use cache directory instead of /tmp
# Note: CACHE_DIR is now dynamic based on run_id - use get_cache_directory() when needed
# For frame cache, we'll initialize it dynamically
FRAME_CACHE_FILE = None  # Will be set dynamically based on run_id
frame_cache_counter = 0
frame_cache_skip_frames = 30  # Only update cache every 30 frames (4x/sec at 120 FPS)

# Map stitcher performance toggle
# Set to False when using ground truth porymap data to avoid expensive updates
ENABLE_MAP_STITCHER = False  # Disabled - using porymap ground truth instead

# State endpoint cache - cache map data by location to avoid expensive regeneration
_state_cache = {"location": None, "map_data": None, "portal_data": None, "timestamp": 0}
_state_cache_ttl = 5.0  # Cache for 5 seconds per location

# Server runs headless - display handled by client

# Threading locks for thread safety
obs_lock = threading.Lock()
step_lock = threading.Lock()
# The emulator lock: every call into the mGBA core (stepping, save/load state, memory
# reads) must hold it. The game loop steps on the main thread while endpoints and the
# milestone thread run on others, and cffi releases the GIL inside C calls, so an
# unguarded load_state can rewrite the core in the middle of run_frame. Reentrant so a
# helper that locks can be called from code that already holds it.
memory_lock = threading.RLock()

# time.monotonic() of the last emulated frame; /health reports unhealthy when it goes
# stale (game loop dead or stuck in the core) instead of answering from a live web thread.
last_frame_monotonic = None
HEALTH_MAX_FRAME_AGE_S = 10.0


# WebSocket connection manager for frame streaming
class ConnectionManager:
    """Manages WebSocket connections for real-time frame streaming"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = threading.Lock()
        self.latest_frame = None  # Shared frame data (thread-safe)
        self.last_sent_frame_count = {}  # Track last sent frame per connection

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self.lock:
            self.active_connections.append(websocket)
            self.last_sent_frame_count[id(websocket)] = 0
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                ws_id = id(websocket)
                if ws_id in self.last_sent_frame_count:
                    del self.last_sent_frame_count[ws_id]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_latest_frame(self, websocket: WebSocket):
        """Send latest frame to a specific connection if new frame available"""
        if not self.latest_frame:
            return

        ws_id = id(websocket)
        frame_count = self.latest_frame.get("frame_count", 0)

        # Only send if this is a new frame for this connection
        if ws_id in self.last_sent_frame_count and self.last_sent_frame_count[ws_id] >= frame_count:
            return

        try:
            message = json.dumps(self.latest_frame)
            await websocket.send_text(message)
            self.last_sent_frame_count[ws_id] = frame_count
        except Exception as e:
            logger.debug(f"Failed to send frame: {e}")
            raise


frame_manager = ConnectionManager()

# Background milestone processing
state_update_thread = None
state_update_running = False

# Button mapping removed - handled by client


# Video recording functions
def init_video_recording(record_enabled=False):
    """Initialize video recording if enabled"""
    global video_writer, video_recording, video_filename, fps, video_frame_skip

    if not record_enabled:
        return

    try:
        # Save directly to run_data/end_state/videos/ to avoid copy corruption
        from utils.data_persistence.run_data_manager import get_run_data_manager
        run_manager = get_run_data_manager()
        if run_manager:
            videos_dir = run_manager.run_dir / "end_state" / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_filename = str(videos_dir / f"{run_manager.run_id}.mp4")
        else:
            # Fallback to cwd if no run manager (e.g. early init)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"pokegent_recording_{timestamp}.mp4"

        # Video settings (GBA resolution is 240x160)
        # Record at 30 FPS (skip every 4th frame from 120 FPS emulator)
        recording_fps = fps / video_frame_skip  # 120 / 4 = 30 FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_filename, fourcc, float(recording_fps), (240, 160))

        if video_writer.isOpened():
            video_recording = True
            print(
                f"📹 Video recording started: {video_filename} at {recording_fps:.0f} FPS (recording every {video_frame_skip} frames)"
            )
        else:
            print("❌ Failed to initialize video recording")
            video_writer = None

    except Exception as e:
        print(f"❌ Video recording initialization error: {e}")
        video_writer = None


def update_frame_cache(screenshot):
    """Update frame cache for WebSocket streaming - NO FILE I/O!

    Stores frames in memory for WebSocket clients to consume.
    Performance: Updates every 2 frames = 40 FPS stream rate at 80 game FPS.
    """
    global frame_cache_counter, _last_viewer_frame_time

    if screenshot is None:
        return

    # Increment counter on every call
    frame_cache_counter += 1

    # At most 40 viewer frames per second of wall time, whatever the game speed
    now = time.monotonic()
    if now - _last_viewer_frame_time < VIEWER_FRAME_INTERVAL_S:
        return
    _last_viewer_frame_time = now

    try:
        # Convert screenshot to base64
        if hasattr(screenshot, "save"):  # PIL image
            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
        elif isinstance(screenshot, np.ndarray):  # Numpy array
            pil_image = Image.fromarray(screenshot)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
        else:
            return

        # Store frame in memory for WebSocket clients (thread-safe)
        frame_manager.latest_frame = {
            "frame_data": img_str,
            "frame_count": frame_cache_counter,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.debug(f"Frame update failed: {e}")


def record_frame(screenshot):
    """Record frame to video if recording is enabled with frame skipping"""
    global video_writer, video_recording, video_frame_counter, video_frame_skip

    if not video_recording or video_writer is None or screenshot is None:
        return

    # Increment frame counter
    video_frame_counter += 1

    # Only record every Nth frame based on frame skip
    if video_frame_counter % video_frame_skip != 0:
        return

    try:
        # Convert PIL image to OpenCV format
        if hasattr(screenshot, "save"):  # PIL image
            # Convert PIL to numpy array
            frame_array = np.array(screenshot)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        elif isinstance(screenshot, np.ndarray):  # Already numpy array
            # Convert RGB to BGR for OpenCV if needed
            if screenshot.shape[2] == 3:  # RGB
                frame_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = screenshot
            video_writer.write(frame_bgr)

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Video recording frame error: {e}")


def cleanup_video_recording():
    """Clean up video recording resources"""
    global video_writer, video_recording

    if video_recording and video_writer is not None:
        try:
            video_writer.release()
            print(f"📹 Video recording saved: {video_filename}")
        except Exception as e:
            print(f"❌ Error saving video recording: {e}")
        finally:
            video_writer = None
            video_recording = False


# Milestone tracking is now handled by the emulator

# FastAPI app
app = FastAPI(
    title="PokeAgent Challenge",
    description="Streamer display FastAPI endpoints",
    version="3.0.0-preview",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models for API requests and responses
class ActionRequest(BaseModel):
    buttons: List[str] = Field(
        default_factory=list
    )  # List of button names: A, B, SELECT, START, UP, DOWN, LEFT, RIGHT, WAIT
    speed: Optional[str] = "normal"  # Action speed: "fast", "normal", or "slow"
    hold_frames: Optional[int] = None  # Optional explicit hold duration (overrides speed preset)
    release_frames: Optional[int] = None  # Optional explicit release duration (overrides speed preset)
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Judge each direction press and stop the call's sequence when one is blocked.
    check_movement: bool = False


class GameStateResponse(BaseModel):
    screenshot_base64: str
    step_number: int
    resolution: list  # [width, height]
    status: str


class ComprehensiveStateResponse(BaseModel):
    visual: dict
    player: dict
    game: dict
    map: dict
    milestones: dict = {}
    location_connections: dict = {}  # Add location connections for portal display
    step_number: int
    status: str
    action_queue_length: int = 0


def periodic_milestone_updater():
    """Lightweight background thread that only updates milestones occasionally"""
    global state_update_running

    last_milestone_update = 0

    while state_update_running and running:
        try:
            current_time = time.time()

            # Update milestones only every 5 seconds (much less frequent)
            if current_time - last_milestone_update >= 5.0:
                if env and env.memory_reader:
                    try:
                        # Use lightweight state for milestone updates only
                        with memory_lock:
                            party = env.get_party_pokemon() or []
                            location = env.get_location()
                            badges = []
                            if hasattr(env, 'memory_reader') and env.memory_reader and hasattr(env.memory_reader, 'read_badges'):
                                try:
                                    badges = env.memory_reader.read_badges() or []
                                except Exception:
                                    badges = []
                            basic_state = {
                                "player": {
                                    "money": env.get_money(),
                                    "party_size": len(env.get_party_pokemon() or []),
                                    "position": env.get_coordinates(),
                                },
                                "map": {"location": env.get_location()},
                            }
                            env.check_and_update_milestones(basic_state, agent_step_count=agent_step_count)
                        last_milestone_update = current_time
                        logger.debug("Lightweight milestone update completed")
                    except Exception as e:
                        logger.debug(f"Milestone update failed: {e}")

            # Sleep for 1 second between checks
            time.sleep(1.0)

        except Exception as e:
            logger.error(f"Error in milestone updater: {e}")
            time.sleep(5.0)  # Wait longer on error


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running, state_update_running, video_recording, video_filename

    # Prevent multiple signal handlers from running simultaneously
    if not running:
        return

    print(f"\nReceived signal {signum}, shutting down gracefully...")
    running = False
    state_update_running = False

    # IMPORTANT: Finalize run data BEFORE cleanup
    # Check video_recording flag BEFORE cleanup_video_recording() resets it
    was_recording = video_recording

    try:
        from utils.data_persistence.run_data_manager import get_run_data_manager
        from utils.data_persistence.llm_logger import get_llm_logger

        run_manager = get_run_data_manager()
        if run_manager:
            print("📦 Finalizing run data...")

            # Get final metrics from LLM logger
            llm_logger = get_llm_logger()
            final_metrics = llm_logger.get_cumulative_metrics() if llm_logger else None

            # Save end-state snapshot (ensures all data is saved)
            run_manager.save_end_state_snapshot()

            # Copy all data to run_data
            logger.info(f"🔍 [DEBUG] Finalizing run data - run_manager: {run_manager is not None}")
            if llm_logger:
                logger.info(f"🔍 [DEBUG] LLM logger available, log_file: {llm_logger.log_file}")
                # Verify LLM log file exists before copying
                if os.path.exists(llm_logger.log_file):
                    logger.info(f"🔍 [DEBUG] LLM log file exists, copying...")
                    run_manager.copy_llm_traces(llm_logger.log_file)
                    logger.info(f"🔍 [DEBUG] Copied LLM traces from: {llm_logger.log_file}")
                else:
                    logger.warning(f"🔍 [DEBUG] LLM log file not found: {llm_logger.log_file}")
                    logger.warning(f"🔍 [DEBUG] Current working directory: {os.getcwd()}")
                    # Try to find the log file by pattern
                    import glob

                    log_pattern = f"llm_logs/llm_log_{run_manager.run_id.split('_', 1)[1] if '_' in run_manager.run_id else '*'}*.jsonl"
                    logger.info(f"🔍 [DEBUG] Searching for log files with pattern: {log_pattern}")
                    log_files = glob.glob(log_pattern)
                    logger.info(f"🔍 [DEBUG] Found {len(log_files)} log files: {log_files}")
                    if log_files:
                        # Use the most recent one
                        log_file = max(log_files, key=os.path.getmtime)
                        logger.info(f"🔍 [DEBUG] Using most recent log file: {log_file}")
                        run_manager.copy_llm_traces(log_file)
                        logger.info(f"🔍 [DEBUG] Found and copied LLM traces from: {log_file}")
            else:
                logger.warning(f"🔍 [DEBUG] LLM logger is None - cannot copy LLM traces")

            if os.environ.get("POKEAGENT_CLI_MODE") != "1":
                run_manager.copy_objectives()
                run_manager.copy_memory()

            # Sync trajectories from cache to run_data before finalizing
            run_manager.sync_trajectories_to_run_data()

            # Copy frame_cache to end_state
            run_manager.copy_frame_cache()

            # Copy video if recording was enabled (check flag BEFORE cleanup)
            logger.info(f"🔍 [DEBUG] Video recording flag: {was_recording}, video_filename: {video_filename}")
            run_manager.copy_video_recording(record_enabled=was_recording)

            # Finalize with metrics
            run_manager.finalize_run(final_metrics=final_metrics)
            print(f"✅ Run data finalized: {run_manager.get_run_directory()}")
    except Exception as e:
        logger.error(f"❌ Error during run data finalization: {e}", exc_info=True)

    # Cleanup video recording AFTER copying (so file is still available)
    cleanup_video_recording()

    if env:
        env.stop()

    sys.exit(0)


def setup_environment(skip_initial_state=False):
    """Initialize the emulator"""
    global env, current_obs, anticheat_tracker

    try:
        if game_type == "red":
            from pokemon_red_env.red_emulator import RedEmulator
            rom_path = "PokemonRed-GBC/pokered.gbc"
            if not os.path.exists(rom_path):
                raise RuntimeError(f"ROM not found at {rom_path}")
            env = RedEmulator(rom_path=rom_path)
        else:
            from pokemon_env.emulator import EmeraldEmulator
            rom_path = "Emerald-GBAdvance/rom.gba"
            if not os.path.exists(rom_path):
                raise RuntimeError(f"ROM not found at {rom_path}")
            env = EmeraldEmulator(rom_path=rom_path)
        env.initialize()

        # Initialize AntiCheat tracker for submission logging
        anticheat_tracker = AntiCheatTracker()
        anticheat_tracker.initialize_submission_log("SERVER_MODE")
        print("AntiCheat tracker initialized for submission logging")

        # Mark GAME_RUNNING milestone as completed at startup
        # But defer the expensive state logging - it will happen on first state request
        if not skip_initial_state:
            try:
                env.milestone_tracker.mark_completed("GAME_RUNNING")
                print("GAME_RUNNING milestone marked - initial state logging deferred")
            except Exception as e:
                print(f"Warning: Could not mark initial milestone: {e}")

        screenshot = env.get_screenshot()
        if screenshot:
            with obs_lock:
                current_obs = np.array(screenshot)
        else:
            with obs_lock:
                current_obs = np.zeros((env.height, env.width, 3), dtype=np.uint8)

        print("Emulator initialized successfully!")
        return True

    except Exception as e:
        print(f"Failed to initialize emulator: {e}")
        return False


def handle_input(manual_mode=False):
    """Handle input - server runs headless, no input handling needed"""
    # Server always runs headless - input handled by client via HTTP API
    return True, []


def step_environment(actions_pressed):
    """Take a step in the environment with optimized locking for better performance"""
    global current_obs, last_frame_monotonic

    # Debug: print what actions are being sent to emulator
    # if actions_pressed:
    # print( Stepping emulator with actions: {actions_pressed}")

    # Only use memory_lock for the essential emulator step
    with memory_lock:
        env.run_frame_with_buttons(actions_pressed)
        last_frame_monotonic = time.monotonic()
        _record_battle_message()

        # Do lightweight area transition detection inside the lock
        if hasattr(env, "memory_reader") and env.memory_reader:
            try:
                transition_detected = env.memory_reader._check_area_transition()
                if transition_detected:
                    logger.info("Area transition detected")
                    env.memory_reader.invalidate_map_cache()
                    if hasattr(env.memory_reader, "_cached_behaviors"):
                        env.memory_reader._cached_behaviors = None
                    if hasattr(env.memory_reader, "_cached_behaviors_map_key"):
                        env.memory_reader._cached_behaviors_map_key = None
                    # Set flag to trigger map stitcher update outside the lock
                    env.memory_reader._area_transition_detected = True
            except Exception as e:
                logger.warning(f"Area transition check failed: {e}")

    # Update screenshot outside the memory lock to reduce contention
    try:
        screenshot = env.get_screenshot()
        if screenshot:
            if not playwright_recording:
                record_frame(screenshot)
            update_frame_cache(screenshot)  # Update frame cache for separate frame server
            with obs_lock:
                current_obs = np.array(screenshot)

            # Update map stitcher on position changes (lightweight approach)
            # This ensures map data stays current as player moves
            # DISABLED: Using porymap ground truth instead for better performance
            if ENABLE_MAP_STITCHER and hasattr(env, "memory_reader") and env.memory_reader:
                try:
                    # Check if player position has changed
                    should_update = False

                    # Get current player coordinates and map info
                    current_coords = env.memory_reader.read_coordinates()
                    current_map_info = env.memory_reader._read_current_map()

                    # Initialize tracking variables if needed
                    if not hasattr(env, "_last_player_coords"):
                        env._last_player_coords = None
                        env._last_map_info = None

                    # Check for position changes
                    if current_coords != env._last_player_coords or current_map_info != env._last_map_info:
                        should_update = True
                        env._last_player_coords = current_coords
                        env._last_map_info = current_map_info
                        logger.debug(f"Position change detected: {current_coords}, map: {current_map_info}")
                        logger.debug(
                            f"Map stitcher update triggered by position change: {current_coords}, map: {current_map_info}"
                        )

                    # Always update on area transitions (already detected above)
                    if (
                        hasattr(env.memory_reader, "_area_transition_detected")
                        and env.memory_reader._area_transition_detected
                    ):
                        should_update = True
                        env.memory_reader._area_transition_detected = False  # Reset flag
                        logger.debug("Map stitcher update triggered by area transition")

                    # Update map stitcher directly when position changes
                    if should_update:
                        # @TODO should do location change warps here too
                        logger.debug("Triggering map stitcher update for position change")
                        # Call map stitcher update directly without full map reading
                        tiles = env.memory_reader.read_map_around_player(radius=7)
                        if tiles:
                            logger.debug(f"Got {len(tiles)} tiles, updating map stitcher")
                            state = {"map": {}}  # Basic state for stitcher
                            env.memory_reader._update_map_stitcher(tiles, state)
                            logger.debug("Map stitcher updated for position change")
                            logger.debug("Map stitcher update completed")
                        else:
                            logger.debug("No tiles found for map stitcher update")

                except Exception as e:
                    logger.error(f"Failed to update map stitcher during movement: {e}")
    except Exception as e:
        logger.warning(f"Error updating screenshot: {e}")


def update_display(manual_mode=False):
    """Update display - server runs headless, no display update needed"""
    # Server runs headless - display handled by client
    pass


def draw_info_overlay():
    """Draw info overlay - server runs headless, no overlay needed"""
    # Server runs headless - overlay handled by client
    pass


def save_screenshot():
    """Save current screenshot"""
    global current_obs

    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None

    if obs_copy is not None:
        timestamp = int(time.time())
        filename = f"simple_test_screenshot_{timestamp}.png"
        img = Image.fromarray(obs_copy)
        img.save(filename)
        print(f"Screenshot saved: {filename}")


def reset_game():
    """Reset the game and all milestones"""
    global env, step_count

    print("Resetting game and milestones...")
    with step_lock, memory_lock:
        env.initialize()
        env.milestone_tracker.reset_all()  # Reset all milestones
        step_count = 0
    print("Game and milestone reset complete")


DIRECTION_BUTTONS = ("UP", "DOWN", "LEFT", "RIGHT")


def _register_sequence(buttons: List[str], check_movement: bool, source: Optional[str]) -> int:
    """Create the report the game loop fills in while this call's buttons run."""
    global _sequence_counter
    with action_queue_lock:
        _sequence_counter += 1
        seq_id = _sequence_counter
        sequence_reports[seq_id] = {
            "seq_id": seq_id,
            "buttons": list(buttons),
            "total": len(buttons),
            "check_movement": bool(check_movement),
            "source": source,
            "executed": 0,
            "skipped": 0,
            "blocked": None,
            "interrupted": None,
            "last_direction": None,
            "done": not buttons,
        }
        while len(sequence_reports) > SEQUENCE_REPORTS_MAX:
            sequence_reports.pop(min(sequence_reports))
    return seq_id


def _report_for(action_data) -> Optional[Dict[str, Any]]:
    if isinstance(action_data, dict) and action_data.get("seq_id") is not None:
        return sequence_reports.get(action_data["seq_id"])
    return None


def _is_checked_direction(action_data) -> bool:
    return (
        game_type == "emerald"
        and isinstance(action_data, dict)
        and bool(action_data.get("check_movement"))
        and action_data.get("button") in DIRECTION_BUTTONS
    )


def _field_snapshot():
    """Overworld snapshot (server.field_control.FieldSnapshot), or None if unreadable."""
    if env is None or getattr(env, "memory_reader", None) is None:
        return None
    try:
        from server.field_control import read_field_snapshot

        with memory_lock:
            return read_field_snapshot(env.memory_reader)
    except Exception as e:
        logger.debug(f"Field snapshot failed: {e}")
        return None


def _refuse_during_battle(tool: str) -> Optional[Dict[str, Any]]:
    """An error result for a walking tool called mid-battle, else None.

    Seen live: between two of a trainer's Pokémon the model decided the battle was
    over (the prompt said BATTLE MODE) and called navigate_to and go_to_exit. Their
    direction presses only moved the battle menu cursor, and each reported "did not
    leave the map", which says nothing about why.
    """
    try:
        with memory_lock:
            busy = env.memory_reader.is_in_battle()
    except Exception:
        return None  # can't tell: let the walk run and report what happened
    if not busy:
        return None
    message = (
        f"{tool} can't walk now: a battle is still going on (game memory says so), so "
        "direction presses would only move the battle menu cursor. Finish the battle "
        "with use_move (the battle section names the best move) or press_buttons, then walk."
    )
    return {"success": False, "error": message, "summary": message}


def _location_name() -> str:
    try:
        with memory_lock:
            return env.memory_reader.read_location()
    except Exception:
        return "this map"


def _wait_for_player_to_stand() -> bool:
    """True while the next checked direction press should wait for a step to finish.

    Starting a press while the player is still mid-step (or mid ledge jump) both loses
    the press (it ends before the jump does) and makes its outcome unjudgeable.
    """
    global _start_wait_frames, _pending_before_snapshot
    head = action_queue[0] if action_queue else None
    if not _is_checked_direction(head):
        _start_wait_frames = 0
        return False
    from server.field_control import SETTLE_MAX_FRAMES

    snap = _field_snapshot()
    if snap is not None and snap.in_transition and _start_wait_frames < SETTLE_MAX_FRAMES:
        _start_wait_frames += 1
        return True
    _start_wait_frames = 0
    _pending_before_snapshot = snap
    return False


def _begin_action(action_data) -> None:
    global _active_action_data, _move_check, _pending_before_snapshot
    _active_action_data = action_data
    report = _report_for(action_data)
    if report is not None:
        report["executed"] += 1
    if _is_checked_direction(action_data):
        _move_check = {
            "button": action_data["button"],
            "before": _pending_before_snapshot or _field_snapshot(),
            "seq_id": action_data.get("seq_id"),
            "previous": report.get("last_direction") if report else None,
            "released": False,
        }
    _pending_before_snapshot = None


def _on_action_released() -> None:
    """The current action's hold and release frames are over."""
    global _active_action_data, _move_check_settle_frames
    if _move_check is not None and not _move_check["released"]:
        # Judged by _settle_move_check once the player stands still again.
        _move_check["released"] = True
        _move_check_settle_frames = 0
        return
    data, _active_action_data = _active_action_data, None
    _finish_action(data)


def _settle_move_check() -> None:
    """Called once per frame after a checked direction press; judges it when the step ends."""
    global _move_check, _move_check_settle_frames, _active_action_data
    from server.field_control import SETTLE_MAX_FRAMES

    after = _field_snapshot()
    if after is not None and after.in_transition and _move_check_settle_frames < SETTLE_MAX_FRAMES:
        _move_check_settle_frames += 1
        return
    check, _move_check = _move_check, None
    data, _active_action_data = _active_action_data, None
    try:
        _judge_move(check, after)
    except Exception as e:
        logger.warning(f"Movement check failed: {e}")
    _finish_action(data)


def _drop_rest_of_sequence(seq_id: int) -> int:
    """Remove this sequence's still-queued presses; returns how many were dropped."""
    global action_queue
    with action_queue_lock:
        kept = [a for a in action_queue if not (isinstance(a, dict) and a.get("seq_id") == seq_id)]
        dropped = len(action_queue) - len(kept)
        action_queue[:] = kept
    # Skipped presses never happened: take them out of the recent-actions list the
    # state text is built from (it also expects an end_pos on every completed press).
    recent_button_presses[:] = [
        entry for entry in recent_button_presses
        if not (entry.get("seq_id") == seq_id and not entry.get("completed"))
    ]
    return dropped


def _judge_move(check: Dict[str, Any], after) -> None:
    global last_blocked_press
    from server.field_control import (
        BLOCKED,
        blocked_target,
        interruption_cause,
        judge_direction_press,
        npc_at,
    )

    button, before, seq_id = check["button"], check["before"], check["seq_id"]
    report = sequence_reports.get(seq_id)
    if report is not None:
        report["last_direction"] = button
    if before is None or after is None:
        return
    verdict = judge_direction_press(button, before, after, check["previous"])
    if verdict is None:
        return
    skipped = _drop_rest_of_sequence(seq_id)
    total = report["total"] if report else skipped + 1
    location = _location_name()
    if verdict == BLOCKED:
        tx, ty = blocked_target(button, before)
        with memory_lock:
            occupied = npc_at(env.memory_reader, tx, ty)
        what = "is occupied by someone" if occupied else "is not walkable"
        message = (
            f"BLOCKED moving {button} at ({before.x},{before.y}) on {location}: ({tx},{ty}) {what}. "
            f"{skipped} of {total} presses skipped."
        )
        if not occupied:
            try:
                from server.game_tools import exit_hint

                hint = exit_hint(location, button)
            except Exception:
                hint = ""
            if hint:
                message += f" {hint}"
        # People walk away, so only a blocking tile makes the same press pointless.
        last_blocked_press = None if occupied else {
            "button": button,
            "x": before.x,
            "y": before.y,
            "map_id": before.map_id,
            "message": message,
        }
        if report is not None:
            report["blocked"] = message
            report["blocked_tile"] = (tx, ty)
            report["skipped"] = skipped
            if occupied:
                report["blocked_by_person"] = (tx, ty)
        logger.info(f"🧱 {message}")
    elif not skipped:
        # The interruption came with the last press: nothing was skipped, but say why
        # the walk ended where it did (e.g. a wild battle started on the last step).
        if report is not None:
            report["note"] = interruption_cause(after)
    else:
        message = (
            f"STOPPED at ({after.x},{after.y}) on {location}: {interruption_cause(after)}. "
            f"{skipped} of {total} presses skipped."
        )
        if report is not None:
            report["interrupted"] = message
            report["skipped"] = skipped
        logger.warning(f"⏹️ {message} (before={before}, after={after})")


def _finish_action(action_data) -> None:
    report = _report_for(action_data)
    if report is None:
        return
    seq_id = report["seq_id"]
    if not any(isinstance(a, dict) and a.get("seq_id") == seq_id for a in list(action_queue)):
        report["done"] = True


def game_loop(manual_mode=False):
    """Main game loop - runs in main thread, always headless"""
    global running, step_count

    print("Starting headless game loop...")
    next_frame_deadline = time.monotonic()

    while running:
        # Handle input
        should_continue, actions_pressed = handle_input(manual_mode)
        if not should_continue:
            break

        # In server mode, handle action queue with proper button hold timing
        action_completed = False
        if not manual_mode:
            global current_action, action_frames_remaining, release_frames_remaining, current_action_release_delay

            if current_action and action_frames_remaining > 0:
                # Continue holding the current action (WAIT actions press nothing)
                if current_action == "WAIT":
                    actions_pressed = []
                else:
                    actions_pressed = [current_action]
                action_frames_remaining -= 1
                if action_frames_remaining == 0:
                    # Action finished, start release delay
                    # Record ending position for this action (skip if queue is large for performance)
                    global recent_button_presses

                    # Only update position tracking if queue is small (<10 actions)
                    # This prevents expensive state reads during batch action processing
                    if len(action_queue) < 10:
                        current_state = env.get_comprehensive_state()
                        player_data = current_state.get("player", {})
                        position = player_data.get("position", {})
                        location = player_data.get("location", "Unknown")
                        end_pos = (
                            (position.get("x"), position.get("y"), location) if position else (None, None, location)
                        )

                        # Find and update the most recent incomplete action matching current_action
                        for i in range(len(recent_button_presses) - 1, -1, -1):
                            if (
                                recent_button_presses[i]["button"] == current_action
                                and not recent_button_presses[i]["completed"]
                            ):
                                recent_button_presses[i]["end_pos"] = end_pos
                                recent_button_presses[i]["completed"] = True
                                break
                    else:
                        # For large queues, just mark as completed without position update
                        for i in range(len(recent_button_presses) - 1, -1, -1):
                            if (
                                recent_button_presses[i]["button"] == current_action
                                and not recent_button_presses[i]["completed"]
                            ):
                                recent_button_presses[i]["end_pos"] = (None, None, "Unknown")
                                recent_button_presses[i]["completed"] = True
                                break

                    current_action = None
                    release_frames_remaining = current_action_release_delay
                    action_completed = True  # Mark action as completed
                    print(f"✅ Action completed: step_count will increment")
                    if release_frames_remaining == 0:
                        _on_action_released()
            elif release_frames_remaining > 0:
                # Release delay (no button pressed)
                actions_pressed = []
                release_frames_remaining -= 1
                if release_frames_remaining == 0:
                    _on_action_released()
            elif _move_check is not None and _move_check["released"]:
                # Judge the last direction press once the player has finished the step.
                actions_pressed = []
                _settle_move_check()
            elif action_queue and _wait_for_player_to_stand():
                # Let a step or ledge jump finish before the next checked direction press.
                actions_pressed = []
            elif action_queue:
                # Start a new action from the queue
                with action_queue_lock:
                    current_action_data = action_queue.pop(0)

                # Handle both old format (string) and new format (dict) for backward compatibility
                if isinstance(current_action_data, str):
                    current_action = current_action_data
                    timing = SPEED_PRESETS[DEFAULT_SPEED]
                else:
                    current_action = current_action_data["button"]
                    speed = current_action_data.get("speed", DEFAULT_SPEED)
                    timing = SPEED_PRESETS.get(speed, SPEED_PRESETS[DEFAULT_SPEED])

                    # Allow explicit frame overrides
                    if current_action_data.get("hold_frames") is not None:
                        timing = timing.copy()  # Don't modify the preset
                        timing["hold"] = current_action_data["hold_frames"]
                    if current_action_data.get("release_frames") is not None:
                        timing = timing.copy()
                        timing["release"] = current_action_data["release_frames"]

                _begin_action(current_action_data)

                # Special handling for WAIT action
                if current_action == "WAIT":
                    # A pure delay: run it as release frames. (It used to leave
                    # current_action == "WAIT" set forever, so /queue_status never
                    # reported the queue empty after a trailing WAIT.)
                    current_action = None
                    action_frames_remaining = 0
                    current_action_release_delay = timing["release"]
                    release_frames_remaining = timing["release"]
                    if release_frames_remaining == 0:
                        _on_action_released()
                    actions_pressed = []
                else:
                    action_frames_remaining = timing["hold"]
                    current_action_release_delay = timing["release"]
                    actions_pressed = [current_action]
                queue_len = len(action_queue)

                # Get current FPS for estimation
                current_fps_for_calc = env.get_current_fps(fps) if env else fps
                estimated_time = queue_len * (timing["hold"] + timing["release"]) / current_fps_for_calc
                speed_indicator = f" [{speed}]" if isinstance(current_action_data, dict) else ""
                print(
                    f"🎮 Server processing action: {current_action}{speed_indicator}, Queue remaining: {queue_len} actions (~{estimated_time:.1f}s)"
                )
            else:
                # No action to process
                actions_pressed = []

        # Step environment
        step_environment(actions_pressed)

        # Milestones are now updated in background thread

        # Server runs headless - no display update needed
        update_display(manual_mode)

        # Only increment step count when an action is completed
        if action_completed:
            with step_lock:
                step_count += 1
                print(f"📈 Step count incremented to: {step_count}")

        # Performance monitoring - log actual FPS every 5 seconds
        global last_fps_log, frame_count_since_log
        frame_count_since_log += 1
        current_time = time.time()
        if current_time - last_fps_log >= 5.0:  # Log every 5 seconds
            actual_fps = frame_count_since_log / (current_time - last_fps_log)
            queue_len = len(action_queue)
            print(f"📊 Server FPS: {actual_fps:.1f} (target: {GAME_FPS:.0f} idle / {TURBO_FPS:.0f} busy), Queue: {queue_len} actions")
            last_fps_log = current_time
            frame_count_since_log = 0

        # Turbo while the harness is executing presses (queue, held button, release
        # delay, or a direction press waiting to settle); normal speed while idle.
        busy = (
            bool(action_queue) or current_action is not None or release_frames_remaining > 0
            or _move_check is not None or _turbo_macros > 0
        )
        target_fps = TURBO_FPS if busy else GAME_FPS
        if target_fps <= 0:
            next_frame_deadline = time.monotonic()
            continue
        # Deadline pacing: sleep only the time left in this frame's slot, so per-frame
        # work doesn't add on top of the sleep (that held 80 fps to ~54). If we fall
        # far behind (e.g. a slow state read), restart the schedule rather than burst.
        now = time.monotonic()
        next_frame_deadline = max(next_frame_deadline + 1.0 / target_fps, now - 0.1)
        if next_frame_deadline > now:
            time.sleep(next_frame_deadline - now)


def run_fastapi_server(port):
    """Run FastAPI server in background thread"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="error",
        access_log=False,
        timeout_keep_alive=60,  # Keep connections alive longer
        timeout_graceful_shutdown=30,  # More time for graceful shutdown
    )


# Serve stream.html
@app.get("/stream")
async def get_stream():
    """Serve the stream.html interface"""
    try:
        with open("server/stream.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Stream interface not found")


# FastAPI endpoints
@app.get("/config")
async def get_config():
    """Return server configuration (game type, resolution) for dynamic UI"""
    return {
        "game": game_type,
        "width": env.width if env else 240,
        "height": env.height if env else 160,
    }


@app.get("/health")
async def get_health():
    """Health check: healthy only while the game loop is emulating frames.

    Deliberately takes no lock: it must answer, and report the stall, even while
    something holds the emulator.
    """
    now = time.time()
    if env is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "emulator not initialized", "timestamp": now})
    if last_frame_monotonic is None:
        return JSONResponse(status_code=503, content={"status": "starting", "reason": "no frame emulated yet", "timestamp": now})
    frame_age = time.monotonic() - last_frame_monotonic
    if frame_age > HEALTH_MAX_FRAME_AGE_S:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": f"no frame emulated for {frame_age:.1f}s (game loop dead or stuck)",
                "last_frame_age_s": round(frame_age, 3),
                "timestamp": now,
            },
        )
    return {"status": "healthy", "last_frame_age_s": round(frame_age, 3), "timestamp": now}


# Local addition: lets a human pause the autonomous agent (agents/PokeAgent.py waits
# while this flag file exists) and play through /action, then hand control back.
AGENT_PAUSE_FLAG = os.path.join(".pokeagent_cache", "AGENT_PAUSED")


@app.get("/play")
async def get_play():
    """Serve the human controller page (server/play.html)."""
    with open("server/play.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/agent_pause")
async def get_agent_pause():
    return {"paused": os.path.exists(AGENT_PAUSE_FLAG)}


@app.post("/agent_pause")
async def set_agent_pause(request: dict):
    if request.get("paused"):
        os.makedirs(os.path.dirname(AGENT_PAUSE_FLAG), exist_ok=True)
        with open(AGENT_PAUSE_FLAG, "w") as f:
            f.write(str(time.time()))
    elif os.path.exists(AGENT_PAUSE_FLAG):
        os.remove(AGENT_PAUSE_FLAG)
    return {"paused": os.path.exists(AGENT_PAUSE_FLAG)}


@app.get("/status")
async def get_status():
    """Get server status"""
    with step_lock:
        current_step = step_count

    # Get current FPS (may be 4x during dialog)
    current_fps = env.get_current_fps(fps) if env else fps
    # Use cached dialog state for consistency with FPS calculation
    is_dialog = env._cached_dialog_state if env else False

    return {
        "status": "running",
        "step_count": current_step,
        "base_fps": fps,
        "current_fps": current_fps,
        "is_dialog": is_dialog,
        "fps_multiplier": 2 if is_dialog else 1,
    }


@app.get("/screenshot")
async def get_screenshot():
    """Get current screenshot"""
    global current_obs, step_count

    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None

    if obs_copy is None:
        raise HTTPException(status_code=500, detail="No screenshot available")

    try:
        # Convert numpy array to PIL image
        pil_image = Image.fromarray(obs_copy)

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        with step_lock:
            current_step = step_count

        return GameStateResponse(
            screenshot_base64=img_str,
            step_number=current_step,
            resolution=[obs_copy.shape[1], obs_copy.shape[0]],
            status="running",
        )

    except Exception as e:
        logger.error(f"Error getting screenshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/frame")
async def get_latest_frame():
    """Get latest game frame in same format as single-process mode"""
    global current_obs, env

    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None

    # If current_obs is None (e.g., after server restart), try to get a fresh screenshot
    if obs_copy is None and env:
        try:
            with memory_lock:
                screenshot = env.get_screenshot()
            if screenshot:
                obs_copy = np.array(screenshot)
                # Update current_obs for future requests
                with obs_lock:
                    current_obs = obs_copy.copy()
                logger.debug("Frame endpoint: Retrieved fresh screenshot after restart")
        except Exception as e:
            logger.warning(f"Frame endpoint: Failed to get fresh screenshot: {e}")

    if obs_copy is None:
        return {"frame": ""}

    try:
        # Convert to base64
        pil_image = Image.fromarray(obs_copy)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return {"frame": img_str}
    except Exception as e:
        logger.warning(f"Frame endpoint: Error encoding frame: {e}")
        return {"frame": ""}


@app.get("/frame")
async def get_frame_from_cache():
    """DISABLED - Use WebSocket /ws/frames for real-time streaming instead"""
    raise HTTPException(
        status_code=410,
        detail="HTTP polling disabled. Use WebSocket endpoint /ws/frames for real-time frame streaming.",
    )


@app.websocket("/ws/frames")
async def websocket_frames(websocket: WebSocket):
    """WebSocket endpoint for real-time frame streaming

    This endpoint continuously sends new frames as they become available.
    The game loop updates frame_manager.latest_frame, and we send it here.
    """
    import asyncio

    await frame_manager.connect(websocket)
    try:
        # Continuously send frames while connection is alive
        while True:
            try:
                # Send latest frame if available (non-blocking check)
                await frame_manager.send_latest_frame(websocket)

                # Small sleep to avoid busy-waiting (check ~60 times per second)
                await asyncio.sleep(0.016)  # ~60 FPS check rate

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"Error in frame streaming: {e}")
                break
    finally:
        frame_manager.disconnect(websocket)


@app.post("/action")
async def take_action(request: ActionRequest):
    """Take an action"""
    global current_obs, step_count, recent_button_presses, action_queue, anticheat_tracker, step_counter, last_action_time

    # print( Action endpoint called with request: {request}")
    # print( Request buttons: {request.buttons}")

    if env is None:
        # print( Emulator not initialized")
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        # Add all actions to the queue (handle both single actions and lists)
        if request.buttons:
            # Get timing parameters
            speed = request.speed or DEFAULT_SPEED
            hold_frames = request.hold_frames
            release_frames = request.release_frames

            # Validate speed parameter
            if speed not in SPEED_PRESETS:
                speed = DEFAULT_SPEED
                print(f"⚠️ Invalid speed '{request.speed}', using '{DEFAULT_SPEED}'")

            # Add ALL actions to the queue - let the game loop handle execution
            speed_info = f" [speed={speed}]" if speed != DEFAULT_SPEED else ""
            frame_info = ""
            if hold_frames is not None or release_frames is not None:
                frame_info = f" [hold={hold_frames}, release={release_frames}]"

            print(f"📡 Server received actions: {request.buttons}{speed_info}{frame_info}")
            print(f"📋 Action queue before extend: {action_queue}")

            seq_id = _register_sequence(request.buttons, request.check_movement, request.source)

            # Create action data with timing info
            with action_queue_lock:
                for idx, button in enumerate(request.buttons):
                    action_data = {
                        "button": button,
                        "speed": speed,
                        "hold_frames": hold_frames,
                        "release_frames": release_frames,
                        "seq_id": seq_id,
                        "seq_index": idx,
                        "check_movement": request.check_movement,
                    }
                    action_queue.append(action_data)

            print(f"📋 Action queue after extend: {len(action_queue)} actions")

            # Track button presses for recent actions display with position tracking
            current_time = time.time()

            # Only read position if queue is small (<20 actions) to avoid performance impact
            # For large queues, skip position tracking to maintain FPS
            if len(action_queue) < 20:
                # Get current player position (use cached state - 100ms cache in emulator)
                with memory_lock:
                    current_state = env.get_comprehensive_state()  # Already cached internally
                player_data = current_state.get("player", {})
                position = player_data.get("position", {})
                location = player_data.get("location", "Unknown")
                start_pos = (position.get("x"), position.get("y"), location) if position else (None, None, location)
            else:
                # Skip expensive state read for large queues
                start_pos = (None, None, "Unknown")

            source = request.source
            metadata = request.metadata or {}
            sequence_length = len(request.buttons)

            for idx, button in enumerate(request.buttons):
                # Add all buttons to recent actions with starting position
                action_entry = {
                    "button": button,
                    "timestamp": current_time,
                    "start_pos": start_pos,
                    "end_pos": None,  # Will be filled when action completes
                    "completed": False,
                    "sequence_index": idx,
                    "sequence_length": sequence_length,
                    "seq_id": seq_id,
                }

                if source:
                    action_entry["source"] = source
                if metadata:
                    action_entry["metadata"] = dict(metadata)

                recent_button_presses.append(action_entry)

            # Update total actions count in metrics
            with step_lock:
                latest_metrics["total_actions"] = latest_metrics.get("total_actions", 0) + len(request.buttons)

                # Also update the LLM logger's action count and gameplay time for checkpoint persistence
                try:
                    from utils.data_persistence.llm_logger import get_llm_logger

                    llm_logger = get_llm_logger()
                    if llm_logger:
                        llm_logger.cumulative_metrics["total_actions"] = latest_metrics["total_actions"]

                        # Update gameplay time for button presses too
                        current_time = time.time()
                        time_since_last_update = current_time - llm_logger.cumulative_metrics.get(
                            "last_update_time", current_time
                        )
                        # Only add time if it's reasonable (less than 5 minutes since last interaction)
                        if time_since_last_update < 300:  # 5 minutes
                            llm_logger.cumulative_metrics["total_run_time"] = (
                                llm_logger.cumulative_metrics.get("total_run_time", 0) + time_since_last_update
                            )
                        llm_logger.cumulative_metrics["last_update_time"] = current_time

                        # Save metrics to cache file
                        llm_logger.save_cumulative_metrics()

                        # Sync LLM logger's cumulative metrics back to latest_metrics
                        # This ensures token usage and costs from LLM interactions are displayed
                        cumulative_metrics_to_sync = [
                            "total_tokens",
                            "prompt_tokens",
                            "completion_tokens",
                            "total_cost",
                            "total_llm_calls",
                            "total_run_time",
                        ]
                        for metric_key in cumulative_metrics_to_sync:
                            if metric_key in llm_logger.cumulative_metrics:
                                latest_metrics[metric_key] = llm_logger.cumulative_metrics[metric_key]
                except Exception as e:
                    logger.debug(f"Failed to sync metrics with LLM logger: {e}")

            # Keep only last 50 button presses to avoid memory issues
            if len(recent_button_presses) > 50:
                recent_button_presses = recent_button_presses[-50:]
        else:
            print(f" No buttons in request")

        # DON'T execute action here - let the game loop handle it from the queue
        # This prevents conflicts between the API thread and pygame thread

        # Return immediate success - avoid all locks to prevent deadlocks
        actions_added = len(request.buttons) if request.buttons else 0

        # print( Returning success, actions_added: {actions_added}, queue_length: {len(action_queue)}")

        # Log action to submission.log if anticheat tracker is available
        if anticheat_tracker and request.buttons:
            try:
                # Calculate decision time
                current_time = time.time()
                if last_action_time is not None:
                    decision_time = current_time - last_action_time
                else:
                    decision_time = 0.0  # First action
                last_action_time = current_time

                # Skip expensive state reads and logging when queue is large (>30 actions)
                # This prevents FPS drops during batch action processing
                if len(action_queue) < 30:
                    # Get current game state for logging
                    with memory_lock:
                        game_state = env.get_comprehensive_state()
                    action_taken = request.buttons[0] if request.buttons else "NONE"  # Log first action

                    # Create simple state hash
                    import hashlib

                    state_str = str(game_state)
                    state_hash = hashlib.md5(state_str.encode()).hexdigest()[:8]

                    # Determine if this is manual mode (from client) or agent mode
                    # For now, assume manual mode if coming through API
                    manual_mode = request.source == "manual" if hasattr(request, "source") else True

                    # Get the latest milestone from the emulator's milestone tracker
                    # First, trigger an immediate milestone check to ensure current state is detected
                    latest_milestone = "NONE"
                    if env and hasattr(env, "milestone_tracker"):
                        try:
                            # Force an immediate milestone check before logging
                            with memory_lock:
                                env.check_and_update_milestones(game_state, agent_step_count=agent_step_count)
                        except Exception as e:
                            logger.debug(f"Error during immediate milestone check: {e}")

                        milestone_name, split_time, total_time = env.milestone_tracker.get_latest_milestone_info()
                        latest_milestone = milestone_name if milestone_name != "NONE" else "NONE"

                    # Log the action
                    step_counter += 1
                    anticheat_tracker.log_submission_data(
                        step=step_counter,
                        state_data=game_state,
                        action_taken=action_taken,
                        decision_time=decision_time,
                        state_hash=state_hash,
                        manual_mode=manual_mode,
                        milestone_override=latest_milestone,
                    )
                else:
                    # For large queues, skip detailed logging to maintain FPS
                    logger.debug(f"Skipping submission logging - queue size: {len(action_queue)}")

            except Exception as e:
                logger.warning(f"Error logging to submission.log: {e}")

        # Return lightweight response without any lock acquisition
        return {
            "status": "success",
            "actions_queued": actions_added,
            "queue_length": len(action_queue),  # action_queue access is atomic for lists
            "message": f"Added {actions_added} actions to queue",
            "sequence_id": seq_id if request.buttons else None,
        }

    except Exception as e:
        # print( Exception in action endpoint: {e}")
        logger.error(f"Error taking action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue_status")
async def get_queue_status():
    """Get action queue status"""
    global action_queue, current_action, action_frames_remaining, release_frames_remaining

    queue_empty = (
        len(action_queue) == 0
        and current_action is None
        and action_frames_remaining == 0
        and release_frames_remaining == 0
        and _move_check is None  # last direction press still being judged
    )

    return {
        "queue_empty": queue_empty,
        "queue_length": len(action_queue),
        "current_action": current_action,
        "action_frames_remaining": action_frames_remaining,
        "release_frames_remaining": release_frames_remaining,
    }


def _build_porymap_visual_map_15x15(state: Dict[str, Any], player_coords: Tuple[int, int]) -> bool:
    """Set ``state['map']['visual_map']`` from elevation-filtered ``porymap.grid`` (15x15 around player).

    Uses whatever grid is already in ``state`` (refreshed while idle, or from /state cache). Safe when
    the action queue is non-empty: recenters the window on *current* player coordinates without
    re-running ``_format_porymap_info``.

    Returns True if ``visual_map`` was written.
    """
    map_section = state.get("map") or {}
    porymap_grid = (map_section.get("porymap") or {}).get("grid")
    if not porymap_grid or not player_coords:
        return False

    px, py = int(player_coords[0]), int(player_coords[1])
    window_size = 15
    half_window = window_size // 2
    height = len(porymap_grid)

    def _slice_row(row: Any, sx: int, ex: int) -> List[str]:
        if isinstance(row, str):
            return list(row[sx:ex])
        return [str(c) for c in row[sx:ex]]

    start_y = max(0, py - half_window)
    end_y = min(height, py + half_window + 1)
    start_x = max(0, px - half_window)

    visual_grid: List[List[str]] = []
    for y in range(start_y, end_y):
        if y < height:
            row = porymap_grid[y]
            end_x = min(len(row), px + half_window + 1)
            window_row = _slice_row(row, start_x, end_x)
            while len(window_row) < window_size:
                window_row.append("#")
            visual_grid.append(window_row[:window_size])
        else:
            visual_grid.append(["#"] * window_size)

    while len(visual_grid) < window_size:
        visual_grid.append(["#"] * window_size)

    player_y_in_window = py - start_y
    player_x_in_window = px - start_x
    if 0 <= player_y_in_window < len(visual_grid) and 0 <= player_x_in_window < len(
        visual_grid[player_y_in_window]
    ):
        visual_grid[player_y_in_window][player_x_in_window] = "P"

    if "map" not in state:
        state["map"] = {}
    state["map"]["visual_map"] = "\n".join(" ".join(r) for r in visual_grid)
    state["map"]["map_source"] = "porymap_with_player_15x15"
    return True


@app.get("/state")
async def get_comprehensive_state():
    """Get comprehensive game state including visual and memory data"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        # Use the emulator's built-in caching (100ms cache)
        # This avoids expensive operations on rapid requests
        with memory_lock:
            state = env.get_comprehensive_state()

        # Ensure game state is consistent with cached dialog state
        # Use the same cached dialog state as the status endpoint
        is_dialog = env._cached_dialog_state if env else False
        if is_dialog:
            state["game"]["game_state"] = "dialog"
        else:
            # Force overworld if not in dialog (respect 5-second timeout)
            state["game"]["game_state"] = "overworld"

        # Include milestones for storyline objective auto-completion
        if env.milestone_tracker:
            state["milestones"] = env.milestone_tracker.milestones

        # Get map stitcher data for enhanced map display
        # Use the memory_reader's MapStitcher instance which has the accumulated data
        map_stitcher = None
        if env and env.memory_reader and hasattr(env.memory_reader, "_map_stitcher"):
            map_stitcher = env.memory_reader._map_stitcher
            num_areas = len(map_stitcher.map_areas) if map_stitcher and hasattr(map_stitcher, "map_areas") else 0
            logger.debug(f"Using memory_reader's MapStitcher with {num_areas} areas")
        else:
            logger.debug("No MapStitcher available from memory_reader")

        # Get current location name
        current_location = state.get("player", {}).get("location", "Unknown")
        player_pos = state.get("player", {}).get("position")
        if player_pos:
            player_coords = (player_pos.get("x", 0), player_pos.get("y", 0))
        else:
            player_coords = None

        # Add stitched map info to the map section
        if not "map" in state:
            state["map"] = {}

        # Check if we can use cached map data (location-based cache)
        current_time = time.time()
        cache_valid = (
            _state_cache["location"] == current_location and current_time - _state_cache["timestamp"] < _state_cache_ttl
        )

        # Track map source states separately:
        # - slam_map_loaded: a real SLAM map was loaded this request
        # - used_cached_map: state.map came from cache and we should skip heavy generation
        slam_map_loaded = False
        used_cached_map = False

        if cache_valid and _state_cache["map_data"]:
            # Use cached map data - skip expensive map generation!
            state["map"] = _state_cache["map_data"].copy()
            if _state_cache["portal_data"]:
                state.update(_state_cache["portal_data"])
            logger.debug(f"✅ Using cached map data for {current_location}")
            used_cached_map = True
        else:
            # Generate fresh map data
            logger.debug(f"🔄 Generating fresh map data for {current_location}")

            # PRIORITY 1: Check for agent's SLAM map first
            if current_location and current_location != "Unknown":
                try:
                    # Path imported at top of file
                    maps_dir = Path(".pokeagent_cache/maps")
                    # Normalize case to title case for consistent filenames
                    normalized_location = current_location.title()
                    safe_name = "".join(c for c in normalized_location if c.isalnum() or c in (" ", "_", "-")).strip()
                    safe_name = safe_name.replace(" ", "_")
                    slam_map_file = maps_dir / f"{safe_name}.txt"

                    logger.info(
                        f"🗺️ Checking for SLAM map: location='{current_location}' (normalized: '{normalized_location}') → file='{slam_map_file}'"
                    )

                    if slam_map_file.exists():
                        slam_map_data = slam_map_file.read_text()
                        state["map"]["visual_map"] = slam_map_data
                        state["map"]["map_source"] = "agent_slam"
                        logger.info(f"✅ Using SLAM map for {current_location} ({len(slam_map_data)} chars)")
                        slam_map_loaded = True
                    else:
                        logger.info(f"   No SLAM map found at {slam_map_file}")
                except Exception as e:
                    logger.error(f"Error loading SLAM map: {e}")

        # PRIORITY 1.5: Build final porymap map for UI parity with LLM map context.
        # This ensures the "Map & Actions" UI uses the same post-filtered map
        # (reconciliation + flag filtering + object markers) when available.
        if game_type != "red" and current_location and current_location != "Unknown" and state.get("map", {}).get("map_source") != "agent_slam":
            try:
                from utils.mapping.porymap_state import _format_porymap_info

                badges = state.get("game", {}).get("badges", [])
                if isinstance(badges, list):
                    badge_count = len(badges)
                elif isinstance(badges, int):
                    badge_count = badges
                else:
                    badge_count = 0

                porymap_result = _format_porymap_info(
                    location_name=current_location,
                    player_coords=player_coords,
                    badge_count=badge_count,
                    memory_reader=env.memory_reader if env else None,
                    runtime_object_events=state.get("map", {}).get("object_events", []),
                )

                porymap_json = porymap_result.json_map if porymap_result else None
                porymap_ascii = porymap_json.get("ascii") if isinstance(porymap_json, dict) else None
                if isinstance(porymap_ascii, str) and porymap_ascii:
                    state["map"].setdefault("porymap", {})
                    state["map"]["porymap"]["ascii"] = porymap_ascii
                    state["map"]["porymap"]["grid"] = porymap_json.get("grid", [])
                    state["map"]["porymap"]["objects"] = porymap_json.get("objects", [])
                    state["map"]["porymap"]["raw_tiles"] = porymap_json.get("raw_tiles", [])
                    state["map"]["porymap"]["dimensions"] = porymap_json.get("dimensions", {})
                    state["map"]["visual_map"] = porymap_ascii
                    state["map"]["map_source"] = "porymap_final"
            except Exception as e:
                logger.debug(f"Could not build final porymap visual map: {e}")

        # PRIORITY 2: Check if visual_map was already generated by memory_reader
        # If so, preserve it as it has the proper accumulated map data
        if not slam_map_loaded and not used_cached_map:
            visual_map_from_memory_reader = state.get("map", {}).get("visual_map")
            if visual_map_from_memory_reader:
                if state.get("map", {}).get("map_source") != "porymap_final":
                    logger.debug("Using visual_map generated by memory_reader")
                    state["map"]["map_source"] = "memory_reader"
                # Keep the visual_map as-is
            elif not ENABLE_MAP_STITCHER and current_location and current_location != "Unknown":
                # PRIORITY 3: Use porymap ground truth data when map stitcher is disabled
                try:
                    from utils.mapping.map_formatter import format_map_for_llm

                    # Get raw tiles from state
                    raw_tiles = state.get("map", {}).get("tiles")
                    player_pos_dict = state.get("player", {}).get("position", {})
                    player_facing = state.get("player", {}).get("facing", "South")

                    if raw_tiles:
                        # Generate visual map using ground truth tiles
                        visual_map = format_map_for_llm(
                            raw_tiles=raw_tiles,
                            player_facing=player_facing,
                            npcs=None,
                            player_coords=player_pos_dict,
                            location_name=current_location,
                        )

                        if visual_map and visual_map != "No map data available":
                            state["map"]["visual_map"] = visual_map
                            state["map"]["map_source"] = "porymap_ground_truth"
                            logger.debug(f"Generated visual_map from porymap ground truth for {current_location}")
                except Exception as e:
                    logger.error(f"Failed to generate visual_map from porymap: {e}")
            elif map_stitcher:
                # Generate visual map if not already present
                state["map"]["map_source"] = "map_stitcher"
                try:
                    # Get NPCs from state if available
                    npcs = state.get("map", {}).get("object_events", [])

                    # Get connections for this location
                    connections_with_coords = []
                    if current_location and current_location != "Unknown":
                        location_connections = map_stitcher.get_location_connections(current_location)
                        for conn in location_connections:
                            if len(conn) >= 3:
                                other_loc, my_coords, their_coords = conn[0], conn[1], conn[2]
                                connections_with_coords.append(
                                    {
                                        "to": other_loc,
                                        "from_pos": list(my_coords) if my_coords else [],
                                        "to_pos": list(their_coords) if their_coords else [],
                                    }
                                )

                    # Generate the map display
                    map_lines = map_stitcher.generate_location_map_display(
                        location_name=current_location,
                        player_pos=player_coords,
                        npcs=npcs,
                        connections=connections_with_coords,
                    )

                    # Store as formatted text
                    if map_lines:
                        state["map"]["visual_map"] = "\n".join(map_lines)
                        logger.debug(f"Generated visual_map with {len(map_lines)} lines")
                except Exception as e:
                    logger.error(f"Failed to generate visual_map: {e}")

        # Add stitched map info for the client/frontend
        if map_stitcher:
            # Get the location grid and connections
            if current_location and current_location != "Unknown":
                location_grid = map_stitcher.get_location_grid(current_location)
                connections = []

                # Get connections for this location
                for other_loc, my_coords, their_coords in map_stitcher.get_location_connections(current_location):
                    connections.append({"to": other_loc, "from_pos": list(my_coords), "to_pos": list(their_coords)})

                state["map"]["stitched_map_info"] = {
                    "available": True,
                    "current_area": {"name": current_location, "connections": connections, "player_pos": player_coords},
                    "player_local_pos": player_coords,
                }
            else:
                state["map"]["stitched_map_info"] = {"available": False, "reason": "Unknown location"}

            # Also include location connections directly for backward compatibility
            try:
                from utils.data_persistence.run_data_manager import get_cache_path
                cache_file = str(get_cache_path("map_stitcher_data.json"))
                if os.path.exists(cache_file):
                    # load_state/save_state rewrite this file in place while holding
                    # the emulator lock; reading under it avoids a half-written file.
                    with memory_lock, open(cache_file, "r") as f:
                        map_data = json.load(f)
                        if "location_connections" in map_data and map_data["location_connections"]:
                            location_connections = map_data["location_connections"]
                            state["location_connections"] = location_connections
                            logger.debug(
                                f"Loaded location connections for {len(location_connections) if location_connections else 0} locations"
                            )
                        elif "warp_connections" in map_data and map_data["warp_connections"]:
                            # Convert warp_connections to portal_connections format for LLM display
                            map_id_connections = {}
                            for conn in map_data["warp_connections"]:
                                from_map = conn["from_map_id"]
                                if from_map not in map_id_connections:
                                    map_id_connections[from_map] = []

                                # Find the location name for the destination map
                                to_map_name = "Unknown Location"
                                if str(conn["to_map_id"]) in map_data.get("map_areas", {}):
                                    to_map_name = map_data["map_areas"][str(conn["to_map_id"])]["location_name"]

                                map_id_connections[from_map].append(
                                    {
                                        "to_name": to_map_name,
                                        "from_pos": conn["from_position"],  # Keep as list for JSON serialization
                                        "to_pos": conn["to_position"],  # Keep as list for JSON serialization
                                    }
                                )

                            state["portal_connections"] = map_id_connections
                            print(f"🗺️ SERVER: Added portal connections to state: {map_id_connections}")
                            print(f"🗺️ SERVER: State now has keys: {list(state.keys())}")
                            logger.debug(
                                f"Loaded portal connections for {len(map_id_connections) if map_id_connections else 0} maps from persistent storage"
                            )
                else:
                    print(f"🗺️ SERVER: Cache file not found at {cache_file}")
                    logger.debug(f"Map stitcher cache file not found: {cache_file}")
            except Exception as e:
                import traceback

                print(f"🗺️ SERVER: Error loading portal connections: {e}")
                print(f"🗺️ SERVER: Full traceback: {traceback.format_exc()}")
                logger.debug(f"Could not load portal connections from persistent storage: {e}")

        # The battle information already contains all necessary data
        # No additional analysis needed - keep it clean

        # Remove MapStitcher instance to avoid serialization issues
        # The instance is only for internal use by state_formatter
        if "_map_stitcher_instance" in state.get("map", {}):
            del state["map"]["_map_stitcher_instance"]

        # Convert screenshot to base64 if available
        if state.get("visual", {}).get("screenshot"):
            buffer = io.BytesIO()
            state["visual"]["screenshot"].save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            state["visual"]["screenshot_base64"] = img_str
            # Remove the PIL image object to avoid serialization issues
            del state["visual"]["screenshot"]

        # Add porymap ground truth data with elevation filtering for frontend display
        # Skip during queued actions to avoid FPS slowdown (Emerald only — Red uses its own map reader)
        current_queue_length = len(action_queue)  # Check queue before expensive porymap operations
        if (
            game_type != "red"
            and current_location
            and current_location != "Unknown"
            and current_location != "TITLE_SEQUENCE"
            and current_queue_length == 0
        ):
            try:
                from utils.state_formatter import _format_porymap_info

                _mr = getattr(env, "memory_reader", None) if env else None
                porymap_result = _format_porymap_info(
                    current_location,
                    player_coords,
                    memory_reader=_mr,
                    runtime_object_events=state.get("map", {}).get("object_events", []),
                )
                porymap_data = getattr(porymap_result, "json_map", None)
                if porymap_data and porymap_data.get("grid"):
                    if "porymap" not in state["map"]:
                        state["map"]["porymap"] = {}
                    state["map"]["porymap"]["grid"] = porymap_data.get("grid")
                    state["map"]["porymap"]["objects"] = porymap_data.get("objects", [])
                    state["map"]["porymap"]["dimensions"] = porymap_data.get("dimensions", {})
                    state["map"]["porymap"]["warps"] = porymap_data.get("warps", [])
                    logger.debug(f"Added elevation-filtered porymap data to /state for {current_location}")
            except Exception as e:
                logger.warning(f"Failed to add porymap data to /state: {e}")

        # Cheap: 15x15 stream UI from cached or freshly refreshed grid (also while actions are queued)
        if game_type != "red" and current_location and current_location not in ("Unknown", "TITLE_SEQUENCE") and player_coords:
            _build_porymap_visual_map_15x15(state, player_coords)
        elif game_type == "red":
            # stream UI map for Red has been built in red_map_reader.py
            visual_map = env.memory_reader.map_reader.format_map_for_llm(radius=7)
            state["map"]["visual_map"] = visual_map
            state["map"]["map_source"] = "red_map_reader"
            whole_map = env.memory_reader.map_reader.get_whole_map_data()
            state["map"]["red_whole_map"] = whole_map

        with step_lock:
            current_step = step_count

        # Include action queue info for multiprocess coordination
        queue_length = len(action_queue)  # Action queue access is atomic for len()

        # Cache map data for this location+position - reduces load on rapid repeated requests
        _state_cache["location"] = current_location
        _state_cache["player_coords"] = player_coords
        _state_cache["map_data"] = state.get("map", {}).copy()
        _state_cache["portal_data"] = {"location_connections": state.get("location_connections", {})}
        _state_cache["timestamp"] = time.time()

        return ComprehensiveStateResponse(
            visual=state["visual"],
            player=state["player"],
            game=state["game"],
            map=state["map"],
            milestones=state.get("milestones", {}),
            location_connections=state.get("location_connections", {}),
            step_number=current_step,
            status="running",
            action_queue_length=queue_length,
        )

    except Exception as e:
        logger.error(f"Error getting comprehensive state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/whole_map")
async def get_whole_map():
    """Get complete map data including full grid, raw tiles, and elevation info for debugging"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    # Red: use RedEmulator.get_whole_map() directly (returns compatible dict)
    if game_type == "red":
        try:
            with memory_lock:
                result = env.get_whole_map()
            if not result or not result.get("grid"):
                raise HTTPException(status_code=400, detail="No valid location loaded")
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting whole map (Red): {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Emerald: use porymap ground truth
    try:
        # Get current location
        with memory_lock:
            state = env.get_comprehensive_state()
        location_name = state.get("player", {}).get("location", "Unknown")
        player_pos = state.get("player", {}).get("position", {})
        px, py = player_pos.get("x", 0), player_pos.get("y", 0)

        if not location_name or location_name in ["Unknown", "TITLE_SEQUENCE"]:
            raise HTTPException(status_code=400, detail="No valid location loaded")

        # Load porymap data with raw tiles
        from utils.mapping.porymap_json_builder import build_json_map_for_llm
        from utils.mapping.pokeemerald_parser import PokeemeraldMapLoader
        from utils.state_formatter import ROM_TO_PORYMAP_MAP
        from pathlib import Path

        # Get pokeemerald root (pokemon_env/porymap or POKEEMERALD_ROOT override)
        from pokemon_env.porymap_paths import get_porymap_root
        pokeemerald_root = get_porymap_root()

        if not pokeemerald_root:
            raise HTTPException(status_code=500, detail="Could not find pokeemerald root")

        # Get porymap name
        porymap_map_name = ROM_TO_PORYMAP_MAP.get(location_name)

        if not porymap_map_name:
            raise HTTPException(status_code=404, detail=f"Could not find porymap for location: {location_name}")

        # Build complete map
        json_map = build_json_map_for_llm(porymap_map_name, pokeemerald_root)

        if not json_map:
            raise HTTPException(status_code=500, detail=f"Failed to build map for {porymap_map_name}")

        # Get player elevation
        raw_tiles = json_map.get("raw_tiles", [])
        player_elevation = 0
        if raw_tiles and 0 <= py < len(raw_tiles) and 0 <= px < len(raw_tiles[py]):
            player_tile = raw_tiles[py][px]
            if len(player_tile) >= 4:
                player_elevation = player_tile[3]

        # Build elevation map
        elevation_map = []
        behavior_map = []
        if raw_tiles:
            from pokemon_env.enums import MetatileBehavior

            for row in raw_tiles:
                elev_row = []
                behav_row = []
                for tile in row:
                    if len(tile) >= 4:
                        elev_row.append(tile[3])  # elevation
                        behavior_id = tile[1]
                        try:
                            behavior_name = MetatileBehavior(behavior_id).name
                        except:
                            behavior_name = f"UNKNOWN_{behavior_id}"
                        behav_row.append(behavior_name)
                    else:
                        elev_row.append(0)
                        behav_row.append("UNKNOWN")
                elevation_map.append(elev_row)
                behavior_map.append(behav_row)

        # Count special behaviors
        from pokemon_env.enums import MetatileBehavior

        special_tiles = {}
        for y, row in enumerate(raw_tiles):
            for x, tile in enumerate(row):
                if len(tile) >= 4:
                    behavior_id = tile[1]
                    elevation = tile[3]
                    try:
                        behavior_name = MetatileBehavior(behavior_id).name
                        if any(keyword in behavior_name for keyword in ["LADDER", "STAIRS", "DOOR", "WARP"]):
                            if behavior_name not in special_tiles:
                                special_tiles[behavior_name] = []
                            special_tiles[behavior_name].append(
                                {"x": x, "y": y, "elevation": elevation, "behavior_id": behavior_id}
                            )
                    except:
                        pass

        return {
            "location": location_name,
            "porymap_name": porymap_map_name,
            "player_position": {"x": px, "y": py},
            "player_elevation": player_elevation,
            "dimensions": json_map.get("dimensions", {}),
            "grid": json_map.get("grid", []),
            "raw_tiles": raw_tiles,
            "elevation_map": elevation_map,
            "behavior_map": behavior_map,
            "special_tiles": special_tiles,
            "warps": json_map.get("warps", []),
            "objects": json_map.get("objects", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting whole map: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/memory")
async def debug_memory():
    """Debug memory reading (basic version)"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        if not env.memory_reader:
            return {"error": "Memory reader not initialized"}

        # Test basic memory access
        with memory_lock:
            diagnostics = env.memory_reader.test_memory_access()

        # Try to read some basic data
        try:
            with memory_lock:
                party_size = env.memory_reader.read_party_size()
                coordinates = env.memory_reader.read_coordinates()
                money = env.memory_reader.read_money()

                # Add new debugging info
                is_in_battle = env.memory_reader.is_in_battle()
                game_state = env.memory_reader.get_game_state()
                player_name = env.memory_reader.read_player_name()

                # Add battle detection debugging
                try:
                    battle_addr = env.memory_reader.IN_BATTLE_BIT_ADDR
                    battle_raw_value = env.memory_reader._read_u8(battle_addr)
                    battle_mask = env.memory_reader.IN_BATTLE_BITMASK
                    battle_result = (battle_raw_value & battle_mask) != 0
                except Exception as e:
                    battle_raw_value = None
                    battle_mask = None
                    battle_result = None

            diagnostics.update(
                {
                    "party_size": party_size,
                    "coordinates": coordinates,
                    "money": money,
                    "is_in_battle": is_in_battle,
                    "game_state": game_state,
                    "player_name": player_name,
                    "battle_detection": {
                        "address": f"0x{battle_addr:08x}" if "battle_addr" in locals() else "unknown",
                        "raw_value": f"0x{battle_raw_value:02x}" if battle_raw_value is not None else "error",
                        "mask": f"0x{battle_mask:02x}" if battle_mask is not None else "unknown",
                        "result": battle_result,
                    },
                    "working_reads": True,
                }
            )
        except Exception as read_error:
            diagnostics["read_error"] = str(read_error)
            diagnostics["working_reads"] = False

        return diagnostics

    except Exception as e:
        logger.error(f"Error debugging memory: {e}")
        return {"error": str(e)}


@app.get("/debug/memory/comprehensive")
async def debug_memory_comprehensive():
    """Comprehensive memory reading test with detailed diagnostics"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        # Use the comprehensive memory testing method
        with memory_lock:
            test_results = env.test_memory_reading()
        return test_results

    except Exception as e:
        logger.error(f"Error running comprehensive memory test: {e}")
        return {"error": str(e)}


@app.get("/debug/memory/dump")
async def debug_memory_dump(start: int = 0x02000000, length: int = 0x1000):
    """Dump raw memory from the emulator"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        if not env.memory_reader:
            return {"error": "Memory reader not initialized"}

        # Read raw memory bytes
        with memory_lock:
            memory_bytes = env.memory_reader._read_bytes(start, length)

        # Convert to hex string for easy viewing
        hex_data = memory_bytes.hex()

        # Also try to decode as text using Pokemon Emerald character mapping
        try:
            from pokemon_env.emerald_utils import EmeraldCharmap

            charmap = EmeraldCharmap()
            decoded_text = charmap.decode(memory_bytes)
        except:
            decoded_text = "Could not decode as text"

        return {
            "start_address": f"0x{start:08X}",
            "length": length,
            "hex_data": hex_data,
            "decoded_text": decoded_text,
            "raw_bytes": [b for b in memory_bytes[:100]],  # First 100 bytes as numbers
        }

    except Exception as e:
        logger.error(f"Error dumping memory: {e}")
        return {"error": str(e)}


@app.get("/test_stream")
async def test_stream():
    """Simple test stream to verify SSE works"""
    from fastapi.responses import StreamingResponse
    import asyncio

    async def simple_stream():
        for i in range(5):
            yield f"data: {{'test': {i}, 'timestamp': {time.time()}}}\n\n"
            await asyncio.sleep(1)
        yield f"data: {{'done': true}}\n\n"

    return StreamingResponse(simple_stream(), media_type="text/event-stream")


@app.get("/agent_stream")
async def stream_agent_thinking():
    """Stream agent thinking in real-time using Server-Sent Events"""
    from fastapi.responses import StreamingResponse
    from utils.data_persistence.llm_logger import get_llm_logger
    import asyncio

    async def event_stream():
        """Generate server-sent events for agent thinking"""
        logger.info("SSE: Starting event stream")
        last_timestamp = ""  # Track last seen timestamp instead of count
        sent_timestamps = set()  # Track all sent timestamps to avoid duplicates
        heartbeat_counter = 0
        llm_logger = get_llm_logger()

        try:
            # Send initial connection message
            yield f"data: {json.dumps({'status': 'connected', 'timestamp': time.time()})}\n\n"

            # On startup, mark all existing interactions as "sent" to avoid flooding with old messages
            # We only want to stream NEW interactions from this point forward
            # Use current session's log file only (not glob of all files - avoids cross-execution bleed)
            try:
                log_file = llm_logger.log_file
                if log_file and os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                if entry.get("type") == "interaction":
                                    timestamp = entry.get("timestamp", "")
                                    if timestamp:
                                        sent_timestamps.add(timestamp)
                            except Exception:
                                continue
                logger.info(f"SSE: Marked {len(sent_timestamps)} existing interactions as already sent")
            except Exception as init_e:
                logger.warning(f"SSE: Error initializing sent timestamps: {init_e}")

            while True:
                try:
                    heartbeat_counter += 1

                    # Use simple file reading instead of complex get_agent_thinking()
                    current_step = 0

                    with step_lock:
                        current_step = agent_step_count

                    new_interactions = []
                    try:
                        # Use current session's log file only (not glob - avoids cross-execution bleed)
                        log_file = llm_logger.log_file
                        if log_file and os.path.exists(log_file):
                            with open(log_file, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                                # Only check last 10 lines for new entries (performance optimization)
                                for line in lines[-10:]:
                                    try:
                                        entry = json.loads(line.strip())
                                        if entry.get("type") == "interaction":
                                            timestamp = entry.get("timestamp", "")
                                            # Only add if we haven't sent this timestamp before
                                            if timestamp and timestamp not in sent_timestamps:
                                                model_name = (
                                                    entry.get("model_info", {}) or {}
                                                ).get("model", "")
                                                new_interactions.append(
                                                    {
                                                        "type": _format_interaction_type_for_ui(
                                                            entry.get("interaction_type", "unknown")
                                                        ),
                                                        "model": model_name,
                                                        "provider_family": _provider_family_for_llm_log_entry(
                                                            entry
                                                        ),
                                                        "response": entry.get("response", ""),
                                                        "duration": entry.get("duration", 0),
                                                        "timestamp": timestamp,
                                                        "agent_step": entry.get("agent_step"),
                                                    }
                                                )
                                    except Exception:
                                        continue
                    except Exception as file_e:
                        logger.warning(f"SSE: File reading error: {file_e}")

                    # Sort by timestamp to ensure chronological order
                    new_interactions.sort(key=lambda x: x.get("timestamp", ""))

                    # Check if there are new interactions
                    if new_interactions:
                        logger.info(f"SSE: Found {len(new_interactions)} new interactions to send")
                        # Send new interactions
                        for interaction in new_interactions:
                            # Prefer per-line global step (looping subagents); else server counter
                            line_step = interaction.get("agent_step")
                            if line_step is None:
                                line_step = current_step
                            event_data = {
                                "step": line_step,
                                "type": interaction.get("type", "unknown"),
                                "model": interaction.get("model", ""),
                                "provider_family": interaction.get(
                                    "provider_family", "other"
                                ),
                                "response": interaction.get("response", ""),
                                "duration": interaction.get("duration", 0),
                                "timestamp": interaction.get("timestamp", ""),
                                "is_new": True,
                            }

                            yield f"data: {json.dumps(event_data)}\n\n"
                            # Mark this timestamp as sent
                            sent_timestamps.add(interaction.get("timestamp", ""))

                    # Send periodic heartbeat to keep connection alive (every 10 cycles = 5 seconds)
                    if not new_interactions and heartbeat_counter % 10 == 0:
                        yield f"data: {json.dumps({'heartbeat': True, 'timestamp': time.time(), 'step': current_step})}\n\n"

                    # Wait before checking again (increased from 500ms to 2s for better performance)
                    await asyncio.sleep(2.0)

                except Exception as e:
                    logger.error(f"SSE: Error in stream loop: {e}")
                    yield f"data: {json.dumps({'error': str(e), 'timestamp': time.time()})}\n\n"
                    await asyncio.sleep(2)

        except Exception as outer_e:
            logger.error(f"SSE: Fatal error in event stream: {outer_e}")
            yield f"data: {json.dumps({'fatal_error': str(outer_e), 'timestamp': time.time()})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/agent")
async def get_agent_thinking():
    """Get current agent thinking status and recent LLM interactions"""
    try:
        # Get the most recent LLM log file
        from utils.data_persistence.llm_logger import get_llm_logger

        # Get recent LLM interactions
        llm_logger = get_llm_logger()
        session_summary = llm_logger.get_session_summary()

        # Use current session's log file only (not glob - avoids cross-execution bleed)
        recent_interactions = []
        log_file = llm_logger.log_file
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("type") == "interaction":
                                model_name = (
                                    entry.get("model_info", {}) or {}
                                ).get("model", "")
                                recent_interactions.append(
                                    {
                                        "type": _format_interaction_type_for_ui(
                                            entry.get("interaction_type", "unknown")
                                        ),
                                        "model": model_name,
                                        "provider_family": _provider_family_for_llm_log_entry(
                                            entry
                                        ),
                                        "prompt": entry.get("prompt", ""),
                                        "response": entry.get("response", ""),
                                        "duration": entry.get("duration", 0),
                                        "timestamp": entry.get("timestamp", ""),
                                        "agent_step": entry.get("agent_step"),
                                    }
                                )
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error reading LLM log {log_file}: {e}")

        # Sort by timestamp and keep only the most recent interaction (current step)
        recent_interactions.sort(key=lambda x: x.get("timestamp", ""))
        recent_interactions = recent_interactions[-1:] if recent_interactions else []
        logger.info(f"Found {len(recent_interactions)} recent interactions (showing current step only)")

        # Format the agent thinking display
        if recent_interactions:
            interaction = recent_interactions[-1]  # Get the most recent interaction
            current_thought = f"Current step LLM output:\n"
            duration = interaction.get("duration", 0)
            current_thought += f"{interaction['type'].upper()} ({duration:.2f}s)\n"
            current_thought += f"Response: {interaction['response']}"
        else:
            current_thought = "No recent LLM interactions. Agent is ready to process game state."

        with step_lock:
            current_step = agent_step_count  # Use agent step count instead of frame step count

        return {
            "status": "thinking",
            "current_thought": current_thought,
            "confidence": 0.85,
            "timestamp": time.time(),
            "llm_session": session_summary,
            "recent_interactions": recent_interactions,
            "current_step": current_step,
        }

    except Exception as e:
        logger.error(f"Error in agent thinking: {e}")
        return {
            "status": "error",
            "current_thought": f"Error getting agent thinking: {str(e)}",
            "confidence": 0.0,
            "timestamp": time.time(),
        }


@app.get("/metrics")
async def get_metrics():
    """Get cumulative metrics for the run"""
    global latest_metrics

    try:
        # Return the latest metrics received from client (with thread safety)
        with step_lock:
            metrics = latest_metrics.copy()
            metrics["agent_step_count"] = agent_step_count

        # If metrics haven't been initialized by client yet, try to load from cumulative_metrics.json
        # BUT only if checkpoint loading is enabled (not for fresh starts with --load-state)
        if metrics.get("total_llm_calls", 0) == 0 and checkpoint_loading_enabled:
            from utils.data_persistence.llm_logger import get_llm_logger
            llm_logger = get_llm_logger()
            if llm_logger and llm_logger.load_cumulative_metrics():
                metrics.update(llm_logger.cumulative_metrics)
            # agent_step_count comes from checkpoint_llm.txt (not cumulative_metrics.json)
            from utils.data_persistence.run_data_manager import get_checkpoint_llm_path
            checkpoint_file = get_checkpoint_llm_path()
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        checkpoint_data = json.load(f)
                        if "agent_step_count" in checkpoint_data:
                            metrics["agent_step_count"] = checkpoint_data["agent_step_count"]
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug(f"Could not read agent_step_count from checkpoint: {e}")

        return metrics

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "total_actions": 0,
            "total_run_time": 0,
            "total_llm_calls": 0,
            "agent_step_count": agent_step_count,
        }


# Store latest metrics from client
latest_metrics = {
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_cost": 0.0,
    "total_actions": 0,
    "total_run_time": 0,
    "total_llm_calls": 0,
    "start_time": time.time(),  # Will be overwritten if checkpoint is loaded
}


@app.get("/termination_condition")
async def get_termination_condition(condition_type: str = "gym_badge_count", threshold: int = 1):
    """Check if a termination condition is met based on ground-truth memory data.
    
    This endpoint reads game state directly from ROM memory to provide reliable
    termination conditions for external CLI agents (Claude Code, Codex, etc.).
    
    Args:
        condition_type: Type of condition to check. Supported types:
            - "gym_badge_count": Check number of gym badges obtained
        threshold: Threshold value for the condition (e.g., 1 for first badge)
    
    Returns:
        JSON with condition status:
        {
            "condition_type": str,
            "threshold": int,
            "current_value": int,
            "condition_met": bool,
            "badge_names": list (for gym_badge_count)
        }
    """
    global env
    
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    if not env.memory_reader:
        raise HTTPException(status_code=500, detail="Memory reader not initialized")
    
    try:
        if condition_type == "gym_badge_count":
            # Read badges directly from ROM memory (ground truth)
            with memory_lock:
                badges = env.memory_reader.read_badges()
            badge_count = len(badges) if badges else 0
            
            return {
                "condition_type": condition_type,
                "threshold": threshold,
                "current_value": badge_count,
                "badge_names": badges,
                "condition_met": badge_count >= threshold
            }
        
        # Future condition types can be added here:
        # elif condition_type == "pokemon_count":
        #     party_size = env.memory_reader.read_party_size()
        #     return {...}
        
        else:
            return {
                "error": f"Unknown condition type: {condition_type}",
                "supported_types": ["gym_badge_count"]
            }
    
    except Exception as e:
        logger.error(f"Error checking termination condition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Flag to track whether checkpoint loading should be enabled
checkpoint_loading_enabled = True  # Will be set based on startup args


@app.post("/reset_metrics")
async def reset_metrics():
    """Reset all metrics to zero for fresh start"""
    global latest_metrics, agent_step_count, checkpoint_loading_enabled

    with step_lock:
        latest_metrics.update(
            {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost": 0.0,
                "total_actions": 0,
                "total_run_time": 0,
                "total_llm_calls": 0,
                "start_time": time.time(),
            }
        )
        agent_step_count = 0
        # Disable checkpoint loading to prevent loading from checkpoint_llm.txt
        checkpoint_loading_enabled = False

    print("🔄 Server metrics reset for fresh start - checkpoint loading disabled")
    return {"status": "reset", "timestamp": time.time()}


@app.post("/agent_step")
async def update_agent_step(request: Request = None):
    """Update the agent step count and metrics"""
    global agent_step_count, latest_metrics

    skip_default_increment = False

    try:
        # Check if this is a direct set operation or has metrics
        if request:
            try:
                request_data = await request.json()

                # Store agent thinking if provided (same path as VLM: log to LLM logger so SSE has one source)
                if "thinking" in request_data:
                    thinking_text = request_data["thinking"]
                    interaction_type = request_data.get("interaction_type", "thinking")
                    duration = float(request_data.get("duration", 0))
                    thinking_step = request_data.get("agent_step")
                    if thinking_step is None:
                        thinking_step = request_data.get("step")
                    thinking_step_int = None
                    if thinking_step is not None:
                        try:
                            thinking_step_int = int(thinking_step)
                        except (TypeError, ValueError):
                            pass
                    try:
                        from utils.data_persistence.llm_logger import get_llm_logger

                        get_llm_logger().log_thinking(
                            thinking_text,
                            interaction_type,
                            duration,
                            agent_step=thinking_step_int,
                        )
                    except Exception as e:
                        logger.debug(f"Could not log thinking: {e}")

                # Update metrics if provided (with thread safety)
                if "metrics" in request_data and isinstance(request_data["metrics"], dict):
                    metrics_payload = request_data["metrics"]
                    with step_lock:  # Use existing lock for thread safety
                        # Safely update each metric individually to avoid race conditions
                        for key, value in metrics_payload.items():
                            if key in latest_metrics:
                                # Always protect total_actions as it's managed by server
                                if key == "total_actions":
                                    continue
                                else:
                                    latest_metrics[key] = value

                    # Full cumulative sync from client: align server step with max global LLM step
                    if metrics_payload.get("total_llm_calls") is not None:
                        steps = metrics_payload.get("steps") or []
                        max_s = 0
                        for s in steps:
                            try:
                                max_s = max(max_s, int(s.get("step") or 0))
                            except (TypeError, ValueError):
                                continue
                        with step_lock:
                            agent_step_count = max(agent_step_count, max_s)
                        # Only skip legacy +1 when we have step rows; empty steps keeps old increment behavior
                        if steps:
                            skip_default_increment = True

                # Handle set_step for initialization
                if "set_step" in request_data:
                    with step_lock:
                        agent_step_count = request_data["set_step"]
                    return {"status": "set", "agent_step": agent_step_count}
            except Exception as e:
                logger.error(f"Error processing agent_step request: {e}")
                # Continue with default increment behavior
    except Exception as e:
        logger.error(f"Error in agent_step endpoint: {e}")
        # Continue with default increment behavior

    if skip_default_increment:
        return {"status": "updated", "agent_step": agent_step_count}

    # Default increment behavior
    with step_lock:
        agent_step_count += 1

        # Save end-state snapshot every 20 steps
        if agent_step_count % 20 == 0:
            try:
                from utils.data_persistence.run_data_manager import get_run_data_manager

                run_manager = get_run_data_manager()
                if run_manager:
                    run_manager.save_end_state_snapshot()
                    logger.info(f"💾 Saved end-state snapshot at step {agent_step_count}")
            except Exception as e:
                logger.debug(f"Could not save end-state snapshot: {e}")

    return {"status": "updated", "agent_step": agent_step_count}


@app.get("/llm_logs")
async def get_llm_logs():
    """Get recent LLM log entries"""
    try:
        from utils.data_persistence.llm_logger import get_llm_logger

        llm_logger = get_llm_logger()
        session_summary = llm_logger.get_session_summary()

        # Get recent log entries
        recent_entries = []
        if os.path.exists(llm_logger.log_file):
            try:
                with open(llm_logger.log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # Get the last 20 entries
                    for line in lines[-20:]:
                        try:
                            entry = json.loads(line.strip())
                            recent_entries.append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error reading LLM log: {e}")

        return {"session_summary": session_summary, "recent_entries": recent_entries, "log_file": llm_logger.log_file}

    except Exception as e:
        logger.error(f"Error getting LLM logs: {e}")
        return {"error": str(e)}


# Helper function to update objectives cache file (for fast reads by stream.html)
def _load_objectives_sequence():
    """Load the configured objective sequence, restoring this run's saved progress first.

    The single loader for every lazy-load path. /mcp/complete_direct_objective used to
    load a fresh sequence at the start index without looking at objectives.json, so a
    completion call that arrived before the first state request reset a resumed run's
    objectives to the beginning and saved that over the real progress.
    """
    from utils.data_persistence.run_data_manager import get_run_data_manager, get_cache_path

    run_manager = get_run_data_manager()
    objectives_run_dir = str(run_manager.get_scratch_space_dir()) if run_manager else None

    restored_from_file = False
    try:
        cache_objectives_path = str(get_cache_path("objectives.json"))
        if os.path.exists(cache_objectives_path):
            with open(cache_objectives_path, "r", encoding="utf-8") as f:
                direct_objectives_manager.restore_from_state(json.load(f))
            restored_from_file = True
            logger.info(f"✅ Restored objectives from {cache_objectives_path}")
    except Exception as e:
        logger.warning(f"Failed to restore objectives from cache, falling back to sequence load: {e}")

    if not restored_from_file:
        if direct_objectives_sequence == "autonomous_objective_creation":
            direct_objectives_manager.load_autonomous_objective_creation_sequence(
                direct_objectives_start_index, run_dir=objectives_run_dir
            )
        elif direct_objectives_sequence == "categorized_full_game":
            direct_objectives_manager.load_categorized_full_game_sequence(
                start_story_index=direct_objectives_start_index,
                start_battling_index=direct_objectives_battling_start_index,
                run_dir=objectives_run_dir,
            )
        else:
            logger.warning(f"Unknown direct objectives sequence: {direct_objectives_sequence}")

        # Persist the freshly-loaded state
        direct_objectives_manager.auto_save()

    # Update objectives cache after loading
    _update_objectives_cache()


# Completion conditions the game itself records as an event flag (pret/pokeemerald
# include/constants/flags.h; SYSTEM_FLAGS = 0x860), with what to call the thing.
_SYSTEM_FLAGS = 0x860
_CONDITION_FLAGS = {
    "pokedex_received": (_SYSTEM_FLAGS + 0x1, "the POKéDEX"),  # FLAG_SYS_POKEDEX_GET
    "pokenav_received": (_SYSTEM_FLAGS + 0x2, "the POKéNAV"),  # FLAG_SYS_POKENAV_GET
    # Seen live: the theft happened while David played the gym exit himself; the AI then
    # sat on "Exit Gym - Devon Researcher reports theft" with nothing left to trigger.
    "theft_event_triggered": (0x8E, "seen the DEVON GOODS being stolen"),  # FLAG_DEVON_GOODS_STOLEN
    "running_shoes_received": (_SYSTEM_FLAGS + 0x60, "the RUNNING SHOES"),  # FLAG_SYS_B_DASH
    "stone_badge_obtained": (_SYSTEM_FLAGS + 0x7, "the STONE BADGE"),  # FLAG_BADGE01_GET ...
    "knuckle_badge_obtained": (_SYSTEM_FLAGS + 0x8, "the KNUCKLE BADGE"),
    "dynamo_badge_obtained": (_SYSTEM_FLAGS + 0x9, "the DYNAMO BADGE"),
    "heat_badge_obtained": (_SYSTEM_FLAGS + 0xA, "the HEAT BADGE"),
    "balance_badge_obtained": (_SYSTEM_FLAGS + 0xB, "the BALANCE BADGE"),
    "feather_badge_obtained": (_SYSTEM_FLAGS + 0xC, "the FEATHER BADGE"),
    "mind_badge_obtained": (_SYSTEM_FLAGS + 0xD, "the MIND BADGE"),
    "rain_badge_obtained": (_SYSTEM_FLAGS + 0xE, "the RAIN BADGE"),
    # Story scenes, each read from the map script that plays it (pret/pokeemerald
    # data/maps/<Map>/scripts.inc). None of these flags is ever cleared. Where a step
    # names part of a scene, the flag is the one its script sets at the end of that scene.
    "steven_found_granite": (0xBD, "found Steven and given him the LETTER"),  # FLAG_DELIVERED_STEVEN_LETTER
    "letter_delivered": (0xBD, "given Steven the LETTER"),  # GraniteCave_StevensRoom
    # The towns' map scripts set these on arrival, and at this point of the story Mr.
    # Briney's boat is the only way there (it lands at Slateport's beach on Route 109).
    "sailed_to_dewford": (_SYSTEM_FLAGS + 0x11, "arrived in DEWFORD TOWN"),  # FLAG_VISITED_DEWFORD_TOWN
    "sailed_to_slateport": (_SYSTEM_FLAGS + 0x17, "arrived in SLATEPORT CITY"),  # FLAG_VISITED_SLATEPORT_CITY
    "shipyard_checked": (0x94, "talked to Dock at the Shipyard"),  # FLAG_DOCK_REJECTED_DEVON_GOODS
    # SlateportCity_OceanicMuseum_2F: grunts, Archie's warning, then the DEVON GOODS handed
    # to Stern, all one scene that ends with this flag; it needs the museum to be entered.
    "museum_entered": (0x95, "been through the Oceanic Museum"),  # FLAG_DELIVERED_DEVON_GOODS
    "archie_first_encounter": (0x95, "met Archie in the Oceanic Museum"),
    "goods_delivered": (0x95, "delivered the DEVON GOODS to Captain Stern"),
    "meteor_falls_event_witnessed": (0xCF, "seen Archie and Maxie at Meteor Falls"),  # FLAG_MET_ARCHIE_METEOR_FALLS
    # MtChimney: set after the Maxie battle once Archie has thanked you and left.
    "archie_chimney_fled": (0x8B, "seen Team Magma and Archie leave Mt. Chimney"),  # FLAG_DEFEATED_EVIL_TEAM_MT_CHIMNEY
    "wynaut_egg_received": (0x10A, "the EGG from the Lavaridge hot springs"),  # FLAG_RECEIVED_LAVARIDGE_EGG
    "castform_received": (0x97, "CASTFORM"),  # FLAG_RECEIVED_CASTFORM
    # Route120: Steven reveals the Kecleon, you battle it, and he then hands over the
    # DEVON SCOPE whatever the battle's outcome (won, ran or caught).
    "steven_route120_found": (0x11D, "found Steven and got the DEVON SCOPE"),  # FLAG_RECEIVED_DEVON_SCOPE
    "kecleon_route120_battled": (0x11D, "battled the Kecleon and got the DEVON SCOPE"),
    "gym_kecleon_cleared": (0x127, "used the DEVON SCOPE on the Kecleon at the Gym"),  # FLAG_KECLEON_FLED_FORTREE
    "submarine_escape_witnessed": (0x70, "seen Team Aqua's submarine leave"),  # FLAG_TEAM_AQUA_ESCAPED_IN_SUBMARINE
    # SeafloorCavern_Room9: Archie wakes Kyogre, it escapes, Maxie arrives and Archie
    # admits his mistake; the flag closes that scene.
    "kyogre_awakened": (0x81, "seen Kyogre awaken"),  # FLAG_KYOGRE_ESCAPED_SEAFLOOR_CAVERN
    "archie_regret_witnessed": (0x81, "seen Archie admit his mistake"),
    "steven_wallace_found": (0x133, "been led to the Cave of Origin by Steven"),  # FLAG_STEVEN_GUIDES_TO_CAVE_OF_ORIGIN
    "sky_pillar_quest_received": (0x137, "been told by Wallace to go to Sky Pillar"),  # FLAG_WALLACE_GOES_TO_SKY_PILLAR
    "crisis_resolved": (0x9E, "talked to Archie and Maxie after Rayquaza"),  # FLAG_SOOTOPOLIS_ARCHIE_MAXIE_LEAVE
    # SootopolisCity: Wallace's thanks and HM07 are one conversation.
    "wallace_thanks": (0x138, "been thanked by Wallace and given HM07"),  # FLAG_RECEIVED_HM_WATERFALL
    "elite_four_entered": (0x107, "passed the League guards"),  # FLAG_ENTERED_ELITE_FOUR
    "hall_of_fame_entered": (_SYSTEM_FLAGS + 0x4, "entered the Hall of Fame"),  # FLAG_SYS_GAME_CLEAR
    # Battling objectives.
    "master_ball_found": (0x464, "picked up the MASTER BALL"),  # FLAG_ITEM_AQUA_HIDEOUT_B1F_MASTER_BALL
    "steven_postgame_defeated": (0x4F8, "beaten Steven at Meteor Falls"),  # FLAG_DEFEATED_METEOR_FALLS_STEVEN
}

# Story scenes the game records as a progress var (pret/pokeemerald include/constants/
# vars.h) that only moves forward: (var id, value the scene's script sets, what you did).
# Met when the var has reached that value.
_CONDITION_VARS = {
    # RustboroCity_DevonCorp_3F: set after Mr. Stone's talk and the POKéNAV.
    "mr_stone_met": (0x408F, 1, "met Mr. Stone"),  # VAR_DEVON_CORP_3F_STATE
    # MtPyre_Summit: set once Archie leaves with the BLUE ORB, before the MAGMA EMBLEM.
    "blue_orb_stolen_learned": (0x40B9, 1, "seen Team Aqua take the BLUE ORB"),  # VAR_MT_PYRE_STATE
    # MossdeepCity_SpaceCenter_2F: 1 when the attack starts (after the Mind Badge), 2
    # after the three grunts upstairs, 3 after Maxie and Tabitha give up. The Maxie and
    # Tabitha fight is a DoSpecialTrainerBattle with Steven, which ends in
    # HandleSpecialTrainerBattleEnd (src/battle_tower.c) and never sets trainer flags, so
    # this var is the only record of that win.
    "space_center_attack_triggered": (0x409F, 2, "reached the Space Center and fought Team Magma"),
    "space_center_magma_defeated": (0x409F, 2, "beaten the Team Magma grunts in the Space Center"),
    "maxie_tabitha_defeated": (0x409F, 3, "beaten Maxie and Tabitha"),  # VAR_MOSSDEEP_SPACE_CENTER_STATE
    "magma_space_retreat": (0x409F, 3, "seen Team Magma give up"),
    # VAR_SOOTOPOLIS_CITY_STATE: 1 Kyogre escaped, 2 the legendaries' fight seen from the
    # city, 3 Wallace sends you to Sky Pillar, 4 its door opened, 5 Rayquaza woken
    # (SkyPillar_Top), 6 Juan beaten.
    "legendary_battle_witnessed": (0x405E, 2, "seen Kyogre and Groudon fight"),
    "rayquaza_awakened": (0x405E, 5, "woken Rayquaza"),
    # SootopolisCity: 1 at Rayquaza's waking, then 2 or 3 once it has calmed the city.
    "rayquaza_intervention_witnessed": (0x40CA, 2, "seen Rayquaza stop Kyogre and Groudon"),  # VAR_SKY_PILLAR_STATE
}

# "Teach <HM move>" objectives: met when a party Pokémon knows the move (names as the
# party reader spells them, pokemon_env/memory_reader.py _read_pokemon_moves_from_decrypted).
_CONDITION_MOVES = {
    "rock_smash_taught": "Rock Smash",
    "surf_taught": "Surf",
    "fly_taught": "Fly",
    "dive_taught": "Dive",
    "waterfall_taught": "Waterfall",
}

# Objectives about being given an item: met when the bag holds it (names as the bag reader
# spells them, pokemon_env/emerald_utils.py). TMs and HMs are one-time gifts here, and
# either bike counts for Rydel's.
_CONDITION_ITEMS = {
    "pokeballs_received": (("Poke Ball",), "POKé BALLS"),
    "great_ball_received": (("Great Ball",), "a GREAT BALL"),
    "devon_goods_obtained": (("Devon Goods",), "the DEVON GOODS"),
    "itemfinder_received": (("Itemfinder",), "the ITEMFINDER"),
    "bicycle_received": (("Mach Bike", "Acro Bike"), "a BIKE"),
    "meteorite_obtained": (("Meteorite",), "the METEORITE"),
    "go_goggles_received": (("Go-Goggles",), "the GO-GOGGLES"),
    "devon_scope_received": (("Devon Scope",), "the DEVON SCOPE"),
    "magma_emblem_received": (("Magma Emblem",), "the MAGMA EMBLEM"),
    **{
        f"{tm.lower()}_received": ((tm,), tm)
        for tm in ("TM39", "TM08", "TM34", "TM50", "TM42", "TM40", "TM04", "TM03",
                   "HM05", "HM06", "HM03", "HM02", "HM08", "HM07")
    },
}

# Story battles against one trainer: the game sets TRAINER_FLAGS_START + the trainer's id
# when that trainer is beaten (pret/pokeemerald include/constants/opponents.h, flags.h).
# A rival battle has one id per rival and starter; "any" is met when one of them is set.
_TRAINER_FLAGS_START = 0x500
_RIVAL_103 = (520, 523, 526, 529, 532, 535)
_RIVAL_110 = (521, 524, 527, 530, 533, 536)
_RIVAL_119 = (522, 525, 528, 531, 534, 537)
_CONDITION_TRAINERS = {
    "rival_battle_1_won": ("any", _RIVAL_103, "your rival on Route 103"),
    "aqua_grunt_woods_defeated": ("any", (10,), "the Team Aqua Grunt in Petalburg Woods"),
    "aqua_grunt_tunnel_defeated": ("any", (16,), "the Team Aqua Grunt in Rusturf Tunnel"),
    # SlateportCity_OceanicMuseum_2F fights both, back to back, in one scene.
    "museum_grunts_defeated": ("all", (20, 21), "the two Team Aqua Grunts in the Oceanic Museum"),
    "rival_battle_2_won": ("any", _RIVAL_110, "your rival on Route 110"),
    "wally_mauville_defeated": ("any", (656,), "Wally in Mauville"),
    "maxie_chimney_defeated": ("any", (602,), "Maxie on Mt. Chimney"),
    "shelly_defeated": ("any", (32,), "Shelly in the Weather Institute"),
    "rival_battle_3_won": ("any", _RIVAL_119, "your rival on Route 119"),
    "matt_hideout_defeated": ("any", (30,), "Matt in the Aqua Hideout"),
    # maxie_tabitha_defeated is in _CONDITION_VARS: that battle sets no trainer flag.
    "archie_found_seafloor": ("any", (34,), "Archie in the Seafloor Cavern"),  # found him, then fought him
    "archie_defeated_seafloor": ("any", (34,), "Archie in the Seafloor Cavern"),
    "wally_victory_road_defeated": ("any", (519, 657, 658, 659, 660), "Wally in Victory Road"),
    "sidney_defeated": ("any", (261,), "Sidney"),
    "phoebe_defeated": ("any", (262,), "Phoebe"),
    "glacia_defeated": ("any", (263,), "Glacia"),
    "drake_defeated": ("any", (264,), "Drake"),
    "champion_wallace_defeated": ("any", (335,), "Wallace"),
}


def _set_grass_cost_for(guidance: Dict[str, Any]) -> None:
    """Walk through tall grass while a training objective is current; go around it otherwise.

    Travelling, a wild battle costs about a minute of model time, so routes avoid grass
    (utils/mapping/pathfinding.py). Training, those battles are the point (David,
    2026-11-03: "make sure it doesn't avoid tall grass now. I want it to battle pokemon").
    A training objective is a story objective whose condition is a level goal not yet met.
    """
    from server import game_tools
    from utils.mapping.pathfinding import GRASS_TILE_COST_MULTIPLIER

    story = guidance.get("story") or {}
    condition = str(story.get("completion_condition") or "")
    training = condition.startswith(("level_", "lead_level_")) and "ALREADY TRUE" not in condition
    game_tools.pathfinder.grass_cost = 1.0 if training else GRASS_TILE_COST_MULTIPLIER


# _trainee_advantage: the trainee fights alone only if it knocks the foe out in this many
# hits or fewer (and sooner than the foe could knock it out).
TRAINEE_SOLO_HITS = 2


def _trainee_advantage(view: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The move the Pokémon in battle (the trainee) can win with itself, or None to switch.

    David (2026-11-03): "You don't need to switch train when the other pokemon is weak
    against shroomish", and later: "I don't think we absolutely need to use combusken to
    fight every time. The pokemon are strong enough to fight on their own." The trainee
    fights on its own when it wins the damage race: its best move (the battle view's
    best_move, by estimated HP) knocks the foe out in at most TRAINEE_SOLO_HITS hits, and
    in fewer hits than the foe needs with its own best move (gBattleMons' move ids, the
    same estimate). Anything else brings the partner in.

    Seen live: the level rule this replaces (foe at most 2 levels up, not super-effective,
    trainee above half HP) let a Lv13 MARILL fight Lv13-14 Route 117 wilds alone; it
    fainted, and the game's replacement menu then cost 30 minutes (M49).
    """
    player, target, best = view.get("player"), view.get("target"), view.get("best_move")
    if not player or not target or not best or not best.get("damage") or not player["hp"]:
        return None
    with memory_lock:
        reader = env.memory_reader
        foe_best = max(
            (reader.estimate_damage(target, player, move) for move, pp in zip(target["move_ids"], target["pp"]) if move and pp),
            default=0.0,
        )
    hits_needed = math.ceil(target["hp"] / best["damage"])
    hits_taken = math.ceil(player["hp"] / foe_best) if foe_best else math.inf
    return best if hits_needed <= TRAINEE_SOLO_HITS and hits_needed < hits_taken else None


def _switch_train_line(guidance: Dict[str, Any]) -> str:
    """Top-of-state line: switch the Pokémon being trained out on its first turn.

    Switch-training (David, 2026-11-03): the trainee leads, a stronger partner comes in
    on turn 1 and wins, and both share the EXP. Seen live: the model attacked with the
    Lv6 SHROOMISH first ("Best move now" in the battle view) and switched only after it
    had taken damage. The trainee is the species in the story objective's unmet
    level_<species>_<n> condition; the partner is the highest-level teammate above half HP.

    Outside a battle the line names train(), which does all of it on its own: David
    (2026-11-03) "It sits for a while after the battle is over" — each battle cost about
    three ~30 s decisions (walk into the grass, switch, attack). Nothing is said while
    the team is weak: the heal-first line is the one to follow then.
    """
    story = guidance.get("story") or {}
    condition = str(story.get("completion_condition") or "")
    if not condition.startswith("level_") or "ALREADY TRUE" in condition:
        return ""
    species, _, level = condition.split(":")[0][len("level_"):].rpartition("_")
    trainee = species.upper()
    with memory_lock:
        if not env.memory_reader.is_in_battle():
            party = env.memory_reader.read_party_pokemon() or []
            names = [str(p.species_name).upper() for p in party if not p.is_egg]
            if not level.isdigit() or trainee not in names or all(n == trainee for n in names):
                return ""  # nothing to train, or no teammate to fight for it
            if env.memory_reader.read_team_condition()["weak"]:
                return ""
            return (
                f'TRAINING: call train("{trainee}", until_level={int(level)}) now. It walks into the tall '
                f"grass and fights wild Pokémon with your strongest teammate while {trainee} shares the "
                "EXP, and answers when it is done or something needs you."
            )
        view = env.memory_reader.read_battle_view() or {}
        party = env.memory_reader.read_party_pokemon() or []
    active = str((view.get("player") or {}).get("name", "")).upper()
    if active != trainee:
        return ""
    move = _trainee_advantage(view)
    if move is not None:
        foe = str(view["target"]["name"]).upper()
        return (
            f"{trainee} can beat {foe} on its own ({move['name']} {move['effectiveness']:g}x, "
            f"~{round(move['damage'])} HP a hit vs {view['target']['hp']} HP left): "
            f'fight with use_move("{move["name"]}") — no switch needed.'
        )
    partners = [
        p for p in party
        if str(p.species_name).upper() != trainee and p.max_hp and p.current_hp * 2 > p.max_hp
    ]
    if not partners:
        return ""
    partner = max(partners, key=lambda p: p.level)
    name = str(partner.species_name).upper()
    return (
        f'SWITCH-TRAINING: {trainee} is being trained and must not fight. Call '
        f'switch_pokemon("{name}") now; {trainee} still shares the EXP. Then win with {name}. '
        f"Ignore the 'Best move' line below until {name} is out."
    )


def _catch_target_line(guidance: Dict[str, Any]) -> str:
    """A top-of-state line when the wild Pokémon in battle is the one an objective wants caught.

    The battle view ends with "Best move now: … call use_move(…)", which a small model
    follows. Against the Shroomish it was sent to catch (David, 2026-11-03), any attack
    from Combusken would knock it out, so this line overrides it.
    """
    wanted = _catch_wanted(guidance)
    if not wanted:
        return ""
    with memory_lock:
        if not env.memory_reader.is_in_battle():
            return ""
        view = env.memory_reader.read_battle_view() or {}
    if view.get("is_trainer"):
        return ""
    for foe in view.get("opponents") or []:
        name = str(foe.get("name", "")).upper()
        if name not in wanted:
            continue
        if foe.get("level", 0) < wanted[name]:
            # "Catch a level 10+ Sableye" (David, 2026-11-03): a lower one isn't it.
            return (
                f"NOT THIS ONE: this wild {name} is Lv{foe.get('level')}, below the Lv{wanted[name]} "
                f"your objective wants. Call run_from_battle and look for another."
            )
        return f"CATCH THIS ONE: the wild {name} Lv{foe.get('level')} in front of you is the Pokémon your objective wants. " + _catch_plan(view, foe)
    return ""


# Moves that put the foe to sleep (x2 catch rate) or paralyze it (x1.5) without hurting
# it, as the battle view names them (pokemon_env Move enum, "_" -> " ").
_CATCH_HELPER_MOVES = (
    "SPORE", "SLEEP POWDER", "HYPNOSIS", "LOVELY KISS", "SING", "GRASS WHISTLE",
    "STUN SPORE", "THUNDER WAVE", "GLARE",
)


def _catch_wanted(guidance: Dict[str, Any]) -> Dict[str, int]:
    """Species an open objective wants caught, with the lowest level that counts.

    "party_has_<species>" wants any level; "level_<species>_<n>" with no such species in
    the party yet wants one caught at Lv<n> or more ("catch a level 10+ Sableye").
    """
    entries = [guidance.get("story"), *(guidance.get("battling_group") or [])]
    wanted: Dict[str, int] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        text = str(e.get("completion_condition", ""))
        if "ALREADY TRUE" in text:
            continue
        condition = text.split(":")[0]
        if condition.startswith("party_has_"):
            wanted[condition[len("party_has_"):].upper()] = 1
        elif condition.startswith("level_"):
            species, _, level = condition[len("level_"):].rpartition("_")
            check = _party_condition(condition)
            if species and level.isdigit() and check is not None and check[1].startswith("no "):
                wanted[species.upper()] = int(level)
    return wanted


def _catch_plan(view: Dict[str, Any], foe: Dict[str, Any]) -> str:
    """What to do this turn to catch the foe: a sleep/paralysis move first if one is at
    hand, then the best ball in the bag. A full-HP Sableye (catch rate 45) breaks out of
    about 94% of POKé BALLs; paralyzed and in a GREAT BALL it is about 1 in 7."""
    helper = None
    if str(foe.get("status", "OK")).upper() in ("OK", "NONE", ""):
        helper = next(
            (m["name"] for m in view.get("moves") or [] if m.get("name") in _CATCH_HELPER_MOVES and m.get("pp")), None
        )
    with memory_lock:
        pockets = env.memory_reader.read_full_inventory() or {}
        weaken = _catch_weaken_move(env.memory_reader, view, foe)
    have = {i.get("name"): i.get("quantity", 0) for pocket in pockets.values() for i in (pocket or [])}
    ball = next((b for b in ("Ultra Ball", "Great Ball", "Poke Ball") if have.get(b)), "Poke Ball")
    first = (
        f'First call use_move("{helper}"): a sleeping or paralyzed Pokémon is much easier to catch. Then '
        if helper else "Call "
    )
    if weaken:
        # Seen live: seven GREAT BALLs at a full-HP ELECTRIKE, which left one GREAT BALL
        # and four POKé BALLs for the NUMEL still to catch.
        return (
            f'It is above half HP: first use_move("{weaken["name"]}") once to weaken it (at most '
            f'~{weaken["max"]} of its {foe["hp"]} HP, never a knockout). '
            f'{first.replace("First call", "Next call") if helper else "Then call "}'
            f'throw_ball("{ball}") and keep throwing if it breaks free. No other damaging moves.'
        )
    return (
        f"Do NOT use damaging moves (you could knock it out). {first}throw_ball(\"{ball}\") and keep "
        "throwing if it breaks free. Ignore the 'Best move' line below."
    )


EFFECT_FALSE_SWIPE = 101  # pret battle_move_effects.h: leaves the foe at 1 HP at worst


def _catch_weaken_move(reader, view: Dict[str, Any], foe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """An attack of the Pokémon out that weakens a foe above half HP but can't knock it
    out: FALSE SWIPE if it has it, else the weakest attack whose highest roll (the
    estimate / 0.925, critical hits aside) leaves it at 1 HP or more. None otherwise:
    go straight to balls. David: "Just catch the sableye. It's not that hard." (kept to
    one line and one move)."""
    if not foe.get("max_hp") or foe["hp"] * 2 <= foe["max_hp"]:
        return None
    usable = [m for m in view.get("moves") or [] if m.get("pp") and m.get("damage")]
    swipe = next((m for m in usable if reader._rom_battle_move(m["id"])["effect"] == EFFECT_FALSE_SWIPE), None)
    if swipe is not None:
        return {"name": swipe["name"], "max": min(foe["hp"] - 1, math.ceil(swipe["damage"] / 0.925))}
    safe = [m for m in usable if math.ceil(m["damage"] / 0.925) < foe["hp"]]
    if not safe:
        return None
    move = min(safe, key=lambda m: m["damage"])
    return {"name": move["name"], "max": math.ceil(move["damage"] / 0.925)}


def _party_condition(condition: str) -> Optional[Tuple[bool, str]]:
    """(met, detail) for "party_has_<species>" / "level_<species>_<n>", else None.

    Species names are compared case-insensitively against the party read from memory.
    The walkthrough's "Heal at <town> Pokemon Center" steps (healed_dewford,
    rustboro_heal_done, final_heal_done, elite_four_final_heal ...) are the same check:
    the game keeps no record of where the team was healed, and a team at full HP is what
    each of them is for.
    """
    healing = condition.startswith("healed_") or condition.endswith(("_heal_done", "_final_heal"))
    if condition == "party_healed" or healing:
        # "Go back to the pokemon center, heal the pokemon" (David, 2026-11-03).
        with memory_lock:
            party = env.memory_reader.read_party_pokemon() or []
        hurt = [
            f"{p.species_name} {p.current_hp}/{p.max_hp}" for p in party
            if p.current_hp < p.max_hp or (getattr(p, "status", None) and p.status.get_status_name() not in ("OK", "Normal", "None"))
        ]
        return (bool(party) and not hurt, "everyone at full HP" if not hurt else "still hurt: " + ", ".join(hurt))
    if condition.startswith("party_has_"):
        species, level = condition[len("party_has_"):], None
    elif condition.startswith("level_"):
        species, _, level = condition[len("level_"):].rpartition("_")
        if not species or not level.isdigit():
            return None
        level = int(level)
    else:
        return None
    with memory_lock:
        party = env.memory_reader.read_party_pokemon() or []
    names = [str(getattr(p, "species_name", "")).upper() for p in party]
    wanted = species.upper()
    # "Train SHROOMISH to Lv23" is still about it once it is a BRELOOM.
    with memory_lock:
        family = env.memory_reader.species_family(wanted)
    matching = [(p, n) for p, n in zip(party, names) if n in family]
    team = ", ".join(f"{n} Lv{p.level}" for p, n in zip(party, names)) or "empty party"
    if level is None:
        return (bool(matching), f"party: {team}")
    if not matching:
        return (False, f"no {wanted} in the party yet; party: {team}")
    best, now = max(((p.level, n) for p, n in matching), key=lambda t: t[0])
    called = wanted if now == wanted else f"{wanted} (now {now})"
    return (best >= level, f"{called} is Lv{best} of {level} needed")


def _note_way_to_story_target(guidance: Dict[str, Any]) -> None:
    """Put the way to the story objective's target map at the front of its navigation hint.

    Only the story objective, to keep the prompt short; see game_tools.way_to_place.
    """
    story = guidance.get("story")
    if env is None or not isinstance(story, dict) or not story.get("target_location"):
        return
    if str(story.get("completion_condition", "")).find("ALREADY TRUE") != -1:
        return  # done already: the call to make is the hint
    from server import game_tools

    snap = _field_snapshot()
    way = game_tools.way_to_place(
        _location_name(), (snap.x, snap.y) if snap is not None else None, story["target_location"]
    )
    if way:
        # One call walks it all (mcp_go_to); the way stays so the model knows the route.
        way += f' Or go_to("{story["target_location"]}") walks the whole way in one call.'
        story["navigation_hint"] = f"{way} {story.get('navigation_hint') or ''}".strip()


def _note_story_scenes(guidance: Dict[str, Any]) -> None:
    """Put the story scenes waiting on this map (game_tools.story_scenes) at the front of
    the story objective's hint, with the tile to walk to.

    Seen live: "Witness Team Magma/Aqua confrontation in Meteor Falls" on METEOR FALLS 1F
    1R, a room with nobody in it until the scene at (14,18) runs; the model walked back
    out to ROUTE 114.
    """
    story = guidance.get("story")
    if env is None or not isinstance(story, dict):
        return
    from server import game_tools
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    map_dir = ROM_TO_PORYMAP_MAP.get(_location_name(), "")
    if not map_dir:
        return
    with memory_lock:
        scenes = game_tools.story_scenes(env, map_dir)
    if not scenes:
        return
    snap = _field_snapshot()
    lines = []
    for name, tiles in scenes:
        if snap is not None:
            tiles = sorted(tiles, key=lambda t: abs(t[0] - snap.x) + abs(t[1] - snap.y))
        where = ", ".join(f"({x},{y})" for x, y in tiles)
        x, y = tiles[0]
        lines.append(
            f"A story scene ({name}) starts when you step on {where} on this map: "
            f"navigate_to({x}, {y}), then read it through."
        )
    story["navigation_hint"] = f"{' '.join(lines)} {story.get('navigation_hint') or ''}".strip()


def _team_weak_line(condition: Dict[str, Any], location: str, position: Optional[Tuple[int, int]]) -> str:
    """'TEAM IS WEAK (TORCHIC 15/35 HP): heal before any trainer — <where to heal>'."""
    from server import game_tools

    head = f"TEAM IS WEAK ({', '.join(condition['reasons'])}): heal before any trainer — "
    way = game_tools.way_to_pokemon_center(location, position)
    if not way:
        return head + f"you are inside {location}: call heal_at_pokecenter() now."
    route = game_tools.pokemon_center_route(location, position)
    if not route:
        return head + way  # none reachable on foot: the note says so
    # Seen live: the model was given the way and an exit to take, and never walked it.
    # heal_at_pokecenter makes the trip itself (_walk_to_pokemon_center).
    return head + f"call heal_at_pokecenter() now: it walks to {route[-1][1]} and heals."


def _note_team_condition(guidance: Optional[Dict[str, Any]]) -> Optional[str]:
    """Put a heal-first line at the front of the story objective while the team is weak.

    Seen live in Rustboro Gym: the model walked from one trainer to the next with a
    worn-down Torchic. David: "It should go to the pokemon center between trainers if
    the HP or PP are low". The story objective's navigation hint is the line the model
    follows best, so the instruction goes first there; on a battle objective the
    description itself starts with it too. Nothing is said in a battle (the battle
    section shows HP and PP). Returns the line, or None when the team is fine.
    """
    if env is None or getattr(env, "memory_reader", None) is None:
        return None
    with memory_lock:
        reader = env.memory_reader
        if reader.is_in_battle():
            return None
        condition = reader.read_team_condition()
    if not condition["weak"]:
        return None
    snap = _field_snapshot()
    line = _team_weak_line(condition, _location_name(), (snap.x, snap.y) if snap is not None else None)
    story = guidance.get("story") if isinstance(guidance, dict) else None
    if isinstance(story, dict):
        story["navigation_hint"] = f"{line} {story.get('navigation_hint') or ''}".strip()
        if story.get("action_type") == "battle":
            story["description"] = f"HEAL FIRST (team is weak), then: {story.get('description') or ''}".strip()
    return line


TRAIN_WALKS_NOTE = "train walks to reachable grass by itself: don't navigate_to the grass first."


def _note_train_walks_itself(guidance: Dict[str, Any]) -> None:
    """Objective text that names train( gets TRAIN_WALKS_NOTE.

    Seen live: a walkthrough hint said "In Route 112 grass call train(…)", and the model
    tried navigate_to into the grass first (failing on unreachable tiles) instead of
    calling train, which finds reachable grass itself.
    """
    entries = [guidance.get("story"), guidance.get("dynamics"), *(guidance.get("battling_group") or [])]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = f"{entry.get('description') or ''} {entry.get('navigation_hint') or ''}"
        if "train(" in text and "walks to reachable grass" not in text:
            entry["navigation_hint"] = f"{entry.get('navigation_hint') or ''} {TRAIN_WALKS_NOTE}".strip()


def _note_conditions_met_in_memory(guidance: Dict[str, Any]) -> None:
    """Say so on an objective whose completion condition the game has already recorded.

    Objectives complete only when the agent calls complete_direct_objective, and the
    agent judges that from the screen. Seen live: Birch handed over the POKéDEX inside a
    cutscene that auto-advance read through, the model never saw it happen, and it
    kept pressing A at May to "receive the Pokedex". The flag is ground truth, so the
    objective line now says it is done and gives the exact call; the agent still makes
    the call, so nothing completes behind its back.

    Getting somewhere is checked the same way: the map name is ground truth. Seen live:
    standing on Route 104 with "Travel west to Route 104 South" as the objective, the
    model read "South" as a direction and set off for Route 105 (all sea).
    """
    if env is None or getattr(env, "memory_reader", None) is None:
        return
    from server import game_tools

    entries = [guidance.get("story"), guidance.get("dynamics"), *(guidance.get("battling_group") or [])]
    place = None
    bag = None  # item names, read once and only when an item condition needs it
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        condition = str(entry.get("completion_condition") or "")
        known = _CONDITION_FLAGS.get(condition)
        if known:
            flag, thing = known
            with memory_lock:
                done = env.memory_reader._is_event_flag_set(flag)
            if done:
                entry["completion_condition"] = (
                    f"{condition}: ALREADY TRUE in game memory (you have {thing}). Call "
                    f'complete_direct_objective(objective_id="{entry.get("id")}") now.'
                )
            continue
        trainers = _CONDITION_TRAINERS.get(condition)
        if trainers:
            # Seen live: the Petalburg Woods Grunt battle ended inside dialogue the
            # model didn't follow; with "Battle Team Aqua Grunt" still open it went
            # back to the walkthrough's first chapter and wandered off to "save Birch".
            mode, ids, who = trainers
            with memory_lock:
                beaten = [env.memory_reader._is_event_flag_set(_TRAINER_FLAGS_START + i) for i in ids]
            if all(beaten) if mode == "all" else any(beaten):
                entry["completion_condition"] = (
                    f"{condition}: ALREADY TRUE in game memory (you have beaten {who}). Call "
                    f'complete_direct_objective(objective_id="{entry.get("id")}") now.'
                )
            continue
        progress = _CONDITION_VARS.get(condition)
        if progress:
            # Seen live: the agent sat on story steps whose scene had already played out
            # in auto-advanced dialogue; scenes that record a var instead of a flag.
            var, reached, did = progress
            with memory_lock:
                value = env.memory_reader.read_game_var(var)
            if value is not None and value >= reached:
                entry["completion_condition"] = (
                    f"{condition}: ALREADY TRUE in game memory (you have {did}). Call "
                    f'complete_direct_objective(objective_id="{entry.get("id")}") now.'
                )
            continue
        move = _CONDITION_MOVES.get(condition)
        if move:
            with memory_lock:
                party = env.memory_reader.read_party_pokemon() or []
            knows = [str(p.species_name).upper() for p in party if move.upper() in (m.upper() for m in p.moves)]
            if knows:
                entry["completion_condition"] = (
                    f"{condition}: ALREADY TRUE in game memory ({knows[0]} knows {move.upper()}). Call "
                    f'complete_direct_objective(objective_id="{entry.get("id")}") now.'
                )
            continue
        party_check = _party_condition(condition)
        if party_check is not None:
            # Team-building steps (David, 2026-11-03, after a whiteout to Roxanne: "we need
            # more pokemon"): "party_has_shroomish", "level_shroomish_12".
            met, detail = party_check
            entry["completion_condition"] = (
                f"{condition}: ALREADY TRUE in game memory ({detail}). Call "
                f'complete_direct_objective(objective_id="{entry.get("id")}") now.'
                if met else f"{condition}: not yet ({detail})."
            )
            species, _, level = condition[len("level_"):].rpartition("_")
            if not met and condition.startswith("level_") and not detail.startswith("no "):
                # The one call that does the grinding (see mcp_train).
                entry["completion_condition"] += (
                    f' Outside a battle, call train("{species.upper()}", until_level={level}).'
                )
            elif not met and condition.startswith("level_") and species and level.isdigit():
                # None in the party: catch one at that level ("a level 10+ Sableye", David).
                # train walks the encounter tiles and stops when one appears.
                with memory_lock:
                    party = env.memory_reader.read_party_pokemon() or []
                if party:
                    lead = str(party[0].species_name).upper()
                    entry["completion_condition"] += (
                        f" Catch one at Lv{level} or higher: where it lives, outside a battle, call "
                        f'train("{lead}", catch="{species.upper()}", catch_min_level={level}).'
                    )
            continue
        if condition.startswith("lead_level_") and condition[len("lead_level_"):].isdigit():
            # "Train TORCHIC to Lv16" (David, 2026-11-03): ground truth is the party's level.
            wanted = int(condition[len("lead_level_"):])
            with memory_lock:
                party = env.memory_reader.read_party_pokemon() or []
            if party:
                lead = party[0]
                name = getattr(lead, "nickname", None) or getattr(lead, "species_name", "your lead")
                if lead.level >= wanted:
                    entry["completion_condition"] = (
                        f"{condition}: ALREADY TRUE in game memory ({name} is Lv{lead.level}). Call "
                        f'complete_direct_objective(objective_id="{entry.get("id")}") now.'
                    )
                else:
                    entry["completion_condition"] = (
                        f"{condition}: not yet ({name} is Lv{lead.level} of {wanted} needed)."
                    )
            continue
        items = _CONDITION_ITEMS.get(condition)
        if items:
            # Seen live: "Receive Great Ball from Devon Researcher" stayed open with the
            # Great Ball already in the bag (handed over in auto-advanced dialogue).
            names, thing = items
            if bag is None:
                with memory_lock:
                    pockets = env.memory_reader.read_full_inventory() or {}
                bag = {i.get("name") for pocket in pockets.values() for i in (pocket or []) if i.get("quantity")}
            if bag & set(names):
                entry["completion_condition"] = (
                    f"{condition}: ALREADY TRUE in game memory (your bag has {thing}). Call "
                    f'complete_direct_objective(objective_id="{entry.get("id")}") now.'
                )
            continue
        # Only conditions that are about being somewhere: "gym_puzzle_solved" also
        # names a place, but standing in the gym doesn't solve it.
        about_place = condition.endswith(("_reached", "_entered")) or condition.startswith(("returned_", "in_"))
        if entry.get("action_type") != "navigate" or not about_place or not entry.get("target_location"):
            continue
        # "_reached"/"_entered" steps are first arrivals in story order, so the game's
        # own first-visit flag answers them after the player has moved on; "returned_"
        # steps revisit places already flagged, so they still need the map name.
        visit_flag = None
        if condition.endswith(("_reached", "_entered")):
            visit_flag = game_tools.first_visit_flag(entry["target_location"])
        if visit_flag is not None:
            with memory_lock:
                been = env.memory_reader._is_event_flag_set(visit_flag)
            if been:
                entry["completion_condition"] = (
                    f"{condition}: ALREADY TRUE in game memory (you have been to "
                    f"{entry['target_location']}). Call "
                    f'complete_direct_objective(objective_id="{entry.get("id")}") now.'
                )
                continue
        if place is None:
            snap = _field_snapshot()
            place = (_location_name(), (snap.x, snap.y) if snap is not None else None)
        if game_tools.arrived_at(place[0], place[1], entry["target_location"]):
            entry["completion_condition"] = (
                f"{condition}: ALREADY TRUE (you are on {place[0]}, which is "
                f"{entry['target_location']}). Call "
                f'complete_direct_objective(objective_id="{entry.get("id")}") now.'
            )


def _update_objectives_cache():
    """Write current objectives state to fast-access cache file"""
    try:
        global direct_objectives_manager
        objectives_data = {
            "mode": "legacy",
            "current": None,
            "recently_completed": [],
            "total_in_sequence": 0,
            "current_index": 0,
        }

        if direct_objectives_manager and direct_objectives_manager.is_sequence_active():
            logger.info(f"📊 Updating objectives cache - mode: {direct_objectives_manager.mode}")
            if direct_objectives_manager.mode == "categorized":
                # Categorized cache: only mode, per-category windows, and status counts.
                # Omit legacy root keys (current, recently_completed, total_in_sequence,
                # current_index); consumers use story/battling/dynamics + categorized_status.

                def get_recent_for_category(category, sequence, index):
                    items = []
                    # Current objective (if exists)
                    if index < len(sequence):
                        current = sequence[index]
                        items.append(
                            {"id": current.id, "description": current.description, "completed": False, "current": True}
                        )

                    # Recently completed (last 5 completed before current, in reverse order - most recent first)
                    for i in range(index - 1, max(-1, index - 6), -1):
                        if i >= 0 and i < len(sequence) and sequence[i].completed:
                            items.append(
                                {
                                    "id": sequence[i].id,
                                    "description": sequence[i].description,
                                    "completed": True,
                                    "current": False,
                                }
                            )

                    return items

                objectives_data = {
                    "mode": "categorized",
                    "story": get_recent_for_category(
                        "story", direct_objectives_manager.story_sequence, direct_objectives_manager.story_index
                    ),
                    "battling": get_recent_for_category(
                        "battling", direct_objectives_manager.battling_sequence, direct_objectives_manager.battling_index
                    ),
                    "dynamics": get_recent_for_category(
                        "dynamics", direct_objectives_manager.dynamics_sequence, direct_objectives_manager.dynamics_index
                    ),
                    "categorized_status": {
                        "story": {
                            "current_index": direct_objectives_manager.story_index,
                            "total": len(direct_objectives_manager.story_sequence),
                        },
                        "battling": {
                            "current_index": direct_objectives_manager.battling_index,
                            "total": len(direct_objectives_manager.battling_sequence),
                        },
                        "dynamics": {
                            "current_index": direct_objectives_manager.dynamics_index,
                            "total": len(direct_objectives_manager.dynamics_sequence),
                        },
                    },
                }

            else:
                # LEGACY: Single objective mode
                current_obj = direct_objectives_manager.get_current_objective()

                # Get recently completed objectives (last 5)
                completed_objectives = []
                if direct_objectives_manager.current_sequence:
                    for i in range(
                        max(0, direct_objectives_manager.current_index - 5), direct_objectives_manager.current_index
                    ):
                        if i < len(direct_objectives_manager.current_sequence):
                            obj = direct_objectives_manager.current_sequence[i]
                            completed_objectives.append(
                                {"id": obj.id, "description": obj.description, "completed": True, "index": i}
                            )

                objectives_data = {
                    "mode": "legacy",
                    "current": {
                        "id": current_obj.id,
                        "description": current_obj.description,
                        "index": direct_objectives_manager.current_index,
                    }
                    if current_obj
                    else None,
                    "recently_completed": completed_objectives,
                    "total_in_sequence": len(direct_objectives_manager.current_sequence),
                    "current_index": direct_objectives_manager.current_index,
                }

        # Write to cache file
        from utils.data_persistence.run_data_manager import get_cache_path
        cache_file = get_cache_path("current_objective.json")
        with open(cache_file, 'w') as f:
            json.dump(objectives_data, f, indent=2)

        logger.info(
            f"✅ Updated objectives cache: mode={objectives_data.get('mode')}, story={len(objectives_data.get('story', []))}, battling={len(objectives_data.get('battling', []))}, dynamics={len(objectives_data.get('dynamics', []))}"
        )

    except Exception as e:
        logger.error(f"Failed to update objectives cache: {e}")
        import traceback

        logger.error(traceback.format_exc())


# Milestone checking is now handled by the emulator


@app.get("/milestones")
async def get_milestones():
    """Get current milestones achieved based on persistent tracking"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        # Get milestones directly from emulator
        with memory_lock:
            result = env.get_milestones(agent_step_count=agent_step_count)

        # Read objectives from cached file (fast, no manager access needed)
        objectives_data = {
            "current": None,
            "recently_completed": [],
            "total_in_sequence": 0,
            "current_index": 0
        }

        try:
            from utils.data_persistence.run_data_manager import get_cache_path
            objectives_cache_file = get_cache_path("current_objective.json")
            if objectives_cache_file.exists():
                with open(objectives_cache_file, 'r') as f:
                    objectives_data = json.load(f)
        except Exception as e:
            logger.debug(f"Could not read objectives cache: {e}")

        result["objectives"] = objectives_data
        return result

    except Exception as e:
        logger.error(f"Error getting milestones: {e}")
        # Fallback to basic milestones if memory reading fails
        basic_milestones = [
            {"id": 1, "name": "GAME_STARTED", "category": "basic", "completed": True, "timestamp": time.time()},
            {"id": 2, "name": "EMULATOR_RUNNING", "category": "basic", "completed": True, "timestamp": time.time()},
        ]
        return {
            "milestones": basic_milestones,
            "completed": 2,
            "total": 2,
            "progress": 1.0,
            "tracking_system": "fallback",
            "objectives": {"current": None, "recently_completed": [], "total_in_sequence": 0, "current_index": 0},
            "error": str(e),
        }


# Global list to track recent button presses
recent_button_presses = []


@app.get("/recent_actions")
async def get_recent_actions():
    """Get recently pressed buttons"""
    global recent_button_presses
    return {
        "recent_buttons": recent_button_presses[-10:],  # Last 10 button presses
        "timestamp": time.time(),
    }


@app.get("/debug/milestones")
async def debug_milestones():
    """Debug milestone tracking system"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        # Get current working directory and list milestone files
        import glob

        current_dir = os.getcwd()
        milestone_files = glob.glob("*milestones*.json")

        # Check if default milestone file exists and get its info
        default_file_info = None
        if os.path.exists(env.milestone_tracker.filename):
            try:
                with open(env.milestone_tracker.filename, "r") as f:
                    default_data = json.load(f)
                default_file_info = {
                    "exists": True,
                    "size": os.path.getsize(env.milestone_tracker.filename),
                    "last_modified": time.ctime(os.path.getmtime(env.milestone_tracker.filename)),
                    "milestone_count": len(default_data.get("milestones", {})),
                    "last_updated": default_data.get("last_updated", "unknown"),
                }
            except Exception as e:
                default_file_info = {"exists": True, "error": str(e)}
        else:
            default_file_info = {"exists": False}

        return {
            "tracking_system": "file_based",
            "current_filename": env.milestone_tracker.filename,
            "current_milestones": len(env.milestone_tracker.milestones),
            "completed_milestones": sum(
                1 for m in env.milestone_tracker.milestones.values() if m.get("completed", False)
            ),
            "default_file_info": default_file_info,
            "milestone_files_in_directory": milestone_files,
            "working_directory": current_dir,
            "milestone_details": env.milestone_tracker.milestones,
        }
    except Exception as e:
        logger.error(f"Error in milestone debug: {e}")
        return {"error": str(e)}


@app.post("/debug/reset_milestones")
async def reset_milestones():
    """Reset all milestones (for testing)"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        env.milestone_tracker.reset_all()
        return {
            "status": "reset",
            "milestone_file": env.milestone_tracker.filename,
            "remaining_milestones": len(env.milestone_tracker.milestones),
        }
    except Exception as e:
        logger.error(f"Error resetting milestones: {e}")
        return {"error": str(e)}


@app.post("/debug/test_milestone_operations")
async def test_milestone_operations():
    """Test milestone loading and saving operations"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        # Test data
        test_milestones = {
            "TEST_MILESTONE_1": {"completed": True, "timestamp": time.time(), "first_completed": time.time()},
            "TEST_MILESTONE_2": {"completed": False, "timestamp": None},
        }

        # Save current state
        original_milestones = env.milestone_tracker.milestones.copy()
        original_filename = env.milestone_tracker.filename

        # Test 1: Save milestones with state filename
        test_state_filename = "test_state_123.sav"
        env.milestone_tracker.milestones = test_milestones.copy()
        saved_filename = env.milestone_tracker.save_milestones_for_state(test_state_filename)

        # Test 2: Load milestones for state
        env.milestone_tracker.milestones = {}  # Clear current milestones
        env.milestone_tracker.load_milestones_for_state(test_state_filename)
        loaded_milestones = env.milestone_tracker.milestones.copy()

        # Test 3: Check if file was created
        file_exists = os.path.exists(saved_filename)
        file_size = os.path.getsize(saved_filename) if file_exists else 0

        # Restore original state
        env.milestone_tracker.milestones = original_milestones
        env.milestone_tracker.filename = original_filename

        return {
            "test_results": {
                "save_operation": {
                    "filename": saved_filename,
                    "file_exists": file_exists,
                    "file_size": file_size,
                    "milestones_saved": len(test_milestones),
                },
                "load_operation": {
                    "milestones_loaded": len(loaded_milestones),
                    "milestones_match": loaded_milestones == test_milestones,
                    "loaded_milestones": loaded_milestones,
                },
            },
            "original_state_restored": True,
        }

    except Exception as e:
        logger.error(f"Error testing milestone operations: {e}")
        return {"error": str(e)}


# ============================================================================
# MCP TOOL ENDPOINTS
# ============================================================================


@app.post("/mcp/get_game_state")
async def mcp_get_game_state():
    """MCP Tool: Get current game state"""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        from utils.state_formatter import format_state_for_llm
        from server import game_tools
        from agents.objectives import DirectObjectiveManager

        # Get recent button presses with position history
        global recent_button_presses, current_obs

        # Get latest frame from game loop to ensure sync between memory and visuals
        with obs_lock:
            obs_copy = current_obs.copy() if current_obs is not None else None

        # Use helper function from game_tools with action history and current frame
        result = game_tools.get_game_state_direct(
            env,
            format_state_for_llm,
            action_history=recent_button_presses,
            current_obs=obs_copy,  # Pass latest frame from game loop
        )

        # Add direct objectives information
        team_line_placed = False
        if result.get("success", False):
            global direct_objectives_manager, current_run_dir

            # Initialize direct objective manager if needed
            if direct_objectives_manager is None:
                direct_objectives_manager = DirectObjectiveManager()

            # Load direct objectives sequence if specified
            if direct_objectives_sequence:
                # Check if we need to load objectives
                needs_loading = not direct_objectives_manager.is_sequence_active()
                logger.info(f"🔍 Checking if objectives need loading:")
                logger.info(f"   - is_active: {direct_objectives_manager.is_sequence_active()}")
                logger.info(f"   - current mode: {direct_objectives_manager.mode}")
                logger.info(f"   - current sequence_name: '{direct_objectives_manager.sequence_name}'")
                logger.info(f"   - requested sequence: '{direct_objectives_sequence}'")
                logger.info(f"   - needs_loading (initial): {needs_loading}")

                # Force reload if the requested sequence doesn't match the loaded sequence
                if direct_objectives_manager.is_sequence_active():
                    if direct_objectives_manager.sequence_name != direct_objectives_sequence:
                        logger.warning(
                            f"⚠️ Sequence mismatch: loaded='{direct_objectives_manager.sequence_name}', requested='{direct_objectives_sequence}' - forcing reload"
                        )
                        needs_loading = True
                    # Also force reload if requesting categorized but manager is in legacy mode
                    elif (
                        direct_objectives_sequence == "categorized_full_game"
                        and direct_objectives_manager.mode == "legacy"
                    ):
                        logger.warning(f"⚠️ Requesting categorized mode but manager is in legacy mode - forcing reload")
                        needs_loading = True

                logger.info(f"   - needs_loading (final): {needs_loading}")

                if needs_loading and os.environ.get("POKEAGENT_CLI_MODE") != "1":
                    # CLI agents do not use objectives; skip when POKEAGENT_CLI_MODE
                    _load_objectives_sequence()

            # Get current objective guidance
            if direct_objectives_manager.is_sequence_active():
                game_state = result.get("raw_state", {})

                # DEBUG: Log manager state
                logger.info(f"📊 Objectives manager state:")
                logger.info(f"   - mode: {direct_objectives_manager.mode}")
                logger.info(f"   - sequence_name: '{direct_objectives_manager.sequence_name}'")
                logger.info(f"   - is_active: {direct_objectives_manager.is_sequence_active()}")

                # Use categorized guidance if in categorized mode
                if direct_objectives_manager.mode == "categorized":
                    result["objectives_mode"] = "categorized"

                    categorized_guidance = direct_objectives_manager.get_categorized_objective_guidance(game_state)
                    if categorized_guidance:
                        try:
                            _note_conditions_met_in_memory(categorized_guidance)
                        except Exception as e:  # a hint must never break the state response
                            logger.warning(f"objective memory check failed: {e}")
                        try:
                            _note_way_to_story_target(categorized_guidance)
                        except Exception as e:
                            logger.warning(f"objective route hint failed: {e}")
                        try:
                            _note_story_scenes(categorized_guidance)
                        except Exception as e:
                            logger.warning(f"story scene note failed: {e}")
                        try:
                            _note_train_walks_itself(categorized_guidance)
                        except Exception as e:
                            logger.warning(f"train note failed: {e}")
                        try:
                            _set_grass_cost_for(categorized_guidance)
                        except Exception as e:
                            logger.warning(f"grass cost update failed: {e}")
                        try:
                            # After the route hint, so the heal line comes first.
                            team_line_placed = bool(_note_team_condition(categorized_guidance))
                        except Exception as e:
                            logger.warning(f"team condition check failed: {e}")
                        try:
                            catch_line = _catch_target_line(categorized_guidance) or _switch_train_line(categorized_guidance)
                            if catch_line and isinstance(result.get("state_text"), str):
                                result["state_text"] = f"{catch_line}\n\n{result['state_text']}"
                        except Exception as e:
                            logger.warning(f"catch target check failed: {e}")
                        result["categorized_objectives"] = categorized_guidance

                    # Add categorized status
                    result["categorized_status"] = {
                        "story": {
                            "current_index": direct_objectives_manager.story_index,
                            "total": len(direct_objectives_manager.story_sequence),
                            "completed": sum(1 for obj in direct_objectives_manager.story_sequence if obj.completed),
                        },
                        "battling": {
                            "current_index": direct_objectives_manager.battling_index,
                            "total": len(direct_objectives_manager.battling_sequence),
                            "completed": sum(1 for obj in direct_objectives_manager.battling_sequence if obj.completed),
                        },
                        "dynamics": {
                            "current_index": direct_objectives_manager.dynamics_index,
                            "total": len(direct_objectives_manager.dynamics_sequence),
                            "completed": sum(1 for obj in direct_objectives_manager.dynamics_sequence if obj.completed),
                        },
                    }
                else:
                    # Legacy mode
                    result["objectives_mode"] = "legacy"

                    current_guidance = direct_objectives_manager.get_current_objective_guidance(game_state)
                    if isinstance(current_guidance, dict):
                        try:
                            team_line_placed = bool(_note_team_condition({"story": current_guidance}))
                        except Exception as e:
                            logger.warning(f"team condition check failed: {e}")
                    if current_guidance:
                        result["direct_objective"] = current_guidance
                        result["direct_objective_status"] = direct_objectives_manager.get_sequence_status()

                    # Add objective context (previous, current, next) - legacy only
                    objective_context = direct_objectives_manager.get_objective_context(game_state)
                    if objective_context:
                        result["direct_objective_context"] = objective_context

        if result.get("success", False) and not team_line_placed and isinstance(result.get("state_text"), str):
            # No objective to carry the heal-first line: put it at the top of the state.
            try:
                line = _note_team_condition(None)
                if line:
                    result["state_text"] = f"{line}\n\n{result['state_text']}"
            except Exception as e:
                logger.warning(f"team condition check failed: {e}")

        # Ensure all data is JSON-serializable (objectives data may contain enums/numpy types)
        from utils.json_utils import serialize_for_json
        return serialize_for_json(result)
    except Exception as e:
        logger.error(f"Error in get_game_state: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_map_data")
async def mcp_get_map_data():
    """MCP Tool: Get structured map data for skill code.

    Uses the same preprocessing as get_game_state (state_text with ASCII map)
    but extracts and returns the grid, warps, objects, etc. as structured data.
    """
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        from utils.state_formatter import format_state_for_llm
        from server import game_tools

        global recent_button_presses, current_obs
        with obs_lock:
            obs_copy = current_obs.copy() if current_obs is not None else None

        state_result = game_tools.get_game_state_direct(
            env, format_state_for_llm,
            action_history=recent_button_presses,
            current_obs=obs_copy,
        )

        if not state_result.get("success"):
            return {"success": False, "error": "Could not get game state"}

        state_text = state_result.get("state_text", "")
        pos = state_result.get("player_position", {})
        location = state_result.get("location", "Unknown")

        # Game-specific player marker and legend
        _gt = os.environ.get("GAME_TYPE", "emerald").lower()
        if _gt == "red":
            _player_marker = "I"
            _grid_legend = (
                "I=player .=walkable #=wall ~=grass W=water D=door "
                "t=cuttable tree G=Card Key gate !=sign ?=hidden item "
                "O=pokeball N=NPC(blocked) ↓/←/→=jump ledge "
                "C=counter *=spinner stop B=bookshelf U=trash "
                "^=display/blueprint P=computer ==bench T=TV/machine"
            )
        else:
            _player_marker = "P"
            _grid_legend = "P=player .=walkable #=blocked ~=grass D=door S=stairs/warp I=item N=NPC(blocked)"

        result = {
            "success": True,
            "location": location,
            "player": {"x": pos.get("x", 0), "y": pos.get("y", 0)},
            "grid_legend": _grid_legend,
        }

        # Get the FULL map grid (not windowed) from the porymap data in raw_state
        # The state_text ASCII map may be cropped, but raw_state has the complete grid
        raw_state = state_result.get("raw_state", {})
        porymap_grid = raw_state.get("map", {}).get("porymap", {}).get("grid")
        grid = None

        if porymap_grid:
            # Use full porymap grid and add player marker
            grid = [list(row) for row in porymap_grid]
            px, py = pos.get("x", 0), pos.get("y", 0)
            if 0 <= py < len(grid) and 0 <= px < len(grid[0]):
                grid[py][px] = _player_marker

            # Mark live NPC positions as 'N' (blocked for pathfinding)
            obj_events = raw_state.get("map", {}).get("object_events", []) # object_events - from runtime memory; objects - from porymap data
            for obj in obj_events:
                ox = obj.get("current_x", -1)
                oy = obj.get("current_y", -1)
                if (ox, oy) != (px, py) and 0 <= oy < len(grid) and 0 <= ox < len(grid[0]):
                    if grid[oy][ox] not in ('#', _player_marker):
                        grid[oy][ox] = 'N'

            grid = ["".join(row) for row in grid]
        elif "ASCII Map:" in state_text:
            # Fallback: extract from state_text if porymap grid unavailable
            map_section = state_text.split("ASCII Map:")[1]
            legend_idx = map_section.find("(Legend:")
            if legend_idx > 0:
                map_section = map_section[:legend_idx]
            grid = [line for line in map_section.strip().split("\n") if line.strip()]

        if grid:
            result["grid"] = grid
            result["dimensions"] = {"width": len(grid[0]) if grid else 0, "height": len(grid)}

        # Extract the JSON map data block if present
        if "Map Data (JSON):" in state_text:
            try:
                import json as _json
                json_section = state_text.split("Map Data (JSON):")[1]
                # Find the JSON object
                start = json_section.find("{")
                if start >= 0:
                    depth = 0
                    end = start
                    for i, c in enumerate(json_section[start:], start):
                        if c == "{": depth += 1
                        elif c == "}": depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                    map_json = _json.loads(json_section[start:end])
                    result["warps"] = map_json.get("warps") or map_json.get("warp_events", [])
                    result["objects"] = map_json.get("objects", [])
                    result["connections"] = map_json.get("connections", [])
                    if not result.get("dimensions"):
                        result["dimensions"] = map_json.get("dimensions", {})
            except Exception:
                pass  # JSON parsing failed, grid is still available

        # Party info
        raw_state = state_result.get("raw_state", {})
        party = raw_state.get("player", {}).get("party") or raw_state.get("game", {}).get("party") or []
        if party:
            result["party"] = [
                {
                    "species": p.get("species_name", "?"),
                    "level": p.get("level", 0),
                    "hp": p.get("current_hp", 0),
                    "max_hp": p.get("max_hp", 0),
                    "moves": p.get("moves", []),
                }
                for p in party
            ]

        return result

    except Exception as e:
        logger.error(f"get_map_data error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _sequence_timeout(n_buttons: int) -> float:
    """Generous upper bound for n presses (hold + release + a ledge-jump settle each)."""
    per_press = SPEED_PRESETS["slow"]["hold"] + SPEED_PRESETS["slow"]["release"] + 48
    return min(80.0, 5.0 + 1.5 * n_buttons * per_press / max(1, fps))


async def _await_sequence(seq_id: Optional[int], timeout_s: float) -> Optional[Dict[str, Any]]:
    """Wait until the game loop has run (or dropped) every press of a sequence."""
    if seq_id is None:
        return None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        report = sequence_reports.get(seq_id)
        if report is None or report.get("done"):
            return report
        await asyncio.sleep(0.05)
    return sequence_reports.get(seq_id)


def _position_text(snap, location: Optional[str] = None) -> str:
    if snap is None:
        return "position unknown"
    text = f"({snap.x},{snap.y}) on {location or _location_name()}"
    if snap.facing:
        text += f", facing {snap.facing}"
    return text


def _screen_note() -> str:
    """Say so when a text box, choice or event is still on screen after the presses.

    Overnight the model spent five minutes on Route 103 pressing B and calling
    navigate_to under the rival's last text box: every result read "Pressed 1 button;
    now at (9,3)", so it kept believing it was still in a battle. Quoting the open box
    lets it act on what is actually on screen.
    """
    if env is None or game_type != "emerald":
        return ""
    from server.field_control import DIALOGUE_BUSY, DIALOGUE_CHOICE, DIALOGUE_PLAIN, dialogue_status, read_field_message

    snap = _field_snapshot()
    if snap is None:
        return ""
    status = dialogue_status(snap)
    if status not in (DIALOGUE_PLAIN, DIALOGUE_CHOICE, DIALOGUE_BUSY):
        return ""
    try:
        with memory_lock:
            text = read_field_message(env.memory_reader)
    except Exception as e:  # extra information only; never break the press result
        logger.debug(f"Reading the field message failed: {e}")
        text = ""
    quoted = f' "{text[-160:]}"' if text else ""
    if status == DIALOGUE_PLAIN:
        return f" A text box is still open{quoted}: press A to continue."
    if status == DIALOGUE_CHOICE:
        return f" A choice or menu is open{quoted}: move the cursor with UP/DOWN and press A, or B to cancel."
    return " An event is running and has control of the player: press_buttons(['WAIT']) and look again."


def _sequence_summary(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One-line outcome of a sequence for the tool result (and the agent's history)."""
    now = _position_text(_field_snapshot())
    screen = _screen_note()
    if report is None:
        return {"summary": f"Now at {now}.{screen}"}
    if report.get("blocked"):
        return {"status": "BLOCKED", "summary": report["blocked"], "now": now}
    if report.get("interrupted"):
        return {"status": "STOPPED", "summary": report["interrupted"] + screen, "now": now}
    if not report.get("done"):
        return {"status": "RUNNING", "summary": f"Still pressing ({report['executed']} of {report['total']} done); now at {now}."}
    plural = "" if report["executed"] == 1 else "s"
    note = f" Then {report['note']}." if report.get("note") else ""
    return {"status": "DONE", "summary": f"Pressed {report['executed']} button{plural}; now at {now}.{note}{screen}"}


_idle_presses: Tuple[Any, int] = (None, 0)  # (map, tile, facing) of the last no-op A press, run length


def _nothing_happened_note(buttons: List[str], before) -> str:
    """'Nothing happened' for A presses in the free overworld that opened nothing.

    In the live test the model pressed A ten times in a row at a Pokemon Center
    entrance, six tiles from the nurse; every result just said "Pressed 1 button".
    """
    global _idle_presses
    if before is None or "A" not in buttons or any(b not in ("A", "B") for b in buttons):
        return ""
    after = _field_snapshot()
    if after is None or not (before.field_free and after.field_free):
        _idle_presses = (None, 0)
        return ""
    if after.position != before.position or after.facing != before.facing or not after.facing:
        _idle_presses = (None, 0)
        return ""
    from server.field_control import DIRECTION_DELTAS

    # Seen live: a battle ended, but the model kept copying its own "Currently in battle
    # with Poochyena" reasoning from short-term memory and pressed A on an empty patch
    # of grass for 13 minutes. "Nothing happened" alone didn't break the belief, so
    # the note states what game memory says the screen is, and counts the repeats.
    spot = (after.map_id, after.position, after.facing)
    count = _idle_presses[1] + 1 if _idle_presses[0] == spot else 1
    _idle_presses = (spot, count)
    dx, dy = DIRECTION_DELTAS[after.facing]
    note = (
        f"Nothing happened: there is no one and nothing to use at ({after.x + dx},{after.y + dy}) "
        f"in front of you. Game memory says you are walking on the map: no battle is going on "
        f"and no text box or menu is open. To talk to someone, stand on a tile next to them and "
        f"face them, then press A."
    )
    if count > 1:
        note += (
            f" A has now done nothing here {count} times in a row: stop pressing A and walk "
            f"(navigate_to or go_to_exit) toward your objective."
        )
    return note


def _refuse_repeat_of_blocked_press(buttons: List[str]) -> Optional[Dict[str, Any]]:
    """Refuse a call that starts with the very press that was just blocked from this tile."""
    global last_blocked_press
    blocked = last_blocked_press
    if not blocked or not buttons:
        return None
    snap = _field_snapshot()
    if snap is None:
        return None
    if (snap.x, snap.y, snap.map_id) != (blocked["x"], blocked["y"], blocked["map_id"]):
        last_blocked_press = None  # moved on: the old block no longer applies
        return None
    if buttons[0] != blocked["button"] or not snap.field_free:
        return None
    message = (
        f"Not pressed: {blocked['button']} from ({snap.x},{snap.y}) was already BLOCKED there, so it "
        f"would do nothing again. To leave this map use go_to_exit (see EXITS); to reach a tile on this "
        f"map use navigate_to(x, y) with a walkable tile; or press a different direction."
    )
    try:
        from server.game_tools import exit_hint

        hint = exit_hint(_location_name(), blocked["button"])
    except Exception:
        hint = ""
    if hint:
        message += f" {hint}"
    logger.info(f"🚫 {message}")
    return {"success": False, "error": message, "status": "REFUSED", "summary": message, "buttons_queued": []}


# go_to_exit: a second look for a map change before a walk is called a failure.
LATE_MAP_CHANGE_S = 1.0


async def _await_map_change(old_map_id, timeout_s: float):
    """Wait for the map to change and the warp fade to finish; returns the last snapshot."""
    deadline = time.monotonic() + timeout_s
    snap = _field_snapshot()
    changed_at = None
    while time.monotonic() < deadline:
        snap = _field_snapshot()
        if snap is not None and snap.map_id != old_map_id:
            changed_at = changed_at or time.monotonic()
            # Hand back control only once the player can walk again (fade-in done),
            # or after a short grace period if something (e.g. a cutscene) holds it.
            if snap.field_free or time.monotonic() - changed_at > 3.0:
                return snap
        await asyncio.sleep(0.1)
    return snap


async def _settle_after_sequence(before, report: Optional[Dict[str, Any]]) -> None:
    """If a press started a warp (door, stairs, exit mat), wait for the new map before
    reporting, so the result names where the player actually is.

    A warp partway through a walk interrupts it (the remaining presses are skipped) and
    is then reported like one. Seen live: walking onto the Oldale Mart's exit mat said
    "STOPPED at (3,7) on OLDALE TOWN MART: an event is running" while the player was
    already outside; the model waited, then walked back in, over and over.
    """
    if before is None or not report or not (report.get("note") or report.get("interrupted")):
        return
    snap = _field_snapshot()
    if snap is None or snap.in_battle or snap.message_box or snap.waiting_for_button:
        return
    if snap.controls_locked or not snap.overworld:
        snap = await _await_map_change(before.map_id, 3.0)
        if snap is not None and snap.map_id != before.map_id:
            report["note"] = "you went through a door/warp to another map"
            if report.pop("interrupted", None):
                report["done"] = True  # the skipped presses were meant for the old map


# Presses allowed for one field move: A on the obstacle, its text, YES, the user's line.
CLEAR_OBSTACLE_PRESSES = 10


async def _clear_obstacle(tile, source: str) -> Optional[str]:
    """If the walk was stopped by a rock or tree the party can clear (game_tools.
    clearable_obstacles), use the field move on it: the blocked press already turned the
    player to face it, so A, through "Would you like to use ROCK SMASH?" (YES is the
    cursor's default), the move's text and animation, until the tile is free and the
    player can walk. Returns the move used, or None.

    Seen live on ROUTE 111: the rocks at (19,100) and (18,101) were cleared by hand with
    exactly these presses after every walk had refused to go near them.
    """
    from server import game_tools
    from server.field_control import DIALOGUE_CHOICE, DIALOGUE_PLAIN, npc_at
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    if not tile:
        return None
    move = game_tools.clearable_obstacles(ROM_TO_PORYMAP_MAP.get(_location_name(), "")).get(tuple(tile))
    if move is None:
        return None
    await _macro_press(["A"], source)
    for _ in range(CLEAR_OBSTACLE_PRESSES):
        await _await_game(
            lambda: _field_status() in (DIALOGUE_PLAIN, DIALOGUE_CHOICE) or _player_can_walk(), MACRO_TEXT_FRAMES
        )
        if _player_can_walk():
            break
        if _field_status() in (DIALOGUE_PLAIN, DIALOGUE_CHOICE):
            await _macro_press(["A"], source)  # the text, and YES at "Would you like to use …?"
    # The object vanishes after the animation; give it a moment.
    await _await_game(lambda: not _mem(lambda r: npc_at(r, tile[0], tile[1])), MACRO_MENU_FRAMES)
    if _mem(lambda r: npc_at(r, tile[0], tile[1])) or not _player_can_walk():
        return None
    return move.replace("_", " ")


@app.post("/mcp/go_to_exit")
async def mcp_go_to_exit(request: dict):
    """MCP Tool: walk to one of the current map's exits (EXITS list) and through it."""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    if game_type != "emerald":
        return {"success": False, "error": "go_to_exit is only implemented for Emerald"}
    try:
        from server import game_tools
        from server.field_control import end_of_path, warp_push_directions

        exit_query = str(request.get("exit") or request.get("destination") or "").strip()
        if not exit_query:
            return {"success": False, "error": "exit is required, e.g. 'north' or 'OLDALE TOWN'"}
        refused = _refuse_during_battle("go_to_exit")
        if refused:
            return refused

        # Seen live (2026-11-03): a person stood at (38,10) on Route 116; every
        # go_to_exit("west") planned through that tile, was blocked there, and the next
        # call planned the same path again. The planner knows only where people stood
        # when the map loaded, so a walk stopped by a person is replanned around the
        # tile they are on now (up to two more tries).
        avoid: List[Tuple[int, int]] = []
        cleared: List[str] = []
        first_from = None
        start = _field_snapshot()
        for _attempt in range(5):
            # Seen live: go_to_exit("METEOR FALLS 1F 1R") from ROUTE 114 went through the
            # door, then planned again from inside METEOR FALLS and answered "No exit of
            # METEOR FALLS 1F 1R matches 'METEOR FALLS 1F 1R'". A retry only makes sense
            # on the map the walk started from.
            here = _field_snapshot()
            if _attempt and start is not None and here is not None and here.map_id != start.map_id:
                snap = here
                break

            def _plan():
                with memory_lock:
                    return game_tools.plan_exit_route(env, exit_query, avoid=tuple(avoid))

            plan = await run_in_threadpool(_plan)
            if not plan.get("success"):
                plan.setdefault("summary", plan.get("error", "go_to_exit failed"))
                return plan

            before = _field_snapshot()
            if first_from is None:
                first_from = _position_text(before)
            buttons = list(plan["buttons"])
            if plan["kind"] == "edge":
                buttons.append(plan["final_button"])
            report = None
            if buttons:
                queued = await take_action(
                    ActionRequest(buttons=buttons, source="go_to_exit", metadata={"exit": plan["label"]}, check_movement=True)
                )
                report = await _await_sequence(queued.get("sequence_id"), _sequence_timeout(len(buttons)))
            snap = await _await_map_change(before.map_id, 4.0 if plan["kind"] == "warp" else 1.5)
            # A rock or tree on the way that the party can clear: clear it, plan again.
            used = await _clear_obstacle((report or {}).get("blocked_tile"), "go_to_exit")
            if used:
                cleared.append(f"used {used} at {tuple(report['blocked_tile'])}")
                continue
            person = (report or {}).get("blocked_by_person")
            if not person or snap is None or snap.map_id != before.map_id or tuple(person) in avoid:
                break
            avoid.append(tuple(person))
        from_text = first_from

        # Arrow warps (house mats, cave mouths) fire when the player walks off them.
        pushes_tried = []
        if (
            plan["kind"] == "warp"
            and snap is not None
            and snap.map_id == before.map_id
            and not (report and (report.get("blocked") or report.get("interrupted")))
        ):
            grid = plan["grid"]
            ox, oy = plan.get("coord_offset") or (0, 0)
            local = (snap.x - ox, snap.y - oy)
            if local == tuple(plan["target"]):
                last_move = buttons[-1] if buttons else None
                for push in warp_push_directions(grid, plan.get("raw_tiles"), local[0], local[1], last_move)[:2]:
                    pushes_tried.append(push)
                    await take_action(ActionRequest(buttons=[push], source="go_to_exit"))
                    snap = await _await_map_change(before.map_id, 4.0)
                    if snap is not None and snap.map_id != before.map_id:
                        break

        if snap is None or snap.map_id == before.map_id:
            # The map can change after the wait above gave up (a slow fade); look again
            # before calling the walk a failure.
            snap = await _await_map_change(before.map_id, LATE_MAP_CHANGE_S)
        location = _location_name()
        now = _position_text(snap, location)
        response = {"exit": plan["label"], "from": from_text, "now": now, "path_length": len(plan["buttons"])}
        on_the_way = f" On the way: {'; '.join(cleared)}." if cleared else ""
        if snap is not None and snap.map_id != before.map_id:
            response["success"] = True
            response["summary"] = f"Arrived: now at {now} (took exit: {plan['label']}).{on_the_way}"
            return response
        response["success"] = False
        if report and (report.get("blocked") or report.get("interrupted")):
            reason = report.get("blocked") or report.get("interrupted")
        elif pushes_tried:
            reason = f"reached the warp tile but it did not fire (tried {', '.join(pushes_tried)})"
        else:
            reason = "the walk ended without leaving the map"
        response["error"] = f"Did not leave the map ({plan['label']}): {reason}"
        response["summary"] = f"{response['error']} Now at {now}."
        return response
    except Exception as e:
        logger.error(f"Error in go_to_exit: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


AUTO_DIALOGUE_MAX_PRESSES = 15
# Wall-clock cap for pressing through one battle turn's text before a model call; the
# agent's request to this endpoint times out at 120 s. A turn with a KO and the win
# money took ~22 s at 60 fps on test_battle_wurmple.state.
AUTO_BATTLE_TEXT_MAX_S = 90.0


async def _auto_advance_battle_text() -> Optional[Dict[str, Any]]:
    """Press A through battle text that only waits for A, up to the next decision.

    Seen live in Rustboro Gym: "Foe GEODUDE fainted!", "TORCHIC gained 34 EXP. Points!"
    and "DAVID got $160 for winning!" each cost a ~30 s press_buttons(['A']) decision.
    Runs while the battle screen shows text or an animation (no battle menu open) and no
    YES/NO is up; A is pressed only while a box shows its waiting-for-A arrow, and no
    question is answered (not even the nickname one). None when not in a battle.
    """
    from server import menu_state as ms

    flow = _mem(ms.read_battle_flow)
    if not flow.in_battle:
        return None
    view = _read_battle_view()
    playing = flow.in_battle_screen and view is not None and view["menu"] == "busy"
    if not playing or flow.asking:
        return {"success": True, "presses": 0, "texts": [], "final_status": "battle"}
    played = await _play_out_battle(
        "auto_dialogue", answer_nickname=False, max_wall_s=AUTO_BATTLE_TEXT_MAX_S
    )
    logger.info(f"⚔️ auto-advance battle text: {played['end']} {_turn_text(played['texts'])}")
    return {
        "success": True,
        "presses": played["presses"],
        "texts": played["texts"],
        "final_status": f"battle_{played['end']}",
        "summary": f"Battle text read (auto-advanced with {played['presses']} A presses). {_battle_report(played)}",
    }


# (map name, tile talked to) -> the lines read there last time (_not_a_trainer_note).
_last_talk: Dict[Tuple[str, Tuple[int, int]], Tuple[str, ...]] = {}
NOT_A_TRAINER_NAMES = 3


def _not_a_trainer_note(texts: List[str]) -> str:
    """After talk that ended without a battle, on a map with trainers still to beat:
    say the person faced is not one of them, and name the nearest who are.

    Seen live on MT CHIMNEY (23,16), 14:52-14:57 UTC: objective "Battle through Team
    Magma Grunts", a Magma grunt at (23,17) who only says "METEORITES pack amazing
    power!", and the model pressed A at him eight times ("grunt next to me → A to
    battle") while the real trainers stood at (13,16) and (9,16). Every result only said
    the text box had closed.
    """
    from server.field_control import DIRECTION_DELTAS
    from utils.mapping.porymap_state import _map_trainers
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    snap = _field_snapshot()
    location = _location_name()
    map_dir = ROM_TO_PORYMAP_MAP.get(location, "")
    if snap is None or not map_dir or not texts or snap.facing not in DIRECTION_DELTAS:
        return ""
    dx, dy = DIRECTION_DELTAS[snap.facing]
    faced = (snap.x + dx, snap.y + dy)
    with memory_lock:
        left = [
            (x, y, name) for x, y, name, trainer_id in _map_trainers(map_dir)
            if not env.memory_reader._is_event_flag_set(_TRAINER_FLAGS_START + trainer_id)
        ]
    if not left or any((x, y) == faced for x, y, _ in left):
        return ""
    said = tuple(texts)
    again = _last_talk.get((location, faced)) == said
    _last_talk[(location, faced)] = said
    left.sort(key=lambda t: abs(t[0] - snap.x) + abs(t[1] - snap.y))
    names = "; ".join(f"{name} at ({x},{y}): talk_to({x}, {y})" for x, y, name in left[:NOT_A_TRAINER_NAMES])
    head = f"The person at ({faced[0]},{faced[1]}) is not a trainer to battle"
    head += ": talking to them again only repeats the same words." if again else "."
    return f"{head} Trainers still to battle on this map: {names}."


@app.post("/mcp/auto_advance_dialogue")
async def mcp_auto_advance_dialogue(request: dict = None):
    """Harness step (not a model tool): press A through a plain text box.

    Stops at a YES/NO or any menu, when the box closes, when the game leaves the
    overworld, or after max_presses. Returns the text it advanced through.
    """
    if env is None or game_type != "emerald":
        return {"success": False, "presses": 0, "error": "only available for Emerald"}
    from server.field_control import (
        DIALOGUE_BUSY,
        DIALOGUE_CHOICE,
        DIALOGUE_NONE,
        DIALOGUE_OTHER,
        DIALOGUE_PLAIN,
        auto_advance_dialogue,
        dialogue_status,
        read_field_message,
    )

    request = request or {}
    max_presses = int(request.get("max_presses", AUTO_DIALOGUE_MAX_PRESSES))
    loop = asyncio.get_running_loop()

    def status():
        snap = _field_snapshot()
        return dialogue_status(snap) if snap is not None else "other"

    first = status()
    battle = None
    if first == DIALOGUE_OTHER:
        battle = await _auto_advance_battle_text()
        if battle is not None:
            if battle["final_status"] != "battle_over":
                return battle
            # A trainer's or gym leader's words after the win are field text: read
            # them too rather than leave them for a model decision.
            first = status()
            if first not in (DIALOGUE_PLAIN, DIALOGUE_BUSY):
                return battle
    if first == DIALOGUE_CHOICE:
        # A YES/NO the model just answered can take a few frames to close.
        await asyncio.sleep(0.3)
        first = status()
    # PLAIN: read it. BUSY: an event is running (e.g. healing after YES) and its next
    # box is worth waiting a few seconds for. Anything else is the model's to handle.
    if first not in (DIALOGUE_PLAIN, DIALOGUE_BUSY):
        snap = _field_snapshot()
        if snap is not None and (snap.message_box or snap.extra_windows or snap.controls_locked):
            # Something is on screen but not judged plain dialogue: record why, so a
            # box the model then gets stuck under (Route 103, rival) can be diagnosed.
            logger.info(
                f"auto-advance declined: status={first} message_box={snap.message_box} "
                f"extra_windows={snap.extra_windows} controls_locked={snap.controls_locked} "
                f"waiting_for_button={snap.waiting_for_button} overworld={snap.overworld} in_battle={snap.in_battle}"
            )
        return {"success": True, "presses": 0, "texts": [], "final_status": first}

    def read_text():
        with memory_lock:
            return read_field_message(env.memory_reader)

    def press_a():
        queued = asyncio.run_coroutine_threadsafe(
            take_action(ActionRequest(buttons=["A"], speed="fast", source="auto_dialogue")), loop
        ).result(timeout=10)
        asyncio.run_coroutine_threadsafe(
            _await_sequence(queued.get("sequence_id"), 10.0), loop
        ).result(timeout=15)

    def settle():
        # >= 12 frames at any game speed: long enough for a YES/NO box to appear.
        time.sleep(max(12.0 / max(1, fps), 0.15))

    result = await run_in_threadpool(
        auto_advance_dialogue, status, read_text, press_a, settle, max_presses
    )
    response = {
        "success": True,
        "presses": result.presses,
        "texts": result.texts,
        "final_status": result.final_status,
        "capped": result.capped,
        "summary": result.summary(),
    }
    if battle is not None:
        response["presses"] += battle["presses"]
        response["texts"] = battle["texts"] + response["texts"]
        response["summary"] = f"{battle['summary']} Then: {response['summary']}"
    elif result.final_status == DIALOGUE_NONE and not _mem(lambda r: r.is_in_battle()):
        try:
            note = _not_a_trainer_note(result.texts)
        except Exception as e:  # a hint must never break the read
            logger.warning(f"not-a-trainer note failed: {e}")
            note = ""
        if note:
            response["summary"] += f" {note}"
    return response


@app.post("/mcp/press_buttons")
async def mcp_press_buttons(request: dict):
    """MCP Tool: Press GBA buttons"""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        buttons = request.get("buttons", [])
        reasoning = request.get("reasoning", "")
        source = request.get("source")
        metadata = request.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}

        # Normalize buttons to always be a list
        if isinstance(buttons, str):
            buttons = [buttons]
        elif isinstance(buttons, dict):
            # Handle dict like {'U': 'P'} -> reconstruct "UP"
            if len(buttons) == 1:
                key, value = next(iter(buttons.items()))
                if isinstance(value, str) and len(value) == 1:
                    buttons = [key + value]  # Reconstruct "UP" from {'U': 'P'}
                else:
                    buttons = list(buttons.keys())
            else:
                buttons = list(buttons.keys())

        if not buttons:
            return {"success": False, "error": "No buttons specified"}

        # GBC (Red) has no shoulder buttons — reject L/R
        if game_type == "red":
            invalid_shoulder = [b for b in buttons if str(b).upper().strip() in ("L", "R")]
            if invalid_shoulder:
                return {"success": False, "error": f"Game Boy has no shoulder buttons: {invalid_shoulder}"}

        # Valid buttons (including WAIT for no-op)
        if game_type == "red":
            valid_buttons = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "WAIT"]
        else:
            valid_buttons = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R", "WAIT"]

        # Validate and normalize buttons with fallback to 'A'
        normalized_buttons = []
        invalid_buttons = []

        for button in buttons:
            # Normalize to uppercase
            button_upper = str(button).upper().strip()

            # Check if valid
            if button_upper in valid_buttons:
                normalized_buttons.append(button_upper)
            else:
                # Invalid button - fallback to A and warn
                invalid_buttons.append(button)
                logger.warning(f"Invalid button '{button}' requested, falling back to 'A'")
                normalized_buttons.append("A")

        # Validate and normalize buttons with fallback to 'A'
        normalized_buttons = []
        invalid_buttons = []

        for button in buttons:
            # Normalize to uppercase
            button_upper = str(button).upper().strip()

            # Check if valid
            if button_upper in valid_buttons:
                normalized_buttons.append(button_upper)
            else:
                # Invalid button - fallback to A and warn
                invalid_buttons.append(button)
                logger.warning(f"Invalid button '{button}' requested, falling back to 'A'")
                normalized_buttons.append("A")

        # Filter out WAIT buttons (they're just for agent decision-making, not actual button presses)
        actual_buttons = [b for b in normalized_buttons if b != "WAIT"]

        # If only WAIT was requested, treat it as a no-op but still complete successfully.
        # (The reasoning is not echoed back: the agent already has it in the call's args,
        # and the echo tripled it in the replayed short-term memory.)
        if not actual_buttons:
            logger.info(f"🎮 Agent chose to WAIT (no buttons pressed) - {reasoning}")
            return {"success": True, "buttons_queued": [], "action": "WAIT", "summary": "Waited (no buttons pressed)."}

        refusal = _refuse_repeat_of_blocked_press(actual_buttons)
        if refusal is not None:
            return refusal

        # Call the existing take_action function to ensure metrics tracking
        action_request = ActionRequest(
            buttons=actual_buttons, source=source, metadata=metadata_dict, check_movement=True
        )
        before = _field_snapshot()
        queued = await take_action(action_request)
        report = await _await_sequence(queued.get("sequence_id"), _sequence_timeout(len(actual_buttons)))
        await _settle_after_sequence(before, report)

        logger.info(f"🎮 Queued buttons via MCP: {actual_buttons} - {reasoning}")
        response_dict = {"success": True, "buttons_queued": actual_buttons}
        response_dict.update(_sequence_summary(report))
        nothing = _nothing_happened_note(actual_buttons, before)
        if nothing:
            response_dict["summary"] += f" {nothing}"

        # Include warning if any buttons were invalid
        if invalid_buttons:
            response_dict["warning"] = f"Invalid buttons replaced with 'A': {invalid_buttons}"

        return response_dict
    except Exception as e:
        logger.error(f"Error pressing buttons: {e}")
        return {"success": False, "error": str(e)}


# How long use_move waits for a battle menu to open after its presses, in game frames
# (the action -> move menu swap takes about 10).
BATTLE_MENU_WAIT_FRAMES = 90


def _read_battle_view() -> Optional[Dict[str, Any]]:
    with memory_lock:
        return env.memory_reader.read_battle_view()


def _game_frame() -> int:
    with memory_lock:
        reader = env.memory_reader
        return reader._read_u32(reader.addresses.MAIN_FRAME_COUNTER)


def _grid_presses(start: int, goal: int, occupied: Optional[List[bool]] = None) -> List[str]:
    """Fewest D-pad presses between two cells of a battle menu's 2x2 grid.

    Cells are 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right. The move menu
    won't move onto an empty move slot, so when one of the two L-shaped paths passes
    through an empty slot the other is used.
    """
    horizontal = [] if (start & 1) == (goal & 1) else ["RIGHT" if goal & 1 else "LEFT"]
    vertical = [] if (start >> 1) == (goal >> 1) else ["DOWN" if goal >> 1 else "UP"]
    if horizontal and vertical and occupied is not None:
        corner_after_horizontal = (start & 2) | (goal & 1)
        if not occupied[corner_after_horizontal]:
            return vertical + horizontal
    return horizontal + vertical


async def _await_battle_menu(menus: Tuple[str, ...], frames: int) -> Optional[Dict[str, Any]]:
    """Poll the battle view until its menu is one of `menus` or `frames` game frames pass."""
    start = _game_frame()
    deadline = time.monotonic() + 10.0  # guard in case the game loop is paused
    view = _read_battle_view()
    with _turbo():
        while view is not None and view["menu"] not in menus:
            if _game_frame() - start >= frames or time.monotonic() > deadline:
                break
            await asyncio.sleep(0.02)
            view = _read_battle_view()
    return view


async def _press_battle_buttons(buttons: List[str], move: str) -> None:
    if not buttons:
        return
    queued = await take_action(ActionRequest(buttons=buttons, source="use_move", metadata={"move": move}))
    await _await_sequence(queued.get("sequence_id"), _sequence_timeout(len(buttons)))


def _move_key(text: str) -> str:
    """'Focus-Energy', 'focus_energy' and 'FOCUS ENERGY' all name the same move."""
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def _effect_text(move: Dict[str, Any], target: Optional[Dict[str, Any]]) -> str:
    if not move["power"]:
        return "status move"
    if move.get("conditional"):
        return "only returns a hit taken"
    against = f" vs {'/'.join(target['types'])}" if target else ""
    if move["effectiveness"] == 0:
        return f"no effect{against}"
    return f"{move['effectiveness']:g}x{against}"


@app.post("/mcp/use_move")
async def mcp_use_move(request: dict):
    """MCP Tool: pick a move in a battle by name or slot; the harness moves the cursors.

    Seen live: every battle opens with the move cursor on slot 1, and the model mashed
    A, so a Lv11 Torchic used SCRATCH (slot 1) on a Lv5 Poochyena for 12+ minutes while
    the super-effective EMBER sat in slot 4. Steering the 2x2 menus by D-pad is exactly
    what an 8B model gets wrong, so this tool does it and confirms the cursor first.

    Then it plays the turn out (_play_out_battle) and answers at the next decision: the
    FIGHT/BAG/POKEMON/RUN menu, the end of the battle, or a question. Seen live in
    Rustboro Gym: with the turn left to play out, the model spent a ~30 s decision on
    every text box, and alternated use_move / press A on a finished battle's text for
    minutes. Text still playing when it is called is pressed through first.
    """
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    if game_type != "emerald":
        return {"success": False, "error": "use_move is only implemented for Emerald"}
    try:
        wanted = request.get("move")
        if wanted is None or str(wanted).strip() == "":
            return {"success": False, "error": "move is required: a move name such as 'EMBER' or a slot number 1-4"}
        view = _read_battle_view()
        if view is None:
            message = "No battle is going on (game memory says so): use_move only works in a battle."
            return {"success": False, "error": message, "summary": message}

        moves = view["moves"]
        listing = ", ".join(f"{m['name']} (slot {m['slot']}, PP {m['pp']})" for m in moves)
        key = _move_key(wanted)
        if key.isdigit():
            move = next((m for m in moves if m["slot"] == int(key)), None)
        else:
            move = next((m for m in moves if _move_key(m["name"]) == key), None)
        if move is None:
            message = f"{wanted} is not one of your moves. Your moves: {listing}."
            return {"success": False, "error": message, "summary": message}
        if move["pp"] == 0:
            message = f"{move['name']} has no PP left. Your moves: {listing}."
            return {"success": False, "error": message, "summary": message}
        with memory_lock:
            useless = env.memory_reader.status_move_failure(move["id"], view.get("target"))
        if useless:
            hint = view.get("switch_hint")
            instead = (
                f'Use a damaging move, or switch_pokemon("{hint["name"]}").' if hint
                else "Use a damaging move, switch Pokémon, or run."
            )
            message = f"Not using {move['name']}: {useless}. {instead} Your moves: {listing}."
            return {"success": False, "error": message, "summary": message}

        from server import menu_state as ms

        menu = view["menu"]
        if menu == "busy":
            played = await _play_out_battle("use_move")
            if played["end"] == "over":
                return _refusal(f"The battle ended before {move['name']} could be used. {_battle_report(played)}")
            if played["end"] != "action":
                return _refusal(f"{move['name']} was not used. {_battle_report(played)}")
            view = _read_battle_view()
            if view is None:
                return _refusal(f"The battle ended before {move['name']} could be used.")
            move = next((m for m in view["moves"] if m["slot"] == move["slot"]), move)
            if move["pp"] == 0:
                return _refusal(f"{move['name']} has no PP left. {_battle_report(played)}")
            menu = view["menu"]
        if menu == "target":
            # A move was already picked and the double battle's target cursor is up:
            # confirm it (it starts on an opposing Pokémon) and play the turn out.
            await _press_battle_buttons(["A"], "target")
            played = await _play_out_battle("use_move")
            return {
                "success": True,
                "summary": (
                    "A target menu was open for the move already chosen: confirmed the target "
                    f"the cursor was on. {_battle_report(played)}"
                ),
                "status": {"action": "YOUR_TURN", "over": "BATTLE_OVER"}.get(played["end"], "STOPPED"),
                "end": played["end"],
            }
        if menu == "reorder":
            message = "The move-reorder mode is on (SELECT was pressed): press B, then use_move again."
            return {"success": False, "error": message, "summary": message}

        before_player, target = view["player"], view["target"]
        if menu == "action":
            await _press_battle_buttons(_grid_presses(view["action_cursor"], 0) + ["A"], move["name"])
            view = await _await_battle_menu(("move",), BATTLE_MENU_WAIT_FRAMES)
            if view is None or view["menu"] != "move":
                message = "Pressed A on FIGHT but the move menu did not open; look at the screen and try again."
                return {"success": False, "error": message, "summary": message}

        occupied = [any(m["slot"] == slot for m in view["moves"]) for slot in (1, 2, 3, 4)]
        presses = _grid_presses(view["move_cursor_slot"] - 1, move["slot"] - 1, occupied)
        await _press_battle_buttons(presses, move["name"])
        view = _read_battle_view()
        if view is None or view["menu"] != "move" or view["move_cursor_slot"] != move["slot"]:
            where = f"slot {view['move_cursor_slot']}" if view else "outside the battle"
            message = f"Could not put the cursor on {move['name']} (it is on {where}); nothing was used."
            return {"success": False, "error": message, "summary": message}

        await _press_battle_buttons(["A"], move["name"])
        view = await _await_battle_menu(("busy", "target", "action"), 30)
        if view is not None and view["menu"] == "move":
            message = f"Pressed A on {move['name']} but the move menu is still open."
            return {"success": False, "error": message, "summary": message}

        logger.info(f"⚔️ use_move: {move['name']} (slot {move['slot']}) - {request.get('reasoning', '')}")
        summary = f"Used {move['name']} ({move['type']}, {_effect_text(move, target)})."
        if before_player:
            summary += f" {before_player['name']} HP {before_player['hp']}/{before_player['max_hp']}"
        if target:
            summary += f"; opponent {target['name'].upper()} HP {target['hp']}/{target['max_hp']} before the hit."
        else:
            summary += "."
        result = {"success": True, "move": move["name"], "slot": move["slot"]}
        if view is not None and view["menu"] == "target":
            # Seen live: in a double battle (PLUSLE and MINUN) use_move stopped here with
            # "choose the target with LEFT/RIGHT and press A"; the model's presses never
            # confirmed one and both foes stayed at 40/40 for seven minutes. The cursor
            # starts on an opposing Pokémon, so A confirms a valid target.
            await _press_battle_buttons(["A"], move["name"])
            summary += " (Double battle: confirmed the target the cursor was on.)"
        played = await _play_out_battle("use_move")
        logger.info(f"⚔️ use_move turn: {played['end']} {_turn_text(played['texts'])}")
        result.update(
            {
                "summary": f"{summary} {_battle_report(played)}",
                "status": {"action": "YOUR_TURN", "over": "BATTLE_OVER"}.get(played["end"], "STOPPED"),
                "end": played["end"],
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error in use_move: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Closed-loop macros: heal_at_pokecenter, run_from_battle, throw_ball,
# switch_pokemon, buy_item.
#
# A model decision takes ~30 s and is where mistakes happen (a cursor one row off,
# A pressed into a YES/NO, B out of a menu half-way). Each macro does one whole chore
# in a single call: before every press it reads from memory which menu and cursor the
# game shows (server/menu_state.py), and it stops with a plain-language reason as soon
# as the screen is not what it expects. Frame budgets only guard against waiting
# forever; the state decides when to press.
# ---------------------------------------------------------------------------

MACRO_MENU_FRAMES = 240     # a menu opening or closing with its fade (~30 frames seen)
MACRO_TEXT_FRAMES = 900     # a message printing at slow text speed before its menu
MACRO_BATTLE_FRAMES = 4800  # a battle turn's text and animations (~400 frames seen)
MACRO_MAX_BATTLE_PRESSES = 40


def _route_flag_reader(flag: int) -> bool:
    """Event flags for route planning (server.game_tools.flag_reader): which HM obstacles
    are gone for good."""
    return bool(_mem(lambda r: r._is_event_flag_set(flag)))


def _mem(fn, *args):
    """Call fn(memory_reader, *args) under the memory lock."""
    with memory_lock:
        return fn(env.memory_reader, *args)


def _refusal(message: str, **extra) -> Dict[str, Any]:
    result = {"success": False, "error": message, "summary": message}
    result.update(extra)
    return result


def _usable_field_moves() -> set:
    """Field moves the party can use outside battle now (server.game_tools.
    field_moves_reader): known by a party Pokémon, with the badge that allows it."""
    from server import game_tools

    with memory_lock:
        reader = env.memory_reader
        known = {m for p in reader.read_party_pokemon() or [] if not p.is_egg for m in p.moves}
        return {
            move for move, badge in game_tools.FIELD_MOVE_OBSTACLES.values()
            if move in known and reader._is_event_flag_set(game_tools.FLAG_BADGE01_GET + badge - 1)
        }


def _held_item_names() -> set:
    """Names of everything in the bag, squashed (server.game_tools.items_reader): which
    turn-back gates the player can pass (the Go-Goggles open Route 111's desert)."""
    from server import game_tools

    inventory = _mem(lambda r: r.read_full_inventory())
    return {game_tools._squash_name(item["name"]) for pocket in inventory.values() for item in pocket}


def _install_route_flag_reader() -> None:
    from server import game_tools

    game_tools.flag_reader = _route_flag_reader
    game_tools.field_moves_reader = _usable_field_moves
    game_tools.items_reader = _held_item_names


_install_route_flag_reader()


async def _macro_press(buttons: List[str], source: str) -> None:
    if not buttons:
        return
    queued = await take_action(ActionRequest(buttons=list(buttons), source=source))
    await _await_sequence(queued.get("sequence_id"), _sequence_timeout(len(buttons)))


async def _await_game(check, frames: int):
    """Poll check() until it returns something truthy or `frames` game frames pass."""
    start = _game_frame()
    deadline = time.monotonic() + max(10.0, frames / 20)  # guard in case the loop is paused
    value = check()
    with _turbo():
        while not value:
            if _game_frame() - start >= frames or time.monotonic() > deadline:
                break
            await asyncio.sleep(0.02)
            value = check()
    return value


def _quote(texts: List[str], limit: int = 4) -> str:
    shown = [t for t in texts if t][-limit:]
    return " / ".join(f'"{t}"' for t in shown)


def _party_lines() -> List[Dict[str, Any]]:
    with memory_lock:
        party = env.memory_reader.read_party_pokemon()
    return [
        {
            "name": p.species_name.upper() if not p.is_egg else "EGG",
            "hp": p.current_hp,
            "max_hp": p.max_hp,
            "status": int(getattr(p.status, "value", p.status) or 0),
            "egg": bool(p.is_egg),
        }
        for p in party
    ]


def _hp_text(party: List[Dict[str, Any]]) -> str:
    return ", ".join(f"{m['name']} {m['hp']}/{m['max_hp']}" for m in party) or "no Pokémon"


def _player_can_walk() -> bool:
    snap = _field_snapshot()
    return snap is not None and snap.field_free


def read_field_text(reader) -> str:
    from server.field_control import read_field_message

    return read_field_message(reader)


# -- battles ---------------------------------------------------------------


async def _play_out_battle(
    source: str,
    frames: int = MACRO_BATTLE_FRAMES,
    answer_nickname: bool = True,
    max_wall_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Press through battle text until the player picks an action again or the battle
    is over and the player stands on the map.

    A is pressed only while a text box shows its down arrow and no YES/NO is up; the
    nickname question is answered NO with B (B always means NO there) when
    `answer_nickname`, the caught Pokémon's Pokédex page is closed with B, and "Will
    DAVID change POKéMON?" is answered by _answer_switch_question. Any other question
    ("Use next POKéMON?", deleting a move to learn a new one), a party menu the game
    opens itself, or an evolution ends the loop: those are the model's decisions.

    `texts` are the battle messages in the order they showed: the string buffer is read
    while the message window prints (BattleFlow.message_printing), so the FIGHT/BAG
    prompt, the move menu's PP/TYPE lines and healthbox names in the same buffer are
    left out.
    """
    with _turbo():
        return await _play_out_battle_steps(source, frames, answer_nickname, max_wall_s)


async def _play_out_battle_steps(
    source: str, frames: int, answer_nickname: bool, max_wall_s: Optional[float]
) -> Dict[str, Any]:
    from server import menu_state as ms

    texts: List[str] = []
    notes: List[str] = []
    presses = replaced = 0
    start = _game_frame()
    deadline = time.monotonic() + (max_wall_s if max_wall_s is not None else max(30.0, frames / 20))
    seen = _frames_run
    flow = _mem(ms.read_battle_flow)
    if flow.text and flow.in_battle and flow.message_printing:
        texts.append(flow.text)  # began printing before this call
    while True:
        # Every message printed since, in order, as the game loop recorded it.
        for at, text in _battle_messages_since(seen):
            seen = at
            if not texts or texts[-1] != text:
                texts.append(text)
        end = None
        if flow.evolving:
            end = "evolution"
        elif flow.in_battle and flow.in_battle_screen and flow.choosing_action:
            end = "action"
        elif not flow.in_battle:
            snap = _field_snapshot()
            if snap is not None and snap.overworld and not _mem(ms.fade_active):
                # Back on the map: give the fade-in a moment to hand control back.
                await _await_game(_player_can_walk, MACRO_MENU_FRAMES)
                end = "over"
        elif flow.nickname_prompt:
            if not answer_nickname:
                end = "question"
            else:
                await _macro_press(["B"], source)
                notes.append("answered NO to giving it a nickname")
                presses += 1
        elif flow.dex_page:
            await _macro_press(["B"], source)
            presses += 1
        elif (flow.yes_no_prompt or flow.script_question) and "CHANGE POK" in str(flow.text).upper():
            # "Will DAVID change POKéMON?" after a trainer's Pokémon faints: a free switch
            # before the next one comes in. Seen live against the Route 110 rival: the
            # model answered YES, picked the Pokémon already battling, and pressed A on
            # SHIFT for two minutes. David: "think about who the next pokemon is and then
            # send out the pokemon that best counters that pokemon." The trainer's line
            # before it names the next one ("MAY is about to use SLUGMA.").
            # It is the battle script's yesnobox (pret BattleScript_HandleFaintedMon), so
            # it reads as script_question. Seen live on Route 117 vs Breeder Isaac: with
            # only yes_no_prompt checked, the question was handed to the model, which
            # opened the START menu. The question shares its text with the "about to use"
            # line, which may not be in `texts` yet.
            notes.append(await _answer_switch_question(texts + [str(flow.text)], source))
            presses += 1
        elif flow.yes_no_prompt:
            end = "question"
        elif flow.script_question:
            end = "learn_move" if flow.learn_move_prompt else "question"
        elif not flow.in_battle_screen and any(
            _mem(ms.callback2_is, cb) for cb in (ms.CB2_UPDATE_PARTY_MENU, ms.CB2_BAG_MENU_RUN)
        ):
            # The game opened the party menu itself: a fainted Pokémon must be replaced.
            # Seen live: MARILL fainted while training and the model spent 30 minutes
            # in this menu (press A on the fainted one, switch_pokemon refused). Any
            # Pokémon able to fight is a valid answer, so the best counter is sent in.
            choice = _replacement() if replaced < 2 and _forced_party_menu() else None
            if choice is None:
                end = "menu"
            else:
                replaced += 1
                error = await _shift_from_party_menu(choice["personality"], choice["name"], source)
                if error:
                    end = "menu"
                else:
                    notes.append(f"sent in {choice['name']} for the fainted Pokémon (the best of the rest here)")
                    presses += 3
        elif flow.level_up_box and flow.in_battle_screen:
            # The stat panel after "X grew to LV. N!" waits for a key without a text
            # printer; A while it is still drawing is ignored, so pressing until the
            # script moves on is safe. Seen on test_wild_two_party.state: after "TORCHIC
            # grew to LV. 12!" use_move waited out its whole frame budget (80 s) and
            # reported a timeout, and every switch-training level-up would do the same.
            await _macro_press(["A"], source)
            presses += 1
        elif flow.text_waiting and flow.in_battle_screen:
            # Inside the learn-move script too: its "is trying to learn X" lines only
            # wait for A; the YES/NO after them is caught by script_question above.
            await _macro_press(["A"], source)
            presses += 1
        if end is None and (
            _game_frame() - start > frames or time.monotonic() > deadline or presses > MACRO_MAX_BATTLE_PRESSES
        ):
            end = "timeout"
        if end is not None:
            return {"end": end, "flow": flow, "texts": texts, "notes": notes, "presses": presses}
        await asyncio.sleep(0.03)
        flow = _mem(ms.read_battle_flow)


def _battle_end_text(played: Dict[str, Any]) -> str:
    """What a played-out battle ended on, for summaries of the cases a macro can't finish."""
    end = played["end"]
    said = _quote(played["texts"])
    asked = _quote(played["texts"][-1:])
    if end == "question":
        return (
            f"The battle asks a YES/NO question ({asked}): answer it yourself with press_buttons "
            "(A = YES, B = NO)."
        )
    if end == "learn_move":
        advice = _learn_move_advice(played["texts"])
        return (
            f"Your Pokémon wants to learn a new move ({asked}): decide with press_buttons "
            f"(A = YES picks a move to forget, B = NO keeps the old moves).{f' {advice}' if advice else ''}"
        )
    if end == "menu":
        choice = _replacement() if _forced_party_menu() else None
        if choice is not None:
            return (
                "The game opened the party menu (a fainted Pokémon must be replaced): call "
                f'switch_pokemon("{choice["name"]}") (the best of the rest here).'
            )
        return "The game opened the party menu (a fainted Pokémon must be replaced): call switch_pokemon with one able to fight."
    if end == "evolution":
        said = _quote([_mem(read_field_text)]) or said
        return f"Your Pokémon is evolving ({said}): press A to let it evolve (B stops it)."
    if end == "timeout":
        return f"The battle did not reach a menu in time; the screen last said {said or 'nothing readable'}."
    return said


# Battle messages quoted in one turn's report; a turn with two KOs and a level-up is ~10.
TURN_REPORT_MAX_MESSAGES = 12


def _turn_text(texts: List[str]) -> str:
    """The turn's battle messages in order: "TORCHIC used EMBER! · It's super effective!"."""
    shown = [t for t in texts if t][-TURN_REPORT_MAX_MESSAGES:]
    return f'"{" · ".join(shown)}"' if shown else ""


def _battle_hp_text(view: Optional[Dict[str, Any]]) -> str:
    if not view or not view.get("player"):
        return ""
    player = view["player"]
    text = f"{player['name'].upper()} HP {player['hp']}/{player['max_hp']}"
    foes = [f"foe {m['name'].upper()} HP {m['hp']}/{m['max_hp']}" for m in view.get("opponents") or []]
    return f"Now: {'; '.join([text] + foes)}."


def _battle_report(played: Dict[str, Any]) -> str:
    """What happened while a battle was played out, and where it stopped.

    Seen live in Rustboro Gym: after each attack the model spent a ~30 s decision per
    text box ("TORCHIC used EMBER!", "It's not very effective…", "Foe GEODUDE fainted!",
    "DAVID got $160 for winning!"). The presses now happen here, so the model gets the
    whole turn as one quote plus the HP of both sides, and is asked only to decide.
    """
    from server import menu_state as ms

    end, flow = played["end"], played["flow"]
    texts = played["texts"]
    if end in ("question", "learn_move"):
        texts = texts[:-1]  # the question itself is quoted where the report asks it
    said = _turn_text(texts)
    happened = f"What happened: {said}. " if said else ""
    # Questions the play-out answered itself. Without this the quote showed "SABLEYE,
    # good! Come back! · Go! COMBUSKEN!" and nothing said why the Pokémon changed.
    if played.get("notes"):
        happened += f"Answered for you: {'; '.join(played['notes'])}. "
    if end == "action":
        hp = _battle_hp_text(_read_battle_view())
        return (
            f"{happened}{hp} Your turn: FIGHT/BAG/POKEMON/RUN is open "
            "(use_move, switch_pokemon, run_from_battle or throw_ball)."
        ).strip()
    if end == "over":
        result = ms.outcome_name(flow.outcome)
        if flow.outcome in (ms.B_OUTCOME_WON, ms.B_OUTCOME_LOST, ms.B_OUTCOME_DREW):
            result = f"you {result}"
        where = _position_text(_field_snapshot())
        return (
            f"{happened}The battle is over ({result}); you are on the map at {where}. "
            f"Party HP: {_hp_text(_party_lines())}.{_screen_note()}"
        ).strip()
    return f"{happened}{_battle_end_text(played)}".strip()


async def _ready_for_battle_action(tool: str, source: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Get to the FIGHT/BAG/POKEMON/RUN menu: (battle view, None) or (None, refusal).

    Battle text that only waits for A (e.g. "Wild SHROOMISH appeared!") is pressed
    through first; the move menu is closed with B.
    """
    view = _read_battle_view()
    if view is None:
        return None, _refusal(f"No battle is going on (game memory says so): {tool} only works in a battle.")
    if view["menu"] == "busy":
        played = await _play_out_battle(source)
        if played["end"] == "over":
            return None, _refusal(
                f"The battle ended before {tool} could act ({_quote(played['texts'])}); you are back on the map."
            )
        if played["end"] != "action":
            return None, _refusal(f"{tool} did nothing: {_battle_end_text(played)}")
        view = _read_battle_view()
    if view is not None and view["menu"] == "move":
        await _macro_press(["B"], source)
        view = await _await_battle_menu(("action",), BATTLE_MENU_WAIT_FRAMES)
    if view is None or view["menu"] != "action":
        where = view["menu"] if view else "outside a battle"
        if where == "target":
            return None, _refusal(f"A target menu is open (double battle): press B first, then {tool} again.")
        if where == "reorder":
            return None, _refusal(f"The move-reorder mode is on: press B, then {tool} again.")
        return None, _refusal(f"Could not reach the FIGHT/BAG/POKEMON/RUN menu (game memory says: {where}); nothing was chosen.")
    return view, None


async def _choose_battle_action(action: int, source: str) -> Optional[str]:
    """Put the action cursor on `action`, confirm it from memory, press A. Error text or None."""
    names = ("FIGHT", "BAG", "POKEMON", "RUN")
    view = _read_battle_view()
    await _macro_press(_grid_presses(view["action_cursor"], action), source)
    view = _read_battle_view()
    if view is None or view["menu"] != "action" or view["action_cursor"] != action:
        where = view["action_cursor_name"] if view else "outside the battle"
        return f"Could not put the cursor on {names[action]} (it is on {where}); nothing was chosen."
    await _macro_press(["A"], source)
    return None


async def _back_to_battle_menu(source: str) -> bool:
    """Leave the bag or party menu with B (never choosing anything) until the battle
    menu shows again."""
    from server import menu_state as ms

    for _ in range(4):
        flow = _mem(ms.read_battle_flow)
        if flow.in_battle_screen and flow.choosing_action:
            return True
        if _mem(ms.text_waiting_for_button):
            await _macro_press(["B"], source)  # dismiss a message inside the menu
        elif not flow.in_battle_screen:
            await _macro_press(["B"], source)
        await _await_game(
            lambda: (lambda f: f.in_battle_screen and f.choosing_action)(_mem(ms.read_battle_flow)),
            MACRO_MENU_FRAMES,
        )
    flow = _mem(ms.read_battle_flow)
    return flow.in_battle_screen and flow.choosing_action


def _battle_type_flags() -> int:
    with memory_lock:
        reader = env.memory_reader
        return reader._read_u32(reader.addresses.BATTLE_TYPE_FLAGS)


@app.post("/mcp/run_from_battle")
async def mcp_run_from_battle(request: dict = None):
    """MCP Tool: flee a wild battle (RUN), confirmed from gBattleOutcome."""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    if game_type != "emerald":
        return {"success": False, "error": "run_from_battle is only implemented for Emerald"}
    try:
        from pokemon_env.memory_reader import BATTLE_TYPE_FIRST_BATTLE
        from server import menu_state as ms

        source = "run_from_battle"
        view = _read_battle_view()
        if view is not None and view["is_trainer"]:
            return _refusal(
                "You can't run from a trainer battle: the game won't allow it. Fight with use_move "
                "(the battle section names the best move) or switch with switch_pokemon."
            )
        if view is not None and _battle_type_flags() & BATTLE_TYPE_FIRST_BATTLE:
            return _refusal("You can't run from this battle (the first battle, with Prof. Birch's Pokémon): use use_move.")
        view, refusal = await _ready_for_battle_action("run_from_battle", source)
        if refusal:
            return refusal
        opponent = view["target"]["name"].upper() if view.get("target") else "the wild Pokémon"
        error = await _choose_battle_action(ms.ACTION_RUN, source)
        if error:
            return _refusal(error)
        played = await _play_out_battle(source)
        flow = played["flow"]
        logger.info(f"🏃 run_from_battle: {played['end']} outcome={flow.outcome} {_turn_text(played['texts'])}")
        if played["end"] == "over" and flow.outcome == ms.B_OUTCOME_RAN:
            now = _position_text(_field_snapshot())
            return {"success": True, "status": "ESCAPED", "summary": f"Got away safely from wild {opponent}. Now at {now}."}
        if played["end"] == "over":
            return {"success": False, "status": "BATTLE_OVER", "summary": f"Chose RUN. {_battle_report(played)}"}
        if played["end"] == "action":
            message = f"Couldn't get away: the battle against wild {opponent} continues. {_battle_report(played)}"
            return _refusal(message, status="STILL_IN_BATTLE")
        return _refusal(f"Chose RUN, then: {_battle_report(played)}", status="STOPPED")
    except Exception as e:
        logger.error(f"Error in run_from_battle: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _ball_listing(balls: List[Dict[str, Any]]) -> str:
    return ", ".join(f"{b['name']} x{b['quantity']}" for b in balls) or "none"


@app.post("/mcp/throw_ball")
async def mcp_throw_ball(request: dict = None):
    """MCP Tool: throw a Poké Ball in a wild battle and report caught / broke free."""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    if game_type != "emerald":
        return {"success": False, "error": "throw_ball is only implemented for Emerald"}
    try:
        from pokemon_env.memory_reader import BATTLE_TYPE_FIRST_BATTLE, BATTLE_TYPE_SAFARI
        from server import menu_state as ms

        source = "throw_ball"
        request = request or {}
        wanted = str(request.get("ball") or "Poke Ball").strip()
        view = _read_battle_view()
        if view is None:
            return _refusal("No battle is going on (game memory says so): throw_ball only works in a wild battle.")
        if view["is_trainer"]:
            return _refusal("You can't catch a trainer's Pokémon: throw_ball only works in a wild battle.")
        flags = _battle_type_flags()
        if flags & BATTLE_TYPE_FIRST_BATTLE:
            return _refusal("This battle (Prof. Birch's first battle) can't be caught in: use use_move.")
        if flags & BATTLE_TYPE_SAFARI:
            return _refusal("The Safari Zone has its own BALL menu: use press_buttons there.")
        alive = [m for m in view["opponents"] if m["hp"] > 0]
        if len(alive) > 1:
            return _refusal("There are two wild Pokémon: the game won't let you throw a ball until one is left.")

        with memory_lock:
            balls = env.memory_reader.read_full_inventory().get("poke_balls", [])
        key = ms.name_key(wanted)
        ball = next((b for b in balls if ms.name_key(b["name"]) == key), None)
        if ball is None:
            # "great" or "GREAT BALL": accept a unique prefix of a ball you have.
            prefixed = [b for b in balls if ms.name_key(b["name"]).startswith(key)] if key else []
            ball = prefixed[0] if len(prefixed) == 1 else None
        if ball is None:
            return _refusal(f"You have no {wanted}. Poké Balls in your bag: {_ball_listing(balls)}.")
        count_before = ball["quantity"]
        party_before = _mem(lambda r: r._read_u8(ms.G_PLAYER_PARTY_COUNT))
        target = view["target"]
        opponent = f"{target['name'].upper()} (Lv{target['level']}, HP {target['hp']}/{target['max_hp']})" if target else "the wild Pokémon"

        view, refusal = await _ready_for_battle_action("throw_ball", source)
        if refusal:
            return refusal
        error = await _choose_battle_action(ms.ACTION_BAG, source)
        if error:
            return _refusal(error)

        def bag_ready():
            return (
                _mem(ms.callback2_is, ms.CB2_BAG_MENU_RUN)
                and _mem(ms.task_running, ms.TASK_BAG_INPUT)
                and not _mem(ms.fade_active)
            )

        async def give_up(message: str) -> Dict[str, Any]:
            back = await _back_to_battle_menu(source)
            tail = " Back at the battle menu; nothing was used." if back else " Could not get back to the battle menu: look at the screen."
            return _refusal(message + tail)

        if not await _await_game(bag_ready, MACRO_MENU_FRAMES):
            return await give_up("Chose BAG but the bag did not open.")
        pocket = _mem(ms.bag_pocket)
        for press in ms.pocket_presses(pocket, ms.BALLS_POCKET):
            await _macro_press([press], source)
            await _await_game(bag_ready, MACRO_MENU_FRAMES)
        if _mem(ms.bag_pocket) != ms.BALLS_POCKET or not bag_ready():
            return await give_up("Could not switch the bag to the POKé BALLS pocket.")
        listed = _mem(ms.bag_pocket_item_ids, ms.BALLS_POCKET)
        if ball["item_id"] not in listed:
            return await give_up(f"{ball['name']} is not listed in the POKé BALLS pocket.")
        goal = listed.index(ball["item_id"])
        await _macro_press(ms.list_presses(_mem(ms.bag_cursor_index, ms.BALLS_POCKET), goal), source)
        index = _mem(ms.bag_cursor_index, ms.BALLS_POCKET)
        if index != goal:
            return await give_up(f"Could not put the bag cursor on {ball['name']} (row {index + 1}, wanted row {goal + 1}).")
        await _macro_press(["A"], source)
        if not await _await_game(lambda: _mem(ms.task_running, *ms.TASK_BAG_CONTEXT), MACRO_MENU_FRAMES):
            return await give_up(f"Pressed A on {ball['name']} but its USE/CANCEL menu did not open.")
        chosen = _mem(lambda r: r._read_u16(ms.G_SPECIAL_VAR_ITEM_ID))
        cursor = _mem(ms.read_menu_cursor)
        if chosen != ball["item_id"] or cursor.cursor != 0 or cursor.max_cursor != 1:
            return await give_up(f"The bag's menu is not USE on {ball['name']} (item {chosen}, cursor {cursor.cursor}).")
        await _macro_press(["A"], source)  # USE

        # The bag closes and the battle screen comes back with the throw.
        if not await _await_game(lambda: _mem(ms.callback2_is, ms.CB2_BATTLE_MAIN), MACRO_MENU_FRAMES * 2):
            said = _mem(read_field_text)
            return await give_up(f"Chose USE on {ball['name']} but the bag stayed open{f' ({said})' if said else ''}.")
        played = await _play_out_battle(source)
        flow = played["flow"]
        with memory_lock:
            balls_after = env.memory_reader.read_full_inventory().get("poke_balls", [])
        count_after = next((b["quantity"] for b in balls_after if b["item_id"] == ball["item_id"]), 0)
        used = f"Threw a {ball['name']} at {opponent}; {ball['name']} {count_before} -> {count_after} left."
        logger.info(f"🎯 throw_ball: {played['end']} outcome={flow.outcome} {_turn_text(played['texts'])}")
        result = {"ball": ball["name"], "balls_before": count_before, "balls_after": count_after}
        if flow.outcome == ms.B_OUTCOME_CAUGHT:
            party_after = _mem(lambda r: r._read_u8(ms.G_PLAYER_PARTY_COUNT))
            where = "joined your party" if party_after > party_before else "was sent to the PC (your party is full)"
            notes = f" ({'; '.join(played['notes'])})" if played["notes"] else ""
            summary = f"Caught it! {used} It {where}{notes}."
            if played["end"] == "over":
                summary += f" The battle is over; now at {_position_text(_field_snapshot())}."
            else:
                summary += f" {_battle_end_text(played)}"
            result.update({"success": True, "status": "CAUGHT", "caught": True, "summary": summary})
            return result
        if played["end"] == "action":
            summary = f"It broke free: not caught. {used} The battle continues. {_battle_report(played)}"
            result.update({"success": True, "status": "BROKE_FREE", "caught": False, "summary": summary})
            return result
        summary = f"{used} Then: {_battle_report(played)}"
        result.update({"success": played["end"] == "over", "status": "STOPPED", "caught": False, "summary": summary})
        if not result["success"]:
            result["error"] = summary
        return result
    except Exception as e:
        logger.error(f"Error in throw_ball: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def _shift_from_party_menu(personality: int, name: str, source: str) -> Optional[str]:
    """With the battle's party menu opening (from POKEMON, from YES to "Will DAVID change
    POKéMON?", or opened by the game to replace a fainted Pokémon), pick the Pokémon with
    this personality and choose SHIFT, or SEND OUT in the replacement menu. Returns None
    once back on the battle screen, else why not (the menu may still be open)."""
    from server import menu_state as ms

    def party_menu_ready():
        return (
            _mem(ms.callback2_is, ms.CB2_UPDATE_PARTY_MENU)
            and _mem(ms.task_running, ms.TASK_PARTY_CHOOSE)
            and not _mem(ms.fade_active)
        )

    if not await _await_game(party_menu_ready, MACRO_MENU_FRAMES):
        return "The party menu did not open."
    # The menu draws gPlayerParty[slot]; find the slot by personality, which is unique,
    # rather than trusting the order the model saw.
    shown = _mem(ms.party_personalities)
    if personality not in shown:
        return f"{name} is not listed in the party menu."
    slot = shown.index(personality)
    await _macro_press(ms.list_presses(_mem(ms.party_menu_slot), slot), source)
    if _mem(ms.party_menu_slot) != slot:
        return f"Could not put the party cursor on {name} (it is on slot {_mem(ms.party_menu_slot) + 1})."
    await _macro_press(["A"], source)
    if not await _await_game(lambda: _mem(ms.task_running, ms.TASK_PARTY_SELECTION), MACRO_MENU_FRAMES):
        said = _mem(read_field_text)
        return f"Pressed A on {name} but no SHIFT menu opened{f' ({said})' if said else ''}."
    actions = _mem(ms.party_menu_actions)
    # Seen live: MARILL fainted in a wild battle and the game's replacement menu offers
    # SEND OUT, not SHIFT; the model spent 30 minutes on press_buttons there.
    action = next((a for a in (ms.MENU_SHIFT, ms.MENU_SEND_OUT) if a in actions), None)
    if action is None:
        return f"{name}'s menu has no SHIFT or SEND OUT option."
    label = "SHIFT" if action == ms.MENU_SHIFT else "SEND OUT"
    await _macro_press(ms.list_presses(_mem(ms.read_menu_cursor).cursor, actions.index(action)), source)
    cursor = _mem(ms.read_menu_cursor)
    if cursor.cursor >= len(actions) or actions[cursor.cursor] != action:
        return f"Could not put the cursor on {label} (it is on entry {cursor.cursor + 1})."
    await _macro_press(["A"], source)
    if not await _await_game(lambda: _mem(ms.callback2_is, ms.CB2_BATTLE_MAIN), MACRO_MENU_FRAMES * 2):
        # Still in the party menu: the game refused (e.g. a trapping move) and says why.
        said = _mem(read_field_text)
        return f"Chose {label} but the game kept the party menu open{f': {said}' if said else ''}."
    return None


def _learn_move_advice(texts: List[str]) -> str:
    """What to do at "X is trying to learn Y": keep the four moves (B) when Y is worth no
    more than the weakest of them (reader.move_worth), else forget the weakest.

    Seen live: at Lv28 BRELOOM learned COUNTER and the model forgot MACH PUNCH, its STAB
    Fighting move and the counter to Wattson's MAGNETON; the prompt was handed over with
    no advice. "" when the texts don't name the Pokémon and the move.
    """
    from server import menu_state as ms

    marker = " is trying to learn "
    line = next((t for t in reversed(texts) if marker in t), "")
    if not line:
        return ""
    who, _, rest = line.partition(marker)
    new_move = rest.split(".")[0].split("!")[0].strip()
    who_key = ms.name_key(who.split("·")[-1].strip())
    with memory_lock:
        reader = env.memory_reader
        party = reader.read_party_pokemon() or []
        mon = next(
            (p for p in party if who_key in (ms.name_key(p.species_name), ms.name_key(p.nickname or ""))), None
        )
        if mon is None or not new_move:
            return ""
        own = reader.party_mon_types(mon)
        current = [(m, reader.move_worth(m, own)) for m in mon.moves if m and m != "NONE"]
        new_worth = reader.move_worth(new_move, own)
    if len(current) < 4:
        return ""
    weakest, weakest_worth = min(current, key=lambda t: t[1])
    name = mon.species_name.upper()
    if new_worth <= weakest_worth:
        why = "it does no damage of its own" if new_worth == 0 else f"it is worth less than {weakest.replace('_', ' ')}"
        return (
            f"Advice: don't learn {new_move} ({why}): press B (NO), then A (YES) to stop learning it; "
            f"{name} keeps its four moves."
        )
    slot = [m for m, _ in current].index(weakest) + 1
    return (
        f"Advice: learn {new_move} and forget {weakest.replace('_', ' ')} (move {slot}, the weakest of "
        f"{name}'s four): press A, pick {weakest.replace('_', ' ')}, A."
    )


def _forced_party_menu() -> bool:
    """In a battle with the party menu up that the player did not open: the game asks
    for a replacement for a fainted Pokémon."""
    from server import menu_state as ms

    return bool(_mem(lambda r: r.is_in_battle())) and _mem(ms.callback2_is, ms.CB2_UPDATE_PARTY_MENU)


def _replacement() -> Optional[Dict[str, Any]]:
    """The teammate to send in for a fainted Pokémon: the battle view's switch hint (best
    counter_score against the foe among those able to fight with damaging PP), with its
    personality; None if nobody qualifies."""
    from server import menu_state as ms

    with memory_lock:
        reader = env.memory_reader
        view = reader.read_battle_view() or {}
        battlers = [0, 2] if view.get("is_double") else [0]
        hint = reader._switch_hint(view.get("target"), battlers)
        if hint is None:
            return None
        personalities = ms.party_personalities(reader)
    if hint["slot"] > len(personalities):
        return None
    return {**hint, "personality": personalities[hint["slot"] - 1]}


def _counter_score(reader, mon, foe_types: List[int]) -> float:
    """reader.counter_score: the battle view uses the same score for its switch hint."""
    return reader.counter_score(mon, foe_types)


# Only switch when the counter is clearly better than the Pokémon already out.
SWITCH_COUNTER_MARGIN = 2.0


def _best_counter(next_foe: str) -> Optional[Tuple[Any, int, float, float]]:
    """(party Pokémon to send in, its personality, its score, the active one's score)
    against next_foe, or
    None when the one out now is as good (within SWITCH_COUNTER_MARGIN) or unknown."""
    from server import menu_state as ms

    with memory_lock:
        reader = env.memory_reader
        foe_types = reader.species_types(next_foe)
        party = reader.read_party_pokemon() or []
        personalities = ms.party_personalities(reader)
        active = ms.active_battler_personality(reader)
        if not foe_types or len(personalities) != len(party):
            return None
        scored = [
            (mon, pid, _counter_score(reader, mon, foe_types))
            for mon, pid in zip(party, personalities)
            if not mon.is_egg and mon.current_hp * 5 > mon.max_hp * 2  # above 40% HP
        ]
    current = next((score for mon, pid, score in scored if pid == active), 0.0)
    bench = [(mon, pid, score) for mon, pid, score in scored if pid != active]
    if not bench:
        return None
    mon, pid, score = max(bench, key=lambda t: t[2])
    if score < max(current, 0.5) * SWITCH_COUNTER_MARGIN:
        return None
    return (mon, pid, score, current)


async def _answer_switch_question(texts: List[str], source: str) -> str:
    """Answer "Will DAVID change POKéMON?": YES and SHIFT in the clear counter to the
    trainer's next Pokémon (_best_counter), else NO. Returns a note for the turn report."""
    from server import menu_state as ms

    foe = _next_foe_from(texts)
    choice = _best_counter(foe) if foe else None
    if choice is None:
        await _macro_press(["B"], source)
        why = f"nobody counters {foe} clearly better" if foe else "the next Pokémon was not named"
        return f"kept the same Pokémon in ({why})"
    mon, personality, score, current = choice
    name = str(mon.species_name).upper()
    await _macro_press(["A"], source)  # YES
    error = await _shift_from_party_menu(personality, name, source)
    if error is None:
        return f"switched to {name} to face {foe} (matchup {score:g} vs {current:g})"
    # Back out of the menu; the game then sends its Pokémon in against the one out now.
    for _ in range(3):
        if _mem(ms.callback2_is, ms.CB2_BATTLE_MAIN):
            break
        await _macro_press(["B"], source)
    return f"tried to switch to {name} for {foe} but {error}"


def _next_foe_from(texts: List[str]) -> str:
    """'MAY is about to use SLUGMA.' -> 'SLUGMA' (the latest such line), else ''."""
    for text in reversed(texts):
        marker = "about to use "
        at = text.find(marker)
        if at != -1:
            name = text[at + len(marker):].split(".")[0].split("!")[0]
            return name.strip().upper()
    return ""


@app.post("/mcp/switch_pokemon")
async def mcp_switch_pokemon(request: dict = None):
    """MCP Tool: in a battle, send out another party Pokémon (POKEMON -> pick -> SHIFT)."""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    if game_type != "emerald":
        return {"success": False, "error": "switch_pokemon is only implemented for Emerald"}
    try:
        from pokemon_env.memory_reader import BATTLE_TYPE_SAFARI
        from server import menu_state as ms

        source = "switch_pokemon"
        request = request or {}
        wanted = request.get("pokemon", request.get("name_or_slot"))
        if wanted is None or str(wanted).strip() == "":
            return _refusal("pokemon is required: a party Pokémon's name (e.g. 'SHROOMISH') or its party slot 1-6.")
        view = _read_battle_view()
        if view is None:
            return _refusal(
                "No battle is going on (game memory says so): switch_pokemon only works in a battle "
                "(reordering the party outside battles is not supported)."
            )
        if _battle_type_flags() & BATTLE_TYPE_SAFARI:
            return _refusal("You can't switch Pokémon in the Safari Zone.")
        party = _party_lines()
        personalities = _mem(ms.party_personalities)
        active = _mem(ms.active_battler_personality)
        listing = ", ".join(f"{i + 1}. {m['name']} HP {m['hp']}/{m['max_hp']}" for i, m in enumerate(party))
        key = ms.name_key(wanted)
        if key.isdigit():
            matches = [int(key) - 1] if 1 <= int(key) <= len(party) else []
        else:
            with memory_lock:
                nicknames = [ms.name_key(p.nickname or "") for p in env.memory_reader.read_party_pokemon()]
            matches = [i for i, m in enumerate(party) if ms.name_key(m["name"]) == key or nicknames[i] == key]
        if not matches:
            return _refusal(f"{wanted} is not in your party. Your party: {listing}.")
        # Two of the same species: prefer one that can actually come in.
        index = next(
            (i for i in matches if personalities[i] != active and party[i]["hp"] > 0 and not party[i]["egg"]),
            matches[0],
        )
        mon = party[index]
        if mon["egg"]:
            return _refusal(f"Slot {index + 1} is an Egg: it can't battle. Your party: {listing}.")
        if personalities[index] == active:
            return _refusal(f"{mon['name']} is already in battle. Your party: {listing}.")
        if mon["hp"] == 0:
            return _refusal(f"{mon['name']} has fainted (HP 0/{mon['max_hp']}) and can't battle. Your party: {listing}.")
        outgoing = view["player"]["name"].upper() if view.get("player") else "your Pokémon"

        # Seen live: the game's own replacement menu (a fainted MARILL) was open, and
        # switch_pokemon said "pick one with press_buttons" for 30 minutes. There the
        # POKEMON step is already done: pick the Pokémon and SEND OUT.
        forced = _forced_party_menu()
        if not forced:
            view, refusal = await _ready_for_battle_action("switch_pokemon", source)
            if refusal and not _forced_party_menu():
                return refusal
            forced = refusal is not None
        if not forced:
            error = await _choose_battle_action(ms.ACTION_POKEMON, source)
            if error:
                return _refusal(error)
        error = await _shift_from_party_menu(personalities[index], mon["name"], source)
        if error:
            back = await _back_to_battle_menu(source)
            tail = " Back at the battle menu; nobody was switched." if back else " Could not get back to the battle menu: look at the screen."
            return _refusal(error + tail)
        played = await _play_out_battle(source)
        now_in = _mem(ms.active_battler_personality)
        logger.info(f"🔁 switch_pokemon: {played['end']} {_turn_text(played['texts'])}")
        if now_in != personalities[index]:
            return _refusal(
                f"Chose SHIFT to {mon['name']}, but it is not the Pokémon in battle now. {_battle_report(played)}",
                status="STOPPED",
            )
        came_back = f"{outgoing} fainted" if forced else f"{outgoing} came back"
        summary = f"Switched: {came_back} and {mon['name']} is in battle now. {_battle_report(played)}"
        return {"success": True, "status": "SWITCHED", "pokemon": mon["name"], "summary": summary, "end": played["end"]}
    except Exception as e:
        logger.error(f"Error in switch_pokemon: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# -- training: train ---------------------------------------------------------

# max_minutes when none is given, and its cap. agents/PokeAgent.py sizes this tool's
# HTTP timeout from the same two numbers.
TRAIN_DEFAULT_MINUTES = 4.0
TRAIN_MAX_MINUTES = 10.0
# A battle still running when the time is up is finished, for at most this long.
TRAIN_FINISH_BATTLE_S = 90.0
# Partner turns in one wild battle before giving up on it; a partner ten levels up
# needs one to three.
TRAIN_MAX_TURNS = 12
# Steps between two encounter tiles without a battle before stopping (a map whose
# grass has no wild Pokémon, or very bad luck).
TRAIN_MAX_QUIET_STEPS = 300
# Presses in a row that did not reach the other tile (a person standing on it, or a
# press that only turned the player) before trying the next pair of tiles.
TRAIN_MAX_STUCK = 4
# Battle ends that are the model's decision: the stop code names them.
TRAIN_QUESTION_ENDS = ("question", "learn_move", "evolution", "menu")


def _train_party() -> List[Tuple[int, Any]]:
    """[(personality, PokemonData)] in party order; the personality names a Pokémon
    even when two share a species."""
    from server import menu_state as ms

    with memory_lock:
        reader = env.memory_reader
        return list(zip(ms.party_personalities(reader), reader.read_party_pokemon() or []))


def _train_member(party: List[Tuple[int, Any]], wanted: Any) -> Optional[int]:
    """Party index for a species name, nickname or party number 1-6; None if not found."""
    from server import menu_state as ms

    key = ms.name_key(wanted)
    if key.isdigit():
        index = int(key) - 1
        return index if 0 <= index < len(party) and not party[index][1].is_egg else None
    for index, (_, mon) in enumerate(party):
        if not mon.is_egg and key in (ms.name_key(mon.species_name), ms.name_key(mon.nickname or "")):
            return index
    # An earlier form's name ("SHROOMISH" for the BRELOOM it became): objectives and the
    # model's notes keep the name from before the evolution.
    with memory_lock:
        family = {ms.name_key(n) for n in env.memory_reader.species_family(str(wanted))}
    for index, (_, mon) in enumerate(party):
        if not mon.is_egg and ms.name_key(mon.species_name) in family:
            return index
    return None


def _train_can_take_a_turn(party: List[Tuple[int, Any]], trainee_pid: int) -> bool:
    """The trainee can come in for one hit to share the EXP: above half HP, not fainted."""
    mon = dict(party).get(trainee_pid)
    return mon is not None and not mon.is_egg and mon.current_hp * 2 > mon.max_hp


def _train_partner(party: List[Tuple[int, Any]], trainee_pid: int, wanted_pid: Optional[int]) -> Optional[int]:
    """Index of the teammate that fights for the trainee: the one asked for, else, in a
    battle, the teammate above half HP that knocks the foe out in the fewest hits
    (reader.turns_to_win, the battle view's damage estimate; level breaks ties), else the
    highest-level one (the partner _switch_train_line names). None when that Pokémon
    can't fight now. Seen live vs MAGNETON: the highest level isn't always the one that
    hurts it (EMBER 2x where MEGA DRAIN and SPARK were 0.5x)."""
    able = [
        i for i, (pid, mon) in enumerate(party)
        if pid != trainee_pid and not mon.is_egg and mon.max_hp and mon.current_hp * 2 > mon.max_hp
    ]
    if wanted_pid is not None:
        return next((i for i in able if party[i][0] == wanted_pid), None)
    if not able:
        return None
    with memory_lock:
        reader = env.memory_reader
        view = reader.read_battle_view() if reader.is_in_battle() else None
        target = (view or {}).get("target")
        if target and target.get("hp"):
            return min(able, key=lambda i: (reader.turns_to_win(i, party[i][1], target), -party[i][1].level))
    return max(able, key=lambda i: party[i][1].level)


def _train_weakness(partner_index: Optional[int]) -> List[str]:
    """The team-weak reasons (read_team_condition), with the partner's PP as the one that
    counts: the trainee leads but never attacks."""
    with memory_lock:
        condition = env.memory_reader.read_team_condition(lead_index=partner_index)
    return list(condition["reasons"]) if condition["weak"] else []


def _heal_way() -> str:
    from server import game_tools

    snap = _field_snapshot()
    location = _location_name()
    way = game_tools.way_to_pokemon_center(location, (snap.x, snap.y) if snap is not None else None)
    return way or f"you are inside {location}: call heal_at_pokecenter() now."


def _mon_text(mon) -> str:
    return f"{mon.species_name.upper()} Lv{mon.level} HP {mon.current_hp}/{mon.max_hp}"


async def _train_battle(
    trainee_pid: int, wanted_partner_pid: Optional[int], deadline: float,
    catch: Optional[Tuple[str, int]] = None,
) -> Tuple[Dict[str, Any], Optional[Tuple[str, str]]]:
    """Fight the wild battle in progress for the trainee: switch it out for the partner on
    its first turn (switch_pokemon), then the best move each turn (use_move, which plays
    the text out). A trainee with the advantage (_trainee_advantage) fights itself until
    it drops to half HP. Returns (record of the battle, stop) where stop is (code, why)
    or None. With catch=(species, min level), a wild one of those at that level stops it
    before the first move, with the catch plan (_catch_plan).
    """
    from server import menu_state as ms

    source = "train"
    before = dict(_train_party())
    trainee_name = before[trainee_pid].species_name.upper()
    view = _read_battle_view()
    foes = [f"{m['name'].upper()} Lv{m['level']}" for m in (view or {}).get("opponents") or []]
    record: Dict[str, Any] = {"opponents": foes, "turns": 0, "switched": False, "trainee_fought": False}
    partner_pid = None
    stop: Optional[Tuple[str, str]] = None
    if view is not None and view["is_trainer"]:
        stop = (
            "trainer_battle",
            f"a trainer battle started ({', '.join(foes)}): train only fights wild Pokémon. "
            "Fight it yourself with use_move.",
        )
    while stop is None and _read_battle_view() is not None:
        if time.monotonic() > deadline:
            stop = ("time", "the time ran out in the middle of a battle: finish it with use_move.")
            break
        if record["turns"] >= TRAIN_MAX_TURNS:
            stop = ("unexpected", f"the battle went on for {TRAIN_MAX_TURNS} turns: finish it yourself.")
            break
        view = _read_battle_view()
        if view is not None and view["menu"] == "busy":
            # Decide only at the menu: during "Wild X appeared!" gBattleMons still holds
            # the last battle's Pokémon. Seen in the tests: train read TORCHIC as the one
            # out, picked its EMBER, and use_move then found SHROOMISH's moves.
            played = await _play_out_battle(source)
            if played["end"] in ("action", "over"):
                continue
            stop = (
                played["end"] if played["end"] in TRAIN_QUESTION_ENDS else "unexpected",
                _battle_report(played),
            )
            break
        if catch is not None and record["turns"] == 0 and not record["switched"] and view is not None:
            # First decision of the battle: gBattleMons now holds this battle's Pokémon.
            found = next(
                (f for f in view.get("opponents") or [] if str(f.get("name", "")).upper() == catch[0] and f.get("level", 0) >= catch[1]),
                None,
            )
            if found is not None:
                record["opponents"] = [f"{m['name'].upper()} Lv{m['level']}" for m in view.get("opponents") or []]
                stop = ("catch_target", f"a wild {catch[0]} Lv{found['level']} appeared. " + _catch_plan(view, found))
                break
        party = _train_party()
        advantage = _trainee_advantage(view) if view is not None else None
        if _mem(ms.active_battler_personality) == trainee_pid and advantage is not None:
            # The foe is weak to the trainee and can't hit it hard: it fights itself, and
            # is switched out as soon as it drops to half HP (checked every turn).
            result = await mcp_use_move({"move": advantage["slot"], "reasoning": f"train: {advantage['name']}"})
            record["turns"] += 1
            record["trainee_fought"] = True
        elif _mem(ms.active_battler_personality) == trainee_pid:
            index = _train_partner(party, trainee_pid, wanted_partner_pid)
            if index is None:
                stop = (
                    "no_partner",
                    f"{trainee_name} is in battle and no teammate above half HP can come in for it: "
                    "fight or run_from_battle yourself.",
                )
                break
            partner_pid = party[index][0]
            result = await mcp_switch_pokemon({"pokemon": index + 1, "reasoning": f"train: {trainee_name} must not fight"})
            record["switched"] = record["switched"] or bool(result.get("success"))
        elif (
            not record.get("trainee_sent_in")
            and not record["switched"]  # it was out already (switched out for the partner)
            and not record["trainee_fought"]
            and _train_can_take_a_turn(party, trainee_pid)
            and _train_partner(party, trainee_pid, wanted_partner_pid) is not None
            # The one out now, by its battle HP (the party copy lags a turn): a weak one
            # fights on rather than make room for a switch nobody can undo.
            and view is not None and view.get("player")
            and view["player"]["hp"] * 2 > view["player"]["max_hp"]
        ):
            # Only Pokémon that were out in the battle get its EXP. Seen live: SABLEYE led
            # the party, train("SHROOMISH") had it fight every battle, and SABLEYE went
            # Lv18 -> 21 while SHROOMISH stayed Lv17. The trainee comes in once (the turn
            # after, the branch above swaps the partner back in), and only while a
            # teammate above half HP can take over from it.
            record["trainee_sent_in"] = True
            index = next(i for i, (pid, _) in enumerate(party) if pid == trainee_pid)
            result = await mcp_switch_pokemon({"pokemon": index + 1, "reasoning": f"train: {trainee_name} joins for the EXP"})
        else:
            partner_pid = _mem(ms.active_battler_personality)
            best = view.get("best_move") if view else None
            if not best:
                name = view["player"]["name"].upper() if view and view.get("player") else "Your Pokémon"
                stop = ("no_moves", f"{name} has no damaging move with PP left: heal first ({_heal_way()}).")
                break
            result = await mcp_use_move({"move": best["slot"], "reasoning": f"train: {best['name']}"})
            record["turns"] += 1
        end = result.get("end")
        if not result.get("success"):
            if _read_battle_view() is None:
                break  # the battle ended before the choice (the wild Pokémon fled)
            stop = ("unexpected", result.get("summary") or result.get("error") or "the battle menu did not respond")
        elif end in TRAIN_QUESTION_ENDS:
            stop = (end, result.get("summary") or "")
        elif end not in ("action", "over"):
            stop = ("unexpected", result.get("summary") or f"the battle stopped ({end})")
    after = dict(_train_party())
    trainee_before, trainee_after = before[trainee_pid], after.get(trainee_pid, before[trainee_pid])
    outcome = "unfinished" if _read_battle_view() is not None else ms.outcome_name(_mem(ms.read_battle_flow).outcome)
    record.update({
        "outcome": outcome,
        "trainee": trainee_name,
        "level_before": trainee_before.level,
        "level_after": trainee_after.level,
        "exp_gained": (trainee_after.experience or 0) - (trainee_before.experience or 0),
        "trainee_hp_before": trainee_before.current_hp,
        "trainee_hp": trainee_after.current_hp,
        "trainee_max_hp": trainee_after.max_hp,
    })
    if partner_pid in after and partner_pid in before:
        record.update({
            "partner": after[partner_pid].species_name.upper(),
            "partner_hp_before": before[partner_pid].current_hp,
            "partner_hp": after[partner_pid].current_hp,
            "partner_max_hp": after[partner_pid].max_hp,
        })
    logger.info(f"🏋️ train battle: {record} stop={stop}")
    return record, stop


def _train_stop_after_battle(
    trainee_pid: int, wanted_partner_pid: Optional[int], until_level: Optional[int], deadline: float, minutes: float
) -> Optional[Tuple[str, str]]:
    """(code, why) when grinding should stop after a battle, else None."""
    party = _train_party()
    trainee = dict(party).get(trainee_pid)
    if trainee is None:
        return ("unexpected", "the Pokémon being trained is no longer in the party.")
    name = trainee.species_name.upper()
    if until_level is not None and trainee.level >= until_level:
        return ("until_level", f"{name} reached Lv{trainee.level} (until_level={until_level}).")
    index = _train_partner(party, trainee_pid, wanted_partner_pid)
    weak = _train_weakness(index)
    if weak:
        return ("team_weak", f"the team is weak ({', '.join(weak)}): heal first. {_heal_way()}")
    if index is None:
        return ("no_partner", f"no teammate above half HP is left to fight for {name}: heal first. {_heal_way()}")
    if time.monotonic() > deadline:
        return ("time", f"the {minutes:g}-minute limit is up (call train again to keep going).")
    return None


# Heal trips train may make on its own in one call (more would let it ping-pong).
TRAIN_MAX_HEALS = 1


async def _train_heal_trip(origin: str) -> Tuple[bool, str]:
    """heal_at_pokecenter, then go_to back to `origin` (the map train was called on).

    Seen live: train refused three times in a row with "the team is weak … call
    heal_at_pokecenter()", the model alternated train and heal attempts, and the stall
    watcher paused it; each refusal cost a ~35 s decision. (ok, note for the report).
    """
    healed = await mcp_heal_at_pokecenter({})
    if not healed.get("success"):
        return False, f"tried to heal first: {healed.get('error') or healed.get('summary')}"
    center = _location_name()
    back = await mcp_go_to({"place": origin, "reasoning": "train: back to the grass after healing"})
    if not back.get("success"):
        return False, f"healed at {center}, but did not get back to {origin}: {back.get('error') or back.get('summary')}"
    return True, f"healed at {center} and came back to {origin}"


def _train_summary(battles: List[Dict[str, Any]], steps: int, elapsed: float, why: str) -> str:
    """'4 wild battles (WHISMUR Lv6, …): SHROOMISH Lv7→9 (+120 EXP shared), COMBUSKEN 45→31/51. Stopped: …'"""
    if not battles:
        return f"No wild battle in {elapsed:.0f} s ({steps} steps in the grass). Stopped: {why}"
    names = [b["opponents"][0] if b["opponents"] else "?" for b in battles]
    shown = ", ".join(names[:6]) + (", …" if len(names) > 6 else "")
    first, last = battles[0], battles[-1]
    exp = sum(b["exp_gained"] for b in battles)
    text = (
        f"{len(battles)} wild battle{'s' if len(battles) != 1 else ''} in {elapsed:.0f} s ({shown}): "
        f"{first['trainee']} Lv{first['level_before']}→{last['level_after']} (+{exp} EXP shared, "
        f"HP {last['trainee_hp']}/{last['trainee_max_hp']})"
    )
    partners: Dict[str, Tuple[int, int, int]] = {}
    for b in battles:
        if "partner" in b:
            start = partners.get(b["partner"], (b["partner_hp_before"],))[0]
            partners[b["partner"]] = (start, b["partner_hp"], b["partner_max_hp"])
    text += "".join(f", {name} {hp0}→{hp1}/{top}" for name, (hp0, hp1, top) in partners.items())
    return f"{text}. Stopped: {why}"


@app.post("/mcp/train")
async def mcp_train(request: dict = None):
    """MCP Tool: grind wild battles for one party Pokémon until a stop condition.

    David (2026-11-03): "Can we speed it up between battles? It sits for a while after
    the battle is over." Measured live: switch-training a Lv7 SHROOMISH with a Lv17
    COMBUSKEN cost about three ~30 s model decisions per wild battle (walk into the grass,
    switch_pokemon("COMBUSKEN"), use_move until it ends), and the pause after each battle
    was the model deciding where to walk next. None of that needs a decision.

    Outside a battle it walks to the nearest two side-by-side encounter tiles (tall
    grass, or any cave floor) and steps between them until a battle starts. In a wild
    battle the trainee goes out on turn 1 for the partner (switch_pokemon), which then
    uses its best move each turn (use_move); both share the EXP. Against a foe it beats
    with a super-effective move and that can't hit it super-effectively, the trainee
    fights itself while above half HP (_trainee_advantage). It returns to the model
    for anything that is a decision or a surprise: until_level reached, the team weak
    (the TEAM IS WEAK check, with the partner's PP), a question (learn a move, evolve,
    "Use next POKéMON?"), a trainer battle, the time limit, or a walk or menu that did
    not go as expected. The game runs at turbo speed for the whole call.
    """
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    if game_type != "emerald":
        return {"success": False, "error": "train is only implemented for Emerald"}
    try:
        with _turbo():
            return await _train(request or {})
    except Exception as e:
        logger.error(f"Error in train: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def _train(request: Dict[str, Any]) -> Dict[str, Any]:
    from server import game_tools
    from server.field_control import DIRECTION_DELTAS

    source = "train"
    wanted = request.get("trainee", request.get("pokemon"))
    if wanted is None or str(wanted).strip() == "":
        return _refusal("trainee is required: the party Pokémon to train, e.g. 'SHROOMISH' (or its party number 1-6).")
    until_level = request.get("until_level")
    if until_level is not None and str(until_level).strip() != "":
        try:
            until_level = int(until_level)
        except (TypeError, ValueError):
            return _refusal(f"until_level must be a level number 2-100, not {until_level!r}.")
        if not 2 <= until_level <= 100:
            return _refusal(f"until_level must be a level number 2-100, not {until_level}.")
    else:
        until_level = None
    try:
        minutes = float(request.get("max_minutes") or TRAIN_DEFAULT_MINUTES)
    except (TypeError, ValueError):
        return _refusal(f"max_minutes must be a number of minutes, not {request.get('max_minutes')!r}.")
    minutes = min(minutes, TRAIN_MAX_MINUTES) if minutes > 0 else TRAIN_DEFAULT_MINUTES
    catch = None
    if request.get("catch") is not None and str(request.get("catch")).strip() != "":
        try:
            catch = (str(request["catch"]).strip().upper(), int(request.get("catch_min_level") or 1))
        except (TypeError, ValueError):
            return _refusal(f"catch_min_level must be a level number, not {request.get('catch_min_level')!r}.")

    party = _train_party()
    listing = ", ".join(f"{i + 1}. {_mon_text(mon)}" for i, (_, mon) in enumerate(party))
    index = _train_member(party, wanted)
    if index is None:
        return _refusal(f"{wanted} is not in your party. Your party: {listing}.")
    trainee_pid, trainee = party[index]
    trainee_name = trainee.species_name.upper()
    if until_level is not None and trainee.level >= until_level:
        return _refusal(f"{trainee_name} is already Lv{trainee.level} (until_level={until_level}): nothing to train.")
    partner_pid = None
    if request.get("partner") is not None and str(request.get("partner")).strip() != "":
        p = _train_member(party, request["partner"])
        if p is None:
            return _refusal(f"{request['partner']} is not in your party. Your party: {listing}.")
        if party[p][0] == trainee_pid:
            return _refusal(f"The partner must be another Pokémon: {trainee_name} is the one being trained and must not fight.")
        partner_pid = party[p][0]

    in_battle = _mem(lambda r: r.is_in_battle())
    spots = None
    if not in_battle:
        refused = _field_refusal(source)
        if refused:
            return refused

        def _spots():
            with memory_lock:
                return game_tools.wild_encounter_spots(env)

        spots = await run_in_threadpool(_spots)
        if not spots["pairs"]:
            snap = _field_snapshot()
            location = _location_name()
            way = await run_in_threadpool(
                game_tools.way_to_wild_encounters, location, (snap.x, snap.y) if snap is not None else None
            )
            if spots.get("unreachable"):
                where = _position_text(snap) if snap is not None else location
                return _refusal(
                    f"{location} has tall grass, but none of it can be walked to from {where}: {way}",
                    stopped_because="no_grass",
                )
            return _refusal(f"No tall grass here ({location}), and it isn't a cave: {way}", stopped_because="no_grass")
    partner_index = _train_partner(party, trainee_pid, partner_pid)
    weak = _train_weakness(partner_index)
    heal_notes: List[str] = []
    origin = _location_name()
    if weak and not in_battle:
        # Heal first, then come back here and train (TRAIN_MAX_HEALS per call).
        ok, note = await _train_heal_trip(origin)
        if not ok:
            return _refusal(f"Not training: the team is weak ({', '.join(weak)}); {note}", stopped_because="team_weak")
        heal_notes.append(note)
        party = _train_party()
        spots = await run_in_threadpool(_spots)
        if not spots["pairs"]:
            return _refusal(f"{note}, but no tall grass can be walked to from here now.", stopped_because="no_grass")
        partner_index = _train_partner(party, trainee_pid, partner_pid)
        weak = _train_weakness(partner_index)
    if weak:
        return _refusal(
            f"Not training: the team is weak ({', '.join(weak)}). Heal first: {_heal_way()}", stopped_because="team_weak"
        )
    if partner_index is None:
        who = f"{_mon_text(dict(party)[partner_pid])} can't" if partner_pid is not None else "No teammate can"
        return _refusal(
            f"{who} fight for {trainee_name}: train needs a teammate above half HP to come in for it "
            f"(your party: {listing}). {_heal_way() if len(party) > 1 else 'Catch a second Pokémon first.'}",
            stopped_because="no_partner",
        )

    logger.info(f"🏋️ train {trainee_name} until_level={until_level} max_minutes={minutes:g} partner={partner_pid}")
    started = time.monotonic()
    deadline = started + minutes * 60
    battles: List[Dict[str, Any]] = []
    stop: Optional[Tuple[str, str]] = None
    steps = stuck = pair_index = 0
    quiet_steps = 0
    last_walk = ""
    while stop is None:
        if _mem(lambda r: r.is_in_battle()):
            record, stop = await _train_battle(trainee_pid, partner_pid, deadline + TRAIN_FINISH_BATTLE_S, catch)
            battles.append(record)
            quiet_steps = 0
            if stop is None:
                stop = _train_stop_after_battle(trainee_pid, partner_pid, until_level, deadline, minutes)
            if stop is not None and stop[0] == "team_weak" and len(heal_notes) < TRAIN_MAX_HEALS:
                await _await_game(_player_can_walk, MACRO_TEXT_FRAMES)  # the battle's end, back on the map
                ok, note = await _train_heal_trip(origin)
                if ok:
                    heal_notes.append(note)
                    stop, spots, pair_index, stuck = None, None, 0, 0  # find the grass again
                else:
                    stop = ("team_weak", f"{stop[1]} {note}")
            continue
        snap = _field_snapshot()
        if snap is None:
            stop = ("unexpected", "the player's position could not be read.")
            break
        if not snap.field_free:
            # A battle's opening transition, a fade, or a script: wait for one to settle.
            if await _await_game(lambda: _mem(lambda r: r.is_in_battle()) or _player_can_walk(), MACRO_TEXT_FRAMES):
                continue
            stop = ("unexpected", f"something else has control of the player at {_position_text(snap)}.{_screen_note()}")
            break
        # Only once nothing is starting: seen in a test, the last step before the deadline
        # began an encounter whose intro wasn't yet "in battle", and train handed back
        # mid-intro. The settle above lets that battle be finished first.
        if time.monotonic() > deadline:
            stop = ("time", f"the {minutes:g}-minute limit is up (call train again to keep going).")
            break
        if spots is None or spots.get("map_id", snap.map_id) != snap.map_id:
            if spots is not None:
                stop = ("unexpected", f"the player ended up on another map ({_position_text(snap)}).")
                break

            def _spots():
                with memory_lock:
                    return game_tools.wild_encounter_spots(env)

            spots = await run_in_threadpool(_spots)
            if not spots["pairs"]:
                stop = ("no_grass", f"no tall grass here ({_location_name()}) to keep training in.")
                break
        spots.setdefault("map_id", snap.map_id)
        if pair_index >= len(spots["pairs"]):
            stop = ("unexpected", f"could not get to the tall grass: {last_walk or 'every spot was blocked'}")
            break
        ox, oy = spots["coord_offset"] or (0, 0)
        here = (snap.x - ox, snap.y - oy)
        a, b = spots["pairs"][pair_index]
        if here not in (a, b):
            walk = await mcp_navigate_to(
                {"x": a[0], "y": a[1], "variance": "none", "reason": "train: walk to the tall grass", "consider_npcs": True}
            )
            snap = _field_snapshot()
            arrived = snap is not None and (snap.x - ox, snap.y - oy) in (a, b)
            busy = snap is None or not snap.field_free or _mem(lambda r: r.is_in_battle())
            if not arrived and not busy:
                # A wild battle on the way is welcome; anything else means try another spot.
                last_walk = walk.get("summary") or walk.get("error") or ""
                pair_index += 1
            continue
        target = b if here == a else a
        button = next(k for k, (dx, dy) in DIRECTION_DELTAS.items() if (here[0] + dx, here[1] + dy) == target)
        queued = await take_action(ActionRequest(buttons=[button], source=source, check_movement=True))
        report = await _await_sequence(queued.get("sequence_id"), _sequence_timeout(1))
        after = _field_snapshot()
        if after is not None and (after.x - ox, after.y - oy) == target:
            steps += 1
            quiet_steps += 1
            stuck = 0
            if quiet_steps >= TRAIN_MAX_QUIET_STEPS:
                stop = ("no_encounters", f"no wild Pokémon appeared in {quiet_steps} steps here.")
        elif after is not None and after.field_free:
            # Only turned, or someone stands on the tile: give them a moment, then move on.
            stuck += 1
            if report and report.get("blocked"):
                last_walk = report["blocked"]
            if stuck >= TRAIN_MAX_STUCK:
                pair_index += 1
                stuck = 0
            else:
                await _await_game(lambda: False, 30)

    code, why = stop
    elapsed = time.monotonic() - started
    summary = _train_summary(battles, steps, elapsed, why)
    if heal_notes:
        summary = f"On the way: {'; '.join(heal_notes)}. {summary}"
    logger.info(f"🏋️ train: {summary}")
    return {
        "success": True,
        "status": code.upper(),
        "summary": summary,
        "battles": battles,
        "stopped_because": code,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
    }


# -- people behind counters (nurse, clerk) ---------------------------------


def _counter_tile_between(x: int, y: int) -> Optional[bool]:
    """Is (x, y) a counter tile? None if it can't be read (out of the visible area)."""
    from pokemon_env.enums import MetatileBehavior

    with memory_lock:
        reader = env.memory_reader
        px, py = reader.read_coordinates()
        grid = reader.read_map_around_player(radius=7)
    row, col = y - py + 7, x - px + 7
    if not grid or not (0 <= row < len(grid)) or not (0 <= col < len(grid[row])):
        return None
    return grid[row][col][1] == MetatileBehavior.COUNTER


async def _walk_up_to(graphics_id: int, who: str, source: str) -> Tuple[Optional[Any], Optional[str]]:
    """Walk to the tile facing the nearest person with this sprite and face them.

    Returns (CounterPerson, None) or (None, why not).
    """
    from server import menu_state as ms
    from server.field_control import DIRECTION_DELTAS

    snap = _field_snapshot()
    people = _mem(ms.find_objects_by_graphics, graphics_id)
    if not people or snap is None:
        return None, f"no {who} is on this map"
    x, y, facing = min(people, key=lambda p: abs(p[0] - snap.x) + abs(p[1] - snap.y))
    dx, dy = DIRECTION_DELTAS[facing]
    counter = _counter_tile_between(x + dx, y + dy)
    # Unreadable (None) is treated as a counter: nurses and clerks all stand behind one.
    spot = ms.talk_spot(x, y, facing, counter is not False)
    if (snap.x, snap.y) != spot.stand:
        walk = await mcp_navigate_to(
            {"x": spot.stand[0], "y": spot.stand[1], "variance": "none", "reason": f"{source}: talk to the {who}", "consider_npcs": True}
        )
        snap = _field_snapshot()
        if snap is None or (snap.x, snap.y) != spot.stand:
            return None, (
                f"could not walk to ({spot.stand[0]},{spot.stand[1]}), the tile facing the {who} at ({x},{y}): "
                f"{walk.get('summary') or walk.get('error')}"
            )
    if snap.facing != spot.face:
        # The counter (or the person) blocks that way, so this press only turns the player.
        await _macro_press([spot.face], source)
        snap = _field_snapshot()
        if snap is None or snap.facing != spot.face or (snap.x, snap.y) != spot.stand:
            return None, f"could not face the {who} from ({spot.stand[0]},{spot.stand[1]})"
    return spot, None


def _field_status() -> str:
    from server.field_control import dialogue_status

    snap = _field_snapshot()
    return dialogue_status(snap) if snap is not None else "other"


async def _read_dialogue(max_presses: int = 12) -> Dict[str, Any]:
    """Press A through plain text (never into a YES/NO or menu): the auto-advance step."""
    return await mcp_auto_advance_dialogue({"max_presses": max_presses})


async def _talk_until(ready, rounds: int = 4) -> Tuple[bool, List[str]]:
    """After talking to someone, read through their plain text until ready() (their
    YES/NO or menu task is running). Returns (ready, texts read).

    Seen in a test run under load: the auto-advance stopped right as the question
    finished printing, a few frames before the YES/NO task started, so a single
    status check after it called a normal greeting a failure.
    """
    from server.field_control import DIALOGUE_BUSY, DIALOGUE_NONE, DIALOGUE_PLAIN

    texts: List[str] = []
    for _ in range(rounds):
        if await _await_game(ready, 60):
            return True, texts
        status = _field_status()
        if status in (DIALOGUE_PLAIN, DIALOGUE_BUSY):
            read = await _read_dialogue()
            texts.extend(t for t in read.get("texts", []) if t not in texts)
        elif status == DIALOGUE_NONE:
            break  # the conversation ended without the menu
    if await _await_game(ready, MACRO_MENU_FRAMES):
        return True, texts
    said = _mem(read_field_text)
    if said and said not in texts:
        texts.append(said)
    return False, texts


async def _close_dialogue(source: str) -> bool:
    """Advance the last plain text until the player can walk again."""
    from server.field_control import DIALOGUE_BUSY, DIALOGUE_PLAIN

    for _ in range(3):
        if _field_status() in (DIALOGUE_PLAIN, DIALOGUE_BUSY):
            await _read_dialogue()
        if await _await_game(_player_can_walk, MACRO_MENU_FRAMES):
            return True
    return False


def _exit_hint(query: str) -> str:
    """'go_to_exit("POKEMON CENTER")' if this map has such an exit, else ''."""
    try:
        from server import game_tools

        with memory_lock:
            plan = game_tools.plan_exit_route(env, query)
        if plan.get("success"):
            return f'go_to_exit("{query}")'
    except Exception as e:  # a hint must never fail the call
        logger.debug(f"exit hint for {query} failed: {e}")
    return ""


def _field_refusal(tool: str) -> Optional[Dict[str, Any]]:
    """Refuse a field macro during a battle or while a text box or menu is open."""
    from server.field_control import DIALOGUE_NONE

    refused = _refuse_during_battle(tool)
    if refused:
        return refused
    status = _field_status()
    if status != DIALOGUE_NONE:
        return _refusal(f"{tool} needs you walking on the map, but{_screen_note() or ' another screen is open.'}")
    return None


HEAL_TRIP_MAX_MAPS = 10
# _walk_along: scripted scenes (a trainer spotting the player) waited out, each up to
# MACRO_TEXT_FRAMES, before the walk reports that something has the player's attention.
WALK_SCENE_WAITS = 3


async def _walk_to_pokemon_center(source: str) -> Tuple[List[str], Optional[str]]:
    """Walk to the nearest Pokémon Center 1F (game_tools.pokemon_center_route), running
    from wild battles on the way. Returns (maps entered, why it stopped short or None).

    Seen live: with "TEAM IS WEAK … go_to_exit("south") takes the first step" in its
    prompt, the model called train (refused: team weak) and navigate_to on its own tile
    over and over while poison drained SHROOMISH. heal_at_pokecenter() now makes the trip
    in the same call.
    """
    from server import game_tools
    from server import menu_state as ms

    return await _walk_along(
        lambda location, position: game_tools.pokemon_center_route(location, position),
        lambda: bool(_mem(ms.find_objects_by_graphics, ms.OBJ_EVENT_GFX_NURSE)),
        "the nearest Pokémon Center",
        source,
        f"{source}() again",
    )


async def _walk_along(
    next_route, arrived, goal: str, source: str, again: str, max_maps: int = HEAL_TRIP_MAX_MAPS
) -> Tuple[List[str], Optional[str]]:
    """Walk map to map until arrived() is true, one go_to_exit per map from
    next_route(location, position) (recomputed on every map, so a detour is
    corrected). Wild battles on the way are run from; a trainer battle or a YES/NO
    stops the walk; plain text from a scene that interrupts (Birch's on ROUTE 110) is
    read through. Returns (maps entered, why it stopped short or None).
    """
    from server.field_control import DIALOGUE_BUSY, DIALOGUE_PLAIN

    entered: List[str] = []
    # (map, go_to_exit argument) of every step that left a map. Seen live: go_to("Route
    # 113") crossed ROUTE 111 → ROUTE 112 → ROUTE 111 … fifteen times before its map budget
    # ran out. Taking the same exit off the same map twice in one walk is a loop.
    crossed = set()
    reads = waits = 0
    for _ in range(max_maps * 3):
        if _mem(lambda r: r.is_in_battle()):
            view = _read_battle_view()
            if view is not None and view.get("is_trainer"):
                return entered, f"a trainer battle started on the way: fight it with use_move, then {again}"
            ran = await mcp_run_from_battle({"reasoning": f"{source}: on the way to {goal}"})
            if _mem(lambda r: r.is_in_battle()):
                return entered, f"could not run from a wild battle on the way ({ran.get('summary') or ran.get('error')})"
            continue
        if arrived():
            return entered, None
        if not _player_can_walk():
            status = _field_status()
            if status in (DIALOGUE_PLAIN, DIALOGUE_BUSY) and reads < 6:
                reads += 1
                await _read_dialogue()
                continue
            # A trainer who spots the player: the "!", the walk up and their line play as
            # a script the field status reads as neither plain nor busy text, and the
            # battle starts by itself ~2 s later. Seen live (Breeder Isaac, ROUTE 117):
            # the walk gave up at once with "something has the player's attention", and
            # the model fought him with a BRELOOM it had been on its way to heal.
            if waits < WALK_SCENE_WAITS:
                waits += 1
                settled = await _await_game(
                    lambda: _mem(lambda r: r.is_in_battle())
                    or _player_can_walk()
                    or _field_status() in (DIALOGUE_PLAIN, DIALOGUE_BUSY),
                    MACRO_TEXT_FRAMES,
                )
                if settled:
                    continue
            return entered, f"something has the player's attention on the way{_screen_note()}"
        snap = _field_snapshot()
        location = _location_name()
        route = await run_in_threadpool(next_route, location, (snap.x, snap.y) if snap is not None else None)
        if route == []:
            return entered, None  # arrived by the map's own measure
        if not route:
            return entered, f"{goal} can't be reached on foot from {location}"
        if len(entered) >= max_maps:
            return entered, f"{goal} is still {len(route)} maps away"
        if (location, route[0][0]) in crossed:
            return entered, (
                f'the way loops: go_to_exit("{route[0][0]}") from {location} was taken once already on this walk. '
                f"The way planned now is {' → '.join(name for _, name in route)}"
            )
        step = await mcp_go_to_exit({"exit": route[0][0], "reasoning": f"{source}: walk to {route[-1][1]}"})
        if _location_name() != location:
            crossed.add((location, route[0][0]))
            entered.append(_location_name())
        elif not _mem(lambda r: r.is_in_battle()) and _player_can_walk():
            return entered, f'go_to_exit("{route[0][0]}") did not get there ({step.get("summary") or step.get("error")})'
    return entered, "the walk took too many steps"


GO_TO_MAX_MAPS = 15


@app.post("/mcp/go_to")
async def mcp_go_to(request: dict = None):
    """MCP Tool: walk to a named place (a town, route, building, cave floor), map after map.

    David (2026-11-04), on what would speed play up: "One call per objective." Each map
    on the way cost a ~35 s model decision (go_to_exit, then think again); this walks the
    whole way game_tools.place_route gives, running from wild battles, reading through
    scenes, and stopping for a trainer battle or a question.
    """
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    from server import game_tools

    source = "go_to"
    request = request or {}
    target = str(request.get("place") or request.get("target") or "").strip()
    if not target:
        return _refusal('place is required: where to go, e.g. "Route 117" or "Mauville City Pokemon Center".')
    refused = _field_refusal(source)
    if refused:
        return refused
    snap = _field_snapshot()
    start = _location_name()
    if game_tools.arrived_at(start, (snap.x, snap.y) if snap is not None else None, target):
        return {"success": True, "status": "ARRIVED", "summary": f"You are already on {start} ({target})."}

    def arrived() -> bool:
        here = _field_snapshot()
        return game_tools.arrived_at(_location_name(), (here.x, here.y) if here is not None else None, target)

    with _turbo():
        entered, why = await _walk_along(
            lambda location, position: game_tools.place_route(location, position, target),
            arrived,
            target,
            source,
            f'go_to("{target}") again',
            GO_TO_MAX_MAPS,
        )
    walked = f" Walked {' → '.join(entered)}." if entered else ""
    if why:
        return _refusal(f"Did not reach {target}: {why}.{walked}")
    return {
        "success": True,
        "status": "ARRIVED",
        "summary": f"Arrived at {target}: now at {_position_text(_field_snapshot())}.{walked}",
    }


@app.post("/mcp/talk_to")
async def mcp_talk_to(request: dict = None):
    """MCP Tool: walk next to the person at (x, y), face them, press A and read what they say.

    Seen live twice: told to talk to Rydel (bike shop) and to Wally (in front of Mauville
    Gym), the model aimed navigate_to at the person's own tile, which can't be stood on,
    and shuffled between the tiles around them for many minutes without facing them and
    pressing A. One call now does all of it; a battle that starts is left to use_move.
    """
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    from server.field_control import DIRECTION_DELTAS

    source = "talk_to"
    request = request or {}
    try:
        x, y = int(request.get("x")), int(request.get("y"))
    except (TypeError, ValueError):
        return _refusal("x and y are required: the tile the person stands on (from the map or OBJECTIVES).")
    refused = _field_refusal(source)
    if refused:
        return refused
    snap = _field_snapshot()
    if snap is None:
        return _refusal("The player's position could not be read.")
    # The four tiles around the person, nearest to the player first.
    sides = sorted(
        ((x - dx, y - dy, button) for button, (dx, dy) in DIRECTION_DELTAS.items()),
        key=lambda t: abs(t[0] - snap.x) + abs(t[1] - snap.y),
    )
    tried = []
    for sx, sy, facing in sides:
        snap = _field_snapshot()
        if snap is None or (snap.x, snap.y) != (sx, sy):
            walk = await mcp_navigate_to({"x": sx, "y": sy, "variance": "none", "reason": f"{source}: next to ({x},{y})", "consider_npcs": True})
            if _mem(lambda r: r.is_in_battle()):
                return _refusal(f"A battle started on the way to ({x},{y}): fight it with use_move, then talk_to again.")
            snap = _field_snapshot()
            if snap is None or (snap.x, snap.y) != (sx, sy):
                tried.append(f"({sx},{sy}): {str(walk.get('summary') or walk.get('error'))[:80]}")
                continue
        from server.menu_state import phone_call_active

        calls = []
        for _ in range(3):
            # A POKéNAV call can ring on the walk (seen in a test: ISABEL called as the
            # player reached Wally, and the A read her call). Finish it, then talk.
            while _mem(phone_call_active):
                calls.extend((await _read_dialogue()).get("texts") or [])
            if snap.facing != facing:
                await _macro_press([facing], source)  # the person blocks the step, so this only turns
                snap = _field_snapshot() or snap
            await _macro_press(["A"], source)
            if await _await_game(lambda: _mem(lambda r: r.is_in_battle()), 60):
                return {"success": True, "status": "BATTLE", "summary": f"Talked to the person at ({x},{y}) from ({sx},{sy}); a battle started: use use_move."}
            if not _mem(phone_call_active):
                break
        # A scene can run many boxes before its battle (Wally and his uncle argue first):
        # keep reading while a script holds the player and no battle has begun.
        from server.field_control import DIALOGUE_BUSY, DIALOGUE_PLAIN

        texts: List[str] = []
        read: Dict[str, Any] = {}
        for _ in range(6):
            read = await _read_dialogue()
            texts.extend(t for t in read.get("texts") or [] if t not in texts)
            if _mem(lambda r: r.is_in_battle()) or _field_status() not in (DIALOGUE_PLAIN, DIALOGUE_BUSY):
                break
        if calls:
            texts = texts + [f"(a POKéNAV call came first: {_quote(calls, 1)})"]
        if _mem(lambda r: r.is_in_battle()):
            return {"success": True, "status": "BATTLE", "summary": f"Talked to ({x},{y}): {_quote(texts)}. A battle started: use use_move."}
        said = _quote(texts) or "nothing was said (nobody there, or they only turned)"
        return {"success": True, "status": "TALKED", "summary": f"Talked to the person at ({x},{y}) from ({sx},{sy}): {said}. {read.get('summary', '')}".strip()}
    return _refusal(f"Could not reach a tile next to ({x},{y}): " + "; ".join(tried))


@app.post("/mcp/heal_at_pokecenter")
async def mcp_heal_at_pokecenter(request: dict = None):
    """MCP Tool: inside a Pokémon Center, talk to the nurse, answer YES, and confirm
    from memory that the whole party is healed."""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    if game_type != "emerald":
        return {"success": False, "error": "heal_at_pokecenter is only implemented for Emerald"}
    try:
        from server import menu_state as ms

        source = "heal_at_pokecenter"
        refused = _field_refusal(source)
        if refused:
            return refused
        trip = []
        if not _mem(ms.find_objects_by_graphics, ms.OBJ_EVENT_GFX_NURSE):
            trip, why = await _walk_to_pokemon_center(source)
            if why:
                walked = f" Walked {' → '.join(trip)} first." if trip else ""
                return _refusal(f"Did not heal: {why}.{walked}")

        before = _party_lines()
        spot, why = await _walk_up_to(ms.OBJ_EVENT_GFX_NURSE, "nurse", source)
        if why:
            return _refusal(f"Did not heal: {why}.")
        await _macro_press(["A"], source)
        asked, texts = await _talk_until(lambda: _mem(ms.task_running, ms.TASK_FIELD_YES_NO))
        if not asked:
            await _close_dialogue(source)
            said = _quote(texts)
            return _refusal(
                f"Talked to the nurse but no YES/NO came up ({said or 'nothing was said'}); nothing was healed.{_screen_note()}"
            )
        cursor = _mem(ms.read_menu_cursor)
        if cursor.max_cursor == 1 and cursor.cursor == 1:
            await _macro_press(["UP"], source)
            cursor = _mem(ms.read_menu_cursor)
        if cursor.max_cursor != 1 or cursor.cursor != 0:
            return _refusal(f"The nurse's YES/NO cursor is not on YES (cursor {cursor.cursor}); answer it yourself.")
        question = _quote(texts[-1:])
        await _macro_press(["A"], source)  # YES
        after_yes = await _read_dialogue()
        closed = await _close_dialogue(source)
        after = _party_lines()
        healed = all(m["egg"] or (m["hp"] == m["max_hp"] and m["status"] == 0) for m in after)
        said = _quote(after_yes.get("texts", []), 3)
        logger.info(f"🏥 heal_at_pokecenter: healed={healed} {said}")
        summary = f"Answered YES to the nurse ({question}). HP before: {_hp_text(before)}; after: {_hp_text(after)}."
        if healed:
            summary = "Healed: every party Pokémon has full HP and no status problem. " + summary
        else:
            summary = "NOT healed (game memory shows HP or status not restored). " + summary
        if trip:
            summary = f"Walked {' → '.join(trip)}. " + summary
        if closed:
            summary += f" You can walk again at {_position_text(_field_snapshot())}."
        else:
            summary += f"{_screen_note()}"
        result = {"success": healed, "status": "HEALED" if healed else "NOT_HEALED", "summary": summary, "before": before, "after": after}
        if not healed:
            result["error"] = summary
        return result
    except Exception as e:
        logger.error(f"Error in heal_at_pokecenter: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _shop_stage() -> str:
    from server import menu_state as ms

    if _mem(ms.callback2_is, ms.CB2_BUY_MENU):
        steps = (
            (ms.TASK_BUY_HOW_MANY, "quantity"),
            (ms.TASK_YES_NO_CALLBACK, "yes/no"),
            (ms.TASK_AFTER_PURCHASE, "thanks"),
            (ms.TASK_BUY_MENU, "list"),
        )
        return "buy screen: " + next((name for task, name in steps if _mem(ms.task_running, task)), "busy")
    if _mem(ms.task_running, ms.TASK_SHOP_MENU):
        return "shop menu"
    return _field_status()


async def _leave_shop(source: str) -> bool:
    """Back out of the shop and close the goodbye text; True once the player can walk.

    On the buy screen B backs out of every step (quantity, YES/NO = NO, the list) and
    never buys; on BUY/SELL/SEE YA! B means SEE YA!.
    """
    from server.field_control import DIALOGUE_BUSY, DIALOGUE_NONE, DIALOGUE_PLAIN

    for _ in range(8):
        stage = _shop_stage()
        if stage == DIALOGUE_NONE:
            return True
        if stage in (DIALOGUE_PLAIN, DIALOGUE_BUSY):
            await _read_dialogue()
        else:
            await _macro_press(["B"], source)
        await _await_game(lambda: _shop_stage() != stage, MACRO_TEXT_FRAMES)
    return _shop_stage() == DIALOGUE_NONE


# teach_move: presses allowed through the texts between two screens.
TEACH_TEXT_PRESSES = 14


def _hm_moves() -> set:
    """Move ids the HMs teach (HM01-HM08): the game refuses to forget them in the field."""
    from server import menu_state as ms

    with memory_lock:
        return {ms.tmhm_move(env.memory_reader, item) for item in range(ms.ITEM_TM01 + 50, ms.ITEM_TM01 + 58)}


@app.post("/mcp/teach_move")
async def mcp_teach_move(request: dict = None):
    """MCP Tool: teach a TM or HM from the bag to a party Pokémon, forgetting a move if needed.

    Seen live: told to "Teach Rock Smash to a Pokémon", the model pressed B over and over in
    the HM man's house. By hand it is START → BAG → TMs & HMs → the HM → USE → "Teach ROCK
    SMASH to a POKéMON?" YES → the party (its cursor is not always on slot 1) → "Should a
    move be deleted…?" YES → the forget screen → the move → A. Each step here waits for its
    screen; the cursors are read from memory (sStartMenuCursorPos, the bag's, gPartyMenu
    slotId, the summary screen's firstMoveIndex). Seen while building it: DOWN presses sent
    before the forget screen had opened left its cursor on move 1, and MACH PUNCH was lost,
    so the forget cursor is checked before A. The result is checked in the party data.
    """
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    from pokemon_env.enums import Move
    from server import menu_state as ms

    source = "teach_move"
    request = request or {}
    wanted_item = str(request.get("item") or "").strip()
    wanted_mon = request.get("pokemon")
    forget = str(request.get("forget") or "").strip()
    if not wanted_item or wanted_mon is None or str(wanted_mon).strip() == "":
        return _refusal('item and pokemon are required, e.g. teach_move("HM06", "BRELOOM").')
    refused = _field_refusal(source)
    if refused:
        return refused

    with memory_lock:
        reader = env.memory_reader
        tms = reader.read_full_inventory().get("tms_hms", [])
        party = reader.read_party_pokemon() or []
        taught = {t["item_id"]: ms.tmhm_move(reader, t["item_id"]) for t in tms}
    move_name = {item: Move(mid).name if mid in Move._value2member_map_ else f"MOVE {mid}" for item, mid in taught.items()}
    key = ms.name_key(wanted_item)
    item = next(
        (t for t in tms if ms.name_key(t["name"]) == key or ms.name_key(move_name[t["item_id"]]) == key), None
    )
    listing = ", ".join(f"{t['name']} ({move_name[t['item_id']].replace('_', ' ')})" for t in tms) or "none"
    if item is None:
        return _refusal(f"{wanted_item} is not in your TMs & HMs pocket. You have: {listing}.")
    new_move = taught[item["item_id"]]
    new_name = move_name[item["item_id"]].replace("_", " ")

    index = _train_member([(0, p) for p in party], wanted_mon)
    if index is None:
        names = ", ".join(f"{i + 1}. {p.species_name.upper()}" for i, p in enumerate(party))
        return _refusal(f"{wanted_mon} is not in your party ({names}).")
    mon = party[index]
    name = mon.species_name.upper()
    with memory_lock:
        able = ms.can_learn_tmhm(env.memory_reader, mon.species_id, item["item_id"])
        personalities = ms.party_personalities(env.memory_reader)
    if not able:
        return _refusal(f"{name} is NOT ABLE to learn {new_name} from {item['name']}.")
    known = [m for m in mon.moves if m and m != "NONE"]
    if move_name[item["item_id"]] in known:
        return _refusal(f"{name} already knows {new_name}.")
    forget_index = None
    if len(known) >= 4:
        hm_moves = {Move(m).name for m in _hm_moves() if m in Move._value2member_map_}
        if forget:
            fkey = ms.name_key(forget)
            forget_index = next((i for i, m in enumerate(known) if ms.name_key(m) == fkey), None)
            if forget_index is None:
                return _refusal(f"{name} doesn't know {forget}. Its moves: {', '.join(m.replace('_', ' ') for m in known)}.")
            if known[forget_index] in hm_moves:
                return _refusal(f"{known[forget_index].replace('_', ' ')} is an HM move: it can't be forgotten here.")
        else:
            # No move named: the weakest that isn't an HM move (the learn-move advice's measure).
            with memory_lock:
                own = env.memory_reader.party_mon_types(mon)
                worth = [(env.memory_reader.move_worth(m, own), i) for i, m in enumerate(known) if m not in hm_moves]
            if not worth:
                return _refusal(f"All of {name}'s moves are HM moves: none can be forgotten here.")
            forget_index = min(worth)[1]

    def screen(check):
        return _mem(lambda r: bool(check(r)))

    async def leave(message: str) -> Dict[str, Any]:
        for _ in range(6):  # B backs out of every screen here
            if _player_can_walk():
                break
            await _macro_press(["B"], source)
            await _await_game(_player_can_walk, MACRO_MENU_FRAMES)
        return _refusal(message + ("" if _player_can_walk() else " Could not close every menu: look at the screen."))

    with _turbo():
        await _macro_press(["START"], source)
        if not await _await_game(lambda: screen(ms.start_menu_open), MACRO_MENU_FRAMES):
            return await leave("The START menu did not open.")
        actions = _mem(ms.start_menu_actions)
        if ms.MENU_ACTION_BAG not in actions:
            return await leave("The START menu has no BAG.")
        await _macro_press(ms.list_presses(_mem(ms.start_menu_cursor), actions.index(ms.MENU_ACTION_BAG)), source)
        if _mem(ms.start_menu_cursor) != actions.index(ms.MENU_ACTION_BAG):
            return await leave("Could not put the START menu cursor on BAG.")
        await _macro_press(["A"], source)

        def bag_ready(r):
            return ms.callback2_is(r, ms.CB2_BAG_MENU_RUN) and ms.task_running(r, ms.TASK_BAG_INPUT) and not ms.fade_active(r)

        if not await _await_game(lambda: screen(bag_ready), MACRO_MENU_FRAMES * 2):
            return await leave("Chose BAG but the bag did not open.")
        for press in ms.pocket_presses(_mem(ms.bag_pocket), ms.TMHM_POCKET):
            await _macro_press([press], source)
            await _await_game(lambda: screen(bag_ready), MACRO_MENU_FRAMES)
        listed = _mem(lambda r: ms.bag_pocket_item_ids(r, ms.TMHM_POCKET))
        if _mem(ms.bag_pocket) != ms.TMHM_POCKET or item["item_id"] not in listed:
            return await leave(f"Could not find {item['name']} in the TMs & HMs pocket.")
        goal = listed.index(item["item_id"])
        await _macro_press(ms.list_presses(_mem(lambda r: ms.bag_cursor_index(r, ms.TMHM_POCKET)), goal), source)
        if _mem(lambda r: ms.bag_cursor_index(r, ms.TMHM_POCKET)) != goal:
            return await leave(f"Could not put the bag cursor on {item['name']}.")
        await _macro_press(["A"], source)
        if not await _await_game(lambda: screen(lambda r: ms.task_running(r, *ms.TASK_BAG_CONTEXT)), MACRO_MENU_FRAMES):
            return await leave(f"Pressed A on {item['name']} but its USE menu did not open.")
        if _mem(lambda r: r._read_u16(ms.G_SPECIAL_VAR_ITEM_ID)) != item["item_id"] or _mem(ms.read_menu_cursor).cursor != 0:
            return await leave(f"The bag's menu is not USE on {item['name']}.")
        await _macro_press(["A"], source)  # USE

        # "Booted up an HM." "It contained X. Teach X to a POKéMON?" YES (the cursor starts there).
        def party_ready(r):
            return ms.callback2_is(r, ms.CB2_UPDATE_PARTY_MENU) and ms.task_running(r, ms.TASK_PARTY_CHOOSE) and not ms.fade_active(r)

        for _ in range(TEACH_TEXT_PRESSES):
            if await _await_game(lambda: screen(party_ready), MACRO_MENU_FRAMES):
                break
            await _macro_press(["A"], source)
        if not screen(party_ready):
            return await leave(f"Used {item['name']} but the party screen never opened ({_mem(read_field_text)}).")
        # The menu lists gPlayerParty in order; the Pokémon is checked by personality too.
        slot = index
        if _mem(ms.party_personalities)[slot] != personalities[index]:
            return await leave(f"The party changed while teaching; {name} is not in slot {slot + 1}.")
        # Seen live: this screen's cursor started on slot 2, and A at once taught COMBUSKEN.
        await _macro_press(ms.list_presses(_mem(ms.party_menu_slot), slot), source)
        if _mem(ms.party_menu_slot) != slot:
            return await leave(f"Could not put the party cursor on {name} (it is on slot {_mem(ms.party_menu_slot) + 1}).")
        await _macro_press(["A"], source)

        forgotten = None
        if forget_index is not None:
            # "...already knows four moves. Should a move be deleted...?" YES → the forget screen.
            for _ in range(TEACH_TEXT_PRESSES):
                if await _await_game(lambda: _mem(ms.replace_move_cursor) is not None, MACRO_MENU_FRAMES):
                    break
                await _macro_press(["A"], source)
            cursor = _mem(ms.replace_move_cursor)
            if cursor is None or cursor[1] != new_move:
                return await leave(f"The forget-a-move screen did not open for {new_name}.")
            await _macro_press(ms.list_presses(cursor[0], forget_index), source)
            cursor = _mem(ms.replace_move_cursor)
            if cursor is None or cursor[0] != forget_index:
                return await leave(f"Could not put the forget cursor on {known[forget_index].replace('_', ' ')}; nothing was forgotten.")
            await _macro_press(["A"], source)
            forgotten = known[forget_index].replace("_", " ")

        # "1, 2, and... Poof! ... learned X!" back to the bag, then out of the menus.
        for _ in range(TEACH_TEXT_PRESSES):
            if await _await_game(lambda: screen(bag_ready), MACRO_MENU_FRAMES):
                break
            await _macro_press(["A"], source)
        for _ in range(4):
            if _player_can_walk():
                break
            await _macro_press(["B"], source)
            await _await_game(_player_can_walk, MACRO_MENU_FRAMES)

    with memory_lock:
        after = (env.memory_reader.read_party_pokemon() or [None] * (index + 1))[index]
    moves_after = [m.replace("_", " ") for m in (after.moves if after else []) if m and m != "NONE"]
    if new_name.replace(" ", "_") not in (after.moves if after else []):
        return _refusal(f"{name} did not learn {new_name}. Its moves: {', '.join(moves_after)}.{_screen_note()}")
    lost = f" (forgot {forgotten})" if forgotten else ""
    walk = "" if _player_can_walk() else " A menu is still open: look at the screen."
    return {
        "success": True,
        "status": "LEARNED",
        "summary": f"{name} learned {new_name}{lost}. Its moves: {', '.join(moves_after)}.{walk}",
    }


@app.post("/mcp/buy_item")
async def mcp_buy_item(request: dict = None):
    """MCP Tool: in a Poké Mart, buy `count` of an item from the clerk; confirmed from
    money and bag in memory."""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}
    if game_type != "emerald":
        return {"success": False, "error": "buy_item is only implemented for Emerald"}
    try:
        from pokemon_env.emerald_utils import _item_id_to_name
        from server import menu_state as ms

        source = "buy_item"
        request = request or {}
        wanted = str(request.get("item") or "").strip()
        if not wanted:
            return _refusal("item is required, e.g. buy_item('POTION', 3) or buy_item('POKE BALL', 5).")
        try:
            count = int(request.get("count", 1))
        except (TypeError, ValueError):
            return _refusal(f"count must be a number from 1 to 99, not {request.get('count')!r}.")
        if not 1 <= count <= 99:
            return _refusal(f"count must be from 1 to 99, not {count}.")
        refused = _field_refusal(source)
        if refused:
            return refused
        location = _location_name()
        if not _mem(ms.find_objects_by_graphics, ms.OBJ_EVENT_GFX_MART_EMPLOYEE):
            hint = _exit_hint("MART")
            where = f" This map has one: {hint}, then buy_item." if hint else " Go to a town with a POKEMON MART first."
            return _refusal(f"There is no shop clerk here ({location}).{where}")

        def bag_counts() -> Dict[int, int]:
            with memory_lock:
                inventory = env.memory_reader.read_full_inventory()
            counts: Dict[int, int] = {}
            for pocket in inventory.values():
                for entry in pocket:
                    counts[entry["item_id"]] = counts.get(entry["item_id"], 0) + entry["quantity"]
            return counts

        money_before = _mem(lambda r: r.read_money())
        spot, why = await _walk_up_to(ms.OBJ_EVENT_GFX_MART_EMPLOYEE, "clerk", source)
        if why:
            return _refusal(f"Bought nothing: {why}.")
        await _macro_press(["A"], source)
        opened, texts = await _talk_until(lambda: _mem(ms.task_running, ms.TASK_SHOP_MENU))
        if not opened:
            said = _quote(texts) or "nothing was said"
            left = await _leave_shop(source)
            return _refusal(f"Talked to the clerk but the BUY/SELL menu did not open ({said}).{'' if left else _screen_note()}")

        stock = _mem(ms.mart_stock)
        names = {item_id: _item_id_to_name(item_id) for item_id in stock}
        listing = ", ".join(f"{names[i]} ${_mem(ms.item_price, i)}" for i in stock)
        key = ms.name_key(wanted)
        item_id = next((i for i in stock if ms.name_key(names[i]) == key), None)
        if item_id is None:
            await _leave_shop(source)
            return _refusal(f"This shop doesn't sell {wanted}. It sells: {listing}.", stock=[names[i] for i in stock])
        name, price = names[item_id], _mem(ms.item_price, item_id)
        if price * count > money_before:
            await _leave_shop(source)
            return _refusal(
                f"{count} {name} cost ${price * count} (${price} each) but you have ${money_before}: "
                f"you can afford {money_before // price}. Nothing was bought."
            )
        bag_before = bag_counts()
        owned_before = bag_before.get(item_id, 0)

        async def give_up(message: str) -> Dict[str, Any]:
            left = await _leave_shop(source)
            money = _mem(lambda r: r.read_money())
            tail = "" if left else " Could not close the shop: look at the screen."
            return _refusal(f"{message} Money ${money_before} -> ${money}.{tail}")

        cursor = _mem(ms.read_menu_cursor)
        await _macro_press(ms.list_presses(cursor.cursor, 0), source)
        if _mem(ms.read_menu_cursor).cursor != 0:
            return await give_up("Could not put the shop menu cursor on BUY.")
        await _macro_press(["A"], source)  # BUY
        if not await _await_game(
            lambda: _mem(ms.callback2_is, ms.CB2_BUY_MENU) and _mem(ms.task_running, ms.TASK_BUY_MENU) and not _mem(ms.fade_active),
            MACRO_MENU_FRAMES,
        ):
            return await give_up("Chose BUY but the item list did not open.")
        goal = stock.index(item_id)
        await _macro_press(ms.list_presses(_mem(ms.shop_list_index), goal), source)
        if _mem(ms.shop_list_index) != goal:
            return await give_up(f"Could not put the list cursor on {name}.")
        await _macro_press(["A"], source)
        if not await _await_game(lambda: _mem(ms.find_task, ms.TASK_BUY_HOW_MANY) is not None, MACRO_TEXT_FRAMES):
            return await give_up(f"Chose {name} but the quantity box did not open.")
        task = _mem(ms.find_task, ms.TASK_BUY_HOW_MANY)
        if _mem(ms.task_data, task, ms.BUY_TASK_ITEM) != item_id:
            return await give_up(f"The quantity box is not for {name}.")
        most = _mem(ms.shop_max_quantity) or 0
        if count > most:
            return await give_up(f"You can buy at most {most} {name} (money or bag space); asked for {count}.")
        await _macro_press(ms.quantity_presses(_mem(ms.task_data, task, ms.BUY_TASK_COUNT), count), source)
        shown = _mem(ms.task_data, task, ms.BUY_TASK_COUNT)
        if shown != count:
            return await give_up(f"Could not set the quantity to {count} (it shows {shown}).")
        await _macro_press(["A"], source)
        if not await _await_game(lambda: _mem(ms.task_running, ms.TASK_YES_NO_CALLBACK), MACRO_TEXT_FRAMES):
            return await give_up("The purchase YES/NO did not come up.")
        cursor = _mem(ms.read_menu_cursor)
        if cursor.max_cursor != 1 or cursor.cursor != 0:
            return await give_up(f"The purchase YES/NO cursor is not on YES (cursor {cursor.cursor}).")
        await _macro_press(["A"], source)  # YES: the money is taken now

        # "Here you go!" (and a Premier Ball bonus message) wait for A, then the list returns.
        def back_at_list():
            return _mem(ms.task_running, ms.TASK_BUY_MENU)

        for _ in range(4):
            if await _await_game(
                lambda: back_at_list() or _mem(ms.task_running, ms.TASK_AFTER_PURCHASE) or _mem(ms.text_waiting_for_button),
                MACRO_TEXT_FRAMES,
            ) and not back_at_list():
                await _macro_press(["A"], source)
            if back_at_list():
                break
        money_after = _mem(lambda r: r.read_money())
        bag_after = bag_counts()
        owned_after = bag_after.get(item_id, 0)
        left = await _leave_shop(source)
        bought = owned_after - owned_before
        paid = money_before - money_after
        ok = bought == count and paid == price * count
        logger.info(f"🛒 buy_item: {name} x{count} paid={paid} bought={bought}")
        summary = (
            f"Bought {bought} {name} for ${paid}: money ${money_before} -> ${money_after}, "
            f"{name} in bag {owned_before} -> {owned_after}."
        )
        # Ten or more POKé BALLs come with a free PREMIER BALL.
        extras = [
            f"{_item_id_to_name(i)} x{n - bag_before.get(i, 0)}"
            for i, n in bag_after.items()
            if i != item_id and n > bag_before.get(i, 0)
        ]
        if extras:
            summary += f" Also received for free: {', '.join(extras)}."
        if not ok:
            summary = f"Purchase NOT confirmed by memory (wanted {count} for ${price * count}). " + summary
        summary += f" You can walk again at {_position_text(_field_snapshot())}." if left else _screen_note()
        result = {
            "success": ok,
            "status": "BOUGHT" if ok else "NOT_CONFIRMED",
            "summary": summary,
            "item": name,
            "money_before": money_before,
            "money_after": money_after,
            "bag_before": owned_before,
            "bag_after": owned_after,
        }
        if not ok:
            result["error"] = summary
        return result
    except Exception as e:
        logger.error(f"Error in buy_item: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.post("/mcp/complete_direct_objective")
async def mcp_complete_direct_objective(request: dict):
    """MCP Tool: Complete current direct objective"""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        from agents.objectives import DirectObjectiveManager

        # Get current game state to check objective completion
        from utils.state_formatter import format_state_for_llm
        from server import game_tools

        game_state_result = game_tools.get_game_state_direct(env, format_state_for_llm)
        if not game_state_result.get("success", False):
            return {"success": False, "error": "Failed to get game state"}

        game_state = game_state_result.get("raw_state", {})

        # Use global direct objective manager and current_run_dir
        global direct_objectives_manager, current_run_dir

        # Initialize direct objective manager if needed
        if direct_objectives_manager is None:
            direct_objectives_manager = DirectObjectiveManager()

        # Load direct objectives sequence if specified
        if direct_objectives_sequence:
            # Check if we need to load objectives
            needs_loading = not direct_objectives_manager.is_sequence_active()

            # Force reload if the requested sequence doesn't match the loaded sequence
            if direct_objectives_manager.is_sequence_active():
                if direct_objectives_manager.sequence_name != direct_objectives_sequence:
                    logger.warning(
                        f"⚠️ Sequence mismatch: loaded='{direct_objectives_manager.sequence_name}', requested='{direct_objectives_sequence}' - forcing reload"
                    )
                    needs_loading = True
                # Also force reload if requesting categorized but manager is in legacy mode
                elif (
                    direct_objectives_sequence == "categorized_full_game" and direct_objectives_manager.mode == "legacy"
                ):
                    logger.warning(f"⚠️ Requesting categorized mode but manager is in legacy mode - forcing reload")
                    needs_loading = True

            if needs_loading and os.environ.get("POKEAGENT_CLI_MODE") != "1":
                # CLI agents do not use objectives; skip when POKEAGENT_CLI_MODE
                _load_objectives_sequence()

        # Check if sequence is active
        if not direct_objectives_manager.is_sequence_active():
            return {"success": False, "error": "No direct objective sequence active"}

        # Extract category parameter if provided
        category = request.get("category")
        objective_id = str(request.get("objective_id") or "").strip()

        # Local addition: complete by objective ID. A small model reliably copies the ID
        # shown in OBJECTIVES but picks the wrong category (it healed at a Pokémon Center,
        # then completed three *battling* objectives instead of the story one, and kept
        # healing because the story objective never advanced). The ID names exactly one
        # current objective, so the category is derived from it, not trusted.
        if objective_id and direct_objectives_manager.mode == "categorized":
            current_ids = {}
            for cat in ("story", "battling", "dynamics"):
                obj = direct_objectives_manager._get_current_objective_for_category(cat)
                if obj is not None:
                    current_ids[cat] = obj.id
            matches = [cat for cat, oid in current_ids.items() if oid == objective_id]
            if not matches:
                listing = ", ".join(f"{cat}: {oid}" for cat, oid in current_ids.items()) or "none"
                return {
                    "success": False,
                    "error": f"'{objective_id}' is not a current objective. Current objectives are {listing}. "
                    "Only complete the objective whose goal you have actually achieved.",
                }
            if category and category != matches[0]:
                logger.warning(f"complete_direct_objective: '{objective_id}' is a {matches[0]} objective, not {category}")
            category = matches[0]

        # Check if we're in categorized mode
        if direct_objectives_manager.mode == "categorized":
            # Categorized mode - category parameter is required
            if not category:
                return {
                    "success": False,
                    "error": "Category parameter required in categorized mode (story, battling, or dynamics)",
                }

            if category not in ["story", "battling", "dynamics"]:
                return {
                    "success": False,
                    "error": f"Invalid category: {category}. Must be story, battling, or dynamics",
                }

            # Get current objective for the specified category
            current_obj = direct_objectives_manager._get_current_objective_for_category(category)
            if not current_obj:
                return {"success": False, "error": f"No current {category} objective to complete"}

            completed_objective_index = getattr(
                direct_objectives_manager, f"{category}_index"
            )

            # Mark objective as completed
            direct_objectives_manager._mark_objective_completed(current_obj)

            # Advance the appropriate index, and inject guidance when a category ends
            if category == "story":
                if direct_objectives_manager.story_index >= len(direct_objectives_manager.story_sequence) - 1:
                    next_obj = _build_story_planning_objective()
                    direct_objectives_manager.story_sequence.append(next_obj)
                    direct_objectives_manager.story_index = len(direct_objectives_manager.story_sequence) - 1
                    logger.info(
                        f"✅ Completed story objective: {current_obj.id} (appended create_new_objectives guidance at story index {direct_objectives_manager.story_index})"
                    )
                else:
                    direct_objectives_manager.story_index += 1
                    logger.info(
                        f"✅ Completed story objective: {current_obj.id} (advanced to story index {direct_objectives_manager.story_index})"
                    )
            elif category == "battling":
                direct_objectives_manager.battling_index += 1
                logger.info(
                    f"✅ Completed battling objective: {current_obj.id} (advanced to battling index {direct_objectives_manager.battling_index})"
                )
            elif category == "dynamics":
                direct_objectives_manager.dynamics_index += 1
                logger.info(
                    f"✅ Completed dynamics objective: {current_obj.id} (advanced to dynamics index {direct_objectives_manager.dynamics_index})"
                )
        else:
            # Legacy mode - ignore category parameter
            current_obj = direct_objectives_manager.get_current_objective()
            if not current_obj:
                return {"success": False, "error": "No current objective to complete"}

            completed_objective_index = direct_objectives_manager.current_index

            # Mark objective as completed
            direct_objectives_manager._mark_objective_completed(current_obj)

            hold_index = (
                current_obj.action_type == "create_new_objectives"
                and direct_objectives_manager.current_index >= len(direct_objectives_manager.current_sequence) - 1
            )
            if hold_index:
                logger.info(
                    f"✅ Completed objective: {current_obj.id} (holding index {direct_objectives_manager.current_index} for create_new_objectives guidance)"
                )
            else:
                direct_objectives_manager.current_index += 1
                logger.info(
                    f"✅ Completed objective: {current_obj.id} (advanced to index {direct_objectives_manager.current_index})"
                )

        # Persist full objectives state after mutation
        direct_objectives_manager.auto_save()

        # Update objectives cache for stream.html (fast file read)
        _update_objectives_cache()

        # Log objective completion to cumulative_metrics.json (objectives column)
        try:
            from utils.data_persistence.llm_logger import log_objective_completion

            log_objective_completion(
                objective_id=current_obj.id,
                category=category if direct_objectives_manager.mode == "categorized" else "legacy",
                objective_index=completed_objective_index,
                step_number=agent_step_count,
            )
        except Exception as e:
            logger.warning("Failed to log objective completion to metrics: %s", e)

        # Persist completed objectives after each completion (for real time execution... get_progress_summary() relies on this run_data)
        # CLI agents do not use objectives; skip when POKEAGENT_CLI_MODE
        if os.environ.get("POKEAGENT_CLI_MODE") != "1":
            try:
                from utils.data_persistence.run_data_manager import get_run_data_manager

                run_manager = get_run_data_manager()
                if not run_manager:
                    logger.warning("Cannot save completed_objectives: run_data_manager not initialized")
                else:
                    scratch_space_dir = str(run_manager.get_scratch_space_dir())
                    if direct_objectives_manager.mode == "categorized":
                        completed_path = os.path.join(scratch_space_dir, "completed_objectives.json")
                        if os.path.exists(completed_path):
                            with open(completed_path, "r") as f:
                                history = json.load(f)
                        else:
                            history = {
                                "mode": "categorized",
                                "sequence_name": direct_objectives_manager.sequence_name,
                                "categories": {"story": [], "battling": [], "dynamics": []},
                            }

                        category_key = category or "dynamics"
                        history.setdefault("categories", {})
                        history["categories"].setdefault("story", [])
                        history["categories"].setdefault("battling", [])
                        history["categories"].setdefault("dynamics", [])
                        history["categories"][category_key].append(
                            {
                                "id": current_obj.id,
                                "description": current_obj.description,
                                "target_location": current_obj.target_location,
                                "action_type": current_obj.action_type,
                                "completed_at": current_obj.completed_at.isoformat()
                                if hasattr(current_obj, "completed_at") and current_obj.completed_at
                                else None,
                                "category": category_key,
                            }
                        )
                        history["last_updated"] = datetime.datetime.now().isoformat()

                        with open(completed_path, "w") as f:
                            json.dump(history, f, indent=2)
                        logger.info(f"💾 Saved completed objectives to {completed_path}")
                    else:
                        saved_file = direct_objectives_manager.save_completed_objectives(run_dir=scratch_space_dir)
                        logger.info(f"💾 Saved completed objectives to {saved_file}")
            except Exception as e:
                logger.warning(f"Failed to save completed objectives: {e}")

        # Create backup of .pokeagent_cache after completing objective
        try:
            from utils.data_persistence.backup_manager import create_cache_backup

            backup_path = create_cache_backup(
                objective_id=current_obj.id, objective_description=current_obj.description
            )
            if backup_path:
                logger.info(f"📦 Created cache backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create cache backup: {e}")

        # Get next objective if available
        if direct_objectives_manager.mode == "categorized":
            next_guidance = direct_objectives_manager.get_categorized_objective_guidance(game_state)
            categorized_status = {
                "story": {
                    "current_index": direct_objectives_manager.story_index,
                    "total": len(direct_objectives_manager.story_sequence),
                    "completed": sum(1 for obj in direct_objectives_manager.story_sequence if obj.completed),
                },
                "battling": {
                    "current_index": direct_objectives_manager.battling_index,
                    "total": len(direct_objectives_manager.battling_sequence),
                    "completed": sum(1 for obj in direct_objectives_manager.battling_sequence if obj.completed),
                },
                "dynamics": {
                    "current_index": direct_objectives_manager.dynamics_index,
                    "total": len(direct_objectives_manager.dynamics_sequence),
                    "completed": sum(1 for obj in direct_objectives_manager.dynamics_sequence if obj.completed),
                },
            }
            from utils.json_utils import serialize_for_json
            return serialize_for_json({
                "success": True,
                "completed_objective": {"id": current_obj.id, "description": current_obj.description},
                "next_objective": next_guidance,
                "sequence_status": categorized_status,
            })
        else:
            next_obj = direct_objectives_manager.get_current_objective()
        if next_obj:
            next_guidance = direct_objectives_manager.get_current_objective_guidance(game_state)
            from utils.json_utils import serialize_for_json
            return serialize_for_json({
                "success": True,
                "completed_objective": {"id": current_obj.id, "description": current_obj.description},
                "next_objective": next_guidance,
                "sequence_status": direct_objectives_manager.get_sequence_status(),
            })
        else:
            # Sequence complete - save to history in timestamped run directory
            # CLI agents do not use objectives; skip when POKEAGENT_CLI_MODE
            if current_run_dir and os.environ.get("POKEAGENT_CLI_MODE") != "1":
                try:
                    # Save to agent_scratch_space in run_data
                    from utils.data_persistence.run_data_manager import get_run_data_manager

                    run_manager = get_run_data_manager()
                    if not run_manager:
                        logger.error("Cannot save completed_objectives: run_data_manager not initialized")
                        raise RuntimeError("run_data_manager must be initialized to save completed_objectives")
                    scratch_space_dir = str(run_manager.get_scratch_space_dir())
                    saved_file = direct_objectives_manager.save_completed_objectives(run_dir=scratch_space_dir)
                    logger.info(f"💾 Saved completed objectives to {saved_file}")
                except Exception as e:
                    logger.warning(f"Failed to save completed objectives: {e}")

            # Automatically create a new objective to guide the agent through next steps
            try:
                # Get current game state for context
                next_step_obj = _build_story_planning_objective()

                # Add the new objective to the sequence
                direct_objectives_manager.current_sequence.append(next_step_obj)

                # Get guidance for the new objective
                next_guidance = direct_objectives_manager.get_current_objective_guidance(game_state)

                from utils.json_utils import serialize_for_json
                return serialize_for_json({
                    "success": True,
                    "completed_objective": {"id": current_obj.id, "description": current_obj.description},
                    "next_objective": next_guidance,
                    "sequence_status": direct_objectives_manager.get_sequence_status(),
                    "message": "All objectives completed! A new objective has been automatically created to guide you through planning more objectives and continuing the run.",
                    "sequence_complete": False,  # Not complete anymore - we added a new objective
                    "auto_objective_created": True,
                })
            except Exception as e:
                logger.error(f"Failed to create auto objective: {e}")
                # Fallback to original behavior if auto-objective creation fails
                from utils.json_utils import serialize_for_json
                return serialize_for_json({
                    "success": True,
                    "completed_objective": {"id": current_obj.id, "description": current_obj.description},
                    "next_objective": None,
                    "sequence_status": direct_objectives_manager.get_sequence_status(),
                    "message": "All objectives completed! Review your progress and plan more objectives to continue the run.",
                    "sequence_complete": True,
                })

    except Exception as e:
        logger.error(f"Error completing direct objective: {e}")
        return {"success": False, "error": str(e)}


async def _navigate_beside_someone(request: dict, x: int, y: int) -> Optional[Dict[str, Any]]:
    """navigate_to whose target tile someone stands on: walk to the nearest free tile
    beside them instead, and say how to talk to them. None when the tile is free.

    Seen live on MT CHIMNEY (12,16), 15:14-15:16 UTC: the model called navigate_to(13,16)
    four times "to get closer" to TABITHA; (13,16) is a beaten grunt who still stands
    there, and each walk ended "BLOCKED … (13,16) is occupied by someone".
    """
    from server.field_control import DIRECTION_DELTAS, npc_at

    if request.get("_beside") or not _mem(npc_at, x, y):
        return None
    snap = _field_snapshot()
    if snap is None:
        return None
    talk = f"to talk to them, talk_to({x}, {y}) (it faces them and presses A)"
    if abs(snap.x - x) + abs(snap.y - y) == 1:
        summary = f"({x},{y}) is where someone stands, and you are already next to them at ({snap.x},{snap.y}): {talk}."
        return {"success": True, "status": "BESIDE", "target": f"({x}, {y})", "summary": summary}
    sides = sorted(
        ((x - dx, y - dy) for dx, dy in DIRECTION_DELTAS.values()),
        key=lambda t: abs(t[0] - snap.x) + abs(t[1] - snap.y),
    )
    walk: Dict[str, Any] = {}
    for sx, sy in sides:
        if _mem(npc_at, sx, sy):
            continue
        walk = await mcp_navigate_to({**request, "x": sx, "y": sy, "_beside": True})
        here = _field_snapshot()
        if here is not None and (here.x, here.y) == (sx, sy):
            walk["summary"] = (
                f"({x},{y}) is where someone stands, so the walk went next to them instead. "
                f"{walk.get('summary', '')} {talk[0].upper()}{talk[1:]}."
            )
            return walk
        if _mem(lambda r: r.is_in_battle()) or (here is not None and here.map_id != snap.map_id):
            break  # a trainer or a door ended the walk: report that walk as it is
    if walk:
        walk["summary"] = (
            f"({x},{y}) is where someone stands; no free tile next to them was reached. "
            f"{walk.get('summary') or walk.get('error', '')} {talk[0].upper()}{talk[1:]}."
        )
        return walk
    return {
        "success": False,
        "error": f"({x},{y}) is where someone stands and every tile next to them is taken; {talk}.",
    }


@app.post("/mcp/navigate_to")
async def mcp_navigate_to(request: dict):
    """MCP Tool: Navigate to coordinates using pathfinding"""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        from server import game_tools

        x = request.get("x")
        y = request.get("y")
        reason = request.get("reason", "")
        variance = request.get("variance")

        if x is None or y is None:
            return {"success": False, "error": "x and y coordinates required"}
        refused = _refuse_during_battle("navigate_to")
        if refused:
            return refused

        # Convert to int in case they come as floats from JSON
        try:
            x = int(x)
            y = int(y)
        except (ValueError, TypeError):
            return {"success": False, "error": f"Invalid coordinates: x={x}, y={y}"}

        beside = await _navigate_beside_someone(request, x, y)
        if beside is not None:
            return beside

        # Calculate path and get buttons
        # Operator use (not in the model's tool list): tiles to walk round, [[x, y], ...].
        # Asked for after the MAUVILLE CITY GYM switches had to be avoided by hand.
        try:
            avoid = [(int(p[0]), int(p[1])) for p in request.get("avoid") or []]
        except (TypeError, ValueError, IndexError):
            return {"success": False, "error": "avoid must be a list of [x, y] pairs."}
        result = game_tools.navigate_to_direct(
            env, x, y, reason=reason, variance=variance, blocked_coords=avoid or None
        )

        def exit_hint() -> str:
            try:
                snap = _field_snapshot()
                position = (snap.x, snap.y) if snap is not None else None
                return game_tools.exit_hint_for_text(_location_name(), reason, position)
            except Exception as e:  # a hint must never fail the walk
                logger.warning(f"navigate_to: exit hint failed: {e}")
                return ""

        if not result.get("success"):
            hint = exit_hint()
            if hint:
                result["error"] = f"{result.get('error', 'navigate_to failed')} {hint}"
            return result

        # Queue buttons via take_action to ensure metrics tracking
        buttons = result.get("buttons", [])
        report = None
        before = None
        if buttons:
            variance_value = result.get("variance")
            if variance_value is None:
                variance_value = variance or "none"
            action_request = ActionRequest(
                buttons=buttons,
                source="navigate_to",
                metadata={"variance": variance_value},
                check_movement=True,
            )
            before = _field_snapshot()
            queued = await take_action(action_request)
            report = await _await_sequence(queued.get("sequence_id"), _sequence_timeout(len(buttons)))
            await _settle_after_sequence(before, report)
            # A rock or tree the party can clear stopped the walk: clear it, walk on.
            tile = (report or {}).get("blocked_tile")
            cleared_before = int(request.get("_cleared", 0))
            used = await _clear_obstacle(tile, "navigate_to") if cleared_before < 3 else None
            if used:
                again = await mcp_navigate_to({**request, "_cleared": cleared_before + 1})
                again["summary"] = f"Used {used} at {tuple(tile)} on the way. {again.get('summary', '')}".strip()
                return again

        response = {
            "success": True,
            "target": result.get("target"),
            "path_length": result.get("path_length"),
            "buttons_queued": len(buttons),
            "variance": result.get("variance", variance or "none"),
        }
        if result.get("note"):
            response["note"] = result["note"]
        response.update(_sequence_summary(report))
        ran_clean = report is None or (
            report.get("done") and not report.get("blocked") and not report.get("interrupted")
        )
        # Judge arrival by where the player stands, not by the plan: variance paths
        # were seen ending a tile short of the target.
        snap = _field_snapshot()
        ox, oy = result.get("coord_offset") or (0, 0)
        at_target = snap is not None and (snap.x - ox, snap.y - oy) == (x, y)
        if before is not None and snap is not None and snap.map_id != before.map_id:
            # Target was a door (or the path crossed one): arriving on another map is
            # the outcome, not "did not reach" (the live model read that as failure and
            # walked back in and out of a shop).
            response["summary"] = f"Went through a door/warp: now at {_position_text(snap)}."
        else:
            if at_target:
                response["summary"] = f"Reached ({x},{y}). " + response["summary"]
            elif ran_clean:
                reason_text = f" {result['note']}." if result.get("note") else ""
                response["summary"] = f"Did not reach ({x},{y}).{reason_text} " + response["summary"]
            hint = exit_hint()
            if hint:
                response["summary"] += f" {hint}"
        return response
    except Exception as e:
        logger.error(f"Error in navigate_to: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/add_memory")
@app.post("/mcp/add_knowledge")  # backward-compat alias
async def mcp_add_memory(request: dict):
    """MCP Tool: Add entry to long-term memory (persistent across runs)"""
    try:
        from server import game_tools

        result = game_tools.add_memory_direct(
            category=request.get("category"),
            title=request.get("title"),
            content=request.get("content"),
            location=request.get("location"),
            coordinates=request.get("coordinates"),
            importance=request.get("importance", 3),
        )
        if os.environ.get("POKEAGENT_CLI_MODE") != "1":
            try:
                from utils.data_persistence.run_data_manager import get_run_data_manager

                run_manager = get_run_data_manager()
                if run_manager:
                    run_manager.copy_memory()
                else:
                    logger.warning("Cannot copy memory: run_data_manager not initialized")
            except Exception as e:
                logger.warning(f"Failed to copy memory: {e}")

        return result
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/search_memory")
@app.post("/mcp/search_knowledge")  # backward-compat alias
async def mcp_search_memory(request: dict):
    """MCP Tool: Search long-term memory (persistent across runs)"""
    try:
        from server import game_tools

        return game_tools.search_memory_direct(
            category=request.get("category"),
            query=request.get("query"),
            location=request.get("location"),
            min_importance=request.get("min_importance", 1),
        )
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_memory_summary")
@app.post("/mcp/get_knowledge_summary")  # backward-compat alias
async def mcp_get_memory_summary(request: dict):
    """MCP Tool: Get long-term memory summary (persistent across runs)"""
    try:
        from server import game_tools

        return game_tools.get_memory_summary_direct(min_importance=request.get("min_importance", 3))
    except Exception as e:
        logger.error(f"Error getting memory summary: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_memory_overview")
async def mcp_get_memory_overview(request: dict):
    """MCP Tool: Get long-term memory tree overview (compact [id] title tree)."""
    try:
        from server import game_tools

        return game_tools.get_memory_overview_direct()
    except Exception as e:
        logger.error(f"Error getting memory overview: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/process_memory")
async def mcp_process_memory(request: dict):
    """MCP Tool: Unified CRUD for long-term memory (read/add/update/delete)."""
    try:
        from server import game_tools

        action = request.get("action", "")
        entries = request.get("entries", [])
        reasoning = request.get("reasoning", "")
        result = game_tools.process_memory_direct(action, entries, reasoning)

        if action in ("add", "update", "delete") and result.get("success"):
            from utils.data_persistence.run_data_manager import get_run_data_manager
            run_manager = get_run_data_manager()
            if run_manager:
                try:
                    run_manager.copy_memory()
                except Exception as sync_err:
                    logger.warning(f"Failed to sync memory to run_data: {sync_err}")

        return result
    except Exception as e:
        logger.error(f"Error in process_memory: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_skill_overview")
async def mcp_get_skill_overview(request: dict):
    """MCP Tool: Get skill library tree overview (compact [id] name tree)."""
    try:
        from server import game_tools

        return game_tools.get_skill_overview_direct()
    except Exception as e:
        logger.error(f"Error getting skill overview: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/process_skill")
async def mcp_process_skill(request: dict):
    """MCP Tool: Unified CRUD for skill library (read/add/update/delete)."""
    try:
        from server import game_tools

        action = request.get("action", "")
        entries = request.get("entries", [])
        reasoning = request.get("reasoning", "")
        result = game_tools.process_skill_direct(action, entries, reasoning)

        if action in ("add", "update", "delete") and result.get("success"):
            from utils.data_persistence.run_data_manager import get_run_data_manager
            run_manager = get_run_data_manager()
            if run_manager:
                try:
                    run_manager.copy_skills()
                except Exception as sync_err:
                    logger.warning(f"Failed to sync skills to run_data: {sync_err}")

        return result
    except Exception as e:
        logger.error(f"Error in process_skill: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_subagent_overview")
async def mcp_get_subagent_overview(request: dict):
    """MCP Tool: Get subagent registry tree overview (compact [id] name tree)."""
    try:
        from server import game_tools

        return game_tools.get_subagent_overview_direct()
    except Exception as e:
        logger.error(f"Error getting subagent overview: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/process_subagent")
async def mcp_process_subagent(request: dict):
    """MCP Tool: Unified CRUD for subagent registry (read/add/update/delete)."""
    try:
        from server import game_tools

        action = request.get("action", "")
        entries = request.get("entries", [])
        reasoning = request.get("reasoning", "")
        result = game_tools.process_subagent_direct(action, entries, reasoning)

        if action in ("add", "update", "delete") and result.get("success"):
            from utils.data_persistence.run_data_manager import get_run_data_manager
            run_manager = get_run_data_manager()
            if run_manager:
                try:
                    run_manager.copy_subagents()
                except Exception as sync_err:
                    logger.warning(f"Failed to sync subagents to run_data: {sync_err}")

        return result
    except Exception as e:
        logger.error(f"Error in process_subagent: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/lookup_pokemon_info")
async def mcp_lookup_pokemon_info(request: dict):
    """MCP Tool: Lookup Pokemon info from wikis"""
    try:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import quote_plus

        topic = request.get("topic")
        source = request.get("source", "bulbapedia")

        if not topic:
            return {"success": False, "error": "topic is required"}

        # Wiki sources configuration
        POKEMON_WIKI_SOURCES = {
            "bulbapedia": {
                "base_url": "https://bulbapedia.bulbagarden.net/wiki/",
                "search_url": "https://bulbapedia.bulbagarden.net/w/index.php?search=",
                "description": "Comprehensive Pokemon encyclopedia",
            },
            "serebii": {
                "base_url": "https://www.serebii.net/",
                "emerald_url": "https://www.serebii.net/emerald/",
                "description": "Detailed Pokemon Emerald guides and data",
            },
            "pokemondb": {"base_url": "https://pokemondb.net/", "description": "Pokemon Database"},
            "marriland": {
                "base_url": "https://marriland.com/",
                "emerald_url": "https://marriland.com/pokemon-emerald/",
                "description": "Marriland guides",
            },
        }

        if source not in POKEMON_WIKI_SOURCES:
            return {
                "success": False,
                "error": f"Unknown source '{source}'. Available: {', '.join(POKEMON_WIKI_SOURCES.keys())}",
            }

        source_info = POKEMON_WIKI_SOURCES[source]

        # Build URL based on source
        if source == "bulbapedia":
            formatted_topic = topic.replace(" ", "_")
            url = f"{source_info['base_url']}{formatted_topic}"
        elif source == "serebii":
            formatted_topic = topic.lower().replace(" ", "")
            url = f"{source_info['emerald_url']}{formatted_topic}.shtml"
        elif source == "pokemondb":
            formatted_topic = topic.lower().replace(" ", "-")
            url = f"{source_info['base_url']}pokedex/{formatted_topic}"
        elif source == "marriland":
            formatted_topic = topic.lower().replace(" ", "-")
            url = f"{source_info['emerald_url']}{formatted_topic}/"
        else:
            url = f"{source_info['base_url']}{topic}"

        logger.info(f"📚 Fetching Pokemon info: {topic} from {source}")

        # Fetch the page
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; PokeAgent/1.0)"})

        # If 404, try search instead
        if response.status_code == 404 and source == "bulbapedia":
            search_url = f"{source_info['search_url']}{quote_plus(topic)}"
            logger.info(f"Page not found, trying search: {search_url}")
            response = requests.get(
                search_url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; PokeAgent/1.0)"}
            )

        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Extract main content based on source
        content = ""
        if source == "bulbapedia":
            main_content = soup.find("div", id="mw-content-text")
            if main_content:
                paragraphs = main_content.find_all("p", limit=5)
                content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        else:
            content = soup.get_text()
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = "\n".join(chunk for chunk in chunks if chunk)

        # Limit content length
        max_chars = 5000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[Content truncated - {len(content)} total characters]"

        if not content or len(content) < 50:
            return {"success": False, "error": f"Could not extract meaningful content from {url}"}

        return {
            "success": True,
            "topic": topic,
            "source": source,
            "url": url,
            "content": content,
            "description": source_info["description"],
        }

    except Exception as e:
        logger.error(f"Error looking up Pokemon info: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/list_wiki_sources")
async def mcp_list_wiki_sources():
    """MCP Tool: List available wiki sources"""
    POKEMON_WIKI_SOURCES = {
        "bulbapedia": {
            "base_url": "https://bulbapedia.bulbagarden.net/wiki/",
            "description": "Comprehensive Pokemon encyclopedia",
        },
        "serebii": {
            "base_url": "https://www.serebii.net/",
            "emerald_url": "https://www.serebii.net/emerald/",
            "description": "Detailed Pokemon Emerald guides and data",
        },
        "pokemondb": {"base_url": "https://pokemondb.net/", "description": "Pokemon Database"},
        "marriland": {
            "base_url": "https://marriland.com/",
            "emerald_url": "https://marriland.com/pokemon-emerald/",
            "description": "Marriland guides",
        },
    }

    sources = []
    for name, info in POKEMON_WIKI_SOURCES.items():
        sources.append(
            {
                "name": name,
                "description": info["description"],
                "base_url": info.get("base_url", ""),
                "emerald_url": info.get("emerald_url", ""),
            }
        )

    return {
        "success": True,
        "sources": sources,
        "count": len(sources),
        "usage": "Use lookup_pokemon_info(topic, source) to fetch information",
    }


@app.post("/mcp/get_walkthrough")
async def mcp_get_walkthrough(request: dict):
    """MCP Tool: Get Pokemon Emerald walkthrough part"""
    try:
        import requests
        from bs4 import BeautifulSoup

        part = request.get("part")

        if part is None:
            return {"success": False, "error": "part is required"}

        # Convert to int in case it comes as float from JSON
        try:
            part = int(part)
        except (ValueError, TypeError):
            return {"success": False, "error": f"Invalid part number: {part}"}

        # Build Bulbapedia walkthrough URL (game-specific)
        if game_type == "red":
            if not 1 <= part <= 17:
                return {"success": False, "error": f"Red walkthrough only has parts 1-17 (got {part})"}
            url = f"https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon_Red_and_Blue/Part_{part}"
            logger.info(f"📖 Fetching Red walkthrough part {part}")
        else:
            if not 1 <= part <= 21:
                return {"success": False, "error": f"Part must be between 1 and 21 (got {part})"}
            url = f"https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon_Emerald/Part_{part}"
            logger.info(f"📖 Fetching Emerald walkthrough part {part}")

        # Fetch the page
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; PokeAgent/1.0)"})
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "table"]):
            element.decompose()

        # Extract main content
        main_content = soup.find("div", id="mw-content-text")
        if not main_content:
            return {"success": False, "error": f"Could not find main content for Part {part}"}

        # Get all paragraphs and headings for structured walkthrough
        content_parts = []
        for element in main_content.find_all(["h2", "h3", "h4", "p", "ul"], limit=50):
            if element.name in ["h2", "h3", "h4"]:
                heading_text = element.get_text(strip=True)
                if heading_text and not heading_text.startswith("[edit]"):
                    level = element.name
                    if level == "h2":
                        content_parts.append(f"\n## {heading_text}")
                    elif level == "h3":
                        content_parts.append(f"\n### {heading_text}")
                    else:
                        content_parts.append(f"\n#### {heading_text}")
            elif element.name == "p":
                para_text = element.get_text(strip=True)
                if para_text and len(para_text) > 20:
                    content_parts.append(para_text)
            elif element.name == "ul":
                for li in element.find_all("li"):
                    li_text = li.get_text(strip=True)
                    if li_text:
                        content_parts.append(f"  - {li_text}")

        content = "\n\n".join(content_parts)

        # Limit content length
        max_chars = 8000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[Content truncated - {len(content)} total characters]"

        if not content or len(content) < 100:
            return {"success": False, "error": f"Could not extract meaningful content from Part {part}"}

        return {
            "success": True,
            "part": part,
            "url": url,
            "content": content,
            "description": f"Pokemon Emerald Walkthrough - Part {part}",
        }

    except Exception as e:
        logger.error(f"Error getting walkthrough: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/save_memory")
async def mcp_save_memory(request: dict):
    """MCP Tool: Save facts to persistent memory (saved to run directory)"""
    global current_run_dir

    fact = request.get("fact")
    if not fact:
        return {"success": False, "error": "fact is required"}

    if not current_run_dir:
        return {"success": False, "error": "No run directory available"}

    try:
        memory_file = os.path.join(current_run_dir, "AGENT.md")

        if os.path.exists(memory_file):
            with open(memory_file, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = "# Agent Memory\n\nThis file stores facts and observations from the AI agent.\n"

        if "## Agent Memories" not in content:
            if content and not content.endswith("\n"):
                content += "\n"
            content += "\n## Agent Memories\n"

        content += f"- {fact}\n"

        with open(memory_file, "w", encoding="utf-8") as f:
            f.write(content)

        return {"success": True, "message": f"Memory saved to {memory_file}", "path": memory_file}
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        return {"success": False, "error": str(e)}


def _coerce_replan_edits_to_list(edits: Any) -> List[Dict[str, Any]]:
    """Normalize ``edits`` from JSON / odd client encodings into ``list[dict]``."""
    return normalize_replan_edits(edits)


@app.post("/insert_objective")
async def insert_objective(request: dict):
    """Operator endpoint (not an agent tool): insert one objective at an index, e.g. a
    training step before the current story objective. Body: {category, index, objective}.
    index defaults to the category's current index."""
    global direct_objectives_manager
    if direct_objectives_manager is None or direct_objectives_manager.mode != "categorized":
        return {"success": False, "error": "needs the categorized objective manager"}
    category = request.get("category", "story")
    index = request.get("index")
    if index is None:
        index = direct_objectives_manager._get_index_for_category(category)
    result = direct_objectives_manager.insert_objective(category, int(index), request.get("objective") or {})
    if result.get("success"):
        _update_objectives_cache()
        logger.info(f"➕ insert_objective({category}@{index}): {result.get('inserted')}")
    return result


@app.post("/mcp/replan_objectives")
async def mcp_replan_objectives(request: dict):
    """MCP Tool: Apply index-based edits to a single objective category (used by subagent_plan_objectives)."""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        from agents.objectives import DirectObjectiveManager

        global direct_objectives_manager
        if direct_objectives_manager is None:
            return {"success": False, "error": "Objective manager not initialized"}

        if direct_objectives_manager.mode != "categorized":
            return {"success": False, "error": "replan_objectives requires categorized mode"}

        category = request.get("category")
        edits = _coerce_replan_edits_to_list(request.get("edits"))
        return_to_orchestrator = request.get("return_to_orchestrator", False)
        rationale = request.get("rationale", "")

        if not category:
            return {"success": False, "error": "Missing required 'category' parameter"}

        result = direct_objectives_manager.replan_category(category, edits)

        if result.get("success"):
            _update_objectives_cache()
            logger.info(
                f"🔄 replan_objectives({category}): {len(result.get('edits_applied', []))} edits applied. "
                f"return_to_orchestrator={return_to_orchestrator}"
            )

        result["return_to_orchestrator"] = return_to_orchestrator
        result["rationale"] = rationale
        return result

    except Exception as e:
        logger.error(f"Error in replan_objectives: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_full_objective_sequence")
async def mcp_get_full_objective_sequence():
    """MCP Tool: Return the complete objective state across all categories (used by subagent_plan_objectives)."""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        global direct_objectives_manager
        if direct_objectives_manager is None:
            return {"success": False, "error": "Objective manager not initialized"}

        from utils.json_utils import serialize_for_json
        snapshot = direct_objectives_manager.get_full_sequence_snapshot()
        return serialize_for_json({"success": True, **snapshot})

    except Exception as e:
        logger.error(f"Error in get_full_objective_sequence: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_progress_summary")
async def mcp_get_progress_summary(request: dict):
    """MCP Tool: Get comprehensive progress summary.

    Request body may include ``compact: true`` to omit ``memory_overview`` and
    ``completed_sequences_history`` (for subagent prompts that already load memory
    via get_memory_overview). Default / orchestrator tool calls with ``{}`` return
    the full payload.
    """
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        req = request if isinstance(request, dict) else {}
        compact = bool(req.get("compact"))
        # Get milestones
        milestones = env.milestone_tracker.milestones if env.milestone_tracker else {}
        completed_milestones = [mid for mid, data in milestones.items() if data.get("completed", False)]

        # Get current game state
        from utils.state_formatter import format_state_for_llm
        from server import game_tools

        game_state_result = game_tools.get_game_state_direct(env, format_state_for_llm)
        game_state = game_state_result.get("raw_state", {})

        # Get direct objective status
        global direct_objectives_manager
        obj_status = {}
        completed_history = []
        if direct_objectives_manager:
            obj_status = direct_objectives_manager.get_sequence_status()

            # Load completed objectives history (skip when compact — subagents use separate memory overview)
            if not compact:
                global current_run_dir
                completed_obj_file = None
                if current_run_dir and os.environ.get("POKEAGENT_CLI_MODE") != "1":
                    from utils.data_persistence.run_data_manager import get_run_data_manager

                    run_manager = get_run_data_manager()
                    if not run_manager:
                        logger.warning("Cannot load completed_objectives: run_data_manager not initialized")
                    else:
                        completed_obj_file = str(run_manager.get_scratch_space_dir() / "completed_objectives.json")
                if completed_obj_file and os.path.exists(completed_obj_file):
                    import json

                    with open(completed_obj_file, "r") as f:
                        history_data = json.load(f)
                        if "categories" in history_data:
                            completed_history = history_data.get("categories", {})
                        else:
                            completed_history = history_data.get("sequences", [])

        memory_overview = ""
        if not compact:
            mem_result = game_tools.get_memory_overview_direct()
            memory_overview = mem_result.get("overview", "No memories yet") if isinstance(mem_result, dict) else "No memories yet"

        # Get location and coordinates
        location = game_state.get("player", {}).get("location", "Unknown")
        player_pos = game_state_result.get("player_position", {}) if game_state_result.get("success") else {}

        from utils.json_utils import serialize_for_json

        progress = {
            "milestones_completed": completed_milestones,
            "total_milestones_completed": len(completed_milestones),
            "current_location": location,
            "player_coordinates": {"x": player_pos.get("x"), "y": player_pos.get("y")} if player_pos else None,
            "direct_objectives": {
                "current_sequence": obj_status.get("sequence_name"),
                "objectives_completed_in_current_sequence": obj_status.get("completed_count", 0),
                "total_in_current_sequence": obj_status.get("total_objectives", 0),
                "is_sequence_complete": obj_status.get("is_complete", False),
                "current_objective": obj_status.get("current_objective"),
            },
            "run_directory": current_run_dir,
        }
        if not compact:
            progress["completed_sequences_history"] = completed_history
            progress["memory_overview"] = memory_overview

        return serialize_for_json({"success": True, "progress": progress})
    except Exception as e:
        logger.error(f"Error getting progress summary: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/mcp/save_map")
async def mcp_save_map(request: dict):
    """MCP Tool: Save agent's mental map of a location to file (SLAM)"""
    try:
        location_name = request.get("location_name", "")
        map_data = request.get("map_data", "")

        if not location_name:
            return {"success": False, "error": "location_name is required"}

        if not map_data:
            return {"success": False, "error": "map_data is required"}

        # Create maps directory (Path imported at top of file)
        from utils.data_persistence.run_data_manager import get_cache_path
        maps_dir = get_cache_path("maps")
        maps_dir.mkdir(parents=True, exist_ok=True)

        # Normalize case to title case for consistent filenames
        normalized_location = location_name.title()
        # Sanitize location name for filename
        safe_name = "".join(c for c in normalized_location if c.isalnum() or c in (" ", "_", "-")).strip()
        safe_name = safe_name.replace(" ", "_")

        map_file = maps_dir / f"{safe_name}.txt"
        map_file.write_text(map_data)

        logger.info(f"🗺️  Saved SLAM map for {location_name} to {map_file}")
        return {
            "success": True,
            "message": f"Map saved to {map_file}",
            "location": location_name,
            "file_path": str(map_file),
        }

    except Exception as e:
        logger.error(f"Error saving map: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/mcp/load_map")
async def mcp_load_map(request: dict):
    """MCP Tool: Load agent's previously saved mental map of a location (SLAM)"""
    try:
        location_name = request.get("location_name", "")

        if not location_name:
            return {"success": False, "error": "location_name is required"}

        # Create maps directory if it doesn't exist (Path imported at top of file)
        from utils.data_persistence.run_data_manager import get_cache_path
        maps_dir = get_cache_path("maps")
        maps_dir.mkdir(parents=True, exist_ok=True)

        # Normalize case to title case for consistent filenames
        normalized_location = location_name.title()
        # Sanitize location name for filename
        safe_name = "".join(c for c in normalized_location if c.isalnum() or c in (" ", "_", "-")).strip()
        safe_name = safe_name.replace(" ", "_")

        map_file = maps_dir / f"{safe_name}.txt"

        if not map_file.exists():
            logger.info(f"🗺️  No existing SLAM map for {location_name}")
            return {
                "success": True,
                "message": f"No existing map for {location_name}",
                "location": location_name,
                "map_data": None,
            }

        map_data = map_file.read_text()
        logger.info(f"🗺️  Loaded SLAM map for {location_name} from {map_file}")
        return {
            "success": True,
            "message": f"Map loaded from {map_file}",
            "location": location_name,
            "map_data": map_data,
            "file_path": str(map_file),
        }

    except Exception as e:
        logger.error(f"Error loading map: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ============================================================================
# SERVER CONTROL
# ============================================================================


@app.post("/stop")
async def stop_server():
    """Stop the server"""
    global running
    running = False
    return {"status": "stopping"}


def _require_state_api_key(request: Request) -> Optional[JSONResponse]:
    """
    When POKEMON_STATE_API_KEY is set (e.g. by run_cli), require X-Internal-API-Key header.
    Prevents the CLI agent from loading/saving state via Bash curl.
    """
    key = os.environ.get("POKEMON_STATE_API_KEY")
    if not key:
        return None
    if request.headers.get("X-Internal-API-Key") != key:
        logger.warning("Rejected state endpoint call: missing or invalid X-Internal-API-Key")
        return JSONResponse(status_code=403, content={"error": "Forbidden: state API protected"})
    return None


# Saving/loading a state waits for the emulator lock and does file I/O for tens to
# hundreds of ms. Run inline in an async endpoint, that would freeze the event loop
# (every other request, /health included), so the endpoints below hand these to
# run_in_threadpool, the same threadpool FastAPI uses for plain `def` endpoints.
def _locked_save_state(path: str):
    with memory_lock:
        env.save_state(path)


def _locked_load_state(path: str):
    with memory_lock:
        env.load_state(path)


@app.post("/save_state")
async def save_state_endpoint(request: Request):
    """Save the current emulator state to a file"""
    if err := _require_state_api_key(request):
        return err
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        from utils.data_persistence.run_data_manager import get_cache_path
        default_filepath = str(get_cache_path("manual_save.state"))
        filepath = body.get("filepath", default_filepath)
        if env:
            await run_in_threadpool(_locked_save_state, filepath)
            logger.info(f"💾 State saved to: {filepath}")
            return {"status": "success", "message": f"State saved to {filepath}"}
        else:
            return JSONResponse(status_code=500, content={"error": "Emulator not initialized"})
    except Exception as e:
        logger.error(f"Error saving state: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/load_state")
async def load_state_endpoint(request: Request):
    """Load an emulator state from a file"""
    if err := _require_state_api_key(request):
        return err
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        from utils.data_persistence.run_data_manager import get_cache_path
        default_filepath = str(get_cache_path("manual_save.state"))
        filepath = body.get("filepath", default_filepath)
        if env:
            if not os.path.exists(filepath):
                return JSONResponse(status_code=404, content={"error": f"State file not found: {filepath}"})
            await run_in_threadpool(_locked_load_state, filepath)
            logger.info(f"📂 State loaded from: {filepath}")
            return {"status": "success", "message": f"State loaded from {filepath}"}
        else:
            return JSONResponse(status_code=500, content={"error": "Emulator not initialized"})
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/checkpoint")
async def save_checkpoint(request_data: dict = None):
    """Save checkpoint - called by client when step count reaches checkpoint interval"""
    try:
        from utils.data_persistence.run_data_manager import get_cache_path
        step_count = request_data.get("step_count", 0) if request_data else 0

        # Save emulator state
        checkpoint_state = str(get_cache_path("checkpoint.state"))
        if env:
            # save_state also writes this state's milestones and map data
            await run_in_threadpool(_locked_save_state, checkpoint_state)
            logger.info(f"💾 Server: Saved checkpoint state at step {step_count}")

            return {
                "status": "checkpoint_saved",
                "step_count": step_count,
                "files": {
                    "state": checkpoint_state,
                    "milestones": str(get_cache_path("checkpoint_milestones.json")),
                    "map": str(get_cache_path("checkpoint_grids.json"))
                }
            }
        else:
            return JSONResponse(status_code=503, content={"status": "error", "message": "No emulator available"})

    except Exception as e:
        # A non-2xx status, so every caller (agent, /play page, play-local.sh stop) sees
        # the failure instead of reporting a checkpoint that was never written.
        logger.error(f"Failed to save checkpoint: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/sync_llm_metrics")
async def sync_llm_metrics(request: Request):
    """Sync LLM cumulative metrics from client to server"""
    try:
        request_data = await request.json()
        cumulative_metrics = request_data.get("cumulative_metrics", {})

        if not cumulative_metrics:
            return {"status": "error", "message": "No metrics provided"}, 400

        # Update server's LLM logger with client's cumulative metrics
        from utils.data_persistence.llm_logger import get_llm_logger

        llm_logger = get_llm_logger()
        if llm_logger is not None:
            # Update cumulative metrics (but preserve server-managed metrics like start_time and total_actions)
            server_start_time = llm_logger.cumulative_metrics.get("start_time")
            server_total_actions = llm_logger.cumulative_metrics.get("total_actions")
            server_total_tokens = llm_logger.cumulative_metrics.get("total_tokens", 0)
            server_prompt_tokens = llm_logger.cumulative_metrics.get("prompt_tokens", 0)
            server_completion_tokens = llm_logger.cumulative_metrics.get("completion_tokens", 0)
            server_cached_tokens = llm_logger.cumulative_metrics.get("cached_tokens", 0)
            server_cache_write_tokens = llm_logger.cumulative_metrics.get("cache_write_tokens", 0)
            server_total_cost = llm_logger.cumulative_metrics.get("total_cost", 0.0)
            server_total_llm_calls = llm_logger.cumulative_metrics.get("total_llm_calls", 0)
            server_milestones = llm_logger.cumulative_metrics.get("milestones")
            server_last_milestone_step = llm_logger.cumulative_metrics.get("_last_milestone_step")
            server_last_milestone_tokens = llm_logger.cumulative_metrics.get("_last_milestone_tokens")
            server_last_milestone_time = llm_logger.cumulative_metrics.get("_last_milestone_time")
            server_last_milestone_actions = llm_logger.cumulative_metrics.get("_last_milestone_actions")
            # Objectives are appended only on server (complete_direct_objective); client has no copy
            server_objectives = llm_logger.cumulative_metrics.get("objectives")
            server_last_objective_step = llm_logger.cumulative_metrics.get("_last_objective_step")
            server_last_objective_tokens = llm_logger.cumulative_metrics.get("_last_objective_tokens")
            server_last_objective_time = llm_logger.cumulative_metrics.get("_last_objective_time")
            server_last_objective_actions = llm_logger.cumulative_metrics.get("_last_objective_actions")
            # total_run_time and last_update_time are updated by server on take_action; run_cli never has them
            server_total_run_time = llm_logger.cumulative_metrics.get("total_run_time")
            server_last_update_time = llm_logger.cumulative_metrics.get("last_update_time")

            llm_logger.cumulative_metrics.update(cumulative_metrics)

            # Restore server-managed metrics
            if server_start_time:
                llm_logger.cumulative_metrics["start_time"] = server_start_time
            if server_total_actions is not None:
                llm_logger.cumulative_metrics["total_actions"] = server_total_actions
            # Preserve milestone tracking from server (authoritative source)
            if server_milestones is not None:
                llm_logger.cumulative_metrics["milestones"] = server_milestones
            if server_last_milestone_step is not None:
                llm_logger.cumulative_metrics["_last_milestone_step"] = server_last_milestone_step
            if server_last_milestone_tokens is not None:
                llm_logger.cumulative_metrics["_last_milestone_tokens"] = server_last_milestone_tokens
            if server_last_milestone_time is not None:
                llm_logger.cumulative_metrics["_last_milestone_time"] = server_last_milestone_time
            if server_last_milestone_actions is not None:
                llm_logger.cumulative_metrics["_last_milestone_actions"] = server_last_milestone_actions
            # Preserve objectives (only server appends on complete_direct_objective)
            if server_objectives is not None:
                llm_logger.cumulative_metrics["objectives"] = server_objectives
            if server_last_objective_step is not None:
                llm_logger.cumulative_metrics["_last_objective_step"] = server_last_objective_step
            if server_last_objective_tokens is not None:
                llm_logger.cumulative_metrics["_last_objective_tokens"] = server_last_objective_tokens
            if server_last_objective_time is not None:
                llm_logger.cumulative_metrics["_last_objective_time"] = server_last_objective_time
            if server_last_objective_actions is not None:
                llm_logger.cumulative_metrics["_last_objective_actions"] = server_last_objective_actions
            # Preserve gameplay time (server updates on take_action; run_cli sync would overwrite with 0)
            if server_total_run_time is not None:
                llm_logger.cumulative_metrics["total_run_time"] = server_total_run_time
            if server_last_update_time is not None:
                llm_logger.cumulative_metrics["last_update_time"] = server_last_update_time

            # Keep cumulative totals monotonic so restored totals are never downgraded by stale client payloads.
            llm_logger.cumulative_metrics["total_tokens"] = max(
                server_total_tokens, llm_logger.cumulative_metrics.get("total_tokens", 0)
            )
            llm_logger.cumulative_metrics["prompt_tokens"] = max(
                server_prompt_tokens, llm_logger.cumulative_metrics.get("prompt_tokens", 0)
            )
            llm_logger.cumulative_metrics["completion_tokens"] = max(
                server_completion_tokens, llm_logger.cumulative_metrics.get("completion_tokens", 0)
            )
            llm_logger.cumulative_metrics["cached_tokens"] = max(
                server_cached_tokens, llm_logger.cumulative_metrics.get("cached_tokens", 0)
            )
            llm_logger.cumulative_metrics["cache_write_tokens"] = max(
                server_cache_write_tokens, llm_logger.cumulative_metrics.get("cache_write_tokens", 0)
            )
            llm_logger.cumulative_metrics["total_cost"] = max(
                server_total_cost, llm_logger.cumulative_metrics.get("total_cost", 0.0)
            )
            llm_logger.cumulative_metrics["total_llm_calls"] = max(
                server_total_llm_calls, llm_logger.cumulative_metrics.get("total_llm_calls", 0)
            )

            # Also sync to latest_metrics for stream.html display (excluding server-managed metrics)
            global latest_metrics
            with step_lock:
                for key, value in cumulative_metrics.items():
                    if key in latest_metrics and key not in [
                        "total_actions",
                        "start_time",
                        "total_run_time",
                        "total_tokens",
                        "prompt_tokens",
                        "completion_tokens",
                        "cached_tokens",
                        "cache_write_tokens",
                        "total_cost",
                        "total_llm_calls",
                    ]:
                        latest_metrics[key] = value

            with step_lock:
                latest_metrics["total_tokens"] = llm_logger.cumulative_metrics.get("total_tokens", 0)
                latest_metrics["prompt_tokens"] = llm_logger.cumulative_metrics.get("prompt_tokens", 0)
                latest_metrics["completion_tokens"] = llm_logger.cumulative_metrics.get("completion_tokens", 0)
                latest_metrics["cached_tokens"] = llm_logger.cumulative_metrics.get("cached_tokens", 0)
                latest_metrics["cache_write_tokens"] = llm_logger.cumulative_metrics.get("cache_write_tokens", 0)
                latest_metrics["total_cost"] = llm_logger.cumulative_metrics.get("total_cost", 0.0)
                latest_metrics["total_llm_calls"] = llm_logger.cumulative_metrics.get("total_llm_calls", 0)

            # Persist to cache so steps/milestones are saved promptly
            llm_logger.save_cumulative_metrics()

            logger.info(
                f"🔄 Synced LLM metrics: {cumulative_metrics.get('total_llm_calls', 0)} calls, {cumulative_metrics.get('total_tokens', 0)} tokens, ${cumulative_metrics.get('total_cost', 0):.6f}"
            )
            return {"status": "metrics_synced"}
        else:
            logger.error("No LLM logger available for sync")
            return {"status": "error", "message": "No LLM logger available"}, 500
    except Exception as e:
        logger.error(f"Error syncing LLM metrics: {e}")
        return {"status": "error", "message": str(e)}, 500


@app.post("/save_agent_history")
async def save_agent_history():
    """Save agent history to checkpoint_llm.txt (called by client after each step)"""
    try:
        # Use server-side LLM logger to save checkpoint
        from utils.data_persistence.llm_logger import get_llm_logger

        llm_logger = get_llm_logger()
        if llm_logger is not None:
            # Save checkpoint using current agent step count
            global agent_step_count
            # Save to cache folder (llm_logger handles path internally now)
            llm_logger.save_checkpoint(agent_step_count=agent_step_count)
            logger.info(f"💾 Saved LLM checkpoint at step {agent_step_count}")
            return {"status": "agent_history_saved", "step_count": agent_step_count}
        else:
            return {"status": "no_logger", "message": "No LLM logger available"}

    except Exception as e:
        logger.error(f"Failed to save agent history: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/load_checkpoint")
async def load_checkpoint():
    """Load checkpoint state - called by client on startup if --load-checkpoint flag is used"""
    try:
        from utils.data_persistence.run_data_manager import get_cache_path
        checkpoint_state = str(get_cache_path("checkpoint.state"))
        
        if not os.path.exists(checkpoint_state):
            return {"status": "no_checkpoint", "message": f"No {checkpoint_state} file found"}
        
        if env:
            def load_state_and_milestones():
                with memory_lock:
                    env.load_state(checkpoint_state)
                    logger.info(f"📂 Server: Loaded checkpoint state")

                    # Load milestones if available
                    if env.milestone_tracker:
                        try:
                            env.milestone_tracker.load_milestones_for_state(checkpoint_state)
                            logger.info(f"📂 Server: Loaded checkpoint milestones")
                        except:
                            logger.warning(f"Could not load checkpoint milestones")

            await run_in_threadpool(load_state_and_milestones)

            return {
                "status": "checkpoint_loaded",
                "files": {
                    "state": checkpoint_state,
                    "milestones": str(get_cache_path("checkpoint_milestones.json")),
                    "map": str(get_cache_path("checkpoint_grids.json"))
                }
            }
        else:
            return {"status": "error", "message": "No emulator available"}

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return {"status": "error", "message": str(e)}


def main():
    """Main function"""
    import argparse

    global running, state_update_running, state_update_thread, latest_metrics, agent_step_count

    # A new server process means a new session: a pause left over from the last one
    # would keep the agent idle with no sign why, however the run was started.
    if os.path.exists(AGENT_PAUSE_FLAG):
        os.remove(AGENT_PAUSE_FLAG)
        print("▶️  Cleared a leftover agent pause from the previous session")

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Simple Pokemon Server")
    parser.add_argument("--game", type=str, default="emerald", choices=["red", "emerald"],
                       help="Which game to run: 'red' (Pokemon Red, Game Boy) or 'emerald' (Pokemon Emerald, GBA)")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
    parser.add_argument("--manual", action="store_true", help="Enable manual mode with keyboard input and overlay")
    parser.add_argument("--load-state", type=str, help="Load a saved state file on startup")
    parser.add_argument("--record", action="store_true", help="Record video of the gameplay")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR dialogue detection")
    parser.add_argument(
        "--direct-objectives",
        type=str,
        help="Load a specific direct objective sequence ('categorized_full_game' or 'autonomous_objective_creation')",
    )
    parser.add_argument(
        "--direct-objectives-start",
        type=int,
        default=0,
        help="Start index for story objectives in legacy mode, or story objectives in categorized mode (for resuming from checkpoints)",
    )
    parser.add_argument(
        "--direct-objectives-battling-start",
        type=int,
        default=0,
        help="Start index for battling objectives (only used in categorized mode)",
    )
    # Server always runs headless - display handled by client

    args = parser.parse_args()

    # Set game type from args (also sync env var for modules like state_formatter)
    global game_type
    game_type = args.game
    os.environ["GAME_TYPE"] = game_type
    print(f"Game type: {game_type}")
    
    # Set global direct objectives sequence
    global direct_objectives_sequence, direct_objectives_start_index, direct_objectives_battling_start_index
    if args.direct_objectives:
        direct_objectives_sequence = args.direct_objectives
        direct_objectives_start_index = args.direct_objectives_start
        direct_objectives_battling_start_index = args.direct_objectives_battling_start
        if direct_objectives_battling_start_index > 0:
            print(
                f"🎯 Direct objectives sequence: {direct_objectives_sequence} (story index: {direct_objectives_start_index}, battling index: {direct_objectives_battling_start_index})"
            )
        else:
            print(
                f"🎯 Direct objectives sequence: {direct_objectives_sequence} (starting at index {direct_objectives_start_index})"
            )

    # Check for environment variables from multiprocess mode
    env_load_state = os.environ.get("LOAD_STATE")
    if env_load_state and not args.load_state:
        args.load_state = env_load_state
        print(f"📂 Using load state from environment: {env_load_state}")
        if env_load_state == ".pokeagent_cache/checkpoint.state":
            from utils.data_persistence.run_data_manager import get_cache_path
            checkpoint_state = get_cache_path("checkpoint.state")
            if checkpoint_state.exists():
                print(f"✅ Server startup: {checkpoint_state} file exists")
            else:
                print(f"❌ Server startup: {checkpoint_state} file MISSING!")
    
    # Set checkpoint loading flag based on whether this is a true checkpoint load
    global checkpoint_loading_enabled
    env_load_checkpoint_mode = os.environ.get("LOAD_CHECKPOINT_MODE")

    if env_load_checkpoint_mode == "true":
        checkpoint_loading_enabled = True
        print("🔄 Checkpoint loading enabled - will restore LLM metrics from cumulative_metrics.json")

        # Initialize LLM logger and load checkpoint immediately during server startup
        from utils.data_persistence.llm_logger import get_llm_logger
        from utils.data_persistence.run_data_manager import get_checkpoint_llm_path
        llm_logger = get_llm_logger()
        checkpoint_file = get_checkpoint_llm_path()
        
        # First try to load lightweight cumulative metrics
        metrics_loaded = llm_logger.load_cumulative_metrics() if llm_logger else False

        # Then load full checkpoint for LLM history
        if llm_logger and checkpoint_file.exists():
            restored_step_count = llm_logger.load_checkpoint(str(checkpoint_file))
            if restored_step_count is not None:
                agent_step_count = restored_step_count
                print(f"✅ Server startup: restored LLM checkpoint with step count {restored_step_count}")

                # Sync latest_metrics with loaded cumulative metrics
                latest_metrics.update(llm_logger.cumulative_metrics)
                print(
                    f"✅ Server startup: synced metrics - actions: {latest_metrics.get('total_actions', 0)}, cost: ${latest_metrics.get('total_cost', 0):.4f}, time: {latest_metrics.get('total_run_time', 0):.0f}s"
                )
            else:
                print("❌ Server startup: failed to load LLM checkpoint")
        elif metrics_loaded:
            # Only metrics file loaded (no checkpoint)
            latest_metrics.update(llm_logger.cumulative_metrics)
            print(
                f"✅ Server startup: loaded cumulative metrics - actions: {latest_metrics.get('total_actions', 0)}, cost: ${latest_metrics.get('total_cost', 0):.4f}, time: {latest_metrics.get('total_run_time', 0):.0f}s"
            )
        else:
            print("ℹ️ Server startup: no checkpoint_llm.txt file found")
    elif env_load_checkpoint_mode == "false":
        checkpoint_loading_enabled = False
        print("✨ Fresh start mode - will NOT load LLM metrics from cumulative_metrics.json")
    else:
        # Default behavior: allow checkpoint loading unless explicitly disabled
        checkpoint_loading_enabled = True
        print(
            "🔄 Checkpoint loading enabled by default - will restore LLM metrics from cumulative_metrics.json if available"
        )

    # ALWAYS try to load cumulative metrics, regardless of checkpoint mode
    # This ensures tokens/cost/actions are preserved even if checkpoint loading is off
    if env_load_checkpoint_mode != "true":
        from utils.data_persistence.llm_logger import get_llm_logger

        llm_logger = get_llm_logger()
        if llm_logger:
            # Only load if we didn't already load above
            metrics_loaded = llm_logger.load_cumulative_metrics()
            if metrics_loaded:
                # latest_metrics is already declared global above, so we can use it here
                latest_metrics.update(llm_logger.cumulative_metrics)
                print(
                    f"✅ Server startup: loaded cumulative metrics - tokens: {latest_metrics.get('total_tokens', 0):,}, cost: ${latest_metrics.get('total_cost', 0):.2f}, actions: {latest_metrics.get('total_actions', 0):,}"
                )

    print("Starting Fixed Simple Pokemon Emerald Server")
    # Initialize run data manager for structured data collection
    # Use run_id from environment if provided (set by client), otherwise create new one
    from utils.data_persistence.run_data_manager import initialize_run_data_manager

    run_id = os.environ.get("RUN_DATA_ID")
    run_name = os.environ.get("RUN_NAME")
    run_manager = initialize_run_data_manager(run_id=run_id, run_name=run_name)
    print(f"📁 Run data directory: {run_manager.get_run_directory()}")

    # Only save metadata if this is a new run (not set by client)
    # If run_id was provided, metadata was already saved by client
    if run_id is None:
        command_args = vars(args)
        run_manager.save_metadata(
            command_args=command_args,
            sys_argv=sys.argv,
            additional_info={"server_mode": True, "checkpoint_loading_enabled": checkpoint_loading_enabled},
        )

    # Initialize run directory for this execution (deprecated, will be moved)
    global current_run_dir
    if current_run_dir is None:
        from utils.data_persistence.run_data_manager import get_cache_directory
        # Use the cache directory directly instead of creating nested subdirectory
        current_run_dir = str(get_cache_directory())
        print(f"📁 Legacy run directory (deprecated): {current_run_dir}")

    # Initialize video recording if requested
    init_video_recording(args.record)
    print("Server mode - headless operation, display handled by client")
    if args.no_ocr:
        print("OCR dialogue detection disabled")
    print("Press Ctrl+C to stop")

    # Initialize emulator
    # Skip initial state reading if we're going to load a state
    if not setup_environment(skip_initial_state=(args.load_state is not None)):
        print("Failed to initialize emulator")
        return

    # Disable dialogue detection if --no-ocr flag is set
    if args.no_ocr:
        if env and env.memory_reader and hasattr(env.memory_reader, '_dialog_detection_enabled'):
            env.memory_reader._dialog_detection_enabled = False
            print("🚫 All dialogue detection disabled (--no-ocr flag)")

    # Load state if specified
    if args.load_state:
        try:
            env.load_state(args.load_state)
            print(f"Loaded state from: {args.load_state}")

            # Milestones and map data are automatically loaded by env.load_state()
            # Check what was loaded
            state_dir = os.path.dirname(args.load_state)
            base_name = os.path.splitext(os.path.basename(args.load_state))[0]

            milestone_file = os.path.join(state_dir, f"{base_name}_milestones.json")
            if os.path.exists(milestone_file):
                print(f"📂 Loaded milestones from: {milestone_file}")

            grids_file = os.path.join(state_dir, f"{base_name}_grids.json")
            if os.path.exists(grids_file):
                print(f"🗺️  Loaded map grids from: {grids_file}")

            # Map buffer should already be found by emulator.load_state() (Emerald only)
            if env.memory_reader and getattr(env.memory_reader, '_map_buffer_addr', None):
                print(f"Map buffer already initialized at 0x{env.memory_reader._map_buffer_addr:08X}")

            # Mark GAME_RUNNING milestone after state load
            # Defer expensive state logging - it will happen on first state request
            try:
                env.milestone_tracker.mark_completed("GAME_RUNNING")
                print("GAME_RUNNING milestone marked after state load - initial logging deferred")
            except Exception as e:
                print(f"Warning: Could not mark initial milestone: {e}")
        except Exception as e:
            # Carrying on from the title screen would let the agent play a different game
            # than the one asked for and overwrite checkpoint.state with it. Exit non-zero
            # so the run stops visibly and the checkpoint stays as it is.
            print(f"❌ Failed to load state from {args.load_state}: {e}")
            print("❌ Refusing to start from a fresh game instead; exiting.")
            sys.exit(1)

    # Start lightweight milestone updater thread
    state_update_running = True
    state_update_thread = threading.Thread(target=periodic_milestone_updater, daemon=True)
    state_update_thread.start()

    # Start FastAPI server in background thread
    server_thread = threading.Thread(target=run_fastapi_server, args=(args.port,), daemon=True)
    server_thread.start()

    # Get local IP for network access
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"

    print(f"🌐 FastAPI server running:")
    print(f"   Local: http://localhost:{args.port}")
    print(f"   Network: http://{local_ip}:{args.port}")
    print(f"📺 Stream interface: http://{local_ip}:{args.port}/stream")

    # Initialize video recording AFTER FastAPI server starts
    if args.record:
        init_video_recording(True)
    print("Available endpoints:")
    print("  /status - Server status")
    print("  /screenshot - Current screenshot")
    print("  /action - Take action (POST)")
    print("  /state - Comprehensive game state (visual + memory data)")
    print("  /agent - Agent thinking status")
    print("  /milestones - Current milestones achieved")
    print("  /recent_actions - Recently pressed buttons")
    print("  /debug/memory - Debug memory reading (basic)")
    print("  /debug/memory/comprehensive - Comprehensive memory diagnostics")
    print("  /debug/milestones - Debug milestone tracking system")
    print("  /debug/reset_milestones - Reset all milestones (POST)")
    print("  /debug/test_milestone_operations - Test milestone save/load (POST)")
    print("  /stop - Stop server")

    try:
        # Run headless game loop in main thread
        game_loop(manual_mode=False)  # Server always runs in server mode
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Backup cleanup (in case signal handler didn't run or failed)
        # The signal handler should handle most finalization, but this is a safety net
        # Note: running and state_update_running are already declared as global at start of main()
        was_running = running  # Read value before modifying
        running = False
        state_update_running = False

        # Only run backup if we weren't already shutting down
        # (signal handler should have handled it, but check if it actually did)
        if was_running:
            print("📦 Running backup finalization in finally block...")
            try:
                from utils.data_persistence.run_data_manager import get_run_data_manager

                run_manager = get_run_data_manager()
                if run_manager:
                    # Quick finalization - just save end state snapshot
                    run_manager.save_end_state_snapshot()
                    print("✅ Backup finalization completed")
            except Exception as e:
                logger.error(f"❌ Error in backup finalization: {e}", exc_info=True)

        if env:
            env.stop()
        print("Server stopped")


# Initialize emulator when imported for multiprocess mode
def init_for_multiprocess():
    """Initialize emulator when server is imported for multiprocess mode"""
    global env

    if env is None:  # Only initialize once
        # Check for environment variables set by agent.py multiprocess mode
        rom_path = os.environ.get("ROM_PATH", "Emerald-GBAdvance/rom.gba")
        load_state = os.environ.get("LOAD_STATE")
        record_video = os.environ.get("RECORD_VIDEO") == "1"
        no_ocr = os.environ.get("NO_OCR") == "1"

        print(f"🔧 Initializing server for multiprocess mode...")
        print(f"   ROM: {rom_path}")
        if load_state:
            print(f"   Load state: {load_state}")

        # Initialize emulator
        try:
            if not os.path.exists(rom_path):
                raise RuntimeError(f"ROM not found at {rom_path}")

            if game_type == "red":
                from pokemon_red_env.red_emulator import RedEmulator
                env = RedEmulator(rom_path=rom_path)
            else:
                from pokemon_env.emulator import EmeraldEmulator
                env = EmeraldEmulator(rom_path=rom_path)
            env.initialize()

            # Initialize video recording if requested
            init_video_recording(record_video)

            # Disable OCR if requested
            if no_ocr and env and env.memory_reader and hasattr(env.memory_reader, '_dialog_detection_enabled'):
                env.memory_reader._dialog_detection_enabled = False
                print("🚫 All dialogue detection disabled (--no-ocr flag)")

            # Load state if specified
            if load_state:
                try:
                    print(f"🔄 Attempting to load state from: {load_state}")
                    env.load_state(load_state)
                    print(f"📂 Successfully loaded state from: {load_state}")

                    # Milestones and map data are automatically loaded by env.load_state()
                    # Check what was loaded
                    state_dir = os.path.dirname(load_state)
                    base_name = os.path.splitext(os.path.basename(load_state))[0]

                    milestone_file = os.path.join(state_dir, f"{base_name}_milestones.json")
                    if os.path.exists(milestone_file):
                        print(f"📋 Loaded milestones from: {milestone_file}")

                    grids_file = os.path.join(state_dir, f"{base_name}_grids.json")
                    if os.path.exists(grids_file):
                        print(f"🗺️  Loaded map grids from: {grids_file}")

                    # Map buffer should already be found by emulator.load_state() (Emerald only)
                    if env.memory_reader and getattr(env.memory_reader, '_map_buffer_addr', None):
                        print(f"📍 Map buffer initialized at 0x{env.memory_reader._map_buffer_addr:08X}")

                    print(f"✅ State loading complete for {load_state}")

                except Exception as e:
                    # Same reasoning as main(): never silently swap in a fresh game.
                    print(f"❌ Failed to load state from {load_state}: {e}")
                    raise

            # Start lightweight milestone updater thread
            global state_update_running, state_update_thread
            state_update_running = True
            state_update_thread = threading.Thread(target=periodic_milestone_updater, daemon=True)
            state_update_thread.start()

            print("✅ Server initialized successfully for multiprocess mode")

        except Exception as e:
            print(f"❌ Failed to initialize server for multiprocess mode: {e}")
            raise


# Auto-initialize when imported for multiprocess mode (when ROM_PATH env var is set)
if os.environ.get("ROM_PATH") and __name__ != "__main__":
    init_for_multiprocess()

if __name__ == "__main__":
    main()