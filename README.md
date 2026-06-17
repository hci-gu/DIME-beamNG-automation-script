# DIME BeamNG Automation Script

This repository contains Python scripts for running a BeamNG.tech driving scenario used for a Driver Impairment Test route. The scripts connect to BeamNG.tech through `beamngpy`, build an East Coast USA scenario, place the ego vehicle, NPC vehicles, and static road signs, then monitor the driver's progress to trigger spoken instructions, NPC traffic behavior, speed warnings, and checkpoint respawns.

The most complete and current implementation is `beamng_updated.py`. The `main_beamngv3.py` and `main_beamngv3_newer.py` files are earlier script versions that contain much of the same scenario logic in a more monolithic form.

## Repository structure

```text
.
+-- README.md
+-- beamng_updated.py
+-- main_beamngv3.py
+-- main_beamngv3_newer.py
+-- Archive.zip
`-- __pycache__/
```

## General codebase summary

The codebase is organized around automating one BeamNG.tech route for an experimental driving task. It launches BeamNG.tech, creates a scenario on the `east_coast_usa` map, adds the participant vehicle and configured NPC vehicles, positions speed-limit signs, sets the driving camera, and waits for the user to start the scenario inside BeamNG.

Once the scenario is running, the main loop continuously updates vehicle state and uses the ego vehicle's position and speed to:

- Play one-time voice prompts at configured route locations.
- Warn the driver when speed exceeds the configured threshold.
- Activate nearby NPC vehicles into BeamNG traffic mode.
- Reposition the ego vehicle to route checkpoints through hotkeys.
- Optionally configure the BeamNG display window and OBS recording capture.

Most configuration is currently hard-coded in Python constants: BeamNG installation path, user profile path, map/scenario metadata, vehicle spawn points, NPC locations, prompt trigger positions, sign placement, display settings, and OBS settings.

## Important modules

### `beamng_updated.py`

Primary and recommended entry point. This is the refactored version of the scenario script, with typed data classes and separated controller/helper functions.

Important responsibilities:

- Defines scenario constants such as `MAP_NAME`, `SCENARIO_NAME`, `BNG_HOME`, player spawn, display settings, OBS settings, and prompt text.
- Uses data classes (`Range2D`, `RespawnCheckpoint`, `PromptTrigger`, `NPCConfig`, `SignConfig`) to keep route configuration readable.
- Uses `OBSController` to connect to OBS through `obsws-python`, create/update a single-window capture scene, and start/stop recording.
- Uses `BeamNGWindowController` to find and resize the BeamNG window on Windows through Win32 APIs.
- Uses `SpeechController` to run text-to-speech prompts asynchronously with `pyttsx3`.
- Uses `RespawnController` to handle checkpoint respawn and full reset hotkeys.
- Builds the BeamNG scenario in `build_scenario()`, adds signs in `add_speed_signs()`, configures vehicle AI in `configure_vehicles()`, then runs the polling loop in `run_main_loop()`.
- Handles cleanup in `finally`: stop window enforcement, stop OBS recording, stop speech, unhook keyboard hotkeys, and close BeamNG.

Run this file for the current behavior:

```powershell
python .\beamng_updated.py
```

### `main_beamngv3_newer.py`

Intermediate version of the scenario script. It adds functionality that was later cleaned up in `beamng_updated.py`, including an ego-speed warning, additional voice prompts, NPC `npc13`, and a suppression zone that avoids triggering NPC traffic near the initial route area.

This file is useful as historical context or for comparing earlier route behavior, but it keeps most scenario data and trigger logic inline inside `main()`.

### `main_beamngv3.py`

Older BeamNG scenario script. It contains the same broad flow: connect to BeamNG.tech, create the East Coast USA scenario, add player/NPC vehicles, add speed signs, wait for the scenario to start, then poll for voice prompts, NPC activation, and respawn hotkeys.

Compared with `beamng_updated.py`, this version has more duplicated state flags, inline checkpoint bounds, and inline prompt checks. Treat it as a legacy version unless you specifically need its older behavior.

### `Archive.zip`

Small archive committed with the repository. It is not referenced by the Python scripts, so it appears to be a backup or historical artifact rather than runtime input.

### `__pycache__/`

Generated Python bytecode cache. It is not source code and is not required to run the project.

## Runtime behavior

The maintained script follows this sequence:

1. Configure logging and controller objects.
2. Resolve the configured BeamNG.tech installation path.
3. Optionally connect to OBS.
4. Persist desired BeamNG display settings into the configured BeamNG user profile.
5. Launch BeamNG.tech with the Vulkan graphics backend.
6. Enforce the configured BeamNG window size and location on Windows.
7. Build and load the `Driver Impairment Test` scenario.
8. Start the text-to-speech worker and play the welcome prompt.
9. Configure vehicle AI and set the player camera to driver view.
10. Wait until the scenario is manually started in BeamNG.tech.
11. Start OBS recording if configured and available.
12. Enter the main loop for prompts, speed checks, respawn handling, and NPC activation.

## Installation

These instructions assume Windows, PowerShell, and a local BeamNG.tech installation.

### 1. Install BeamNG.tech

Install BeamNG.tech and update `BNG_HOME` in `beamng_updated.py` if your install path differs from the current default:

```python
BNG_HOME = r"C:\BeamNG.tech.v0.38.3.0\BeamNG.tech.v0.38.3.0"
```

The script looks for BeamNG executables under `Bin64`, including:

- `BeamNG.tech.x64.exe`
- `BeamNG.x64.exe`
- `BeamNG.drive.x64.exe`

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install Python dependencies

There is no checked-in `requirements.txt`, so install the packages imported by the scripts:

```powershell
python -m pip install beamngpy numpy pillow pyttsx3 keyboard obsws-python
```

Notes:

- `beamngpy`, `numpy`, `pillow`, and `pyttsx3` are required for the maintained script.
- `keyboard` is optional, but without it the respawn/reset hotkeys are disabled.
- `obsws-python` is optional, but without it OBS recording setup is skipped.

### 4. Configure optional runtime settings

Edit constants near the top of `beamng_updated.py` as needed:

- `BNG_USER_PATH`: BeamNG user profile path used for this experiment.
- `BNG_RESOLUTION`, `BNG_WINDOW_X`, `BNG_WINDOW_Y`, `BNG_WINDOW_WIDTH`, `BNG_WINDOW_HEIGHT`: display/window layout.
- `OBS_ENABLE_RECORDING`, `OBS_HOST`, `OBS_PORT`, `OBS_PASSWORD`, `OBS_RECORD_DIRECTORY`: OBS integration.
- `PLAYER_SPAWN`, `PLAYER_ROT_QUAT`, `RESPAWN_CHECKPOINTS`, `VOICE_PROMPTS`, `NPC_CONFIGS`, `SIGN_CONFIGS`: scenario route behavior.

If OBS recording is enabled, start OBS Studio first and enable obs-websocket on the configured port.

## Running

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python .\beamng_updated.py
```

BeamNG.tech should launch and load the generated scenario. Start the scenario from inside BeamNG.tech when prompted. The Python script will then continue running in the terminal and drive the route automation.

Useful controls:

- `S`: respawn the ego vehicle at the checkpoint matching the current route region.
- `Home`: reset the ego vehicle to the initial spawn.

Stop the script with `Ctrl+C` in the terminal. The script attempts to stop OBS recording, stop speech playback, unhook keyboard hotkeys, and close the BeamNG connection during cleanup.

## Development notes

- Prefer editing `beamng_updated.py` for new behavior.
- Keep route data in the existing configuration lists where possible instead of adding more inline checks.
- If adding new prompts, use `PromptTrigger`.
- If adding NPC vehicles, use `NPCConfig` and update any route-specific trigger behavior in `process_npc_triggers()`.
- If adding signs or static objects, follow the `SignConfig`/`add_speed_signs()` pattern.
- The current code is Windows-oriented because it uses Windows paths, PowerShell-oriented setup, `keyboard`, and Win32 window-management APIs.
