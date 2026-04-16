import logging
import random
import threading


import beamngpy.api.beamng.ui
#from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy import BeamNGpy, Scenario, Vehicle, StaticObject

from beamngpy.api.beamng.ui import UiApi
import math
import time
from PIL import Image
import numpy as np
import pyttsx3

from collections import deque

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    logging.warning("'keyboard' module not found — respawn hotkeys disabled. Install with: pip install keyboard")

run_at_home = False

# Respawn history window (seconds)
HISTORY_SECONDS = 5.0#5.0
REWIND_SECONDS  = 5.0# 3.0   # how far back R-key rewinds

VOICE_TRIGGER_POS = (500, -875, 41.0)   # change to your route position
VOICE_TRIGGER_DIST = 20.0                   # metres
VOICE_PROMPT_TEXT = "Welcome. Please continue to drive along the route until the experimenter stops the simulation. You are required to drive safely and obey driving rules consistent with Swedish law. If you damage the vehicle, your vehicle will be repositioned at an earlier point on the route.  You may continue when ready. "

VOICE_TRIGGER_POS2 = (470, -831, 43)


bng_home = r'C:\BeamNG.tech.v0.38.3.0\BeamNG.tech.v0.38.3.0' #r'C:\Users\Rob\BeamNG\BeamNG.tech.v0.35.5.0\BeamNG.tech.v0.35.5.0' 

CONE_SHAPE = "/levels/east_coast_usa/art/shapes/misc/cone.dae"


bridge_pos = (799.127869, 785.006409, 77.3248138)


# see line 354 for player spawn (starting point) - should have a variable with the x,y,z coords
# and rotation values assigned that are randomly selected checkpoints (or we use a latin matrix perhaps so a 
# pre-given script, or .json file is read determining what is the starting point)
# the starting point should always be at a check point : 
# so start coords and rot vals for: 
# CP4 Event 1 ((500, -875, 41.0), (0.0087653, 0.00096168, 0.93361151, 0.35821542))
# CP1 Event 1 ((-722, 482, 22), (0.007477, -0.0027, 0.883, 0.4677))
# CÅ6 Event 1 (-920.711, 662.963, 24.557), (0.0208, 0.0216, 0.952, 0.302)
# CP2 Event 1 (-800, 950, 35), (0.0022, 0.0166, -0.736, 0.676)
# CP3 Event 1 (-571, 941, 35), (0.004, -0.0054, -0.694, 0.719)
# CP3 Event 2 (-175, 910, 35), (0.004, -0.0054, -0.694, 0.719) # most likely inaccurate
# CP6 Event 2 (-70, 873, 25), (0.026, 0.025, -0.06, 0.78)
# CP4 Event 2 (811.88, 581.11, 65.38), (-0.0072, -0.000579, -0.147, 0.989)
# CP! Event 2 (892, -611, 40), (0.005, 0.0007, 0.1946, 0.981)
# CP2 Event 2 (829.9, -759.14, 42.25), (-0.0048, 0.001898, 0.198, 0.98)

tts_lock = threading.Lock()
tts_engine = None


def dist3(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2) ** 0.5

def init_tts():
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 170)   # speech speed
        tts_engine.setProperty('volume', 1.0) # 0.0 to 1.0
        logging.info("TTS engine initialized.")
    except Exception as e:
        logging.warning("TTS init failed: %s", e)
        tts_engine = None

import threading
import queue
import pyttsx3
import logging

speech_queue = queue.Queue()

# ---------------------------------------------------------------------------
# Speech / verbal prompts
# ---------------------------------------------------------------------------

speech_queue = queue.Queue()
speech_worker_running = True

def speech_worker():
    """Continuously plays queued speech prompts, one at a time."""
    while speech_worker_running:
        text = speech_queue.get()
        if text is None:
            break

        engine = None
        try:
            logging.info("Speaking: %s", text)
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.warning("TTS worker failed: %s", e)
        finally:
            try:
                if engine is not None:
                    engine.stop()
            except Exception:
                pass
            speech_queue.task_done()

def start_speech_worker():
    t = threading.Thread(target=speech_worker, daemon=True)
    t.start()
    return t

def stop_speech_worker():
    global speech_worker_running
    speech_worker_running = False
    speech_queue.put(None)

def speak_async(text):
    """Queue speech without blocking the main BeamNG loop."""
    speech_queue.put(text)

#def speak_async(text):
#    """Speak without blocking the main BeamNG loop."""
#    if tts_engine is None:
#        logging.warning("TTS engine unavailable. Prompt was: %s", text)
#        return

#    def _worker():
#        with tts_lock:
#            try:
#                tts_engine.say(text)
#                tts_engine.runAndWait()
#            except Exception as e:
#                logging.warning("TTS speak failed: %s", e)

#    threading.Thread(target=_worker, daemon=True).start()


def add_cone_line(scenario, start_pos, spacing_m=5.0, count=10, yaw_deg=0.0):
    """
    Place a line of static traffic cones.

    start_pos: (x, y, z)
    spacing_m: spacing between cones
    count: number of cones
    yaw_deg: heading angle in degrees
    """
    yaw_rad = math.radians(yaw_deg)
    dx = math.cos(yaw_rad) * spacing_m
    dy = math.sin(yaw_rad) * spacing_m

    # Quaternion for yaw about Z axis
    half = yaw_rad / 2.0
    rot_quat = (0.0, 0.0, math.sin(half), math.cos(half))

    for i in range(count):
        x = start_pos[0] + i * dx
        y = start_pos[1] + i * dy
        z = start_pos[2]

        cone = StaticObject(
            name=f"cone_{i}",
            pos=(x, y, z),
            rot_quat=rot_quat,
            scale=(1.0, 1.0, 1.0),
            shape=CONE_SHAPE,
        )
        scenario.add_object(cone)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    )

def quat_conj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)

def quat_rotate(q, v):
    vx, vy, vz = v
    vq = (vx, vy, vz, 0.0)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[:3]

def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def vscale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)

def quat_from_z_rotation_degrees(deg):
    rad = math.radians(deg)
    half = rad / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))




def save_birdseye_image(bng, center_pos, filename='birdseye_route_view.png', height=200):
    """
    Capture a single bird's-eye (top-down) screenshot centered at center_pos.
    """
    camera_pos = (center_pos[0], center_pos[1], center_pos[2] + height)  # height meters above
    camera_dir = (0, 0, -1)  # looking straight down

    cam = bng.camera.render(
        pos=camera_pos,
        dir=camera_dir,
        fov=70,
        resolution=(1920, 1080),
        render_annotations=False
    )

    # BeamNGpy sometimes uses 'colour' vs 'color' depending on version
    colour_key = 'colour' if 'colour' in cam else 'color'
    img = Image.fromarray(np.asarray(cam[colour_key]))
    img.save(filename)

class PositionHistory:
    """Rolling buffer of (pos, rot_quat, timestamp) entries."""

    def __init__(self, window_seconds=HISTORY_SECONDS):
        self._buf     = deque()
        self._window  = window_seconds
        self._lock    = threading.Lock()

    def record(self, pos, rot):
        now = time.time()
        with self._lock:
            self._buf.append((tuple(pos), tuple(rot), now))
            cutoff = now - self._window
            while self._buf and self._buf[0][2] < cutoff:
                self._buf.popleft()

    def get_state_seconds_ago(self, seconds):
        target = time.time() - seconds
        with self._lock:
            if not self._buf:
                return None, None
            best = min(self._buf, key=lambda e: abs(e[2] - target))
            return best[0], best[1]   # pos, rot


class RespawnController:
    def __init__(self, spawn_pos, spawn_rot):
        self.rewind_requested = False
        self.reset_requested = False
        self.spawn_pos = spawn_pos
        self.spawn_rot = spawn_rot

        if KEYBOARD_AVAILABLE:
            keyboard.add_hotkey('S', self._request_rewind, suppress=False)
            keyboard.add_hotkey('home', self._request_reset, suppress=False)
            logging.info("Hotkeys active: F7 = checkpoint respawn, Home = full reset")

    def _request_rewind(self):
        self.rewind_requested = True

    def _request_reset(self):
        self.reset_requested = True

    def stop(self):
        if KEYBOARD_AVAILABLE:
            keyboard.unhook_all()

    def process(self, vehicle, current_pos):
        if self.rewind_requested:
            self.rewind_requested = False
            x, y, z = current_pos[0], current_pos[1], current_pos[2]

            logging.info("F7 pressed at (%f, %f, %f)", x, y, z)

            if (-488 < x < 501) and (-877 < y < 249):
                # CP4 event 1 start
                vehicle.teleport((500, -875, 41.0), (0.0087653, 0.00096168, 0.93361151, 0.35821542))
                return True
            elif (-876 < x < -488) and (-877 < y < 608):
                # CP1 event 1 start
                vehicle.teleport((-722, 482, 22), (0.007477, -0.0027, 0.883, 0.4677))
                return True
            elif (-876 < x < -570): #and (-950 < y < 877)
                # CP2 event 1 start
                vehicle.teleport((-800, 950, 35), (0.0022, 0.0166, -0.736, 0.676))
                return True
            elif (-570 < x < -175) and (910 < y < 980):
                # CP3 event 1 start
                vehicle.teleport((-571, 941, 35), (0.004, -0.0054, -0.694, 0.719))
            elif (-175 < x < -70) and (910 < y < 860):    
                # CP3 event 2 start - needs testing, rot vals off here
                vehicle.teleport((-175, 910, 35), (0.004, -0.0054, -0.694, 0.719))
                return True
            elif (608 < y < 950): # bit hacked 
                # CP6 event 1 start
                vehicle.teleport((-920.711, 662.963, 24.557), (0.0208, 0.0216, 0.952, 0.302))
                return True
            elif (-70 < x < 78) and (880 > y > 860):
                # CP6 event 2 start
                vehicle.teleport((-70, 873, 25), (0.026, 0.025, -0.06, 0.78))
                return True
            elif (902 < x < 812) and (581 > y > -420):
                # CP4 event 2 start
                vehicle.teleport((811.88, 581.11, 65.38), (-0.0072, -0.000579, -0.147, 0.989))
                return True 
            elif (902 < x < 892) and (581 > y > -611):
                # CP1 event 2 start
                vehicle.teleport((892, -611, 40), (0.005, 0.0007, 0.1946, 0.981))
                return True     


            logging.info("No checkpoint matched")
            return False

        if self.reset_requested:
            self.reset_requested = False
            vehicle.teleport(self.spawn_pos, self.spawn_rot)
            return True

        return False


def ego_speed_mps(vehicle):
    vehicle.sensors.poll()
    vx, vy, vz = vehicle.state["vel"]
    speed_mps = math.sqrt(vx**2 + vy**2 + vz**2)
    speed_mph = speed_mps * 2.23693629
    return speed_mph
    

def main():
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    traffic_spawned = False

    history          = PositionHistory(HISTORY_SECONDS)
    respawn_ctrl     = None	

    try:
        # Connect to BeamNG.tech
        bng = BeamNGpy('localhost', 64256, home=bng_home,
                       user=None)
        bng.open(None, '-gfx', 'vk')

        logging.info("Connected to BeamNG.tech")

        #bng.display_gui_message(VOICE_PROMPT_TEXT)
        
        # Create a scenario in East Coast USA map
        scenario = Scenario('east_coast_usa', 'Driver Impairment Test',
                            description='Please listen to the voice instructions before starting your drive. You may continue when prompted.', authors='University of Gothenburg',
                            settings={'Start Time': 0})

        #add_cone_line(
       # 	scenario,
       # 	start_pos=(500, -875, 41.0),   # adjust to your road
       # 	spacing_m=4.0,
       # 	count=12,
       # 	yaw_deg=90.0,
       # ) 	


        # Player spawn point coordinates
        player_spawn = (501, -877, 41)#(438.334015, -810.008179, 42.9711952)

        # Player rotation (180 degrees from previous rotation)
        player_rot_quat = (0.0087653, 0.00096168, 0.93361151, 0.35821542)

        # Create and add the player's vehicle
        player_vehicle = Vehicle('ego_vehicle', model='etk800', license='UVT 6xY')
        scenario.add_vehicle(player_vehicle, pos=player_spawn, rot_quat=player_rot_quat)

        

        # NPC vehicles information
        npc_vehicles = [
                        {
                'name': 'npc6',
                'pos': (-879.305, 612.342, 24.706),#(-745.856569408033, 501.04896083922904, 23.04104474254609), #(-813.315186, 553.499817, 25.5619831),
                'rot': (-0.0006, -0.00053, 0.92952, 0.36877)#(0.00747713865712285, -0.002740494441241026, 0.8838406205177307, 0.4677203595638275) #(0, 0, 0.938254, 0.345945)  # Same rotation as npc5
            },
            {
                'name': 'npc7',
                'pos': (-852.8610016107559, 586.0869419574738, 26.510813504457474),#(-808.043823, 950.198364, 35.0526657),
                'rot': (0, 0, -0.662620, 0.748956)  # Converted from Euler rotation
            },
            {
                'name': 'npc8',
                'pos': (17.7, 865.67, 26.178),#(-176.642288, 918.723938, 23.2221851),
                'rot': (-0.003, -0.0026, -0.717, 0.696)#(0, 0, -0.662620, 0.748956)  # Same rotation as npc7
            },
            {
                'name': 'npc9',
                'pos': (455.650055, 902.720032, 51.4617844),
                'rot': (0, 0, -0.750111, 0.661312)  # Converted from Euler rotation
            },
            {
                'name': 'npc10',
                'pos': (-748, 488, 23),
                'rot': (-0.026407821103930473, -0.01194380410015583, -0.8407955765724182, 0.5405763387680054) 
            },
            {
                'name': 'npc11',
                'pos': (-667.2760881692193, 950.6957388893661, 37.89792846151886),
                'rot': (0.0018872566288337111, 0.005335998721420765, 0.7313352823257446, 0.681994616985321) 
            },
            {
                'name': 'npc12',
                'pos': (529.8497040759812, 931.7092222183737, 61.2222659783788),
                'rot': (0.03721029683947563, 0.06410301476716995, -0.810825526714325, 0.5805758237838745) 
            },
            {
                'name': 'npc13',
                'pos': (498.96, -881.98, 39.886),
                'rot': (-0.074618, -0.001525, -0.4296, 0.89989) 
            }
        ]    


        # Create and add NPC vehicles
        for npc in npc_vehicles:
            if npc['name'] == 'npc6':
               vehicle = Vehicle(npc['name'], model='scintilla', license=f'RNO LR8')
            elif npc['name'] == 'npc7':
               vehicle = Vehicle(npc['name'], model='autobello', license=f'GNX 566')		
            elif npc['name'] == 'npc11':
               vehicle = Vehicle(npc['name'], model='etki', license=f'YRU 593')
            elif npc['name'] == 'npc12':
               vehicle = Vehicle(npc['name'], model='etki', license=f'NRL 002')   
            elif npc['name'] == 'npc8':
               vehicle = Vehicle(npc['name'], model='etk800', license=f'JYT 7T1')
            elif npc['name'] == 'npc9':
               vehicle = Vehicle(npc['name'], model='etk800', license=f'PII 377')   
            elif npc['name'] == 'npc10':
               vehicle = Vehicle(npc['name'], model='autobello', license=f'HJG 89U')
            elif npc['name'] == 'npc13':
               vehicle = Vehicle(npc['name'], model='scintilla', license=f'NOM 756')      
            else:   
               vehicle = Vehicle(npc['name'], model='etk800', license=f'NPC {npc["name"][-1]}')
            #if not npc['name'] == 'npc13':
            scenario.add_vehicle(vehicle, pos=npc['pos'], rot_quat=npc['rot'])

        logging.info("Vehicles added to scenario")
        SIGN_SHAPE = "/levels/east_coast_usa/art/shapes/signs_usa/sign_speedlimit_50.dae"
        SIGN_SHAPE2 = "/levels/east_coast_usa/art/shapes/signs_usa/sign_speedlimit_40.dae"

        # Work out forward/right directions from player spawn rotation
        forward = quat_rotate(player_rot_quat, (0, 1, 0))
        right = quat_rotate(player_rot_quat, (1, 0, 0))

        # Put sign a bit ahead of the player and on the right-hand side of the road
        sign_pos =  ((-721.0817284743873, 481.14044092518566, 22.124332515832975))
        sign_pos = vadd(sign_pos, vscale(forward, -2.5))#-6.0))   # 12 m ahead
        sign_pos = vadd(sign_pos, vscale(right, 0))# 12.0))      # 4 m to the right
        sign_pos = vadd(sign_pos, (0, 0, 0.2))             # slight lift above ground


        forward = quat_rotate(player_rot_quat, (0, 1, 0))
        right = quat_rotate(player_rot_quat, (1, 0, 0))
        sign_pos2 =  ((-559.9609605189526, 941.1932155510767, 34.83220948548933))
        sign_pos2 = vadd(sign_pos2, vscale(forward, -2))#-6.0))   # 12 m ahead
        sign_pos2 = vadd(sign_pos2, vscale(right, -6))# 12.0))      # 4 m to the right
        sign_pos2 = vadd(sign_pos2, (0, 0, 0.2))             # slight lift above ground
        
        sign_pos3 =  ((-386.5015354813413, 918.9126735679774, 30.40501675377709))
        sign_pos3 = vadd(sign_pos3, vscale(forward, -0))# -6.0))   # 12 m ahead
        sign_pos3 = vadd(sign_pos3, vscale(right, -7))# 12.0))      # 4 m to the right
        sign_pos3 = vadd(sign_pos3, (0, 0, 0.2))             # slight lift above ground

        sign_rot = quat_mul(player_rot_quat, quat_from_z_rotation_degrees(180))# this works
        sign_rot2 = quat_mul(player_rot_quat, quat_from_z_rotation_degrees(0))		

        speed_sign = StaticObject(
            name='speed_sign_70',
            pos=sign_pos,
            rot_quat=sign_rot,
            scale=(1, 1, 1),
            shape=SIGN_SHAPE
        )
        speed_sign2 = StaticObject(
            name='speed_sign_40',
            pos=sign_pos2,
            rot_quat=sign_rot2,
            scale=(1, 1, 1),
            shape=SIGN_SHAPE2
        )
        speed_sign3 = StaticObject(
            name='speed_sign_50',
            pos=sign_pos3,
            rot_quat=sign_rot2,
            scale=(1, 1, 1),
            shape=SIGN_SHAPE
        )

        scenario.add_object(speed_sign)
        scenario.add_object(speed_sign2)
        scenario.add_object(speed_sign3)

        ## add second sign : 'pos': [-559.9609605189526, 941.1932155510767, 34.83220948548933]

#### respawning
        respawn_ctrl = RespawnController(player_spawn, player_rot_quat)
        
        

        # Load the scenario
        scenario.make(bng)
        logging.info("Scenario created")

        # Load the scenario in BeamNG.tech
        bng.load_scenario(scenario)
        logging.info("Scenario loaded in BeamNG.tech")

        speech_thread = None
        speech_thread = start_speech_worker()
        speak_async(VOICE_PROMPT_TEXT)

        # Set up initial state
        player_vehicle.ai.set_mode('manual')
        for npc in npc_vehicles:
            scenario.get_vehicle(npc['name']).ai.set_mode('disabled')
        logging.info("Vehicle AI modes set")

        # Set the default camera to interior view
        bng.camera.set_player_mode(player_vehicle, 'driver', {'fov': 60})
        logging.info("Camera set to interior view")

        logging.info("Waiting for the user to start the scenario in BeamNG.tech...")

        # Wait for the scenario to start
        scenario_started = False
        traffic_spawned = False
        
        

        while not scenario_started:
            try:
                gamestate = bng.get_gamestate()
                if gamestate['state'] == 'scenario' and gamestate['scenario_state'] == 'running':
                    logging.info("Scenario started by user.")
                    scenario_started = True
                    #if not traffic_spawned:
                    #    bng.traffic.spawn(max_amount=3)
                    #    traffic_spawned = True

                else:
                    time.sleep(1)  # Check every second
            except Exception as e:
                logging.warning(f"Error checking game state: {e}")
                time.sleep(1)  # Wait a bit before retrying
                
                
        # Make sure we have at least one update so vehicle state is populated
        scenario.update()

        
        npc_triggered = [False] * len(npc_vehicles)
        voice_prompt_played = False
        voice_prompt_played_1 = False
        voice_prompt_played_2 = False
        voice_prompt_played_3 = False
        voice_prompt_played_4 = False
        voice_prompt_played_5 = False
        voice_prompt_played_6 = False
        voice_prompt_played_7 = False
        voice_prompt_played_8 = False
        voice_prompt_played_9 = False
        voice_prompt_played_10 = False
        voice_prompt_played_11 = False
        voice_prompt_played_12 = False
        voice_prompt_played_13 = False
        voice_prompt_played_14 = False
        voice_prompt_played_15 = False
        voice_prompt_played_16 = False
        voice_prompt_played_17 = False
        voice_prompt_played_18 = False
        voice_prompt_played_19 = False
        voice_prompt_played_20 = False
        voice_prompt_played_21 = False
        #init_tts()

        #speech_thread = start_speech_worker()

        while True:
            # Update vehicle states
            scenario.update()

            # Get player position		
            player_pos = player_vehicle.state['pos']
            player_rot = player_vehicle.state.get('rotation', player_rot_quat)

            # Record position for rewind
            history.record(player_pos, player_rot)

            # Handle respawn hotkeys	
            #respawn_ctrl.process(player_vehicle, history)
            respawn_ctrl.process(player_vehicle, player_pos)

	    # Voice prompt trigger

            ## CP4 Event 1
            #if not voice_prompt_played and dist3(player_pos, VOICE_TRIGGER_POS) < VOICE_TRIGGER_DIST: # (500, -875, 41.0)
            #    bng.display_gui_message(VOICE_PROMPT_TEXT)
            #    speak_async(VOICE_PROMPT_TEXT)
            #    voice_prompt_played = True
            #    logging.info("Voice prompt triggered at %s", player_pos)

            if not voice_prompt_played_1 and ego_speed_mps(player_vehicle) > 55.0:
                voice_prompt_played_1 = True
                speak_async("Please slow down!")

            if voice_prompt_played_1 and ego_speed_mps(player_vehicle) < 50.0:
                voice_prompt_played_1 = False    

            if not voice_prompt_played_2 and dist3(player_pos, VOICE_TRIGGER_POS2) < 10: #(470, -831, 43)
                #bng.display_gui_message("Continue Straight")
                speak_async("Continue Straight")
                voice_prompt_played_2 = True
                logging.info("Voice prompt triggered at %s", player_pos)		

            ## CP4 Event 1            
            if not voice_prompt_played_8 and dist3(player_pos, (430, -800, 41)) < 20: #(470, -831, 43)
                #bng.display_gui_message("Attention! Please spell 'Norway'")
                speak_async("Attention! Please spell 'Norway'")
                voice_prompt_played_8 = True
                logging.info("Voice prompt triggered at %s", player_pos)
                
    
            if not voice_prompt_played_3 and dist3(player_pos, (-450,226,28)) < 20:
                #bng.display_gui_message("Continue Straight")
                speak_async("Continue Straight")
                voice_prompt_played_3 = True
                logging.info("Voice prompt triggered at %s", player_pos) 	  	

            if not voice_prompt_played_4 and dist3(player_pos, (-605,395,21)) < 20:
                #bng.display_gui_message("Continue Straight")
                speak_async("Continue Straight")
                voice_prompt_played_4 = True
                logging.info("Voice prompt triggered at %s", player_pos) 	    
	   	

            if not voice_prompt_played_5 and dist3(player_pos, (-655,446,23)) < 20:
                #bng.display_gui_message("Continue Straight")
                speak_async("Continue Straight")
                voice_prompt_played_5 = True
                logging.info("Voice prompt triggered at %s", player_pos)  

            
            ## CP1 Event 1
            if not voice_prompt_played_11 and dist3(player_pos, ((-731.8,481.8,22.4))) < 20:
                #bng.display_gui_message("Attention! Please spell 'alone'")
                speak_async("Attention! Please spell 'alone'")
                voice_prompt_played_11 = True
                logging.info("Voice prompt triggered at %s", player_pos)

            ## CP6 Event 1
            if not voice_prompt_played_6 and dist3(player_pos, (-920.711, 662.963, 24.557)) < 10:
                #bng.display_gui_message("Please read the license plate of the car in front of you and then overtake when possible")
                speak_async("Attention! Please read the license plate of the car in front of you and then overtake when possible")
                voice_prompt_played_6 = True
                logging.info("Voice prompt triggered at %s", player_pos)

            if not voice_prompt_played_7 and dist3(player_pos, (-852.8610016107559, 586.0869419574738, 26.510813504457474)) < 10:
                #bng.display_gui_message("Continue Straight")
                speak_async("Continue Straight")
                voice_prompt_played_7 = True
                logging.info("Voice prompt triggered at %s", player_pos)
    
            

            ## CP3 Event 1
            if not voice_prompt_played_9 and dist3(player_pos, ((-571, 941, 35))) < 20:
                #bng.display_gui_message("I want to play a game. Say red lorry, yellow lorry twice")
                speak_async("Attention! Say red lorry, yellow lorry twice")
                voice_prompt_played_9 = True
                logging.info("Voice prompt triggered at %s", player_pos)


            ## CP2 Event 1
            if not voice_prompt_played_10 and dist3(player_pos, ((-800, 950, 35))) < 30:
                #bng.display_gui_message("Attention! Please spell 'Dollar'")

                speak_async("Attention! Please spell 'Dollar'")
                voice_prompt_played_10 = True
                logging.info("Voice prompt triggered at %s", player_pos)

            if not voice_prompt_played_12 and dist3(player_pos, (-14.27, 866.40, 26.56)) < 20:
                #bng.display_gui_message("Continue Straight")
                speak_async("Continue Straight")
                voice_prompt_played_12 = True
                logging.info("Voice prompt triggered at %s", player_pos)
            
            if not voice_prompt_played_13 and dist3(player_pos, (225, 880, 37)) < 20:
                #bng.display_gui_message("Continue Straight")
                speak_async("Continue Straight")
                voice_prompt_played_13 = True
                logging.info("Voice prompt triggered at %s", player_pos)

            ## CP6 Event 2
            if not voice_prompt_played_14 and dist3(player_pos, (492.3,914.3,55.69)) < 10:
                #bng.display_gui_message("Please read the license plate of the car in front of you and then overtake when possible")
                speak_async("Attention! Please read the license plate of the car in front of you and then overtake when possible")
                voice_prompt_played_14 = True
                logging.info("Voice prompt triggered at %s", player_pos)


            # CP3 Event 2 (-175, 910, 35), (0.004, -0.0054, -0.694, 0.719) # most likely inaccurate
            if not voice_prompt_played_19 and dist3(player_pos, (-175, 910, 35)) < 20:
                #bng.display_gui_message("Please turn right.")
                speak_async("Attention. Please tell me about the last meal you had.")
                voice_prompt_played_19 = True
                logging.info("Voice prompt triggered at %s", player_pos) 

            # CP4 Event 2 (811.88, 581.11, 65.38), (-0.0072, -0.000579, -0.147, 0.989)    
            if not voice_prompt_played_20 and dist3(player_pos, (811.88, 581.11, 65.38)) < 20:
                #bng.display_gui_message("Please turn right.")
                speak_async("Attention. Please tell me what you did on your last birthday.")
                voice_prompt_played_20 = True
                logging.info("Voice prompt triggered at %s", player_pos)

            # CP2 Event 2 (829.9, -759.14, 42.25), (-0.0048, 0.001898, 0.198, 0.98)
            if not voice_prompt_played_21 and dist3(player_pos, (829.9, -759.14, 42.25)) < 20:
                #bng.display_gui_message("Please turn right.")
                speak_async("Attention. Please spell lonely.")
                voice_prompt_played_21 = True
                logging.info("Voice prompt triggered at %s", player_pos)
            


            if not voice_prompt_played_15 and dist3(player_pos, (902, -420, 44)) < 20:
                #bng.display_gui_message("Continue Straight")
                speak_async("Continue Straight")
                voice_prompt_played_15 = True
                logging.info("Voice prompt triggered at %s", player_pos)

            # CP1 Event 2
            if not voice_prompt_played_16 and dist3(player_pos, (909, -258, 40)) < 20:
                #bng.display_gui_message("Attention! Please say 'Unique New York' twice")
                speak_async("Attention! Please say 'Unique New York' twice")
                voice_prompt_played_16 = True
                logging.info("Voice prompt triggered at %s", player_pos)

            if not voice_prompt_played_17 and dist3(player_pos, (544, -892, 39)) < 20:
                #bng.display_gui_message("Please turn right.")
                speak_async("Please turn right.")
                voice_prompt_played_17 = True
                logging.info("Voice prompt triggered at %s", player_pos)

            if not voice_prompt_played_18 and dist3(player_pos, (827.84, -765.47, 42.2)) < 20:
                #bng.display_gui_message("Please turn right.")
                speak_async("Continue straight.")
                voice_prompt_played_18 = True
                logging.info("Voice prompt triggered at %s", player_pos)

            


            time.sleep(0.1)	 		

            for i, npc in enumerate(npc_vehicles):
                npc_vehicle = scenario.get_vehicle(npc['name'])
                if npc['name'] == 'npc6': 
                   npc_vehicle.set_color((0.0, 0.0, 1.0, 0.2))
                elif npc['name'] == 'npc11':
                   npc_vehicle.set_color((0.5, 0.5, 0.0, 0.6))
                elif npc['name'] == 'npc12':
                   npc_vehicle.set_color((1.0, 0.0, 0.0, 0.6))  
                elif npc['name'] == 'npc8':
                   npc_vehicle.set_color((1.0, 1.0, 0.0, 1.0))
                elif npc['name'] == 'npc9':
                   npc_vehicle.set_color((1.0, 1.0, 1.0, 0.0))
                elif npc['name'] == 'npc10':
                   npc_vehicle.set_color((0.5, 0.5, 0.5, 0.0)) 
                elif npc['name'] == 'npc13':
                   npc_vehicle.set_color((0.0, 0.0, 0.0, 1.0))                  
                npc_pos = npc_vehicle.state['pos']

                # Calculate distance between player and NPC
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(player_pos, npc_pos)))

                # If player is between 5 and 100 meters and NPC hasn't been triggered yet, make NPC drive
                if distance < 100 and not npc_triggered[i] and not ((502 > player_pos[0] > 350) and (-800 > player_pos[1] > -910)):
                    npc_vehicle.ai.set_mode('traffic')
                    npc_vehicle.ai.drive_in_lane(True)
                    npc_vehicle.ai.set_speed(12, mode='limit')  # Set speed limit to 12 m/s (about 43 km/h or 27 mph)

                    npc_pos = [npc_pos[0] + 2, npc_pos[1] + 2, npc_pos[2] + 2]
                    print(npc_pos)
                    print("\n\n\n\n\n\n\n\n\n\n\n\n")

                    npc_triggered[i] = True
                    logging.info(f"{npc['name']} has been triggered and is now driving in traffic mode!")
                    bng.ui.display_message(f"Overtake the vehicle in front of you")

            bridge_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(player_pos, bridge_pos)))

            #if bridge_distance < 50 and not traffic_spawned:
            #    bng.traffic.spawn(max_amount=3)
            #    traffic_spawned = True

            time.sleep(0.1)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # Close the connection
        stop_speech_worker()
        if 'bng' in locals():
            bng.close()

if __name__ == "__main__":
    main()