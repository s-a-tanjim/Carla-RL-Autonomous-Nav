import carla
import time
import random
import numpy as np
import math
import cv2
import gym
from gym import spaces

SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 20 # Per episode max time
CARLA_RESPONSE_TIMEOUT = 15.0

THROTTLE_MAX = 1.0
THROTTLE_MIN = -1.0
STEER_MAX = 1.0
STEER_MIN = -1.0
# For discrete action space
STEER_ROTATION = 1.0 
N_DISCRETE_ACTIONS = 3

class CarEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  SHOW_CAM = SHOW_PREVIEW
  im_width = IM_WIDTH
  im_height = IM_HEIGHT
  front_camera = None
  car_navigation_map = None


  def __init__(self, action_space_type):
    super(CarEnv, self).__init__()

    self.action_space_type = action_space_type

    # Carla Config
    self.client = carla.Client("localhost", 2000)
    self.client.set_timeout(CARLA_RESPONSE_TIMEOUT)
    self.world = self.client.get_world()
    self.map = self.world.get_map()
    weather = carla.WeatherParameters(
        cloudiness=99.0,
        precipitation=30.0,
        sun_altitude_angle=80.0,
        rayleigh_scattering_scale=0,
    )
    self.world.set_weather(weather)
    self.blueprint_library = self.world.get_blueprint_library()
    self.model_3 = self.blueprint_library.filter("model3")[0] # Change Car model from here
    self.actor_list = []

    # Map variable
    self.car_navigation_map = np.zeros((512,512,3), dtype=np.uint8)
    self.map_color = (255,255,255)
    self.reset_step_count = 0
    self.CARLA_WAYPOINT_COLOR = carla.Color(r=3, g=211, b=252, a=255)
    self.previous_vehicle_location = None

    # RL Environment variable
    if self.action_space_type == "continious":
      self.action_space = spaces.Box(np.array([THROTTLE_MIN, STEER_MIN]), np.array([THROTTLE_MAX, STEER_MAX]), dtype=np.float32)

    elif self.action_space_type == "discrete":
      self.action_space = self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    
    self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.im_height, self.im_width, 3), dtype=np.uint8)

    if self.SHOW_CAM:
      cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
      cv2.namedWindow("Feedback", cv2.WINDOW_NORMAL)


  def reset(self):
    self.collision_hist = []
    
    for actor in self.actor_list:
      actor.destroy()

    self.actor_list = []

    # Spawn Car
    # self.transform = random.choice(self.map.get_spawn_points())
    self.transform = self.map.get_spawn_points()[0]
    self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
    self.actor_list.append(self.vehicle)

    # Add Camera
    self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
    self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
    self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
    self.rgb_cam.set_attribute("fov", f"110")

    transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
    self.actor_list.append(self.sensor)
    self.sensor.listen(lambda data: self.process_img(data))

    self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0)) # Make car faster to response (Video feed, and fall from sky)
    time.sleep(4) # Waits, Cz car falls from sky, Needs time for Video feeds, Collision sensor gives value as it falls from sky :3

    colsensor = self.blueprint_library.find("sensor.other.collision")
    self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
    self.actor_list.append(self.colsensor)
    self.colsensor.listen(lambda event: self.collision_data(event))

    # On each episode the color of the path in map will be different
    self.map_color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
    self.reset_step_count += 1

    # After each 10 reset, map will reset
    if(self.reset_step_count%10==0):
      self.car_navigation_map = np.zeros((512,512,3), dtype=np.uint8)
    self.CARLA_WAYPOINT_COLOR = carla.Color(r=random.randint(1,255), g=random.randint(1,255), b=random.randint(1,255), a=0)
    self.previous_vehicle_location = self.vehicle.get_location()
    
    while self.front_camera is None:
      time.sleep(0.01)

    self.episode_start = time.time()
    self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

    # Return Observation
    return self.front_camera


  def process_img(self, image):
    i = np.array(image.raw_data, dtype = np.uint8)
    # print(i.shape)
    i2 = i.reshape((self.im_height, self.im_width, 4))
    i3 = i2[:, :, :3] # Remove Alpha value from Image
    # print(i3.shape)
    self.front_camera = i3


  def collision_data(self, event):
    self.collision_hist.append(event)


  def draw_line(self, start, end):
    self.world.debug.draw_line(start, end, thickness = 0.2, color = self.CARLA_WAYPOINT_COLOR, life_time = 120.0)


  def _step(self, throttle, steer):
    # __init__(_object*, float throttle=0.0, float steer=0.0, float brake=0.0, bool hand_brake=False, bool reverse=False, bool manual_gear_shift=False, int gear=0)
    self.vehicle.apply_control(carla.VehicleControl(throttle = throttle, steer = steer))

    v = self.vehicle.get_velocity()
    kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

    # Updating the map
    vehicle_new_location = self.vehicle.get_location()
    vehicle_waypoint = self.map.get_waypoint(vehicle_new_location, project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
    x_val = int(vehicle_waypoint.transform.location.x)+256
    y_val = int(vehicle_waypoint.transform.location.y)+256
    self.car_navigation_map[x_val, y_val] = self.map_color

    self.draw_line(self.previous_vehicle_location, vehicle_new_location)
    self.previous_vehicle_location = vehicle_new_location

    if len(self.collision_hist) != 0:
      done = True
      reward = -200
    elif kmh < 25:
      done = False
      reward = -1
    else:
      done = False
      reward = 1

    if self.episode_start + SECONDS_PER_EPISODE < time.time(): # Limit the length of episode
      done = True

    self.render()
    info = {}
    return self.front_camera, reward, done, info


  def step(self, action):
    if action is not None:
      if type(action)==np.ndarray:
        return self._step((action[0].item()+1)/2, action[1].item())
      else:
        if action == 0: # Left
          steer = -1 * STEER_ROTATION
        elif action == 1: # Straight
          steer = 0
        elif action == 2:# Right
          steer = 1 * STEER_ROTATION
        return self._step(1.0, steer)


  def render(self, mode='human'):
    if self.SHOW_CAM:
      #if cv2.getWindowProperty('Feedback', cv2.WND_PROP_VISIBLE) != 0.0:
      #    cv2.destroyAllWindows()
      # cv2.imwrite("Feedback.jpg", self.front_camera)
      # img = cv2.imread("Feedback.jpg", 0)
      # print(self.front_camera.shape)
      cv2.imshow("Map", self.car_navigation_map)
      cv2.imshow("Feedback", self.front_camera)
      cv2.waitKey(1)


  def close(self):
    cv2.destroyAllWindows()
    
    for actor in self.actor_list:
      actor.destroy()
  
  
  def __del__(self):
    self.close()
