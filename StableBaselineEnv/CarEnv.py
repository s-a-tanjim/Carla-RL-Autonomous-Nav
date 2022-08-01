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
N_DISCRETE_ACTIONS = 3
CARLA_RESPONSE_TIMEOUT = 5.0

class CarEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  SHOW_CAM = SHOW_PREVIEW
  STEER_AMT = 1.0
  im_width = IM_WIDTH
  im_height = IM_HEIGHT
  front_camera = None

  def __init__(self):
    super(CarEnv, self).__init__()

    # Carla Config
    self.client = carla.Client("localhost", 2000)
    self.client.set_timeout(CARLA_RESPONSE_TIMEOUT)
    self.world = self.client.get_world()
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


    # RL Environment variable
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.im_height, self.im_width, 3), dtype=np.uint8)

    

  def reset(self):
    self.collision_hist = []
    
    for actor in self.actor_list:
      actor.destroy()

    self.actor_list = []

    # Spawn Car
    self.transform = random.choice(self.world.get_map().get_spawn_points())
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



  def step(self, action):
    if action == 0: # Left
      self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
    elif action == 1: # Straight
      self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
    elif action == 2: # Right
      self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

    v = self.vehicle.get_velocity()
    kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

    if len(self.collision_hist) != 0:
      done = True
      reward = -200
    elif kmh < 50:
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


  def render(self, mode='human'):
    if self.SHOW_CAM:
      #if cv2.getWindowProperty('Feedback', cv2.WND_PROP_VISIBLE) != 0.0:
      #    cv2.destroyAllWindows()
      # cv2.imwrite("Feedback.jpg", self.front_camera)
      # img = cv2.imread("Feedback.jpg", 0)
      # print(self.front_camera.shape)
      cv2.imshow("Feedback", self.front_camera)
      cv2.waitKey(1)


  def close(self):
    cv2.destroyAllWindows()
    

