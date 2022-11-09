import os
from CarEnv import CarEnv
import time
import numpy as np
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# ALGORITHM_TYPE = "ddpg"
ALGORITHM_TYPE = "ppo"

# with open(f"{ALGORITHM_TYPE}/hyperparams.yaml", "r") as stream:
# 	try:
# 		hyperparams = yaml.safe_load(stream)
# 	except yaml.YAMLError as exc:
# 		print(exc)

# print(hyperparams["policy"])

models_dir = f"models/{ALGORITHM_TYPE}/{int(time.time())}/"
logdir = f"logs/{ALGORITHM_TYPE}/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)



if ALGORITHM_TYPE=='ddpg':
	env = CarEnv(action_space_type="continious")
	env.reset()
	n_actions = env.action_space.shape[-1]
	action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

	model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log=logdir, buffer_size=100)

elif ALGORITHM_TYPE=='ppo':
	env = CarEnv(action_space_type="discrete")
	env.reset()
	model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 50 # 5000
MAX_ITERS = 20 # 2000

iters = 0
while iters < MAX_ITERS:
	try:
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{ALGORITHM_TYPE}")
		model.save(f"{models_dir}/{TIMESTEPS*iters}")
	
	except KeyboardInterrupt:
		print("Exit!")
		break

	except:
		break