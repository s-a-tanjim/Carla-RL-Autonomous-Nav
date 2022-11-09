# Carla RL With Stable-Baseline

## Install Dependencies
```sh
$ pip3 install stable-baselines3
$ pip3 install gym
$ pip3 install numpy
```

## Running Commands
```sh
# Open carla folder
$ cd /opt/carla-simulator

# Run Carla simulator
$ ./CarlaUE4.sh
# Or run in low res mode
$ ./CarlaUE4.sh -quality-level=Low
# Or run without rendering
$ ./CarlaUE4.sh -quality-level=Low -RenderOffScreen


# Run Script
# Check Env if needed
$ python3 env_checker.py

# Run RL training with PPO/DDPG algorithm
$ python3 main.py
```

## Requirements
1. Install Carla [Link](https://carla.readthedocs.io/en/latest/start_quickstart/)
2. Install Python dependencies
