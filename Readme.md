## Installing Carla
official Documentation: [Link](https://carla.readthedocs.io/en/latest/start_quickstart)
```sh
$ pip install --user pygame numpy && pip3 install --user pygame numpy
$ sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
$ sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"
$ sudo apt-get update
$ sudo apt-get install carla-simulator
$ cd /opt/carla-simulator # Open the folder where CARLA is installed
```

## Running Scripts
```sh
# Run Carla
$ cd /opt/carla-simulator
$ ./CarlaUE4.sh
# Or
$ ./CarlaUE4.sh -quality-level=Low -RenderOffScreen

# Run RL Agent
# DQN
$ python3 DQN-Agent.py

# Show result in tensorboard
$ tensorboard --logdir=log_dir_path

```

## Folder Structure
```
| StableBaseline > Used stable-baseline with gym environment
| DNQ-Agent.py > Custom DQN implementation
| CarlaTest.py > Carla environment test
```