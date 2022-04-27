import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import envs
from envs.path_plan_env import PathPlanEnv

start = np.array([60, 120])
goal = np.array([490, 440])
env: PathPlanEnv = gym.make("envs/PathPlanEnv-v0", file="Map_1_obs.png", start=start, goal=goal)

obs = env.reset()
imshow(obs["map"], cmap="gray")
plt.show()

done = False
action_map = {"d": 0, "r": 1, "u": 2, "l": 3}
while not done:
    action = input("Enter an action (u/d/l/r):")
    action = action_map[action]
    obs, reward, done, _ = env.step(action)
    print("Reward:", reward)
    imshow(obs["map"], cmap="gray")
    plt.show()
