
from envs.path_plan_env import PathPlanEnv
from gym.envs.registration import register

register(
    id='envs/PathPlanEnv-v0',
    entry_point='envs:PathPlanEnv',
    max_episode_steps=1000,
)
