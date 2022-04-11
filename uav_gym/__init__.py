from gym.envs.registration import register

register(
    id='uav_env-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='uav_gym.envs:uavEnv',              # Expalined in envs/__init__.py
    max_episode_steps=160,
)
