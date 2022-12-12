from gym.envs.registration import register
from bus_bunch.environments import *

register(
    id='Busbunch-v0',
    entry_point='bus_bunch:Env',
)
# register(
#     id='Sliding-v0',
#     entry_point='gym_hybrid:SlidingEnv',
# )