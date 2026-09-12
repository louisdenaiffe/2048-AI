from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from environment import Game2048Env

env = Monitor(Game2048Env())

model = DQN.load("dqn_2048", env=env)
model.learn(
    total_timesteps=5_000_000,
    reset_num_timesteps=False,
    progress_bar=True
)
model.save("dqn_2048")

env.close()