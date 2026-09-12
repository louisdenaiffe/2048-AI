from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from environment import Game2048Env


env = Monitor(Game2048Env())


model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0001,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=128,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
    device="auto"
)


model.learn(total_timesteps=200_000)
model.save("dqn_2048")


env.close()