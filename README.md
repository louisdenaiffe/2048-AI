# 2048-AI

The goal of this project is to train an AI model to play 2048 using Deep-Q Learning, which is a form of reinforcement learning.
2048 is a sliding-tile puzzle where you merge equal-numbered tiles to create tile 2048.

[![nfy8A8X.md.png](https://iili.io/nfy8A8X.md.png)](https://freeimage.host/i/nfy8A8X)


# Project structure

--> `game.py` : the actual game, reproduced in Python and GUI using Tkinter. If run by itself, playable using keyboard inputs.

--> `environment.py` : the Gymnasium environment class, which imports the Game2048 class from `game.py` and sets the action/observation spaces.

--> `train.py` : for training the AI in the env with stable_baseline3. Using the flags, one can train a new model, continue training an existing one, or use another model name.

--> `evaluate.py` : for testing the AI once it is trained. This is useful for getting metrics which can determine whether the AI has improved or not.

--> `watch_ai.py` : the model plays the game live with a GUI so the user can see what moves it is doing. Very useful for verifying what strategies the AI is following, and to see where to give it a helping hand.
