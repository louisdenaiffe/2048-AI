# 2048-AI

The goal of this project is to train an AI model to play 2048 using Deep-Q Learning, which is a form of reinforcement learning.
2048 is a sliding-tile puzzle where you merge equal-numbered tiles to create tile 2048.

[![nfy8A8X.md.png](https://iili.io/nfy8A8X.md.png)](https://freeimage.host/i/nfy8A8X)

# Table of contents :
 - [Project structure](#project-structure)
 - [Q-Learning](#q-learning-)
 - [Evolution of the model's capabilities](#evolution-of-the-models-capabilities)


## Project structure

--> `game.py` : the actual game, reproduced in Python and GUI using Tkinter. If run by itself, playable using keyboard inputs.

--> `environment.py` : the Gymnasium environment class, which imports the Game2048 class from `game.py` and sets the action/observation spaces.

--> `train.py` : for training the AI in the env with stable_baseline3. Using the flags, one can train a new model, continue training an existing one, or use another model name.

--> `evaluate.py` : for testing the AI once it is trained. This is useful for getting metrics which can determine whether the AI has improved or not.

--> `watch_ai.py` : the model plays the game live with a GUI so the user can see what moves it is doing. Very useful for verifying what strategies the AI is following, and to see where to give it a helping hand.

## Q-Learning :
1) Observes the 2048 board.
2) Chooses a direction.
3) Receives a reward—merge points or -2 for blocked moves.
4) Stores the experience in its replay buffer.
5) Learns which actions tend to produce higher future rewards.

## Evolution of the model's capabilities:

 - Trained it for 500k steps. Results:
 ---

    - Games: 100
    - Mean score: 817.56
    - Median score: 754.0
    - Best score: 2128 
    - Mean largest tile: 87.64 
    - Best tile: 256 
    - Reached 64: 85.0%
    - Reached 128: 42.0%
    - Reached 256: 2.0% 
    - Reached 512: 0.0%
    - Reached 1024: 0.0%
    - Reached 2048: 0.0%

 ---
- Trained it for 1M steps. 
---
    - Games: 100
    - Mean score: 1164.32
    - Median score: 1150.0
    - Best score: 2804
    - Mean largest tile: 121.6
    - Best tile: 256
    - Reached 64: 91.0%
    - Reached 128: 67.0%
    - Reached 256: 14.0%
    - Reached 512: 0.0%
    - Reached 1024: 0.0%
    - Reached 2048: 0.0%

---

 - Trained it for 5M steps. Conclusion: the AI got even worse.
 Output:
---

    - Games: 100
    - Mean score: 1040.68
    - Median score: 1062.0
    - Best score: 2500
    - Mean largest tile: 104.7
    - Best tile: 256
    - Reached 64: 90.0%
    - Reached 128: 60.0%
    - Reached 256: 5.0%
    - Reached 512: 0.0%
    - Reached 1024: 0.0%
    - Reached 2048: 0.0%

---
 - Next change to be made: the reward function. Thanks to `watch_ai.py`, I've noticed the AI just presses keys in a circular motion, without any sense of what's really going on.