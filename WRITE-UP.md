# Teaching an AI to play 2048

*From a model that never reached 512 to an agent that reaches 2048 in 98% of games*

## Summary

The final agent reaches the 2048 tile in 98% of 100 games it had never seen, and the 4096 tile in 54% of them, with a mean score of about 51,000. It has two parts: a neural network trained with Deep Q-Learning (DQN) that judges how good a board is, and an expectimax search that uses those judgments to plan up to four moves ahead.

The largest gains came from changing what the agent learns (its reward) and how it decides (the search), while tuning hyperparameters barely moved the results. And careful measurement (learning curves, medians, and new games kept aside for the final test) stopped me from mistaking luck for progress several times.

![Mean score of the main versions, from the first model to the final agent](figures/2_progression.png)

*Figure 1. Mean score of the main versions, from the first model to the final agent. Blue: the network plays alone. Orange: the network plus the search.*

## The setup

2048 is played on a 4×4 grid. Each move slides every tile in one direction, equal tiles merge, and a new tile (a 2 with 90% probability, a 4 with 10%) appears in a random empty cell. The game ends when no move is possible.

In reinforcement learning, an agent learns by playing. After each move it receives a reward, and it learns to estimate, for each possible move, the total reward it will collect afterwards: the Q-value. In DQN, a neural network computes the Q-values of the four moves from the board, and is trained on millions of past moves stored in a replay buffer. I used Python, Gymnasium for the game environment and Stable-Baselines3 (PyTorch) for DQN, all on a laptop CPU.

## Getting the basics right

The first model scored 914 points on average and never reached 512, and more training made no difference. Instead of guessing why, I added counters to the evaluation. They showed that 88% of games were not lost but cut off by the move limit, with 8,714 blocked moves per game on average: the agent kept choosing moves that did nothing.

- **Action masking:** invalid moves can no longer be chosen. Applied to the same model at play time, it raised the mean from 914 to 1,374. Retraining with masking (masked_v1) took it to 2,507, with 49% of games reaching 256 instead of 4%.
- **Log2 reward:** merge points range from 4 to several thousand, which makes Q-values hard to predict. Rewarding the log2 of each merged tile keeps them in a small range.
- **One-hot input:** each cell is given as 16 slots (empty, 2, 4, up to 32,768) instead of a single number, so the network does not have to work out from raw values that two tiles are equal. The median rose from 2,102 to 2,682.

## Measuring progress reliably

A bigger network (two layers of 256 neurons, big_v1) first did worse, and a single final score could not tell me why. So I added evaluations during training: every 100,000 moves the model plays a fixed set of games, which gives a learning curve. I later raised these evaluations from 20 to 100 games, because 20 were too noisy to compare runs.

The curves changed several decisions. big_v3 scored 4,219 after 2 million moves, but two runs with a decaying learning rate and slower target updates (big_v4 and big_v5) scored about 3,460, and the three medians were within about 200 points of each other (3,324, 3,108 and 3,200). The 4,219 came mostly from a few lucky games on the final checkpoint. From then on I compared medians as well as means, and concluded that tuning was not raising the ceiling.

## Changing what the agent learns

The first change that raised the ceiling was the reward. I added a bonus of 0.1 per empty cell after each move, since empty cells are what keep a game alive. The median rose from about 3,200 to 4,164 at 2 million moves (empty_v1). With a constant learning rate and 4 million moves of training, empty_v2 reached a median of 7,406, with 72% of games reaching 512 and 19% reaching 1,024.

![Learning curves of five versions playing without search](figures/1_plateau.png)

*Figure 2. Learning curves of five versions playing without search (100 games every 100,000 moves). The empty-cell bonus (empty_v1, empty_v2) moved the curve up; Double DQN and n-step returns level off at the same 7,000 to 7,500.*

## The plateau

I then tried three standard DQN improvements:

- **Double DQN,** which reduces the overestimation of Q-values: a median of 7,034 at 3 million moves (double_v1), the same as empty_v2 at that point.
- **3-step returns,** which pass the consequences of a move back to it faster: faster learning, but no higher plateau (a median of 6,978 at 3 million moves).
- **Less random exploration** at the end of training (1% instead of 5%): no measurable difference.

All three ended in the same 7,000 to 7,800 range. They changed how fast the network learned, not what it learned.

## Adding search

A network that only looks at the current board makes short-term mistakes, such as a move that leaves the board one tile away from game over. Expectimax search corrects them. Before each move, the agent tries all four moves. For each one, it lists every possible new tile (each empty cell, with a 2 or a 4), scores each resulting board with the network or searches one move deeper from it, and plays the move with the best probability-weighted average.

The search only needs the network's judgment of boards, so it works on top of any trained model. With a 1-move search, eps_v1 went from 5,686 to 12,810 points and reached 2048 in 8% of games. With a 2-move search, empty_v2 went from 7,413 to 22,736 points and reached 2048 in 45% of games, against 0% without search.

## Searching during training

Without search, training games rarely reached the late game (1,024 in under 20% of games), so the network had little experience of the crowded boards where games are decided. In lookahead_v1, the moves played during training are chosen by a 1-move search instead of the network alone. Training games last longer, and the network learns from many more late-game boards. With a 2-move search at play time, it reached 2048 in 75% of games (mean 29,584).

This has a side effect. The network is only trained on the moves the search picks, so its Q-values for the other moves drift, and on its own it scores only about 1,500 points. Its judgment of how good a board is keeps improving, and that is the only part the search uses. Training with a deeper 2-move search (lookahead_v2) made things worse, 61% instead of 75%, probably because the training data then covered an even narrower set of moves.

## Adaptive depth and final results

Each extra level of search multiplies the number of boards to evaluate by about 50, so the depth depends on how crowded the board is: 2 moves when 5 or more cells are empty, 3 moves when 3 or 4 are empty, and 4 moves when 2 or fewer are empty. Games are lost on crowded boards, and those boards have fewer possible new tiles, so a deep search there stays affordable.

![Share of games reaching 2048 by search depth](figures/3_search_depth.png)

*Figure 3. Share of games reaching 2048 for lookahead_v1 by search depth, with the time per game on a laptop CPU.*

Every decision so far had been made on the same 100 games (seeds 0 to 99), so results on them could be partly luck. The final agent was therefore tested on 100 new games (seeds 1000 to 1099).

| Agent | Games | Mean score | Reached 2048 | Reached 4096 |
|---|---|---|---|---|
| empty_v2, network alone | Seeds 0-99 | 7,413 | 0% | 0% |
| lookahead_v1, 2-move search | Seeds 0-99 | 29,584 | 75% | n/a |
| lookahead_v1, adaptive search (2 to 3 moves) | Seeds 0-99 | 41,207 | 95% | 30% |
| lookahead_v1, adaptive search (2 to 4 moves) | Seeds 1000-1099 | 51,222 | 98% | 54% |

The two failed games both ended at 1,024. A deeper setting fixed those two games, but on another 100 new games (seeds 2000 to 2099) it reached 2048 in 97%, no better, at about four times the cost. Fixing specific games had only fitted those games, so I kept the faster setting.

## What I learned

The most useful thing I did was add two counters to the evaluation. For weeks, I had assumed the model just needed more training. The counters showed that most games weren't being lost at all: they ran into the move limit because the agent kept pressing keys that did nothing. This could not have been fixed through more training.

I also learned not to trust a single score. big_v3 looked like a clear improvement until two similar runs came in about 750 points lower and I compared medians instead of means. Learning curves, and testing the final agent on games I had never used, kept me honest.

Most of the standard tricks I tried (tuning the learning rate, Double DQN, n-step returns) didn't change much. Every version ended up around the same 7,000 points. The two changes that actually mattered were the empty-cell bonus and the search, because they changed what the agent was learning and how it chose its moves.

Finally, I had to accept when to stop. Going from a 2-move search to an adaptive 3-move search took the 2048 rate from 75% to 95%, but searching even deeper made no difference, and one evaluation run took eight hours to tell me that.

## Limitations

98% is not 100%: about 2 to 3 games in 100 still end at 1,024. The search does much of the work, and with up to 4 moves of search a game takes about 3 minutes on a laptop CPU. The next improvements would be a network that judges crowded boards better, and a faster search, since many boards are reached through several move orders and are currently evaluated more than once.

Full results for every version are in the [Appendix](APPENDIX.md).
