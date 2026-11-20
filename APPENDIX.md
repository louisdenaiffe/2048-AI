# Appendix

Detailed results and settings for the [write-up](WRITE-UP.md).

- [Evaluation method](#evaluation-method)
- [All evaluations](#all-evaluations)
- [What each version changed](#what-each-version-changed)
- [Final agent settings](#final-agent-settings)
- [Glossary](#glossary)

## Evaluation method

Each evaluation plays 100 games with `evaluate.py`, always choosing the move the agent rates best. Each game uses a fixed seed, so every version is tested on exactly the same games. Unless stated otherwise, the games are seeds 0 to 99. Those games were used to make every decision during the project, so the final agent was also tested on seeds 1000 to 1099 and 2000 to 2099, which no decision had used.

With 100 games, a percentage near 50% is uncertain by about 5 points either way, and one near 98% by about 1.5 points, so small differences between versions are noise. The median is often more reliable than the mean, which a few long games can pull up.

## All evaluations

In the order they were run. Steps are training moves. Play is how the agent chose its moves: the network alone, or the network with a search of that many moves. Percentages are the share of games reaching each tile.

| Version | Steps | Play | Mean | Median | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|
| smoke | n/a | Network, no masking | 914 | 858 | 0% | 0% | 0% | 0% |
| smoke | n/a | Network, masking | 1,374 | 1,314 | 0% | 0% | 0% | 0% |
| masked_v1 | 1M | Network | 2,507 | 2,272 | 7% | 0% | 0% | 0% |
| masked_v2 | 1M | Network | 2,547 | 2,102 | 10% | 1% | 0% | 0% |
| onehot_v1 | n/a | Network | 2,572 | 2,682 | 7% | 0% | 0% | 0% |
| big_v1 | n/a | Network | 2,132 | 1,888 | 3% | 0% | 0% | 0% |
| big_v3 | 1M | Network | 3,021 | 3,022 | 13% | 0% | 0% | 0% |
| big_v3 | 2M | Network | 4,219 | 3,324 | 38% | 3% | 0% | 0% |
| big_v4 | 2M | Network | 3,477 | 3,108 | 25% | 0% | 0% | 0% |
| big_v5 | 2M | Network | 3,457 | 3,200 | 21% | 1% | 0% | 0% |
| empty_v1 | 2M | Network | 4,803 | 4,164 | 45% | 2% | 0% | 0% |
| empty_v2 | 4M | Network | 7,413 | 7,406 | 72% | 19% | 0% | 0% |
| double_v1 | 3M | Network | 6,956 | 7,034 | 74% | 10% | 0% | 0% |
| nstep_v1 | 2M | Network | 6,301 | 6,090 | 65% | 6% | 0% | 0% |
| nstep_v1 | 3M | Network | 6,674 | 6,978 | 69% | 10% | 0% | 0% |
| eps_v1 | 2M | Network | 5,686 | 5,408 | 58% | 7% | 0% | 0% |
| eps_v1 | 2M | 1-move search | 12,810 | 12,300 | 99% | 64% | 8% | 0% |
| empty_v2 | 4M | 2-move search | 22,736 | 17,120 | 100% | 96% | 45% | 0% |
| lookahead_v1 | 2M | 2-move search | 29,584 | 31,656 | 100% | 100% | 75% | n/a |
| lookahead_v2 | 2M | 2-move search | 25,108 | 27,260 | 98% | 94% | 61% | 4% |
| lookahead_v1 | 3M | Adaptive, 2 to 3 moves | 41,207 | 36,250 | 100% | 100% | 95% | 30% |
| lookahead_v1 | 3M | Adaptive, 2 to 4 moves (seeds 1000-1099) | 51,222 | n/a | 100% | 100% | 98% | 54% |
| lookahead_v1 | 3M | Deeper adaptive, 2 to 4 moves (seeds 2000-2099) | n/a | n/a | n/a | n/a | 97% | n/a |

The run on seeds 1000 to 1099 was split into four runs of 25 games; their means combine, their medians do not. For seeds 2000 to 2099, only the games that missed 2048 were recorded. The deeper setting used 4 moves at 3 or fewer empty cells and 3 moves at 6 or fewer, instead of 2 and 4.

## What each version changed

| Version | Change | Result |
|---|---|---|
| smoke | First DQN: raw tile values as input, merge points as reward | Never reached 512; 88% of games hit the move limit |
| masked_v1 | Action masking in training | Mean from 914 to 2,507 |
| masked_v2 | Log2 reward | Similar mean, first 1,024 tile |
| onehot_v1 | One-hot input (256 numbers) | Median from 2,102 to 2,682 |
| big_v1 | Two hidden layers of 256 neurons | Worse at first (mean 2,132) |
| big_v2 | Learning curves during training (20 games per evaluation) | Plateau after about 300,000 moves, below the corner-rule baseline |
| big_v3 | 100-game evaluations, longer exploration, 500,000-move replay buffer | Steadier learning curve; mean 3,021 after 1M moves, the best so far |
| big_v4, big_v5 | Decaying learning rate, target network updated every 30,000 moves | No gain: tuning did not raise the ceiling |
| empty_v1 | Reward bonus of 0.1 per empty cell | Median from about 3,200 to 4,164 |
| empty_v2 | Same, constant learning rate, 4M moves | Best version without search (median 7,406) |
| double_v1 | Double DQN | No measurable change; kept as standard practice |
| nstep_v1 | 3-step returns | Faster learning, same plateau; removed |
| eps_v1 | Final exploration of 1% instead of 5% | No measurable change |
| lookahead_v1 | 1-move search choosing the moves during training | Best version: 75% reach 2048 with a 2-move search |
| lookahead_v2 | 2-move search during training | Worse: 61% |

## Final agent settings

| Setting | Value |
|---|---|
| Network input | The board as 256 numbers: 16 cells, each with 16 slots (one-hot) |
| Network | Two hidden layers of 256 neurons (ReLU), one output per move (Q-values) |
| Reward | Log2 of each merged tile, plus 0.1 per empty cell |
| Training length | 3 million moves (2 million, then continued for 1 million) |
| Moves during training | Chosen by a 1-move search, apart from random exploration moves |
| Exploration | 100% random moves at first, down to 1% over the first 25% of training |
| Replay buffer | Last 500,000 moves; learning starts after 10,000 |
| Updates | Every 4 moves, batch of 128, Adam optimizer, constant learning rate of 0.0001 |
| Discount factor | 0.99 |
| Target network | Double DQN, copy updated every 30,000 moves |
| Invalid moves | Masked when playing and in training targets |
| Search when playing | 2 moves with 5 or more empty cells, 3 moves with 3 or 4, 4 moves with 2 or fewer |

Built with Python, Gymnasium, Stable-Baselines3 (PyTorch) and NumPy. All training and evaluation ran on a laptop CPU.

## Glossary

**Q-value.** The network's estimate of the total reward the agent will collect from a board if it plays a given move, then keeps playing well. Rewards further in the future count a bit less (× 0.99 per move, the discount factor *gamma*).

**ReLU.** The function applied by each hidden neuron to its result: `ReLU(x) = max(0, x)`. It keeps positive numbers and turns negative ones into 0. Without it, stacking layers would be pointless: a chain of linear layers is equivalent to a single linear layer, which can only represent very simple relationships ("each tile adds or removes a fixed amount of value"). Cutting at 0 lets each neuron act like a switch that only activates for certain patterns (e.g. "big tile in the corner and the row next to it full"), and combining many such switches lets the network represent complex judgments of a board.

**Adam.** The method used to adjust the network's weights during training. Each update, the error of the network's Q-values gives, for each weight, a direction that would reduce that error (the gradient). Plain gradient descent moves every weight a fixed small step in that direction. Adam also keeps, for each weight, a running average of its recent gradients (momentum: it keeps going in consistent directions and ignores noise) and of their size (each weight gets its own step size). This makes training faster and more stable, which matters in DQN because the targets themselves are noisy estimates.

**Expectimax.** A search for games with chance: on the agent's turns it takes the **max** (the agent picks its best move), and on the random tile's turns it takes the **expected** value (the probability-weighted average of what can appear).
