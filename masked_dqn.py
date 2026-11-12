import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN


# DQN with action masking: the agent can only choose moves that change the board,
# while exploring, while playing, and when computing its learning targets.


def valid_action_mask(observations):
    # observations: a torch tensor of shape (N, 16), N encoded boards (log2 values, 0 = empty cell).
    # Returns a (N, 4) tensor of True/False: is action 0 (up), 1 (down), 2 (left), 3 (right) valid on each board?
    boards = observations.reshape(-1, 4, 4) # back into 4x4 grids. The first dimension (N) is the board number.

    # Every pair of horizontal neighbours: west is the left cell, east is the right cell of each pair.
    west, east = boards[:, :, :-1], boards[:, :, 1:]
    # Every pair of vertical neighbours: north is the upper cell, south is the lower cell of each pair.
    north, south = boards[:, :-1, :], boards[:, 1:, :]

    # Two equal, non-empty neighbours can merge, whichever direction the move goes.
    merge_horizontal = (west == east) & (west != 0)
    merge_vertical = (north == south) & (north != 0)

    # A tile can slide into an empty neighbour cell in the direction of the move.
    # Example for left: the left cell (west) is empty and the right one (east) isn't.
    up = ((north == 0) & (south != 0)) | merge_vertical
    down = ((south == 0) & (north != 0)) | merge_vertical
    left = ((west == 0) & (east != 0)) | merge_horizontal
    right = ((east == 0) & (west != 0)) | merge_horizontal

    # A move is valid if this happens for at least one pair on the board.
    # .flatten(1) merges each board's pairs into one list, .any(dim=1) checks if any of them is True.
    return torch.stack(
        [up.flatten(1).any(dim=1), down.flatten(1).any(dim=1), left.flatten(1).any(dim=1), right.flatten(1).any(dim=1)],
        dim=1,
    )

class MaskedDQN(DQN): # Inherits everything from SB3's DQN, we only replace how actions are chosen and how targets are computed.

    def _choose_actions(self, observation, exploration_rate):
        # Picks one action per board: the valid action with the highest Q-value,
        # or, with probability exploration_rate, a random valid action.
        observation_tensor, vectorized = self.policy.obs_to_tensor(observation)
        # obs_to_tensor also tells us if the input was a batch of boards (vectorized=True) or a single board.
        with torch.no_grad(): # only using the network, not training it
            q_values = self.q_net(observation_tensor) # shape (N, 4): 4 Q-values per board
        mask = valid_action_mask(observation_tensor) # shape (N, 4): True where the move is valid

        q_values[~mask] = -torch.inf # ~ means "not": blocked moves get a Q-value of minus infinity, so argmax never picks them
        actions = q_values.argmax(dim=1).cpu().numpy() # best valid action for each board

        mask = mask.cpu().numpy()
        for i in range(len(actions)): # exploration, board by board
            valid_actions = np.flatnonzero(mask[i]) # indices of the valid actions, e.g. [0, 1, 3]
            if valid_actions.size and np.random.rand() < exploration_rate:
                actions[i] = np.random.choice(valid_actions)
        return actions, vectorized

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # Replaces DQN.predict(). Used by evaluate.py and watch_ai.py, and during training after the warm-up phase.
        exploration_rate = 0.0 if deterministic else self.exploration_rate
        actions, vectorized = self._choose_actions(observation, exploration_rate)
        if not vectorized: # a single board was given, so return a single action, not a list of one
            actions = actions[0]
        return actions, state # state is only used by recurrent policies, we just pass it through like DQN does

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Replaces how SB3 picks actions while collecting experience during training.
        # Before learning_starts (warm-up), the original picks a fully random action, which may be blocked.
        # Here: fully random but valid during warm-up, then epsilon-greedy with masking.
        exploration_rate = 1.0 if self.num_timesteps < learning_starts else self.exploration_rate
        actions, _ = self._choose_actions(self._last_obs, exploration_rate)
        return actions, actions
        # SB3 expects two versions of the action: the one sent to the env, and the one stored in the replay buffer.
        # They only differ for continuous actions, for us they are the same.

    def train(self, gradient_steps, batch_size=100):
        # Copy of SB3's DQN.train() (stable-baselines3 2.9), with one change, marked CHANGED below:
        # the "best Q-value of the next board" only considers moves that are valid on that next board.
        self.policy.set_training_mode(True) # switches the network to training mode
        self._update_learning_rate(self.policy.optimizer) # applies the learning rate schedule

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # A random batch of past experiences: board, action, reward, next board, done.
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with torch.no_grad():
                # The target: what the Q-value of (board, action) should be = reward + gamma * best Q-value of the next board.
                next_q_values = self.q_net_target(replay_data.next_observations) # (batch_size, 4)

                # CHANGED: blocked moves on the next board get -inf, so max() ignores them.
                next_mask = valid_action_mask(replay_data.next_observations)
                next_q_values[~next_mask] = -torch.inf
                next_q_values, _ = next_q_values.max(dim=1)
                # If the next board has no valid move at all (game over), every value was -inf.
                # The target should then just be the reward, so we replace -inf with 0
                # (otherwise -inf * 0 later would give NaN, "not a number", and break the training).
                next_q_values = torch.where(next_mask.any(dim=1), next_q_values, torch.zeros_like(next_q_values))

                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                # (1 - dones) removes the future part when the game ended.

            # What the network currently predicts for the actions that were actually taken.
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # The loss measures how far the predictions are from the targets (Huber loss, less sensitive to outliers).
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Updates the network to reduce the loss.
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm) # limits the size of each update, for stability
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
