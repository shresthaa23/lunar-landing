# lunar-landing-reinforce-learning
Training an artificial agent to land Gymnasium's `LunarLander-v3` (`continuous=True`) using Deep Deterministic Policy Gradient (DDPG). The landing machine has 8 continuous values (position, velocity, angle, angular velocity, leg contacts) and has 2 continuous engines [-1, 1] (main and side engines). The goal is to land softly between the flags of the planet. 

The code trains an actor and critic network for 2000 episodes with a replay buffer, target network for softer updates, and decaying Gaussian noise. Once training is completed, the frozen actor is evaluated with 0 noise in 3 fresh environments with 10 episodes each. The results are printed in the `results/` folder and comes with a graph of the training process and evaluation returns. The actor weights are saved to `results/actor.pt`.

## To Run

```
uv venv
uv pip install -r requirements.txt
uv run src/lunar/project3.py
```

## Setup

| | |
|---|---|
| Python | 3.12.3 |
| gymnasium | 1.4.0 (with box2d 2.3.10) |
| torch | 2.15.0 (run on CPU) |
| numpy | 2.5.5 |
| matplotlib | 3.11.2 |
| Hardware | Intel Core i7-1360P laptop CPU (16 threads), 7.6 GB RAM, Ubuntu on WSL2, no GPU |
| Runtime | about 8-10 min of compute end-to-end |

Other downloaded items in the requirements are installed automatically along with the packages above.

## Given Hyperparameters

All starting settings from the project description were used unchanged.

| warmup_steps | gamma | tau | actor_lr | critic_lr | buffer_size | batch_size | noise_std | noise_decay | min_noise | update_every | num_episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0.99 | 0.01 | 0.001 | 0.002 | 1,000,000 | 64 | 0.2 | 0.999995 | 0.01 | 25 | 2000 |

## Random Seeds

- Training seed - 42
- Evaluation seeds - 101, 202, 303 (3 fresh environments)

## High Level Architecture

Built with PyTorch:

- **Actor** - 8 inputs → 128 ReLU → 128 ReLU → 2 outputs with tanh
- **Critic** - 8 state + 2 action values concatenated (10 inputs) → 128 ReLU → 128 ReLU → 1 linear output (Q-value)
- **Target networks** - exact copies of the actor and critic at the start, then soft updated - not directly with gradient descent.
- **Optimizers** - Adam optimizer for both actor and critic, learning rates .001 and .002 respectively

## Evaluation Results

Frozen actor on 3 seeds × 10 episodes:

| Seed | Mean return |
|---|---|
| 101 | -41.59 |
| 202 | -49.25 |
| 303 | -48.75 |
| **Overall (30 episodes)** | **-46.53** |

Exceeds 200 benchmark? **No.** 

## Results / Graph Explanation
**Training (top panel):** Returns are very low at the start, around -250 and can spike down to -800 from episodes 0-180. This is from the random warmup and the start of learning when random noise and effects are expected. From episode 180 to 300, the average climbs up quickly to -50 and the agent stops crashing hard. From then on, the average stays between roughly -20 to -90. Some of the middle-stage episodes get close to 1000 full steps, suggesting that the agent is learning to hover safely and burn fuel, avoiding the penalty but never getting the landing bonus.

**Evaluation (bottom panel):** The frozen policy is consistent across all 3 environments, with only 2 of the 30 being positive and the rest hovering around -20 to -80. This avoids crashing reliably but doesn't finish the landings successfully. This returns an overall mean of -46.53 which is well below the benchmark 200. 

**Why low score**
- **Few updates.** One update every 25 environment steps may not be enough updates for over a million environment steps. The critic and actor get little training compared to experience. 
- **Hovering local.** Hovering avoids the -100 crash penalty, so the critic learns that that is a safe position, and doesn't expand. The noise is tiny by the time the policy tries landing, so the agent doesn't discover the reward.
- **One run, final policy.** The policy evaluated is just the last episode, not necessarily the best one seen.

**Ways to improve**
- Update more often instead of every 25 steps to best use experience
- Slower noise decay or a higher minimum noise
- Training for more episodes and with many training seeds to check robustness
