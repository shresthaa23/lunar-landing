import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import collections
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import os
import time

# hyperparams 
WARMUP_STEPS = 1_000
GAMMA = 0.99
TAU = 0.01
ACTOR_LR = 0.001
CRITIC_LR = 0.002
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 64
NOISE_DECAY = 0.999995
MIN_NOISE = 0.01
NUM_EPISODES = 2_000
NOISE_STD = 0.2

UPDATE = 25
HIDDEN = 128
STATE_DIM = 8
ACTION_DIM = 2
EVAL_EPISODES_PER_SEED = 10
BENCHMARK = 200
TRAIN_SEED = 42
EVAL_SEEDS = [101, 202, 303]

# seeds
random.seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)
torch.manual_seed(TRAIN_SEED)
rng = np.random.default_rng(TRAIN_SEED)
device = torch.device("cpu")

# environment
env = gym.make("LunarLander-v3", continuous=True)
state, info = env.reset(seed=TRAIN_SEED)
env.action_space.seed(TRAIN_SEED)
ACTION_LOW = env.action_space.low
ACTION_HIGH = env.action_space.high
buffer = collections.deque(maxlen=BUFFER_SIZE)

# variable setups
env_steps = 0
grad_updates = 0
noise_std = NOISE_STD
episode_returns = []
episode_lengths = []
batch_prints = []

# actor network
actor = nn.Sequential(
    nn.Linear(STATE_DIM, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, ACTION_DIM),
    nn.Tanh()
)

# critic network
critic = nn.Sequential(
    nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, 1)
)
# targets
target_actor = copy.deepcopy(actor)
target_critic = copy.deepcopy(critic)
actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

# def to fix random batches
def fix_batches(buffer):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, terminated, truncated = zip(*batch)
    f32 = torch.float32
    return (
        torch.as_tensor(np.array(states), dtype=f32),
        torch.as_tensor(np.array(actions), dtype=f32),
        torch.as_tensor(np.array(rewards).reshape(-1, 1), dtype=f32),
        torch.as_tensor(np.array(next_states), dtype=f32),
        torch.as_tensor(np.array(terminated, dtype=np.float32).reshape(-1, 1), dtype=f32),
        torch.as_tensor(np.array(truncated), dtype=f32),
    )

# def returns q value
def q_value(critic_net, states, actions):
    return critic_net(torch.cat([states, actions], dim=1))

# def to move each weight TAU 
def soft_update(target_net, online_net):
    with torch.no_grad():
        for t, o in zip(target_net.parameters(), online_net.parameters()):
            t.copy_((1 - TAU) * t + TAU * o)

# def to choose best action depending on stepcount
def choose_action(actor, state, noise_std, steps):
    if steps < WARMUP_STEPS:
        return env.action_space.sample()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = actor(state_tensor).detach().numpy()[0]
    noise = rng.normal(0, noise_std, size=ACTION_DIM)
    return np.clip(action + noise, ACTION_LOW, ACTION_HIGH)

# main update
def update():
    # critic update
    s, a, r, s_next, term, trunc = fix_batches(buffer)
    with torch.no_grad():
        next_a = target_actor(s_next)
        next_q = q_value(target_critic, s_next, next_a)
        y = r + GAMMA * (1 - term) * next_q
    q = q_value(critic, s, a)
    critic_loss = F.mse_loss(q, y)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    # actor update
    new_a = actor(s)
    actor_loss = -q_value(critic, s, new_a).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # soft update both
    soft_update(target_actor, actor)
    soft_update(target_critic, critic)

    return critic_loss.item(), actor_loss.item()

#################### TRAINING ####################
train_start = time.time()
for episode in range(NUM_EPISODES):
    done = False
    episode_return = 0
    steps = 0

    while not done:
        action = choose_action(actor, state, noise_std, env_steps)
        obs, reward, term, trunc, info = env.step(action)
        buffer.append((state, action, reward, obs, term, trunc))
        episode_return += reward
        env_steps += 1
        steps += 1

        noise_std = max(MIN_NOISE, noise_std * NOISE_DECAY)

        if env_steps > WARMUP_STEPS and env_steps % UPDATE == 0:
            update()
            grad_updates += 1

        state = obs
        if term or trunc:
            done = True

    state, _ = env.reset()
    episode_returns.append(episode_return)
    episode_lengths.append(steps)

    # progress every 50
    if (episode + 1) % 50 == 0:
        statement = f"Episode {episode + 1:4d} | noise: {noise_std:.3f} | env steps: {env_steps:7d} | mean return (last 50): {np.mean(episode_returns[-50:]):7.2f} | {time.time() - train_start:5.0f} s"
        batch_prints.append(statement)
        print(statement)

env.close()
train_time = time.time() - train_start

os.makedirs("results/", exist_ok=True)
torch.save(actor.state_dict(), "results/actor.pt")

#################### EVAL ####################
eval_returns = {}

for seed in EVAL_SEEDS:
    eval_env = gym.make("LunarLander-v3", continuous=True)
    state, info = eval_env.reset(seed=seed)
    returns = []

    for episode in range(EVAL_EPISODES_PER_SEED):
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                action = actor(torch.as_tensor(state, dtype=torch.float32)).numpy()
            obs, reward, term, trunc, _ = eval_env.step(action)
            total_reward += reward
            state = obs
            done = term or trunc

        returns.append(total_reward)
        state, _ = eval_env.reset()

    eval_returns[seed] = returns
    eval_env.close()

#################### RESULTS ####################
all_returns = [r for seed in EVAL_SEEDS for r in eval_returns[seed]]
overall_mean = np.mean(all_returns)

results = (
    "Evaluation: frozen deterministic actor (no exploration noise, no learning)\n"
    f"Training seed: {TRAIN_SEED} | Eval seeds: {EVAL_SEEDS}\n"
    f"Training episodes: {len(episode_returns)} | Env steps: {env_steps} | Gradient updates: {grad_updates}\n"
    f"Training time: {train_time:.0f} s\n\n"
)
for seed in EVAL_SEEDS:
    returns_str = ", ".join(f"{r:.1f}" for r in eval_returns[seed])
    results += f"Seed {seed}: mean {np.mean(eval_returns[seed]):7.2f} | returns: {returns_str}\n"
results += (
    f"\nOverall mean (30 episodes): {overall_mean:.2f}\n"
    f"Exceeds {BENCHMARK}: {'Yes' if overall_mean > BENCHMARK else 'No'}\n"
)
results += "\nTraining progress (every 50 episodes):\n" + "\n".join(batch_prints) + "\n"

print("\n" + results)

with open("results/results.txt", "w") as f:
    f.write(results)


#################### PLOTTING ####################
window = 100
moving_average = np.convolve(episode_returns, np.ones(window) / window, mode="valid")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# training progress
ax1.plot(episode_returns, alpha=0.3, label="Episode return")
ax1.plot(np.arange(window - 1, len(episode_returns)), moving_average, label=f"Average (last {window} episodes)")
ax1.axhline(BENCHMARK, color="green", linestyle="--", label=f"Benchmark ({BENCHMARK})")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Return")
ax1.set_title("Lunar Lander DDPG Training Progress")
ax1.legend()
ax1.grid(True)

# evaluation, colored by seed
for i, seed in enumerate(EVAL_SEEDS):
    x = np.arange(1, EVAL_EPISODES_PER_SEED + 1) + i * EVAL_EPISODES_PER_SEED
    ax2.bar(x, eval_returns[seed], label=f"Seed {seed} (mean {np.mean(eval_returns[seed]):.1f})")
ax2.axhline(overall_mean, color="black", label=f"Overall mean ({overall_mean:.1f})")
ax2.axhline(BENCHMARK, color="green", linestyle="--", label=f"Benchmark ({BENCHMARK})")
ax2.set_xlabel("Eval episode")
ax2.set_ylabel("Return")
ax2.set_title("Evaluation (3 seeds x 10 episodes)")
ax2.legend()
ax2.grid(True)

fig.tight_layout()
fig.savefig("results/training_graph.png")
plt.close(fig)
