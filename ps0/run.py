from env import ApartmentEnv
from policies import (OptimalPolicy, RandomPolicy, ThresholdPolicy)

import numpy as np
import pandas as pd
import plotly.express as px

NUM_EPISODES = 10000

def run_episode(env, policy) -> int | np.int64:
    obs, _ = env.reset()

    final_reward = 0
    terminated = False

    while not terminated:
        action = policy.act(obs)

        obs, reward, terminated, _, _ = env.step(action)

        # reward will only ever be nonzero in a terminal state, so we can just overwrite
        final_reward = reward
    
    return final_reward

def evaluate_policy(policy, std_dev):
    rewards = []

    env = ApartmentEnv(4, 4, std_dev)

    for _ in range(NUM_EPISODES):
        reward = run_episode(env, policy)
        rewards.append(reward)
    
    env.close()
    
    return pd.Series(rewards, dtype=np.int64)

policies = {
    "Random": {"policy": RandomPolicy(4)},
    "Threshold_1": {"policy": ThresholdPolicy(1)},
    "Threshold_2": {"policy": ThresholdPolicy(2)},
    "Threshold_3": {"policy": ThresholdPolicy(3)},
    "Threshold_4": {"policy": ThresholdPolicy(4)},
    "Optimal": { "policy": OptimalPolicy()}
}

dfs = []
summary = []
for name, policy in policies.items():
    for std_dev in [0.0, 0.5, 1.0, 2.0]:
        rewards = evaluate_policy(policy["policy"], std_dev)

        result = {"name": name, "std_dev": std_dev}
        result["mean"] = rewards.mean()
        result["std_error"] = rewards.sem()
        result["full_rejection_ratio"] = rewards[rewards == 0].size / NUM_EPISODES
        summary.append(result)

        policy_df = pd.DataFrame(rewards, columns=["reward"])
        policy_df = policy_df.assign(policy=name, std_dev=std_dev)
        dfs.append(policy_df)

summary_df = pd.DataFrame(summary)
print(summary_df)

df = pd.concat(dfs, ignore_index=True)
fig = px.histogram(df, x="reward", facet_col="policy", facet_row="std_dev", color="reward")
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("policy=", "")))
fig.write_image("histogram_noise.png")
# fig.show()