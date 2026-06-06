from vi import value_iteration
from pi import policy_iteration

import datetime
from typing import (Dict, List, Tuple)

import gymnasium as gym
import pandas as pd
import plotly.express as px

env = gym.make("FrozenLake-v1", is_slippery=True)

num_states = len(env.unwrapped.P)

data = []
for gamma in [0.5, 0.9, 0.99, 0.999]:
    vi_start = datetime.datetime.now()
    _, _, vi_iteration_count = value_iteration(env.unwrapped.P, gamma, 10**-4)
    vi_end = datetime.datetime.now()

    data.append(
        {
            "Gamma": gamma,
            "Algorithm": "Value Iteration",
            "Time (microseconds)": (vi_end - vi_start).microseconds,
            "Iterations": vi_iteration_count,
            # |S| backups per iteration plus |S| backups for the policy extraction
            "Bellman Backups": num_states * (vi_iteration_count + 1)
        })

    pi_start = datetime.datetime.now()
    _, _, pi_iteration_count = policy_iteration(env.unwrapped.P, gamma)
    pi_end = datetime.datetime.now()

    data.append(
        {
            "Gamma": gamma,
            "Algorithm": "Policy Iteration",
            "Time (microseconds)": (pi_end - pi_start).microseconds,
            "Iterations": pi_iteration_count,
            # It's doesn't seem to make sense to count the number of Bellman
            # backups performed by policy iteration when we're doing the
            # policy iteration by solving a system of linear equations. While
            # we know that is O(|S|^3), it's not clear how to map that to
            # a number of Bellman backups.
            # Since I counted the O(|S|^2*|A|) work per iteration performed
            # by value iteration as |S| Bellman backups per iteration, I'll
            # count this as |S|^2 backups per iteration, but it's not clear
            # if that is really an accurate representation
            #
            #|S|^2 backups per iteration for evalutation and |S| backups per
            # iteration for policy improvement
            "Bellman Backups": (num_states**2 + num_states) * pi_iteration_count
        })

print(pd.DataFrame(data))

# fig = px.line(data, x="Gamma", y="Iterations", color="Algorithm")
fig = px.line(data, x="Gamma", y="Bellman Backups", color="Algorithm")
# fig.write_image("iteration_counts.png")
fig.show()
