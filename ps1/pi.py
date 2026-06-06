from typing import (Dict, List, Tuple)

import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)

def _get_matrices(P, policy: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    probability_list: List[List[float]] = []
    reward_list: List[float] = []
    for state, possible_actions in P.items():
        action = policy[state]

        probabilities_for_state: List[float] = [0.0 for _ in range(len(P))]
        reward_for_state: float = 0.0
        for probability, next_state, reward, _ in possible_actions[action]:
            probabilities_for_state[next_state] += probability
            reward_for_state += probability * reward

        probability_list.append(probabilities_for_state)
        reward_list.append(reward_for_state)
    
    return np.array(probability_list), np.array(reward_list)

def _q_value(gamma: float, v: np.ndarray, next_states: List[Tuple[float, int, int, bool]]) -> float:
    def get_expected_value_for_next_state(next_state: Tuple[float, int, int, bool]) -> float:
        probability, s, reward, terminated = next_state
        if terminated:
            return probability * reward

        return probability * (reward + (gamma * v[s]))
    
    return sum(get_expected_value_for_next_state(next_state) for next_state in next_states)

def _get_best_action(possible_actions, V: np.ndarray, gamma: float) -> int:
    max_expected_value: float = -1.0
    best_action: int = -1

    for action, possible_results in possible_actions.items():
        expected_value: float = _q_value(gamma, V, possible_results)
        if expected_value > max_expected_value:
            max_expected_value = expected_value
            best_action = action

    return best_action

def policy_iteration(P, gamma: float) -> Tuple[Dict[int, float], Dict[int, int], int]:
    num_states = len(P)
    I = np.identity(num_states)
    
    # initialize policy[0] to down everywhere
    policy: List[Dict[int, int]] = [{state: 1 for state in P}]
    V: List[np.ndarray] = []

    i = 0
    while True:
        p, r = _get_matrices(P, policy[i])
        V.append(np.linalg.solve((I - (gamma * p)), r))

        new_policy: Dict[int, int] = {}
        for state, possible_actions in P.items():
            new_policy[state] = _get_best_action(possible_actions, V[i], gamma)
        
        if new_policy == policy[i]:
            break

        policy.append(new_policy)
        i += 1

    V_optimal = {
        state: value
        for state, value
        in enumerate(V[i].tolist())
    }
    return V_optimal, policy[i], i + 1


V, policy, iteration_count = policy_iteration(env.unwrapped.P, 0.99)

print("V*:")
print("+--------"*4+"+")
for row in range(4):
    line = "|"
    for col in range(4):
        line += f" {V[row*4 + col]:.4f} |"
    
    print(line)
    print("+--------"*4+"+")

print()

def action_to_string(action: int) -> str:
    if action == 0:
        return "LEFT "
    elif action == 1:
        return "DOWN "
    elif action == 2:
        return "RIGHT"
    else:
        return " UP  "

print("pi*:")
print("+-------"*4+"+")
for row in range(4):
    line = "|"
    for col in range(4):
        action = "     "
        if row*4 + col in policy:
            action = action_to_string(policy[row*4 + col])

        line += f" {action} |"
    
    print(line)
    print("+-------"*4+"+")

print()
print(f"Iteration count: {iteration_count}")
