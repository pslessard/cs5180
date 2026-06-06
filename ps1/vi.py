from typing import (Dict, List, Tuple)

import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)

def _q_value(gamma: float, v: Dict[int, float], next_states: List[Tuple[float, int, int, bool]]) -> float:
    def get_expected_value_for_next_state(next_state: Tuple[float, int, int, bool]) -> float:
        probability, s, reward, terminated = next_state
        if terminated:
            return probability * reward

        return probability * (reward + (gamma * v[s]))
    
    return sum(
        get_expected_value_for_next_state(next_state)
        for next_state
        in next_states
    )

def _improvement(new_v: Dict[int, float], old_v: Dict[int, float]):
    max_norm = 0.0
    for i in range(len(new_v)):
        diff = abs(new_v[i] - old_v[i])
        if diff > max_norm:
            max_norm = diff
    
    return max_norm

def _get_greedy_policy(P, v: Dict[int, float], gamma: float) -> Dict[int, int]:
    policy: Dict[int, int] = {}
    for state, actions in P.items():
        max_expected_value: float = -1.0
        for action, result in actions.items():
            expected_value: float = _q_value(gamma, v, result)

            if expected_value > max_expected_value:
                max_expected_value = expected_value
                policy[state] = action
    
    return policy

def value_iteration(P, gamma: float, theta: float, return_policy_convergence_point: bool = False) -> Tuple[Dict[int, float], Dict[int, int], int]:
    V: Dict[int, Dict[int, float]] = {
        0: {s: 0 for s in P}
    }

    i = 0
    while True:
        V[i+1] = {}
        for s, actions in P.items():
            V[i+1][s] = max(
                _q_value(gamma, V[i], result)
                for result
                in actions.values()
            )
               
        if _improvement(V[i+1], V[i]) < (theta * (1 - gamma) / gamma):
            break

        i += 1
    
    V_optimal: Dict[int, float] = V[i]
    optimal_policy: Dict[int, int] = _get_greedy_policy(P, V[i], gamma)

    if return_policy_convergence_point:
        for k in range(i):
            policy = _get_greedy_policy(P, V[k+1], gamma)

            if policy == optimal_policy:
                return V[k+1], policy, k+1
    
    return V_optimal, optimal_policy, i + 1

V, policy, iteration_count = value_iteration(env.unwrapped.P, 0.99, 10**(-4))

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

print()
V_k, policy_k, k = value_iteration(env.unwrapped.P, 0.99, 10**-4, True)
print(f"Policy converged at iteration {k}.")

difference = _improvement(V_k, V)
print(f"||V_k - V*|| = {difference}")
