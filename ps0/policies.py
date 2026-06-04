from env import (ACCEPT, REJECT)

import numpy as np

class RandomPolicy():
    def __init__(self, T: int, seed=None):
        self._accept_probability = 1/T

        self.np_random = np.random.default_rng(seed=seed)

    def act(self, _: dict[str, int | np.int64]) -> int:
        return self.np_random.choice([ACCEPT, REJECT], p=[self._accept_probability, 1-self._accept_probability])
    
class ThresholdPolicy():
    def __init__(self, u_min: int):
        self._threshold = u_min
    
    def act(self, obs: dict[str, int | np.int64]) -> int:
        assert "quality" in obs

        if obs["quality"] >= self._threshold:
            return ACCEPT
        
        return REJECT
    
class OptimalPolicy():
    def __init__(self) -> None:
        self._policy: dict[int, dict[int | np.int64, int]] = {
            1: {1: REJECT, 2: REJECT, 3: REJECT, 4: ACCEPT},
            2: {1: REJECT, 2: REJECT, 3: REJECT, 4: ACCEPT},
            3: {1: REJECT, 2: REJECT, 3: ACCEPT, 4: ACCEPT},
            4: {1: ACCEPT, 2: ACCEPT, 3: ACCEPT, 4: ACCEPT}
        }
    
    def act(self, obs: dict[str, int | np.int64 | float]) -> int:
        assert "week" in obs and obs["week"] > 0 and obs["week"] <= 4
        assert "quality" in obs

        week = int(obs["week"])
        estimated_quality = round(obs["quality"])
        if estimated_quality < 1:
            estimated_quality = 1
        if estimated_quality > 4:
            estimated_quality = 4    

        return self._policy[week][estimated_quality]