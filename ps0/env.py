from typing import (Any, SupportsFloat, Tuple)

import gymnasium
from gymnasium import spaces
import numpy as np

REJECT = 0
ACCEPT = 1

class ApartmentEnv(gymnasium.Env):
    def __init__ (self, T: int, K: int, noise_std: float = 0.0, seed=None):
        self._max_weeks: int = T
        self._max_quality: int = K
        self._noise_std: float = noise_std

        self._week: int = 0
        self._quality: int | np.int64 = 0
        self._observed_quality = 0

        self.action_space = spaces.Discrete(2) # 0 = reject, 1 = accept
        self.observation_space = spaces.Dict(
            {
                "week": spaces.Discrete(self._max_weeks, start=1),
                "quality": spaces.Discrete(self._max_quality, start=1)
            }
        )

        super().reset(seed=seed)

    def _get_obs(self) -> dict[str, float]:
        return {"week": self._week, "quality": self._observed_quality}
    
    def _generate_new_quality(self):
        self._quality = self.np_random.integers(low=1, high=self._max_quality+1)
        self._observed_quality = float(self._quality) + self.np_random.normal(0, scale=self._noise_std)

    def reset(self, seed=None, options=None) -> Tuple[dict[str, float], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self._week = 1
        self._generate_new_quality()

        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[dict[str, float], SupportsFloat, bool, bool, dict[str, Any]]:
        terminated = False
        truncated = False

        if action == ACCEPT:
            terminated = True
            return self._get_obs(), self._quality, terminated, truncated, {}
        
        if self._week == self._max_weeks:
            terminated = True
            return self._get_obs(), 0, terminated, truncated, {}
        
        self._week += 1
        self._generate_new_quality()
        return self._get_obs(), 0, terminated, truncated, {}