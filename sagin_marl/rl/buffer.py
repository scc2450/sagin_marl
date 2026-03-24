from __future__ import annotations

from typing import List, Optional

import numpy as np


class RolloutBuffer:
    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = int(capacity) if capacity is not None else None
        self._use_list = self.capacity is None
        self._idx = 0

        self._obs: Optional[np.ndarray] = None
        self._actions: Optional[np.ndarray] = None
        self._logprobs: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._terminated: Optional[np.ndarray] = None
        self._truncated: Optional[np.ndarray] = None
        self._global_states: Optional[np.ndarray] = None
        self._next_global_states: Optional[np.ndarray] = None
        self._imitation: Optional[np.ndarray] = None
        self._danger_imitation_target: Optional[np.ndarray] = None
        self._danger_imitation_mask: Optional[np.ndarray] = None
        self._sat_indices: Optional[np.ndarray] = None

        if self._use_list:
            self.obs: List[np.ndarray] = []
            self.actions: List[np.ndarray] = []
            self.logprobs: List[np.ndarray] = []
            self.rewards: List[float] = []
            self.values: List[float] = []
            self.terminated: List[bool] = []
            self.truncated: List[bool] = []
            self.global_states: List[np.ndarray] = []
            self.next_global_states: List[np.ndarray] = []
            self.imitation: List[np.ndarray] = []
            self.danger_imitation_target: List[np.ndarray] = []
            self.danger_imitation_mask: List[np.ndarray] = []
            self.sat_indices: List[np.ndarray] = []

    def _allocate(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        global_state: np.ndarray,
        next_global_state: np.ndarray,
        imitation: np.ndarray,
        danger_imitation_target: np.ndarray,
        danger_imitation_mask: np.ndarray,
        sat_indices: np.ndarray,
    ) -> None:
        if self.capacity is None:
            return
        cap = self.capacity
        self._obs = np.empty((cap,) + obs.shape, dtype=np.float32)
        self._actions = np.empty((cap,) + actions.shape, dtype=np.float32)
        self._logprobs = np.empty((cap,) + logprobs.shape, dtype=np.float32)
        self._rewards = np.empty((cap,), dtype=np.float32)
        self._values = np.empty((cap,), dtype=np.float32)
        self._terminated = np.empty((cap,), dtype=np.float32)
        self._truncated = np.empty((cap,), dtype=np.float32)
        self._global_states = np.empty((cap,) + global_state.shape, dtype=np.float32)
        self._next_global_states = np.empty((cap,) + next_global_state.shape, dtype=np.float32)
        self._imitation = np.empty((cap,) + imitation.shape, dtype=np.float32)
        self._danger_imitation_target = np.empty((cap,) + danger_imitation_target.shape, dtype=np.float32)
        self._danger_imitation_mask = np.empty((cap,) + danger_imitation_mask.shape, dtype=np.float32)
        self._sat_indices = np.empty((cap,) + sat_indices.shape, dtype=np.int64)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        reward: float,
        value: float,
        terminated: bool,
        truncated: bool,
        global_state: np.ndarray,
        next_global_state: np.ndarray,
        imitation: np.ndarray | None = None,
        danger_imitation_target: np.ndarray | None = None,
        danger_imitation_mask: np.ndarray | None = None,
        sat_indices: np.ndarray | None = None,
    ) -> None:
        if imitation is None:
            imitation = np.zeros_like(actions, dtype=np.float32)
        if danger_imitation_target is None:
            danger_imitation_target = np.zeros((actions.shape[0], 2), dtype=np.float32)
        if danger_imitation_mask is None:
            danger_imitation_mask = np.zeros((actions.shape[0], 2), dtype=np.float32)
        if sat_indices is None:
            sat_indices = np.full((actions.shape[0], 0), -1, dtype=np.int64)
        if self._use_list:
            self.obs.append(obs)
            self.actions.append(actions)
            self.logprobs.append(logprobs)
            self.rewards.append(float(reward))
            self.values.append(float(value))
            self.terminated.append(bool(terminated))
            self.truncated.append(bool(truncated))
            self.global_states.append(global_state)
            self.next_global_states.append(next_global_state)
            self.imitation.append(imitation)
            self.danger_imitation_target.append(danger_imitation_target)
            self.danger_imitation_mask.append(danger_imitation_mask)
            self.sat_indices.append(sat_indices)
            return

        if self._obs is None:
            self._allocate(
                obs,
                actions,
                logprobs,
                global_state,
                next_global_state,
                imitation,
                danger_imitation_target,
                danger_imitation_mask,
                sat_indices,
            )

        if self.capacity is not None and self._idx >= self.capacity:
            raise IndexError('RolloutBuffer capacity exceeded.')

        self._obs[self._idx] = obs
        self._actions[self._idx] = actions
        self._logprobs[self._idx] = logprobs
        self._rewards[self._idx] = float(reward)
        self._values[self._idx] = float(value)
        self._terminated[self._idx] = float(terminated)
        self._truncated[self._idx] = float(truncated)
        self._global_states[self._idx] = global_state
        self._next_global_states[self._idx] = next_global_state
        self._imitation[self._idx] = imitation
        self._danger_imitation_target[self._idx] = danger_imitation_target
        self._danger_imitation_mask[self._idx] = danger_imitation_mask
        self._sat_indices[self._idx] = sat_indices
        self._idx += 1

    def as_arrays(self):
        if self._use_list:
            return (
                np.stack(self.obs, axis=0),
                np.stack(self.actions, axis=0),
                np.stack(self.logprobs, axis=0),
                np.array(self.rewards, dtype=np.float32),
                np.array(self.values, dtype=np.float32),
                np.array(self.terminated, dtype=np.float32),
                np.array(self.truncated, dtype=np.float32),
                np.stack(self.global_states, axis=0),
                np.stack(self.next_global_states, axis=0),
                np.stack(self.imitation, axis=0),
                np.stack(self.danger_imitation_target, axis=0),
                np.stack(self.danger_imitation_mask, axis=0),
                np.stack(self.sat_indices, axis=0),
            )
        end = self._idx
        return (
            self._obs[:end],
            self._actions[:end],
            self._logprobs[:end],
            self._rewards[:end],
            self._values[:end],
            self._terminated[:end],
            self._truncated[:end],
            self._global_states[:end],
            self._next_global_states[:end],
            self._imitation[:end],
            self._danger_imitation_target[:end],
            self._danger_imitation_mask[:end],
            self._sat_indices[:end],
        )
