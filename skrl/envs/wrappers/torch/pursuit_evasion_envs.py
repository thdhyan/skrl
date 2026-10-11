from __future__ import annotations

from typing import Any

import torch

from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper


class PursuitEvasionWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Wrapper for the native torch pursuit-evasion grid environment.

        See :class:`skrl.envs.pursuit_evasion.PursuitEvasionEnv`. The environment already speaks torch and
        resets its sub-environments automatically, so this wrapper is a thin pass-through.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

    def step(self, actions: dict[str, torch.Tensor]) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, Any],
    ]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        with torch.no_grad():
            return self._env.step(actions)

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        return self._env.reset()

    def state(self) -> dict[str, torch.Tensor | None]:
        """Get the environment (global) state, shared by all agents.

        :return: State.
        """
        state = self._env.state()
        return {uid: state for uid in self.possible_agents}

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
