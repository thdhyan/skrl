from __future__ import annotations

from typing import Any, Literal, Type

import copy
import dataclasses
import importlib
import inspect
import math  # noqa

import gymnasium

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.models.torch import Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise  # noqa
from skrl.resources.preprocessors.torch import DictStandardScaler, RunningStandardScaler  # noqa
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa
from skrl.trainers.torch import Trainer
from skrl.utils import set_seed


def _import_string(name: str) -> Any:
    """Import ``"package.module:attribute"`` (the attribute may be dotted, e.g. ``"module:Class.method"``)."""
    module, _, attribute = name.partition(":")
    obj = importlib.import_module(module)
    for part in attribute.split("."):
        obj = getattr(obj, part)
    return obj


def _call(fn: Any, **kwargs) -> Any:
    """Call ``fn`` with the keyword arguments its signature accepts (all of them if it takes ``**kwargs``)."""
    parameters = inspect.signature(fn).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in parameters}
    return fn(**kwargs)


_BELIEF_CALLABLES = ["belief_factory", "belief_prior_fn", "belief_true_state_fn", "belief_state_to_position"]


class Runner:
    def __init__(self, env: Wrapper | MultiAgentEnvWrapper, cfg: dict[str, Any], *, verbose: bool = False) -> None:
        """Experiment runner.

        Configure and instantiate skrl components to execute training/evaluation workflows in a few lines of code.

        :param env: Environment to train on.
        :param cfg: Runner configuration.
        :param verbose: Whether to print extra information about the setup.
        """
        self._env = env
        self._verbose = verbose

        # check for configuration compatibility
        self._cfg = self._check_cfg_compatibility(copy.deepcopy(cfg))

        # set random seed
        set_seed(self._cfg.get("seed", None))

        # PrivilegedPPO: strategy preset, shared mask curriculum and observation-history wrapper
        self._curriculum = None
        self._privileged_layout = None
        # BeliefPPO / BeliefDQN: belief tracker and the callables named in the cfg
        self._belief_callables = {}
        self._belief_space = None
        agent_class = self._cfg.get("agent", {}).get("class", "").lower()
        if agent_class == "privilegedppo":
            self._env, self._cfg = self._setup_privileged(self._env, self._cfg)
        elif agent_class in ["beliefppo", "beliefdqn"]:
            self._cfg = self._setup_belief(self._env, self._cfg)

        self._models = self._generate_models(self._env, copy.deepcopy(self._cfg))
        self._agent = self._generate_agent(self._env, copy.deepcopy(self._cfg), self._models)
        self._trainer = self._generate_trainer(self._env, copy.deepcopy(self._cfg), self._agent)

    @property
    def trainer(self) -> Trainer:
        """Trainer instance."""
        return self._trainer

    @property
    def agent(self) -> Agent:
        """Agent instance."""
        return self._agent

    @staticmethod
    def load_cfg_from_yaml(path: str) -> dict:
        """Load a runner configuration from a yaml file.

        :param path: File path.

        :return: Loaded configuration, or an empty dict if an error has occurred.
        """
        try:
            import yaml
        except Exception as e:
            logger.error(f"{e}. Install PyYAML with 'pip install pyyaml'")
            return {}

        try:
            with open(path) as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Loading yaml error: {e}")
            return {}

    def _check_cfg_compatibility(self, cfg: dict) -> dict:
        """Check for configuration compatibility.

        :param cfg: Configuration dictionary to check for compatibility.

        :return: Updated dictionary.
        """
        # rename 'lambda' to 'gae_lambda'
        if "lambda" in cfg.get("agent", {}):
            logger.warning("The 'lambda' field in the configuration is deprecated. Use 'gae_lambda' instead")
            cfg["agent"]["gae_lambda"] = cfg["agent"]["lambda"]
            del cfg["agent"]["lambda"]
        # remove 'clip_predicted_values' redundant configuration by using 'value_clip'
        if "clip_predicted_values" in cfg.get("agent", {}):
            logger.warning(
                "The 'clip_predicted_values' field in the configuration is deprecated. "
                "Use a 'value_clip' value greater than 0 to clip the predicted values instead"
            )
            value_clip = cfg["agent"].get("value_clip", 0.2)
            cfg["agent"]["value_clip"] = value_clip if cfg["agent"]["clip_predicted_values"] else 0.0
            del cfg["agent"]["clip_predicted_values"]
        # replace `state_preprocessor` by `observation_preprocessor` if the latter is not defined
        if "state_preprocessor" in cfg.get("agent", {}):
            if "observation_preprocessor" not in cfg.get("agent", {}):
                logger.warning(
                    "The 'state_preprocessor' field in the configuration has been replaced by 'observation_preprocessor'. "
                    "If the 'state_preprocessor' definition is desired but the `observation_preprocessor` is not, "
                    "define the last one to None (null) to avoid the automatic replacement"
                )
                cfg["agent"]["observation_preprocessor"] = cfg["agent"]["state_preprocessor"]
                cfg["agent"]["observation_preprocessor_kwargs"] = cfg["agent"].get("state_preprocessor_kwargs")
                del cfg["agent"]["state_preprocessor"]
                if "state_preprocessor_kwargs" in cfg["agent"]:
                    del cfg["agent"]["state_preprocessor_kwargs"]
        # remove `shared_state_preprocessor` by using `state_preprocessor`
        if "shared_state_preprocessor" in cfg.get("agent", {}):
            logger.warning(
                "The 'shared_state_preprocessor' field in the configuration is deprecated. Use 'state_preprocessor' instead"
            )
            cfg["agent"]["state_preprocessor"] = cfg["agent"]["shared_state_preprocessor"]
            cfg["agent"]["state_preprocessor_kwargs"] = cfg["agent"].get("shared_state_preprocessor_kwargs")
            del cfg["agent"]["shared_state_preprocessor"]
            if "shared_state_preprocessor_kwargs" in cfg["agent"]:
                del cfg["agent"]["shared_state_preprocessor_kwargs"]
        return cfg

    def _setup_privileged(self, env: Wrapper, cfg: dict) -> tuple[Wrapper, dict]:
        """Apply the PrivilegedPPO strategy preset, build the mask curriculum and (optionally) wrap the env.

        Config keys (besides the PrivilegedPPO_CFG fields):

        - ``agent.strategy``: name of a preset in :data:`skrl.agents.torch.privileged_ppo.STRATEGIES` (``roa``,
          ``sysid``, ``contrastive``, ``whiten``, ``osi``, ``rma``, ``uposi``). Explicit values override it.
        - ``agent.curriculum``: ``{class: <curriculum name>, **kwargs}`` (e.g. ``SoftmaxMaskCurriculum``).
        - ``privileged_history``: kwargs of :class:`PrivilegedHistoryVecWrapper` (``history_length``,
          ``include_last_action``, ``privileged_indices``, ``privileged_fn: "module:function"``). The env is wrapped
          with it and the privileged models get its ``layout``. Without it, the env must already produce the packed
          observation and the models need ``layout: {obs_dim, history_length, privileged_dim}``.
        """
        from skrl.agents.torch.privileged_ppo import apply_strategy, make_curriculum
        from skrl.envs.wrappers.torch.privileged_history_vec_wrapper import PrivilegedHistoryVecWrapper

        policy_cfg, agent_cfg = apply_strategy(
            cfg["agent"].pop("strategy", None), cfg.get("models", {}).get("policy", {}), cfg["agent"]
        )
        if "policy" in cfg.get("models", {}):
            cfg["models"]["policy"] = policy_cfg
        cfg["agent"] = agent_cfg
        self._curriculum = make_curriculum(agent_cfg.pop("curriculum", None))
        history = cfg.pop("privileged_history", None)
        if history is not None:
            history = dict(history)
            privileged_fn = history.pop("privileged_fn", None)
            env = PrivilegedHistoryVecWrapper(
                env,
                curriculum=self._curriculum,
                privileged_fn=_import_string(privileged_fn) if isinstance(privileged_fn, str) else privileged_fn,
                **history,
            )
            self._privileged_layout = env.layout
        return env, cfg

    def _setup_belief(self, env: Wrapper, cfg: dict) -> dict:
        """Resolve the belief callables of a BeliefPPO / BeliefDQN cfg and the belief-input space of its models.

        - ``agent.belief_factory`` (and ``belief_prior_fn``, ``belief_true_state_fn``, ``belief_state_to_position``):
          ``"module:attribute"`` import strings of the callables.
        - or a top-level ``belief`` section that builds the particle filter::

            belief:
              class: ParticleFilter              # or "module:Class" (a Belief subclass)
              num_particles: 300
              state_dim: 10
              num_gmm_components: 2
              n_substeps: 1                      # passed to backend.step()
              sigma_enc: 1.0                     # passed to backend.get_measurement()
              backend:
                class: skrl.belief.backends.stub_backend:StubBackend   # class or factory, gets env/device if accepted
                n_joints: 2

        - ``agent.belief_dim``: size of the compressed belief b̃ (the models' input). Default: from the filter.
        """
        agent_cfg = cfg["agent"]
        for key in _BELIEF_CALLABLES:
            value = agent_cfg.pop(key, None)
            self._belief_callables[key] = _import_string(value) if isinstance(value, str) else value
        if self._belief_callables["belief_factory"] is None and "belief" in cfg:
            self._belief_callables["belief_factory"] = self._belief_factory_from_cfg(env, dict(cfg.pop("belief")))
        belief_dim = agent_cfg.pop("belief_dim", None)
        if belief_dim is None:
            if self._belief_callables["belief_factory"] is None:
                raise ValueError("Belief agents need 'agent.belief_factory' or a 'belief' section in the cfg")
            belief_dim = self._belief_callables["belief_factory"]().belief_dim
        self._belief_space = gymnasium.spaces.Box(low=-math.inf, high=math.inf, shape=(belief_dim,))
        return cfg

    def _belief_factory_from_cfg(self, env: Wrapper, spec: dict) -> Any:
        from skrl.belief.torch.particle_filter import ParticleFilter

        belief_class = spec.pop("class", "ParticleFilter")
        belief_class = ParticleFilter if belief_class.lower() == "particlefilter" else _import_string(belief_class)
        backend_spec = dict(spec.pop("backend"))
        backend_class = backend_spec.pop("class")
        backend_class = _import_string(backend_class) if isinstance(backend_class, str) else backend_class
        device = spec.pop("device", env.device)
        n_substeps, sigma_enc = spec.pop("n_substeps", 1), spec.pop("sigma_enc", 1.0)
        backend = _call(backend_class, env=env, device=device, **backend_spec)

        def factory():
            return belief_class(
                motion_model_fn=lambda particles, control: backend.step(particles, control, n_substeps=n_substeps),
                measurement_model_fn=lambda particles, measurement: backend.get_measurement(
                    particles, sigma_enc=sigma_enc
                ),
                device=device,
                **spec,
            )

        return factory

    def _instantiate_model(self, model_class: Any, role: str, kwargs: dict, **spaces) -> Model:
        """Instantiate a model: skrl instantiator functions take every argument, model classes (e.g.
        ``PrivilegedGaussianPolicy``) only the ones in their signature. Privileged models get the env ``layout``
        (or build it from a ``layout`` dict), belief models get the compressed-belief space as observation space."""
        if not inspect.isclass(model_class):
            return model_class(**spaces, **kwargs)
        parameters = inspect.signature(model_class).parameters
        if "layout" in parameters:
            from skrl.agents.torch.privileged_ppo import ObsLayout

            layout = kwargs.get("layout", self._privileged_layout)
            if layout is None:
                raise ValueError(
                    f"The 'models.{role}' model needs a 'layout' (or a 'privileged_history' section in the cfg)"
                )
            kwargs["layout"] = ObsLayout(**layout) if isinstance(layout, dict) else layout
        if self._belief_space is not None:
            spaces["observation_space"] = self._belief_space
        spaces = {k: v for k, v in spaces.items() if k in parameters}  # e.g. no state_space
        return model_class(**spaces, **kwargs)

    def _component(self, name: str) -> Type:
        """Get skrl component (e.g.: agent, trainer, etc..) from string identifier.

        :return: skrl component.
        """
        from skrl.agents.torch.a2c import A2C, A2C_CFG
        from skrl.agents.torch.amp import AMP, AMP_CFG
        from skrl.agents.torch.belief_dqn import BeliefDQN, BeliefDQN_CFG, ResNetDuelingQNetwork
        from skrl.agents.torch.belief_ppo import BeliefPPO, BeliefPPO_CFG, ResNetCategoricalPolicy, ResNetValue
        from skrl.agents.torch.cem import CEM, CEM_CFG
        from skrl.agents.torch.ddpg import DDPG, DDPG_CFG
        from skrl.agents.torch.ddqn import DDQN, DDQN_CFG
        from skrl.agents.torch.dqn import DQN, DQN_CFG
        from skrl.agents.torch.ppo import PPO, PPO_CFG
        from skrl.agents.torch.privileged_ppo import (
            PrivilegedCategoricalPolicy,
            PrivilegedGaussianPolicy,
            PrivilegedPPO,
            PrivilegedPPO_CFG,
            PrivilegedValue,
        )
        from skrl.agents.torch.rpo import RPO, RPO_CFG
        from skrl.agents.torch.sac import SAC, SAC_CFG
        from skrl.agents.torch.td3 import TD3, TD3_CFG
        from skrl.agents.torch.trpo import TRPO, TRPO_CFG
        from skrl.memories.torch import RandomMemory
        from skrl.models.torch.encoders import EncoderCategoricalPolicy, EncoderDeterministic, EncoderGaussianPolicy
        from skrl.multi_agents.torch.ippo import IPPO, IPPO_CFG
        from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG
        from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
        from skrl.utils.model_instantiators.torch import (
            categorical_model,
            deterministic_model,
            gaussian_model,
            multicategorical_model,
            multivariate_gaussian_model,
            shared_model,
        )

        component = {
            # models
            "gaussianmixin": gaussian_model,
            "categoricalmixin": categorical_model,
            "multicategoricalmixin": multicategorical_model,
            "deterministicmixin": deterministic_model,
            "multivariategaussianmixin": multivariate_gaussian_model,
            "shared": shared_model,
            # model classes (instantiated with the spaces, the device and the remaining cfg as kwargs)
            "privilegedgaussianpolicy": PrivilegedGaussianPolicy,
            "privilegedcategoricalpolicy": PrivilegedCategoricalPolicy,
            "privilegedvalue": PrivilegedValue,
            "resnetcategoricalpolicy": ResNetCategoricalPolicy,
            "resnetvalue": ResNetValue,
            "resnetduelingqnetwork": ResNetDuelingQNetwork,
            "encodergaussianpolicy": EncoderGaussianPolicy,
            "encodercategoricalpolicy": EncoderCategoricalPolicy,
            "encoderdeterministic": EncoderDeterministic,
            # memories
            "randommemory": RandomMemory,
            # agents
            "a2c": A2C,
            "a2c_cfg": A2C_CFG,
            "amp": AMP,
            "amp_cfg": AMP_CFG,
            "beliefdqn": BeliefDQN,
            "beliefdqn_cfg": BeliefDQN_CFG,
            "beliefppo": BeliefPPO,
            "beliefppo_cfg": BeliefPPO_CFG,
            "cem": CEM,
            "cem_cfg": CEM_CFG,
            "ddpg": DDPG,
            "ddpg_cfg": DDPG_CFG,
            "ddqn": DDQN,
            "ddqn_cfg": DDQN_CFG,
            "dqn": DQN,
            "dqn_cfg": DQN_CFG,
            "ppo": PPO,
            "ppo_cfg": PPO_CFG,
            "privilegedppo": PrivilegedPPO,
            "privilegedppo_cfg": PrivilegedPPO_CFG,
            "rpo": RPO,
            "rpo_cfg": RPO_CFG,
            "sac": SAC,
            "sac_cfg": SAC_CFG,
            "td3": TD3,
            "td3_cfg": TD3_CFG,
            "trpo": TRPO,
            "trpo_cfg": TRPO_CFG,
            # multi-agents
            "ippo": IPPO,
            "ippo_cfg": IPPO_CFG,
            "mappo": MAPPO,
            "mappo_cfg": MAPPO_CFG,
            # trainers
            "sequentialtrainer": SequentialTrainer,
            "sequentialtrainer_cfg": SequentialTrainerCfg,
        }.get(name.lower())

        if component is None and ":" in name:  # "package.module:attribute", e.g. a custom model class
            component = _import_string(name)
        if component is None:
            raise ValueError(f"Component '{name}' is not supported in the runner cfg")
        return component

    def _process_cfg(self, cfg: dict) -> dict:
        """Convert simple types to skrl classes/components.

        :param cfg: A configuration dictionary.

        :return: Updated dictionary.
        """
        _direct_eval = [
            "learning_rate_scheduler",
            "observation_preprocessor",
            "state_preprocessor",
            "value_preprocessor",
            "amp_observation_preprocessor",
            "exploration_noise",
            "smooth_regularization_noise",
        ]

        def update_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    update_dict(value)
                else:
                    if key in _direct_eval:
                        if isinstance(value, str):
                            d[key] = eval(value)
                    elif key.endswith("_kwargs"):
                        d[key] = value if value is not None else {}
            return d

        cfg = update_dict(copy.deepcopy(cfg))
        if "class" in cfg:
            del cfg["class"]

        # materialize exploration scheduler
        if "exploration_scheduler" in cfg:
            cfg["exploration_scheduler"] = eval(f"lambda timestep, timesteps: {cfg['exploration_scheduler']}")
        # materialize rewards shaper
        if "rewards_shaper_scale" in cfg:
            scale = cfg["rewards_shaper_scale"]
            if scale is not None and scale != 1.0:
                cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * scale
            del cfg["rewards_shaper_scale"]

        return cfg

    def _generate_models(self, env: Wrapper | MultiAgentEnvWrapper, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
        """Generate model instances according to the environment specification and the given config.

        :param env: Wrapped environment.
        :param cfg: A configuration dictionary.

        :return: Model instances.
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        agent_class = cfg.get("agent", {}).get("class")
        if not agent_class:
            raise ValueError(f"The 'agent.class' field is not defined in the specified configuration")
        agent_class = agent_class.lower()

        # instantiate models
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("The 'models' field is not defined in the specified configuration")
            # get separate (non-shared) configuration and remove 'separate' key
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True
                logger.warning(
                    "The 'models.separate' field is not defined in the specified configuration. Falling back to True by default"
                )
            # non-shared models
            if separate:
                for role in models_cfg:
                    # get instantiator function and remove 'class' key
                    model_class = models_cfg[role].get("class")
                    if not model_class:
                        raise ValueError(
                            f"The 'models.{role}.class' field is not defined in the specified configuration"
                        )
                    del models_cfg[role]["class"]
                    model_class = self._component(model_class)
                    # get specific spaces according to agent/model cfg
                    observation_space = observation_spaces[agent_id]
                    if agent_class == "amp" and role == "discriminator":
                        try:
                            observation_space = env.amp_observation_space
                        except Exception as e:
                            logger.warning(
                                "Unable to get AMP space via 'env.amp_observation_space'. Using 'env.observation_space' instead"
                            )
                    # print model source
                    if self._verbose and not inspect.isclass(model_class):
                        source = model_class(
                            observation_space=observation_space,
                            state_space=state_spaces[agent_id],
                            action_space=action_spaces[agent_id],
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                            return_source=True,
                        )
                        print("==================================================")
                        print(f"Model (role): {role}")
                        print("==================================================\n")
                        print(source)
                        print("--------------------------------------------------")
                    # instantiate model
                    models[agent_id][role] = self._instantiate_model(
                        model_class,
                        role,
                        self._process_cfg(models_cfg[role]),
                        observation_space=observation_space,
                        state_space=state_spaces[agent_id],
                        action_space=action_spaces[agent_id],
                        device=device,
                    )
            # shared models
            else:
                roles = list(models_cfg.keys())
                if len(roles) != 2:
                    raise ValueError(
                        "Runner currently only supports shared models, made up of exactly two models. "
                        "Set 'separate' field to True to create non-shared models for the given cfg"
                    )
                # get shared model structure and parameters
                structure = []
                parameters = []
                for role in roles:
                    # get instantiator function and remove 'class' key
                    model_structure = models_cfg[role].get("class")
                    if not model_structure:
                        raise ValueError(
                            f"The 'models.{role}.class' field is not defined in the specified configuration"
                        )
                    del models_cfg[role]["class"]
                    structure.append(model_structure)
                    parameters.append(self._process_cfg(models_cfg[role]))
                model_class = self._component("Shared")
                # print model source
                if self._verbose:
                    source = model_class(
                        observation_space=observation_spaces[agent_id],
                        state_space=state_spaces[agent_id],
                        action_space=action_spaces[agent_id],
                        device=device,
                        structure=structure,
                        roles=roles,
                        parameters=parameters,
                        return_source=True,
                    )
                    print("==================================================")
                    print(f"Shared model (roles): {roles}")
                    print("==================================================\n")
                    print(source)
                    print("--------------------------------------------------")
                # instantiate model
                models[agent_id][roles[0]] = model_class(
                    observation_space=observation_spaces[agent_id],
                    state_space=state_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=structure,
                    roles=roles,
                    parameters=parameters,
                )
                models[agent_id][roles[1]] = models[agent_id][roles[0]]

        # initialize lazy modules' parameters
        for agent_id in possible_agents:
            for role, model in models[agent_id].items():
                model.init_state_dict(role=role)

        return models

    def _generate_agent(
        self,
        env: Wrapper | MultiAgentEnvWrapper,
        cfg: dict[str, Any],
        models: dict[str, dict[str, Model]],
    ) -> Agent:
        """Generate agent instance according to the environment specification and the given config and models.

        :param env: Wrapped environment.
        :param cfg: A configuration dictionary.
        :param models: Agent's model instances.

        :return: Agent instances.
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        num_envs = env.num_envs
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        # get agent class
        if "agent" not in cfg:
            raise ValueError(f"The 'agent' field is not defined in the specified configuration")
        if "class" not in cfg["agent"]:
            raise ValueError(f"The 'agent.class' field is not defined in the specified configuration")
        agent_class = cfg["agent"]["class"].lower()

        # get memory class
        if "memory" not in cfg:
            raise ValueError(f"The 'memory' field is not defined in the specified configuration")
        if "class" not in cfg["memory"]:
            raise ValueError(f"The 'memory.class' field is not defined in the specified configuration")
        memory_class = self._component(cfg["memory"]["class"])
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
        memories = {
            agent_id: memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))
            for agent_id in possible_agents
        }

        # single-agent configuration and instantiation
        if agent_class in ["amp"]:
            agent_id = possible_agents[0]
            try:
                amp_observation_space = env.amp_observation_space
            except Exception as e:
                logger.warning(
                    "Unable to get AMP space via 'env.amp_observation_space'. Using 'env.observation_space' instead"
                )
                amp_observation_space = observation_spaces[agent_id]
            agent_cfg = dataclasses.asdict(self._component(f"{agent_class}_CFG")(**self._process_cfg(cfg["agent"])))
            agent_cfg.get("observation_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": state_spaces[agent_id], "device": device})
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            agent_cfg.get("amp_observation_preprocessor_kwargs", {}).update(
                {"size": amp_observation_space, "device": device}
            )

            motion_dataset = None
            if "motion_dataset" in cfg:
                if "class" not in cfg["motion_dataset"]:
                    raise ValueError(f"The 'motion_dataset.class' field is not defined in the specified configuration")
                motion_dataset_class = self._component(cfg["motion_dataset"]["class"])
                motion_dataset = motion_dataset_class(device=device, **self._process_cfg(cfg["motion_dataset"]))
            reply_buffer = None
            if "reply_buffer" in cfg:
                if "class" not in cfg["reply_buffer"]:
                    raise ValueError(f"The 'reply_buffer.class' field is not defined in the specified configuration")
                reply_buffer_class = self._component(cfg["reply_buffer"]["class"])
                reply_buffer = reply_buffer_class(device=device, **self._process_cfg(cfg["reply_buffer"]))

            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "state_space": state_spaces[agent_id],
                "action_space": action_spaces[agent_id],
                "amp_observation_space": amp_observation_space,
                "motion_dataset": motion_dataset,
                "reply_buffer": reply_buffer,
                "collect_reference_motions": lambda num_samples: env.collect_reference_motions(num_samples),
            }
        elif agent_class in [
            "a2c", "beliefdqn", "beliefppo", "cem", "ddpg", "ddqn", "dqn", "ppo", "privilegedppo", "rpo", "sac", "td3", "trpo"
        ]:
            agent_id = possible_agents[0]
            agent_cfg = dataclasses.asdict(self._component(f"{agent_class}_CFG")(**self._process_cfg(cfg["agent"])))
            agent_cfg.get("observation_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": state_spaces[agent_id], "device": device})
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            agent_cfg.get("exploration_noise_kwargs", {}).update({"device": device})
            agent_cfg.get("smooth_regularization_noise_kwargs", {}).update({"device": device})
            # objects shared with the env or holding state: set after `dataclasses.asdict` (it deep-copies values)
            if agent_class == "privilegedppo":
                agent_cfg["curriculum"] = self._curriculum
            agent_cfg.update({k: v for k, v in self._belief_callables.items() if k in agent_cfg})
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "state_space": state_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        # multi-agent configuration and instantiation
        elif agent_class in ["ippo", "mappo"]:
            agent_cfg = dataclasses.asdict(self._component(f"{agent_class}_CFG")(**self._process_cfg(cfg["agent"])))
            agent_cfg.get("observation_preprocessor_kwargs", {}).update(
                {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {agent_id: {"size": state_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "state_spaces": state_spaces,
                "action_spaces": action_spaces,
                "possible_agents": possible_agents,
            }
        return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)

    def _generate_trainer(self, env: Wrapper | MultiAgentEnvWrapper, cfg: dict[str, Any], agent: Agent) -> Trainer:
        """Generate trainer instance according to the environment specification and the given config and agent.

        :param env: Wrapped environment.
        :param cfg: A configuration dictionary.
        :param agent: Agent's model instances.

        :return: Trainer instances.
        """
        # get trainer class
        if "trainer" not in cfg:
            raise ValueError(f"The 'trainer' field is not defined in the specified configuration")
        if "class" not in cfg["trainer"]:
            raise ValueError(f"The 'trainer.class' field is not defined in the specified configuration")
        trainer_class = cfg["trainer"]["class"].lower()
        # instantiate trainer
        trainer_cfg = self._component(f"{trainer_class}_CFG")(**self._process_cfg(cfg["trainer"]))
        return self._component(trainer_class)(env=env, agents=agent, cfg=trainer_cfg)

    def run(self, mode: Literal["train", "eval"] = "train") -> None:
        """Run the training/evaluation.

        :param mode: Running mode: ``"train"`` for training or ``"eval"`` for evaluation.

        :raises ValueError: The specified running mode is not valid.
        """
        if mode == "train":
            self._trainer.train()
        elif mode == "eval":
            self._trainer.eval()
        else:
            raise ValueError(f"Unknown running mode: {mode}")
