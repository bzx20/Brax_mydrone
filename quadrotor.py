from brax import base
from brax.envs import env
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class Quadrotor(env.PipelineEnv):

  def __init__(self, backend='generalized', **kwargs):
    path = epath.resource_path('brax') / 'envs/assets/quadrotor.xml'
    # path = "./quadrotor.xml"
    sys = mjcf.load(path)

    n_frames = 2

    if backend in ['spring', 'positional']:
      sys = sys.replace(dt=0.005)
      n_frames = 4

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    q = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
    )
    qd = jax.random.uniform(
        rng2, (self.sys.qd_size(),), minval=-0.01, maxval=0.01
    )
    pipeline_state = self.pipeline_init(q, qd)
    obs = self._get_obs(pipeline_state)
    reward, done = jp.zeros(2)
    metrics = {}

    return env.State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    pipeline_state = self.pipeline_step(state.pipeline_state, action)
    obs = self._get_obs(pipeline_state)
    reward = 1.0
    done = jp.where(jp.abs(obs[1]) > 0.2, 1.0, 0.0)
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  @property
  def action_size(self):
    return 1

  def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
    """Observe cartpole body position and velocities."""
    return jp.concatenate([pipeline_state.q, pipeline_state.qd])



env = Quadrotor(debug=False)
state = env.reset(rng=jp.random_prngkey(seed=0))