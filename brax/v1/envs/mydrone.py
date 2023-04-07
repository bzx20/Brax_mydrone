# from brax import envs
import brax.v1 as brax
import jax
import jax.numpy as jnp
from brax.v1.envs import env
from brax.v1.envs.env import State
from brax.v1 import jumpy as jp
# from brax.envs.to_torch import JaxToTorchWrapper
import numpy as np
from typing import Optional, Tuple


class MyDroneEnv(env.Env):
  # def __init__(self, **kwargs):
  #   config = _SYSTEM_CONFIG
  #   super().__init__(config=config,**kwargs)
  
  def __init__(self, legacy_spring=False, **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)

    qpos = self.sys.default_angle() + self._noise(rng1)
    qvel = self._noise(rng2)

    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    obs = self._get_obs(qp, self.sys.info(qp))
    reward, done = jp.zeros(2)
    metrics = {}

    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)
    reward = 1.0
    done = jp.where(jp.abs(obs[1]) > .2, 1.0, 0.0)  # pytype: disable=wrong-arg-types  # jax-ndarray

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  @property
  def action_size(self):
    return 1

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe cartpole body position and velocities."""
    joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)

    # [cart pos, joint angle, cart vel, joint vel]
    obs = [qp.pos[0, :1], joint_angle, qp.vel[0, :1], joint_vel]

    return jp.concatenate(obs)

  def _noise(self, rng):
    return jp.random_uniform(rng, (self.sys.num_joint_dof,), -0.01, 0.01)


_SYSTEM_CONFIG = """
  bodies {
    name: "cart"
    colliders {
      rotation {
      x: 90
      z: 90
      }
      box {
        halfsize { x: 0.05 y: 0.2 z: 0.2 }
      }
    }
    mass: 10.471975
  }
  bodies {
    name: "wing1"
    colliders {
      position { x: 0.2 y:0.2 z:0.05 }
      sphere {
        radius: 0.01
      }
    }
    mass: 1.0
  }
  bodies {
    name: "wing2"
    colliders {
      position { x: 0.2 y:-0.2 z:0.05 }
      sphere {
        radius: 0.01
      }
    }
    mass: 1.0
  }
  bodies {
    name: "wing3"
    colliders {
      position { x: -0.2 y:0.2 z:0.05 }
      sphere {
        radius: 0.01
      }
    }
    mass: 1.0
  }
  bodies {
    name: "wing4"
    colliders {
      position { x: -0.2 y:-0.2 z:0.05 }
      sphere {
        radius: 0.01
      }
    }
    mass: 1.0
  }
  bodies {
    name: "pole"
    colliders {
      capsule {
        radius: 0.01
        length: 0.6
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 5.0185914
  }
  bodies {
    name: "object"
    colliders {
      sphere {
        radius: 0.1
      }
    }
    mass: 10.0
  }
  bodies {
    name: "Ground"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  joints {
    name: "hinge"
    parent: "cart"
    child: "pole"
    child_offset { z: 0.3 }
    rotation {
      z: 90.0
    }
    angle_limit { min: -360 max: 360 }
  }
  joints {
    name: "objectjoint"
    parent: "pole"
    child: "object"
    child_offset { z: 0.3 }
    rotation {
      z: 90.0
    }
    angle_limit { min: -360 max: 360 }
  }
  joints {
    name: "joint1"
    parent: "cart"
    child: "wing1"
    rotation {
      z: 90.0
    }
    angle_limit { min: -360 max: 360 }
  }
  joints {
    name: "joint2"
    parent: "cart"
    child: "wing2"
    rotation {
      z: 90.0
    }
    angle_limit { min: -360 max: 360 }
  }
  joints {
    name: "joint3"
    parent: "cart"
    child: "wing3"
    rotation {
      z: 90.0
    }
    angle_limit { min: -360 max: 360 }
  }
  joints {
    name: "joint4"
    parent: "cart"
    child: "wing4"
    rotation {
      z: 90.0
    }
    angle_limit { min: -360 max: 360 }
  }
  forces {
    name: "thruster1"
    body: "wing1"
    strength: 500.0
    thruster{
    }
  }
  forces {
    name: "thruster2"
    body: "wing2"
    strength: 400.0
    thruster{
    }
  }
  forces {
    name: "thruster3"
    body: "wing3"
    strength: 800.0
    thruster{
    }
  }
  forces {
    name: "thruster4"
    body: "wing4"
    strength: 500.0
    thruster{
    }
  }
  
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.04
  substeps: 8
  dynamics_mode: "pbd"
  """


_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "cart"
    colliders {
      rotation {
      x: 90
      z: 90
      }
      capsule {
        radius: 0.1
        length: 0.4
      }
    }
    frozen { position { x:0 y:1 z:1 } rotation { x:1 y:1 z:1 } }
    mass: 10.471975
  }
  bodies {
    name: "pole"
    colliders {
      capsule {
        radius: 0.049
        length: 0.69800085
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 5.0185914
  }
  joints {
    name: "hinge"
    stiffness: 10000.0
    parent: "cart"
    child: "pole"
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    limit_strength: 0.0
    angle_limit { min: 0.0 max: 0.0 }
  }
  forces {
    name: "thruster"
    body: "cart"
    strength: 100.0
    thruster {}
  }
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.04
  substeps: 8
  dynamics_mode: "legacy_spring"
  """
#   def reset(self, rng: jnp.ndarray) -> env.State:
#     """Resets the environment to an initial state."""
#     rng, rng1, rng2 = jp.random_split(rng, 3)
#     qpos = self.sys.default_angle()
#     qp = self.sys.default_qp(joint_angle=qpos)
#     obs = self._get_obs(qp, self.sys.info(qp))
#     reward, done, zero = jp.zeros(3)
#     metrics = {'reward_dist': zero, 'reward_ctrl': zero, 'reward_near': zero}
#     return env.State(qp, obs, reward, done, metrics)

#   def step(self, state: env.State, action: jnp.ndarray) -> env.State:
#     """Runs one timestep of the environment dynamics."""
#     qp, qdot = self.sys.step(state.qp, state.qdot, action)
#     return env.State(qp)

#   def observation_size(self) -> int:
#     """Returns the size of the observation."""
#     return self.sys.observation_size

#   def action_size(self) -> int:
#     """Returns the size of the action."""
#     return self.sys.action_size

#   def default_action(self) -> jnp.ndarray:
#     """Returns the default action."""
#     return jnp.zeros((self.action_size(),), dtype=np.float32)

#   def max_episode_steps(self) -> Optional[int]:
#     """Returns the maximum number of steps per episode."""
#     return None
#   def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
#     """Observe body position and velocities."""
#     # qpos: position and orientation of the torso and the joint angles
#     qpos = [qp.pos[0, (0, 2)], qp.rot[0, (0, 2)]]
#     # qvel: velocity of the torso and the joint angle velocities
#     qvel = [qp.vel[0, (0, 2)], qp.ang[0, 1:2]]
#     return jp.concatenate(qpos + qvel)


#   def render(self, state: env.State) -> np.ndarray:
#     """Renders the environment state."""
#     return np.zeros((256, 256, 3), dtype=np.uint8)


# _SYSTEM_CONFIG = """
#   bodies {
#     name:"box"
#     colliders {
#       box {
#         halfsize { x: 1.0 y: 1.0 z: 0.3 }
#       }
#     }
#     mass: 10
#   }
#   joints {
#     name: "joint1"
#     stiffness: 100.0
#     parent_offset { x: 0.2 y: 0.2 }
#     child_offset { x: -0.1 y: -0.1 }
#     parent: "box"
#     angle_limit { min: -360 max: 360 }
#     rotation { y: -90 }
#   }
#   joints {
#     name: "joint2"
#     stiffness: 100.0
#     parent_offset { x: -1 y: 1 z: 0.3 }
#     child_offset { x: 0 y: 0 z: 0 }
#     parent: "box"
#     angle_limit { min: -360 max: 360 }
#     rotation { y: -90 }
#   }

#   defaults {
#     angles {
#         name: "ball"
#         angle {}
#     }
#   }
#   collide_include {}
#   gravity {
#     z: -9.81
#   }
#   dt: 0.01
#   substeps: 4
# """

# _SYSTEM_CONFIG = """
#   bodies {
#     name:"box"
#     colliders {
#       box {
#         halfsize { x: 1.0 y: 1.0 z: 0.3 }
#       }
#     }
#     frozen { position { x:1 y:1 z:1 } rotation { x:1 y:1 z:1 } }
#     mass: 10
#   }
#   defaults {
#     angles {
#         name: "hinge"
#         angle{ x: 180.0 y: 0.0 z: 0.0}
#     }
#   }
#   collide_include {}
#   gravity {
#     z: -9.81
#   }
#   dt: 0.01
#   substeps: 4
# """



# from IPython.display import HTML, Image 

# env = MyBoxEnv()
# state = env.reset(rng=jp.random_prngkey(seed=0))
# HTML(html.render(env.sys, [state.qp]))
