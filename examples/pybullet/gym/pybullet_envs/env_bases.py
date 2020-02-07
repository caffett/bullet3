import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
from pybullet_utils import bullet_client

from pybullet_envs.utils import reset_current_system_state, get_current_system_state

from pkg_resources import parse_version

from os import path
import tensorflow as tf
from tensorflow.keras.models import load_model

ROOT = path.dirname(path.abspath(gym.__file__))+"/envs/env_approx/"
from tensorflow.keras import backend as K


class MJCFBaseBulletEnv(gym.Env):
  """
	Base class for Bullet physics simulation loading MJCF (MuJoCo .xml) environments in a Scene.
	These environments create single-player scenes and behave like normal Gym environments, if
	you don't use multiplayer.
	"""

  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

  def __init__(self, robot, render=False):
    self.scene = None
    self.physicsClientId = -1
    self.ownsPhysicsClient = 0
    self.camera = Camera()
    self.isRender = render
    self.robot = robot
    self.seed()
    self._cam_dist = 3
    self._cam_yaw = 0
    self._cam_pitch = -30
    self._render_width = 320
    self._render_height = 240

    self.action_space = robot.action_space
    self.observation_space = robot.observation_space

    # self.previou_parts = None

  def configure(self, args):
    self.robot.args = args

  def seed(self, seed=None):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
    return [seed]

  def reset(self, x0=None):
    if (self.physicsClientId < 0):
      self.ownsPhysicsClient = True

      if self.isRender:
        self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
      else:
        self._p = bullet_client.BulletClient()

      # set the simulation as deteministic
      pybullet.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

      self.physicsClientId = self._p._client
      self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    if self.scene is None:
      self.scene = self.create_single_player_scene(self._p)
    if not self.scene.multiplayer and self.ownsPhysicsClient:
      self.scene.episode_restart(self._p)

    self.robot.scene = self.scene

    self.frame = 0
    self.done = 0
    self.reward = 0
    dump = 0

    s = self.robot.reset(self._p)
    
    if x0 is not None:
      # first reset do not consider floor,
      # the others should not consider floor
      self.robot.parts.pop("floor", None)

      self._p = reset_current_system_state(client=self._p, state=x0)
      s = self.robot.calc_state()

    self.potential = self.robot.calc_potential()

    return s

  @property
  def state(self):
    return np.array(get_current_system_state(client=self._p, flatten=True))

  # def approximator(self, x0, step, algo, *args, **kwargs):
  #   if "initial_space" not in self.__dir__():
  #     raise Exception("no initial space specified")

  #   dim = np.sum((self.initial_space.high-self.initial_space.low) != 0)

  #   model_name = self.__class__.__name__+"-v0"
  #   self.approx = load_model(ROOT+model_name+"/"+algo+"_vra_approx_approx"+str(step)+".model")

  #   new_model = tf.keras.Sequential()
  #   new_input = tf.keras.Input(tensor=tf.reshape(x0, (-1,dim)))
  #   new_model.add(new_input)
  #   for layer in self.approx.layers:
  #       new_model.add(layer)

  #   sess = K.get_session()

  #   return tf.reshape(new_model.output, (-1, 1)), sess

  def render(self, mode='human', close=False):
    if mode == "human":
      self.isRender = True
    if mode != "rgb_array":
      return np.array([])

    base_pos = [0, 0, 0]
    if (hasattr(self, 'robot')):
      if (hasattr(self.robot, 'body_xyz')):
        base_pos = self.robot.body_xyz

    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(self._render_width) /
                                                     self._render_height,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=self._render_width,
                                              height=self._render_height,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def close(self):
    if (self.ownsPhysicsClient):
      if (self.physicsClientId >= 0):
        self._p.disconnect()
    self.physicsClientId = -1

  def HUD(self, state, a, done):
    pass

  # def step(self, *args, **kwargs):
  # 	if self.isRender:
  # 		base_pos=[0,0,0]
  # 		if (hasattr(self,'robot')):
  # 			if (hasattr(self.robot,'body_xyz')):
  # 				base_pos = self.robot.body_xyz
  # 				# Keep the previous orientation of the camera set by the user.
  # 				#[yaw, pitch, dist] = self._p.getDebugVisualizerCamera()[8:11]
  # 				self._p.resetDebugVisualizerCamera(3,0,0, base_pos)
  #
  #
  # 	return self.step(*args, **kwargs)
  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed


class Camera:

  def __init__(self):
    pass

  def move_and_look_at(self, i, j, k, x, y, z):
    lookat = [x, y, z]
    distance = 10
    yaw = 10
    self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)
