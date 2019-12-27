# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-10-29 10:49:28
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-12-27 16:07:56
# -------------------------------
import pybullet_envs
import gym
import numpy as np

import os
ROOT = os.path.dirname(os.path.abspath(pybullet_envs.__file__))

import multiprocessing as mp


def _iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def _flatten(state):
    flatten_state = []

    def flatten_helper(state):
        for item in state:
            if _iterable(item):
                flatten_helper(item)
            else:
                flatten_state.append(item)
    flatten_helper(state)

    return flatten_state


def _fill(template, flatten_state):
    sys_state = template
    global flatten_index
    flatten_index = 0

    def fill_helper(state):
        global flatten_index
        for i in range(len(state)):
            if _iterable(state[i]):
                fill_helper(state[i])
            else:
                state[i] = flatten_state[flatten_index]
                flatten_index += 1

    fill_helper(sys_state)

    return sys_state


def get_current_system_state(client, flatten=False):
    # print("call get_current_system_state")
    state = []
    num_bodies = client.getNumBodies()

    for i in range(num_bodies):
        body_state = []
        for j in range(client.getNumJoints(i)):
            jointPosition, jointVelocity, jointReaction, \
                ForcesappliedJointMotorTorque = client.getJointState(i, j)
            joint_state = [jointPosition, jointVelocity,
                           list(jointReaction), ForcesappliedJointMotorTorque]
            body_state.append(joint_state)

        pos, orn = client.getBasePositionAndOrientation(i)
        linVel, angVel = client.getBaseVelocity(i)
        body_state.append([list(pos), list(orn), list(linVel), list(angVel)])

        state.append(body_state)

    return _flatten(state) if flatten else state


def reset_current_system_state(client, state):
    # print("call reset_current_system_state")
    # set the simulation as deteministic
    client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
    if not _iterable(state[-1]):
        template = get_current_system_state(client)
        state = _fill(template, state)

    for i in range(client.getNumBodies()):
        for j in range(client.getNumJoints(i)):
            pos = state[i][j][0]
            vel = state[i][j][1]
            # Why does this function just provide reseting position and
            # velocity
            client.resetJointState(i, j, pos, vel)
        pos, orn, linVel, angVel = state[i][-1]
        client.resetBasePositionAndOrientation(i, pos, orn)
        client.resetBaseVelocity(i, linVel, angVel)

    return client

# Initial boundary
def core_func(env_name, iteration):
    np.random.seed()

    env = gym.make(env_name)
    env.seed()
    env.reset()
    state = env.env.state
    state_shape = np.array(state).shape
    low_boundary = np.full(state_shape, np.inf)
    high_boundary = np.full(state_shape, np.NINF)

    for i in range(iteration):
        env.reset()
        env.step(env.action_space.sample())
        state = env.env.state
        low_boundary[state < low_boundary] = state[state < low_boundary]
        high_boundary[state > high_boundary] = state[state > high_boundary]
    env.close()

    return low_boundary, high_boundary


def initial_boundary_estimate(env_name, iteration=100000):
    proc_num = 64
    pool = mp.Pool(processes=proc_num)
    # print(env_name)
    res = [pool.apply_async(core_func, args=(env_name, iteration))
           for i in range(proc_num)]
    state_shape = np.array(res[0].get()[0]).shape
    low_boundary = np.full(state_shape, np.inf)
    high_boundary = np.full(state_shape, np.NINF)

    for p in res:
        low, high = p.get()
        low_boundary[low < low_boundary] = low[low < low_boundary]
        high_boundary[high > high_boundary] = high[high > high_boundary]
    pool.close()

    # print(low, high)
    print(env_name, ":", np.sum((high_boundary - low_boundary) != 0))

    path = ROOT + "/initial_space/" + env_name
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(path + "/boundary.npy", np.array([low_boundary, high_boundary]))


def get_and_reset_state_test(client, env_name):
    client.resetSimulation()

    env = gym.make(env_name)
    env.reset()
    state = get_current_system_state(client)
    # print(state)

    flatten_state = _flatten(state)
    # print(flatten_state)

    env.reset()
    filled_state = _fill(state, flatten_state)
    print("filled_state check: ", filled_state == state)

    env.reset()
    reset_current_system_state(client, state)
    restored_sys_state = get_current_system_state(client)
    print("restore_sys_state check: ", restored_sys_state == state)

    env.reset()
    reset_current_system_state(client, flatten_state)
    restored_sys_state = get_current_system_state(client)
    print("restore_flatten_state check: ", restored_sys_state == state)

    env.close()


def reset_test(client, env_name):
    env = gym.make(env_name)
    ori_obs = env.reset()
    ori_state = get_current_system_state(client, flatten=True)

    reset_obs = env.reset(x0=ori_state)
    reset_state = get_current_system_state(client, flatten=True)

    print("reset_state == ori_state:", reset_state == ori_state)
    print("reset_obs == ori_obs:", (reset_obs == ori_obs).all())
    assert (reset_state == ori_state) == True
    assert (reset_obs == ori_obs).all() == True

    env.close()

if __name__ == "__main__":
    benchmarks = ["AntBulletEnv-v0", "HalfCheetahBulletEnv-v0",
                  "ReacherBulletEnv-v0", "HopperBulletEnv-v0", "HumanoidBulletEnv-v0",
                  "InvertedDoublePendulumBulletEnv-v0", "InvertedPendulumSwingupBulletEnv-v0", "Walker2DBulletEnv-v0"]
    for name in benchmarks:
        initial_boundary_estimate(name, int(1))
        print("done")
