# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-03-15 13:32:20
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-03-16 00:15:14
# -------------------------------
# Get states of all bodies and joints and reset them to specific state
# Note, this cannot ensure the deterministic of a system,
# because the contact(collision) information is not stored.
# pybullet.saveState and pybullet.restoreState do store the contact information. 
# A way to get deterministic excution is store a state with pybullet.saveState and modify the state with this util.
# Note, please call pybullet.setPhysicsEngineParameter(deterministicOverlappingPairs=1) 
# to make sure the collision tree deterministic.

import numpy as np

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

    return np.array(_flatten(state)) if flatten else np.array(state)


def reset_current_system_state(client, state):
    # print("call reset_current_system_state")
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