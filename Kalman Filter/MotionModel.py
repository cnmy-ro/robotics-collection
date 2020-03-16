import math
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


'''
Simple velocity-based motion model
2 modes: 'true' -- True model used by the environment to accurately simulate the robot movement
         'noisy' -- Noisy model used by the robot in its Kalman update
'''


def update_robot_position(robot_pose, velocities, mode='true', time_step=1):
    # INPUT: Current pose and current velocities
    # Output: New pose

    angle = robot_pose[2]
    velocities = np.array(velocities)

    # Define matrices:
    #   A - for transformation of robot state into new state excluding the control information
    #   B - for applying the controls (i.e. the velocities) in the transformation
    A = np.identity(3)
    B = np.array([[math.cos(math.radians(angle))*time_step, 0],
                  [math.sin(math.radians(angle))*time_step, 0],
                  [0, time_step]])
    # Update the robot state
    robot_pose_new = np.dot(A, robot_pose) + np.dot(B, velocities)

    if mode == 'true': # For the environement
        return robot_pose_new, A, B

    if mode == 'noisy': # For the robot
        R = np.identity(3) * np.array([5, 5, 1])
#        noise = np.random.multivariate_normal(mean=[0,0,0], cov=R)
#        robot_pose_new += noise
        return robot_pose_new, A, B, R