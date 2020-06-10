import math
import numpy as np

def euclidean_distance(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.linalg.norm(pt1-pt2, ord=2)


###############################################################################
#  Simple constant-velocity motion model

def forward_kinematics_update(robot_pose, velocities, mode='true', time_step=0.3):
    angle = robot_pose[2]
    velocities = np.array(velocities)

    A = np.identity(3) # Transformation without controls
    B = np.array([[math.cos(math.radians(angle))*time_step, 0], # Control matrix
                  [math.sin(math.radians(angle))*time_step, 0],
                  [0, time_step]])
    # Update the robot state
    robot_pose_new = np.dot(A, robot_pose) + np.dot(B, velocities)

    if mode == 'true': # For the environement
        robot_pose_new[0] = max(robot_pose_new[0], 0+20)
        robot_pose_new[0] = min(robot_pose_new[0], 1000-20)

        robot_pose_new[1] = max(robot_pose_new[1], 0+20)
        robot_pose_new[1] = min(robot_pose_new[1], 700-20)
        return robot_pose_new, A, B

    if mode == 'noisy': # For the robot
        R = np.identity(3) * np.array([10, 10, 5])
        return robot_pose_new, A, B, R



###############################################################################
#  Sensor Model

'''
Sensor Model class:
    Uses features/keypoints with known correspondence
'''

class SensorModel:
    def __init__(self, robot_actual_pose, beacons):
        self.robot_actual_pose = robot_actual_pose
        self.beacons = beacons
        self.sensor_range = 350

        # Control params
        self.C = np.identity(3)
        self.Q = np.identity(3) * np.array([5, 5, 0.1])

        self.nearest_features = None

    def set_actual_pose(self, robot_actual_pose):
        self.robot_actual_pose = robot_actual_pose

    def get_C_matrix(self):
        return self.C

    def get_noise_covar(self):
        return self.Q

    def get_nearest_features(self):
        return self.nearest_features

    def get_sensor_range(self):
        return self.sensor_range

    # Robot can only access these functions ######################################

    def sense_keypoints(self):
        robot_angle = self.robot_actual_pose[2]

        # Get feature data from beacons
        feature_vector = []
        for i in range(len(self.beacons)):
            range_i = euclidean_distance(self.beacons[i], self.robot_actual_pose[:2])
            bearing_i = math.atan2(self.beacons[i][1]-self.robot_actual_pose[1], self.beacons[i][0]-self.robot_actual_pose[0]) - math.radians(robot_angle)
            signature_i = i

            feature = np.array([range_i, math.degrees(bearing_i), signature_i])
            feature_vector.append(feature.tolist())

        feature_vector = sorted(feature_vector, key=lambda element:element[0])
        # Features (beacons) within sensor range
        nearest_features = [f[:] for f in feature_vector if f[0]<self.sensor_range]
        if len(nearest_features) > 0:
            nearest_features = np.array(nearest_features)

        self.nearest_features = nearest_features

        return nearest_features, self.beacons