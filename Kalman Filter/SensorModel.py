import math
import numpy as np

def sort_key(element):
    return element[0]

def get_circle_circle_intersection(c0, r0, c1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    x0, y0 = c0
    x1, y1 = c1
    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d
        y2=y0+a*(y1-y0)/d
        x3=x2+h*(y1-y0)/d
        y3=y2-h*(x1-x0)/d
        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        pt1 = (x3, y3)
        pt2 = (x4, y4)
        return pt1, pt2

def euclidean_distance(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.linalg.norm(pt1-pt2, ord=2)

################################################################################
#                                 Sensor Model
################################################################################
'''
Sensor Model class:
    Uses the given beacons like GPS satellites to trianglute robot's position
'''

class SensorModel:
    def __init__(self, robot_actual_pose, beacons):
        # The robot CANNOT access the actual pose.
        # It can only get the readings based on it -- i.e. a (indirect) measurement that's used by the Kalman Filter to estimate the pose
        self.robot_actual_pose = robot_actual_pose
        self.beacons = beacons
        self.sensor_range = 350

        # Control params
        self.C = np.identity(3)
        self.Q = np.identity(3) * np.array([10, 10, 2])

        # Triangulation stuff
        self.nearest_features = None
        self.tri_features = None
        self.tolerance = 0.1

    def set_actual_pose(self, robot_actual_pose):
        self.robot_actual_pose = robot_actual_pose

    def get_C_matrix(self):
        return self.C

    def get_noise_covar(self):
        return self.Q

    def get_tri_features(self):
        return self.tri_features

    def get_sensor_range(self):
        return self.sensor_range

    # Robot can only access this function ######################################
    def triangulate(self):
        robot_angle = self.robot_actual_pose[2]

        # Get feature data from beacons
        feature_vector = []
        for i in range(len(self.beacons)):
            range_i = euclidean_distance(self.beacons[i], self.robot_actual_pose[:2])
            bearing_i = math.atan2(self.beacons[i][1]-self.robot_actual_pose[1], self.beacons[i][0]-self.robot_actual_pose[0]) - math.radians(robot_angle)
            signature_i = i

            feature = np.array([range_i, math.degrees(bearing_i), signature_i])
            feature_vector.append(feature.tolist())

        feature_vector = sorted(feature_vector, key=sort_key)

        # Features (beacons) within sensor range
        nearest_features = [f[:] for f in feature_vector if f[0]<self.sensor_range]

        # Define the noise
        noise = np.random.multivariate_normal(mean=[0,0,0], cov=self.Q)

        # Triangulation --------------------------------------------------------
        tri_features = [] # List of features that were used for triangulation
        Z = None
        for i in range(len(nearest_features)):
            feature_a = nearest_features[i]
            sig_a = round(feature_a[2])
            beacon_a_coord = self.beacons[sig_a]
            distance_a = feature_a[0]

            z_x, z_y = beacon_a_coord
            z_robot_angle = math.atan2(beacon_a_coord[1]-z_y, beacon_a_coord[0]-z_x) - math.radians(feature_a[1])
            z_robot_angle = math.degrees(z_robot_angle)
#
#            tri_features = list([nearest_features[i]])
#            Z = np.dot(self.C, np.array([z_x, z_y, z_robot_angle])) + noise
#            self.tri_features = tri_features

            # Look for a 2nd beacon whose 'circle' intersects with the 1st
            intersect = False
            for j in range(len(nearest_features)):
                intersection_pts = None
                if j != i:
                    feature_b = nearest_features[j]
                    sig_b = round(feature_b[2])
                    beacon_b_coord = self.beacons[sig_b]
                    distance_b = feature_b[0]
                    intersection_pts = get_circle_circle_intersection(beacon_a_coord, distance_a, beacon_b_coord, distance_b)
                    if intersection_pts:
                        intersect = True
                        intersection_pts = np.array(intersection_pts)

                        # Estimate robot's position using 2 beacons' "intersection", just in case a 3rd beacon isn't found to triangulate
                        z_x = np.mean(intersection_pts[:,0])
                        z_y = np.mean(intersection_pts[:,1])
                        z_robot_angle = math.atan2(beacon_a_coord[1]-z_y, beacon_a_coord[0]-z_x) - math.radians(feature_a[1])
                        z_robot_angle = math.degrees(z_robot_angle)

                        tri_features = list([nearest_features[i], nearest_features[j]])
                        Z = np.dot(self.C, np.array([z_x, z_y, z_robot_angle])) + noise
                        break

            # Check for a 3rd beacon only if the first 2 circles intersect
            if intersect == True:
                for k in range(len(nearest_features)):
                    if k != i and k != j:
                        feature_c = nearest_features[k]
                        sig_c = round(feature_c[2])
                        beacon_c_coord = self.beacons[sig_c]
                        distance_c = feature_c[0]
                        for pt in intersection_pts:
                            if abs(euclidean_distance(pt, beacon_c_coord) - distance_c) < self.tolerance: # If the robot is at point 'pt'
                                z_x, z_y = pt
                                z_robot_angle = math.atan2(beacon_c_coord[1]-z_y, beacon_c_coord[0]-z_x) - math.radians(feature_c[1])
                                z_robot_angle = math.degrees(z_robot_angle)

                                tri_features = list([nearest_features[i], nearest_features[j], nearest_features[k]])
                                Z = np.dot(self.C, np.array([z_x, z_y, z_robot_angle])) + noise

#                                return Z
            self.tri_features = tri_features
            return Z

