import math
import numpy as np
from geometer.shapes import Segment
from geometer.point import Point, Line
from shapely.geometry import LineString, Polygon


def collision_activation(collision):
    return np.power((np.e/2), -collision)

class CollisionHandler:
    def __init__(self, robot_radius, obstacle_list):
        self.robot_radius = robot_radius
        self.obstacle_list = obstacle_list

    def _euc_distance(self, p1, p2):
        dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        return dist


    def _get_angle(self, obs, adj_obs):
        # Get corner
        if obs[0] == adj_obs[0]: corner, E, F = obs[0], obs[1], adj_obs[1]
        elif obs[0] == adj_obs[1]: corner, E, F = obs[0], obs[1], adj_obs[0]
        elif obs[1] == adj_obs[0]: corner, E, F = obs[1], obs[0], adj_obs[1]
        elif obs[1] == adj_obs[1]: corner, E, F = obs[1], obs[0], adj_obs[0]

        CE_vec = np.array(E) - np.array(corner)
        CF_vec = np.array(F) - np.array(corner)

        CE_len = np.linalg.norm(CE_vec,ord=2)
        CF_len = np.linalg.norm(CF_vec,ord=2)

        corner_angle = math.acos(np.dot(CE_vec, CF_vec)/(CE_len*CF_len))

        return corner_angle, corner, CE_vec, CF_vec



    def set_obstacles(self, obstacle_list):
        self.obstacle_list = obstacle_list


    def _detect_collision(self, obs, robot_coord_curr, robot_coord_prev):
        obs_line = Line(Point(obs[0][0], obs[0][1]), Point(obs[1][0], obs[1][1]))
        projected_point = obs_line.project( Point(robot_coord_curr[0], robot_coord_curr[1]) )

        circle_obs_cross = Segment(Point(obs[0][0],obs[0][1]), Point(obs[1][0],obs[1][1])).contains(projected_point) and \
                           self._euc_distance(robot_coord_curr, projected_point.normalized_array) < self.robot_radius

        displacement_obs_cross = LineString([(robot_coord_curr[0], robot_coord_curr[1]), (robot_coord_prev[0], robot_coord_prev[1])]).crosses(LineString([(obs[0][0], obs[0][1]),(obs[1][0], obs[1][1])]))

        return obs_line, circle_obs_cross, displacement_obs_cross


    def handle_collision(self, robot_coord_curr, robot_coord_prev, prev_obs): # New new function -- can handle corners
        robot_coord_corrected = robot_coord_curr.copy()
        # Get intersection of robot and obstacle
        if prev_obs:
            obs = prev_obs
            obs_line, circle_obs_cross, displacement_obs_cross = self._detect_collision(obs, robot_coord_curr, robot_coord_prev)

            if circle_obs_cross or displacement_obs_cross: # If collision is detected
                obs_vector = np.array(obs[1])-np.array(obs[0])
                parallel_comp = ( np.dot(robot_coord_curr-robot_coord_prev, obs_vector) )  * obs_vector/np.linalg.norm(obs_vector,ord=2)**2

                projected_point_2 = obs_line.project( Point(robot_coord_prev[0], robot_coord_prev[1]) )
                MP = robot_coord_prev - projected_point_2.normalized_array[:2]

                robot_coord_corrected = projected_point_2.normalized_array[:2] + \
                                        parallel_comp + \
                                        (MP/np.linalg.norm(MP,ord=2)) * self.robot_radius

                return robot_coord_corrected, obs


        for obs in self.obstacle_list:
            adjacent_obs = [o for o in self.obstacle_list if o!=obs and (o[0]==obs[0] or o[0]==obs[1] or o[1]==obs[0] or o[1]==obs[1]) ]

            obs_line, circle_obs_cross, displacement_obs_cross = self._detect_collision(obs, robot_coord_curr, robot_coord_prev)

            if circle_obs_cross or displacement_obs_cross: # If collision is detected
                obs_vector = np.array(obs[1])-np.array(obs[0])
                parallel_comp = ( np.dot(robot_coord_curr-robot_coord_prev, obs_vector) )  * obs_vector/np.linalg.norm(obs_vector,ord=2)**2

                projected_point_2 = obs_line.project( Point(robot_coord_prev[0], robot_coord_prev[1]) )
                MP = robot_coord_prev - projected_point_2.normalized_array[:2]

                robot_coord_corrected = projected_point_2.normalized_array[:2] + \
                                        parallel_comp + \
                                        (MP/np.linalg.norm(MP,ord=2)) * self.robot_radius

                # Corner correction -- Basically, if corner angle >= 90 and if there is a collision with the adjacent wall after applying the above correction
                for adj_obs in adjacent_obs:
                    adj_obs_line, circle_adj_obs_cross, displacement_adj_obs_cross = self._detect_collision(adj_obs, robot_coord_corrected, robot_coord_prev)

                    corner_angle, corner, CE_vec, CF_vec = self._get_angle(obs, adj_obs)

                    if (circle_adj_obs_cross or displacement_adj_obs_cross) and math.degrees(corner_angle) <= 90.0:
                        #print("\n")
                        #print("corner: ", corner)
                        #print("CE_vec: ", CE_vec)
                        #print("CF_vec: ", CF_vec)
                        #print("Sharp Corner: ", math.degrees(corner_angle), "degrees")
                        #print("h: ", self.robot_radius/math.sin(corner_angle/2))

                        C_Pcc_vec = CE_vec/np.linalg.norm(CE_vec,ord=2) + CF_vec/np.linalg.norm(CF_vec,ord=2)
                        C_Pcc_unit_vec = C_Pcc_vec/np.linalg.norm(C_Pcc_vec,ord=2)

                        #print("C_Pcc_vec: ", C_Pcc_vec)

                        robot_coord_corrected = np.array(corner) + \
                                                ( C_Pcc_unit_vec ) * (self.robot_radius / math.sin(corner_angle/2))

                        return robot_coord_corrected, None

        return robot_coord_corrected, None