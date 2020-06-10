import math
import numpy as np
import pygame
import MotionModel, CollisionModel, SensorModel

np.random.seed(0)

'''
Environment class:
    Defines the physical environemnt of the simulation
    Includes the use of motion model and collision handling to simulate physics

'''
class Environment:

    def __init__(self, env_dims=(700,400), obstacle_list=[], beacons=[], robot_pos=[30,30,0]):

        # Define pygame stuff
        pygame.init()
        self.environment = pygame.display.set_mode(env_dims)
        self.background_surface = pygame.Surface(env_dims)
        self.background_surface.fill((255, 255, 255))

        pygame.display.set_caption('Robot Simulator')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 12)

        self.render_counter = 0

        # Define environment parameters
        self.env_dims = env_dims # Dims of the window
        self.obstacle_list = obstacle_list
        self.beacons = beacons # List of beacon points -- each one is identified by its index
        self.draw_obstacles()

        # Define robot parameters
        self.robot_radius = 20
        self.robot_actual_pose = [30,30,45] # [x,y,angle] -- Origin at bottom-left
        self.collision_handler = CollisionModel.CollisionHandler(self.robot_radius,
                                                                 self.obstacle_list)
        self.collision_history = None
        self.belief_history = []

        ## Sensor model actually belongs to the robot. But the sensor readings it'd get depend on its actual pose which it doesn't know
        self.sensor_model = SensorModel.SensorModel(self.robot_actual_pose, self.beacons)


    def __del__(self):
        print("[env] Killing the environment")
        pygame.quit()


    # Setters and Getters ------------------------------------------------------

    def get_env_dims(self):
        return self.env_dims

    def set_obstacle_list(self, obstacle_list):
        self.obstacle_list = obstacle_list

    # Main methods -------------------------------------------------------------

    def draw_obstacles(self):
        for obs in self.obstacle_list:
            pygame.draw.line(self.background_surface, (0,0,0), (obs[0][0], self.env_dims[1]-obs[0][1]), (obs[1][0], self.env_dims[1]-obs[1][1]), 1)
        for b in self.beacons:
            pygame.draw.circle(self.background_surface, (0, 0, 255), (b[0], self.env_dims[1]-b[1]), 3)
            pygame.draw.circle(self.background_surface, (0, 0, 255), (b[0], self.env_dims[1]-b[1]), 7, 1)



    def draw_robot(self):
        robot_coord = [int(round(self.robot_actual_pose[0])), self.env_dims[1] - int(round(self.robot_actual_pose[1]))] # Transform to image-style coordinates (where (0,0) is at top left)
        robot_angle = self.robot_actual_pose[2]   # Orientation angle in radians
        robot_angle = - robot_angle  # In reference to x-axis as mentioned in the assignment, the default render is in a clockwise direction

        # Draw robot's sense range (transparent)
        sensor_range = self.sensor_model.get_sensor_range()
        transparent_surface = pygame.Surface((1000,750))
        transparent_surface.set_alpha(50)
        transparent_surface.fill((255,255,255))
        pygame.draw.circle(transparent_surface, (0, 255, 100), robot_coord, sensor_range)

        # Front facing robot light (transparent)
        pygame.draw.polygon(transparent_surface, (255, 255, 0), (robot_coord, (robot_coord[0] + math.cos(math.radians(robot_angle+15)) * self.robot_radius * 2.5, robot_coord[1] + math.sin(math.radians(robot_angle+15)) * self.robot_radius* 2.5), \
                    (robot_coord[0] + math.cos(math.radians(robot_angle-15)) * self.robot_radius * 2.5, robot_coord[1] + math.sin(math.radians(robot_angle-15)) * self.robot_radius* 2.5)))

        self.environment.blit(transparent_surface, (0,0))

        # Robot Design Structure
        pygame.draw.circle(self.environment, (100, 0, 0), robot_coord, self.robot_radius)
        pygame.draw.circle(self.environment, (0, 0, 0), robot_coord, round(0.75 * self.robot_radius))
        pygame.draw.circle(self.environment, (255, 255, 255), robot_coord, round(0.45 * self.robot_radius))

        ## aaline is anti-aliased line and the end_pos is determined by the angle made with the x-axis
        pygame.draw.aaline(self.environment, (255, 255, 255), robot_coord, (robot_coord[0] + math.cos(math.radians(robot_angle)) * self.robot_radius, robot_coord[1] + math.sin(math.radians(robot_angle)) * self.robot_radius))


    def draw_sensor_lines(self):
        tri_features = self.sensor_model.get_tri_features()
        if tri_features is not None:
            for i in range(len(tri_features)):
                sig = round(tri_features[i][2])
                pygame.draw.line(self.environment, (0,255,0), (self.robot_actual_pose[0], self.env_dims[1] - self.robot_actual_pose[1]), (self.beacons[sig][0], self.env_dims[1] - self.beacons[sig][1]), 2)


    def draw_robot_beliefs(self, robot):
        bel_pose_mean, bel_pose_covar = robot.get_believed_pose()

        # Draw trails
        pygame.draw.circle(self.background_surface, (0, 0, 0), (int(self.robot_actual_pose[0]), int(self.env_dims[1]-self.robot_actual_pose[1])), 1)
        pygame.draw.circle(self.background_surface, (150, 150, 150), (int(bel_pose_mean[0]), int(self.env_dims[1]-bel_pose_mean[1])), 1)

        # Draw error ellipse
        ellipse_x_axis = 2*bel_pose_covar[0][0]*math.sqrt(5.991) # For 95% connfidence
        ellipse_y_axis = 2*bel_pose_covar[1][1]*math.sqrt(5.991) #
        pygame.draw.ellipse(self.environment, (0,0,0),
                            pygame.Rect(round(bel_pose_mean[0]-ellipse_x_axis/2), round(self.env_dims[1]-(bel_pose_mean[1]+ellipse_y_axis/2)), ellipse_x_axis, ellipse_x_axis),
                            1)
        self.render_counter += 1

        if self.render_counter == 30:
            pygame.draw.ellipse(self.background_surface, (0,0,0),
                                pygame.Rect(round(bel_pose_mean[0]-ellipse_x_axis/2), round(self.env_dims[1]-(bel_pose_mean[1]+ellipse_y_axis/2)), ellipse_x_axis, ellipse_x_axis),
                                1)
            pygame.draw.aaline(self.background_surface, (0,0,0),
                              ( round(bel_pose_mean[0]), round(self.env_dims[1]-bel_pose_mean[1]) ),
                              ( round(bel_pose_mean[0]+25*math.cos(math.radians(bel_pose_mean[2]))), self.env_dims[1] - round(bel_pose_mean[1] + 25*math.sin(math.radians(bel_pose_mean[2]))) ),
                              5)
            self.render_counter = 0


    def update(self, robot):
        self.environment.blit(self.background_surface, (0,0))
        velocities = robot.get_velocities()

        # Update the robot position using the motion model --------
        robot_actual_pose_prev = self.robot_actual_pose[:]
        self.robot_actual_pose, _, _ = MotionModel.update_robot_position(self.robot_actual_pose, velocities, mode='true')

        # Handle collision --------
        corrected_coords, self.collision_history = self.collision_handler.handle_collision(np.array([self.robot_actual_pose[0], self.robot_actual_pose[1]]),
                                                                                           np.array([robot_actual_pose_prev[0], robot_actual_pose_prev[1]]),
                                                                                           self.collision_history)
        self.robot_actual_pose[:2] = corrected_coords[:]

        # Give the sensor model the actual pose so it can give sensor readings to the robot --------
        self.sensor_model.set_actual_pose(self.robot_actual_pose)
        robot.set_sensor_model(self.sensor_model)

        # Update robot's beliefs --------
        robot.kalman_update()

        # Draw sensor lines --------
        self.draw_sensor_lines()

        # Draw robot's beliefs --------
        self.draw_robot_beliefs(robot)

        # Render the robot --------
        self.draw_robot()

        pygame.display.update()
        self.clock.tick(120)

################################################################################
