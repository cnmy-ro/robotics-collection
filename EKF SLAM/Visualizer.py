import math
import pygame

class Visualizer:

    def __init__(self, env_dims=(700,500), robot_pos=None, beacons=[]):
        # Define pygame stuff
        pygame.init()
        self.environment = pygame.display.set_mode(env_dims)
        self.background_surface = pygame.Surface(env_dims)
        self.background_surface.fill((255, 255, 255))

        pygame.display.set_caption('EKF SLAM simulation')
        self.clock = pygame.time.Clock()
        self.render_counter = 0

        # Environment parameters
        self.env_dims = env_dims # Dims of the window
        self.beacons = beacons # List of beacon points -- each one is identified by its index
        self.draw_beacons()

        # Robot parameters
        self.robot_radius = 20
        self.robot_actual_pose = robot_pos# [x,y,angle] -- Origin at bottom-left

    def set_sensor_model(self, sensor_model):
        self.sensor_model = sensor_model

    def set_actual_pose(self,robot_actual_pose):
        self.robot_actual_pose = robot_actual_pose
        self.sensor_model.set_actual_pose(robot_actual_pose)


    def draw_beacons(self):
        for b in self.beacons:
            pygame.draw.circle(self.background_surface, (255, 0, 0), (b[0], self.env_dims[1]-b[1]), 3)
            pygame.draw.rect(self.background_surface, (255, 0, 0), pygame.Rect((b[0]-10), (b[1]-10), 20, 20), 3)


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
        self.environment.blit(transparent_surface, (0,0))

        # Robot Design Structure
        pygame.draw.circle(self.environment, (0, 100, 0), robot_coord, self.robot_radius)
        pygame.draw.circle(self.environment, (0, 0, 0), robot_coord, round(0.70 * self.robot_radius))
        ## aaline is anti-aliased line and the end_pos is determined by the angle made with the x-axis
        pygame.draw.aaline(self.environment, (255, 255, 255), robot_coord, (robot_coord[0] + math.cos(math.radians(robot_angle)) * self.robot_radius, robot_coord[1] + math.sin(math.radians(robot_angle)) * self.robot_radius))


    def draw_sensor_lines(self):
        nearest_features = self.sensor_model.get_nearest_features()
        if nearest_features is not None:
            for i in range(len(nearest_features)):
                sig = int(nearest_features[i][2])
                pygame.draw.line(self.environment, (0,255,0), (self.robot_actual_pose[0], self.env_dims[1] - self.robot_actual_pose[1]), (self.beacons[sig][0], self.env_dims[1] - self.beacons[sig][1]), 2)


    def draw_robot_beliefs(self, robot):
        state_mean, state_covar = robot.get_believed_state()
        pose_mean, pose_covar = state_mean[:3], state_covar[:3,:3]

        self.render_counter += 1

        if self.render_counter % 10 == 0:
            # Draw trails
            pygame.draw.circle(self.background_surface, (0, 0, 0), (int(self.robot_actual_pose[0]), int(self.env_dims[1]-self.robot_actual_pose[1])), 1)
            pygame.draw.circle(self.background_surface, (150, 150, 150), (int(pose_mean[0]), int(self.env_dims[1]-pose_mean[1])), 1)

            # Draw error ellipse
            ellipse_x_axis = 2*pose_covar[0][0]*math.sqrt(5.991) # For 95% confidence
            ellipse_y_axis = 2*pose_covar[1][1]*math.sqrt(5.991) #
            pygame.draw.ellipse(self.environment, (0,0,0),
                                pygame.Rect(round(pose_mean[0]-ellipse_x_axis/2), round(self.env_dims[1]-(pose_mean[1]+ellipse_y_axis/2)), ellipse_x_axis, ellipse_y_axis),
                                1)

        if self.render_counter == 100:
            # Robot pose stuff
            pygame.draw.ellipse(self.background_surface, (50,50,50),
                                pygame.Rect(round(pose_mean[0]-ellipse_x_axis/2), round(self.env_dims[1]-(pose_mean[1]+ellipse_y_axis/2)), ellipse_x_axis, ellipse_y_axis),
                                1)
            pygame.draw.aaline(self.background_surface, (0,0,0),
                              ( round(pose_mean[0]), round(self.env_dims[1]-pose_mean[1]) ),
                              ( round(pose_mean[0]+25*math.cos(math.radians(pose_mean[2]))), self.env_dims[1] - round(pose_mean[1] + 25*math.sin(math.radians(pose_mean[2]))) ),
                              5)

            # Map stuff
            for i in range(len(self.beacons)):
                sig_idx = 5 + 3*i
                kptx, kpty, kpts = state_mean[sig_idx-2:sig_idx+1]
                kpt_covar = state_covar[sig_idx-2:sig_idx, sig_idx-2:sig_idx]
                kpt_ellipse_x_axis = 2*kpt_covar[0][0]*math.sqrt(5.991)  * 10 # Inflated to be big enough
                kpt_ellipse_y_axis = 2*kpt_covar[1][1]*math.sqrt(5.991)  * 10 #

                if kptx != -999 and kpty != -999:
                    pygame.draw.ellipse(self.background_surface, (128,100,255),
                                        pygame.Rect(round(kptx-kpt_ellipse_x_axis/2), round(self.env_dims[1]-(kpty+kpt_ellipse_y_axis/2)), kpt_ellipse_x_axis, kpt_ellipse_y_axis),
                                        1)

            self.render_counter = 0


    def update(self, robot):
        self.environment.blit(self.background_surface, (0,0))

        self.draw_sensor_lines()
        self.draw_robot_beliefs(robot)
        self.draw_robot()

        pygame.display.update()
        self.clock.tick(120)


