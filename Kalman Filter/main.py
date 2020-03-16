import pygame
from Environment import Environment
from Robot import Robot
import logging

logging.basicConfig(level=logging.INFO)
pygame.init()

################################################################################
# Initialize the Simulation parameters
################################################################################

def init_sim_parameters(env_dims=(700, 400)):
    # Define obstacles, other than the boundaries
    obstacles = [
                  [ (int(env_dims[0]/6), int(env_dims[1]/2)),  (int(env_dims[0]-env_dims[0]/6), int(env_dims[1]/2)) ],
                  [ (int(env_dims[0]/6), int(env_dims[1]/2)),  (int(env_dims[0]/6), int(env_dims[1]/3)) ],
                  [ (int(env_dims[0]-env_dims[0]/6), int(env_dims[1]/2)),  (int(env_dims[0]-env_dims[0]/6), int(env_dims[1]-env_dims[1]/3)) ]
                ]
    # Get full list of obstacles, including the boundaries
    obstacle_list = [((0, 0), (env_dims[0], 0)),
                     ((0, 0), (0, env_dims[1])),
                     ((0, env_dims[1]),(env_dims[0], env_dims[1])),
                     ((env_dims[0], env_dims[1]),(env_dims[0], 0))]
    for obs in obstacles:
        obstacle_list.append(tuple(obs))

    # Set the beacon points
    beacons = []  # List of beacon points -- each one is identified by its index
    for segment in obstacle_list:
        for point in segment:
            if point not in beacons:
                beacons.append(point)

    # Initialize Environment
    init_robot_pose = [30,30,0]
    environment = Environment(env_dims, obstacle_list, beacons, init_robot_pose)

    # Create robot object
    robot = Robot(beacons)

    return environment, robot

################################################################################
# MANUAL CONTROL
################################################################################
'''
CONTROLS --
    >> W,S,A,D - As in the slides
'''
def manual_control(environment, robot):
    # Start scanning for user input
    crashed = False
    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN:
                # Handle 'W' -- increase V_tr
                if event.key == pygame.K_w:
                    logging.debug('[main] User command: V_tr ++')
                    robot.increment_velocity('TRANSLATION')
                # Handle 'S' -- decrease V_tr
                elif event.key == pygame.K_s:
                    logging.debug('[main] User command: V_tr --')
                    robot.decrement_velocity('TRANSLATION')
                # Handle 'D' -- increase V_ro
                elif event.key == pygame.K_d:
                    logging.debug('[main] User command: V_ro ++')
                    robot.increment_velocity('ROTATION')
                # Handle 'A' -- decrease V_ro
                elif event.key == pygame.K_a:
                    logging.debug('[main] User command: V_ro --')
                    robot.decrement_velocity('ROTATION')
                # Handle 'X' -- set V_tr = V_ro = 0
                elif event.key == pygame.K_x:
                    logging.debug('[main] User command: Stop')
                    robot.reset_velocities()

        # Update environment
        environment.update(robot)
    pygame.quit()

################################################################################

if __name__ == '__main__':
    env_dims = (1000,600)
    environment, robot = init_sim_parameters(env_dims)
    manual_control(environment, robot)