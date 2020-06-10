import pygame
from Visualizer import Visualizer
from Robot import Robot
import Physics

pygame.init()

# Initialize the Simulation parameters
def initialize_sim_params():
    env_dims = (1000,700)
    X,Y = env_dims[0], env_dims[1]
    beacons = [(0, 0),
               (X, 0),
               (0, Y),
               (X, Y),
               (int(X/6), int(Y/3)),
               (int(3*X/6), int(Y/3)),
               (int(5*X/6), int(Y/3)),
               (int(X/6), int(2*Y/3)),
               (int(3*X/6), int(2*Y/3)),
               (int(5*X/6), int(2*Y/3)) ]
    # Initial robot position
    init_robot_pose = [X/6,Y/2,0]
    return env_dims, init_robot_pose, beacons

###############################################################################

if __name__ == '__main__':

    env_dims, init_robot_pose, beacons = initialize_sim_params()

    sensor_model = Physics.SensorModel(init_robot_pose, beacons)
    robot = Robot(init_robot_pose, beacons)
    robot.set_sensor_model(sensor_model)

    visualizer = Visualizer(env_dims, init_robot_pose, beacons)
    visualizer.set_sensor_model(sensor_model)

    robot_actual_pose = init_robot_pose
    crashed = False
    while not crashed: # Loop to scan user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYDOWN:
                # Handle 'W' -- increase V_tr
                if event.key == pygame.K_w:
                    robot.increment_velocity('TRANSLATION')
                # Handle 'S' -- decrease V_tr
                elif event.key == pygame.K_s:
                    robot.decrement_velocity('TRANSLATION')
                # Handle 'D' -- increase V_ro
                elif event.key == pygame.K_d:
                    robot.increment_velocity('ROTATION')
                # Handle 'A' -- decrease V_ro
                elif event.key == pygame.K_a:
                    robot.decrement_velocity('ROTATION')
                # Handle 'X' -- set V_tr = V_ro = 0
                elif event.key == pygame.K_x:
                    robot.reset_velocities()

        velocities = robot.get_velocities()
        robot_actual_pose, _, _ = Physics.forward_kinematics_update(robot_actual_pose, velocities, mode='true')
        visualizer.set_actual_pose(robot_actual_pose)
        robot.EKF_SLAM_update() # Run SLAM update
        visualizer.update(robot)

    pygame.quit()