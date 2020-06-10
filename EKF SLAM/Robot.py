import math
import numpy as np
import Physics

'''
Robot class with EKF SLAM implementation
'''

class Robot:
    def __init__(self, init_robot_pose, beacons):
        self.velocities = [0,0] # [Translation vel, Rotation vel]
        self.velocity_step = 2
        self.beacons = beacons # List of beacon points
        self.init_pose = init_robot_pose
        self.initialize_state()
        self.sensor_model = None

    def initialize_state(self):
        init_pose = self.init_pose
        init_map = np.ones([3*len(self.beacons)]) * -999
        self.state_mean = np.hstack([init_pose, init_map])
        self.state_covar = np.identity(3 + 3*len(self.beacons))

    def get_velocities(self):
        return self.velocities

    def get_believed_state(self):
        return self.state_mean, self.state_covar

    def increment_velocity(self, vel_type):
        if vel_type == 'TRANSLATION':
            self.velocities[0] += self.velocity_step
        elif vel_type == 'ROTATION':
            self.velocities[1] += self.velocity_step

    def decrement_velocity(self, vel_type):
        if vel_type == 'TRANSLATION':
            self.velocities[0] -= self.velocity_step
        elif vel_type == 'ROTATION':
            self.velocities[1] -= self.velocity_step

    def reset_velocities(self):
        self.velocities = [0, 0]

    def set_sensor_model(self, sensor_model):
        self.sensor_model = sensor_model

    #--------------------------------------------------------------------------
    # EKF SLAM

    def EKF_SLAM_update(self):
        # Prediction -- Uses Linear motion model ------------------------------
        pose_mean = self.state_mean[:3].copy()
        pose_covar = self.state_covar[:3, :3].copy()
        pose_mean_pred, A, B, R = Physics.forward_kinematics_update(pose_mean, self.velocities, mode='noisy')
        pose_covar_pred = np.dot(A, np.dot(pose_covar, A.T)) + R

        state_mean_pred = self.state_mean
        state_mean_pred[:3] = pose_mean_pred
        state_covar_pred = self.state_covar
        state_covar_pred[:3, :3] = pose_covar_pred

        # Correction -- Uses Jacobian for control matrix ----------------------
        Q = self.sensor_model.get_noise_covar()
        Z, m = self.sensor_model.sense_keypoints()

        if len(Z) > 0: # If any feature is in sensor range
            for z_i in Z:
                j = int(z_i[2])

                sig_idx = 5 + 3*j
                if state_mean_pred[sig_idx-2] == -999 and state_mean_pred[sig_idx-1] == -999:
                    state_mean_pred[sig_idx-2] = state_mean_pred[0] + z_i[0]*math.cos( math.radians(z_i[1]) + math.radians(state_mean_pred[2]) )
                    state_mean_pred[sig_idx-1] = state_mean_pred[1] + z_i[0]*math.sin( math.radians(z_i[1]) + math.radians(state_mean_pred[2]) )
                    state_mean_pred[sig_idx] = z_i[2]


                delta = np.array([state_mean_pred[sig_idx-2] - state_mean_pred[0],
                                  state_mean_pred[sig_idx-1] - state_mean_pred[1]])

                q = np.dot(delta.T, delta)

                z_i_hat = np.array([math.sqrt(q),
                                    math.degrees(math.atan2(delta[1], delta[0]) - state_mean_pred[2]),
                                    state_mean_pred[sig_idx]])

                z_i_hat[1] = ( z_i_hat[1]/abs(z_i_hat[1]) ) * (z_i_hat[1] % 180)

                F = np.zeros([6, 3 + 3*len(self.beacons)])
                F[:3, :3] = np.identity(3)
                F[3:, sig_idx-2:sig_idx+1] = np.identity(3)

                H =  np.array([[-math.sqrt(q)*delta[0], -math.sqrt(q)*delta[1], 0, math.sqrt(q)*delta[0], math.sqrt(q)*delta[1], 0],
                               [delta[1], -delta[0], -q, -delta[1], delta[0], 0],
                               [0, 0, 0, 0, 0, q]])
                H = (1/q) * np.dot(H,F)

                tempMat = np.dot(H, np.dot(state_covar_pred, H.T)) + Q
                tempMat = np.linalg.inv(tempMat)
                K = np.dot(state_covar_pred, np.dot(H.T, tempMat))

                state_mean_pred += np.dot(K, z_i-z_i_hat)
                state_covar_pred = np.dot(np.identity(state_mean_pred.shape[0])-np.dot(K,H), state_covar_pred)

            self.state_mean = state_mean_pred
            self.state_covar = state_covar_pred

        else:
            self.state_mean = state_mean_pred
            self.state_covar = state_covar_pred