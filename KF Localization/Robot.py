import numpy as np
import MotionModel

'''
Robot class:
    Defines a (realistic) robot -- i.e. it has access to same data as a real robot would
'''

class Robot:
    def __init__(self, beacons):
        self.velocities = [0,0] # [Translation vel, Rotation vel]
        self.velocity_step = 2
        self.beacons = beacons # List of beacon points -- each one is identified by its index
        self.believed_pose_mean = np.array([200,200,0])
        self.believed_pose_covar = np.identity(3)
        self.sensor_model = None

    # Getters amd Setters
    def get_velocities(self):
        return self.velocities

    def get_believed_pose(self):
        return self.believed_pose_mean, self.believed_pose_covar

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

    # Kalman Filter
    def kalman_update(self):
        # Prediction -- based on values from previous time step

        bel_pose_mean_pred, A, B, R = MotionModel.update_robot_position(self.believed_pose_mean, self.velocities, mode='noisy')
        bel_pose_covar_pred = np.dot(A, np.dot(self.believed_pose_covar, A.T)) + R

        # Correction -- incorporating sensor readings from current time step
        C = self.sensor_model.get_C_matrix()
        Q = self.sensor_model.get_noise_covar()
        Z = self.sensor_model.estimate_self_pos()
        if Z is not None:
            M = np.dot(C, np.dot(bel_pose_covar_pred, C.T)) + Q
            M_inv = np.linalg.inv(M)

            kalman_gain = np.dot(bel_pose_covar_pred,np.dot(C.T, M_inv))
            self.believed_pose_mean = bel_pose_mean_pred + np.dot(kalman_gain,(Z - np.dot(C, bel_pose_mean_pred)))
            self.believed_pose_covar = np.dot(np.identity(3)-np.dot(kalman_gain,C), bel_pose_covar_pred)
        else:
            self.believed_pose_mean = bel_pose_mean_pred
            self.believed_pose_covar = bel_pose_covar_pred


	def kalman_update_2(self):
        # Prediction -- based on values from previous time step

        bel_pose_mean_pred, A, B, R = MotionModel.update_robot_position(self.believed_pose_mean, self.velocities, mode='noisy')
        bel_pose_covar_pred = np.dot(A, np.dot(self.believed_pose_covar, A.T)) + R

        # Correction -- incorporating sensor readings from current time step
        Q = self.sensor_model.get_noise_covar()
        Z, m = self.sensor_model.sense_keypoints()

        print(bel_pose_mean_pred)
        if len(Z) > 0:
            for z_i in Z:

                j = int(z_i[2])

                q = np.linalg.norm(np.array(m[j])-bel_pose_mean_pred[:2], ord=2) ** 2

                z_i_hat = np.array([math.sqrt(q),
                                     math.degrees(math.atan2(m[j][1]-bel_pose_mean_pred[1], m[j][0]-bel_pose_mean_pred[0]) - bel_pose_mean_pred[2]),
                                     j])

                z_i_hat[1] = ( z_i_hat[1]/abs(z_i_hat[1]) ) * (z_i_hat[1] % 180)

                H = np.array([[-(m[j][0]-bel_pose_mean_pred[0])/math.sqrt(q), -(m[j][1]-bel_pose_mean_pred[1])/math.sqrt(q), 0],
                             [(m[j][1]-bel_pose_mean_pred[1])/q, -(m[j][0]-bel_pose_mean_pred[0])/q, -1],
                             [0, 0, 0]])

                S = np.dot(H, np.dot(bel_pose_covar_pred, H.T)) + Q
                S_inv = np.linalg.inv(S)
                kalman_gain = np.dot(bel_pose_covar_pred,np.dot(H.T, S_inv))

                bel_pose_mean_pred = bel_pose_mean_pred + np.dot(kalman_gain,(z_i - z_i_hat))
                bel_pose_covar_pred = np.dot(np.identity(3)-np.dot(kalman_gain,H), bel_pose_covar_pred)

            self.believed_pose_mean = bel_pose_mean_pred
            self.believed_pose_covar = bel_pose_covar_pred
            print("\n")


        else:
            self.believed_pose_mean = bel_pose_mean_pred
            self.believed_pose_covar = bel_pose_covar_pred