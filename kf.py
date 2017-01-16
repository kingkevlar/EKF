#!/usr/bin/env python
from __future__ import division
import numpy as np

import rospy
import tf

from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from sensor_msgs.msg import Imu


class KalmanFilter(object):
    def __init__(self, F, H, x0, P0, dims=2, sensors=1):
        '''
        F  := relates last state to this state
        G  := relates last input vector to this state
        H  := relates this state to what sensors should see
        x0 := Initial state estimate
        P0 := Initial state cov
        '''
        self._dims = dims
        self._sensors = sensors

        self.F = F
        self.H = H

        self.x = x0.reshape(self._dims, 1).astype(np.float64)
        self.P = P0.astype(np.float64)

    def predict(self, u, Q, dt):
        '''
        u := input vector
        Q := process noise cov matrix
        '''
        u = np.array(u).reshape(-1, 1)
        assert Q.shape == (self._dims, self._dims), Q.shape
        self.x, Jf = self.F(self.x, u, dt)
        self.P = Jf.dot(self.P).dot(Jf.T) + Q

    def correct(self, z, R):
        '''
        z := sensor vector
        R := sensor noise model
        '''
        z = np.array(z).reshape(self._sensors, 1)
        assert R.shape == (self._sensors, self._sensors)
        
        y, Jh = self.H(self.x, z)
        S = Jh.dot(self.P).dot(Jh.T) + R
        K = self.P.dot(Jh.T).dot(np.linalg.inv(S))

        self.x = self.x + K.dot(y)
        self.P = (np.eye(self._dims) - K.dot(Jh)).dot(self.P)


class Aquire(object):
    def __init__(self, kf):
        self.pub_est = rospy.Publisher("pose_est", PoseStamped, queue_size=10)
        self.kf = kf
        
        self.imu_data = None
        self.last_imu = None

        rospy.Subscriber("/imu", Imu, self.got_imu)
        rospy.Subscriber("/sensor", Twist, self.got_sensor)

    def got_imu(self, msg):
        '''
        Got an imu message, do a predict step and save correct data for sensor measurements
        '''
        if self.last_imu is None:
            self.last_imu = msg
            return

        u = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.angular_velocity.z]
        # Not sure how this should be filled out
        Q = np.zeros(6)
        Q[0] = Q[1] = msg.linear_acceleration_covariance[0]  # x0
        Q[3] = Q[4] = msg.linear_acceleration_covariance[4]  # x1
        Q[2] = Q[5] = msg.angular_velocity_covariance[8]

        dt = (msg.header.stamp - self.last_imu.header.stamp).to_sec()

        self.kf.predict(u, np.diag(Q), dt)
        self.imu_data = msg
        self.last_imu = msg
        self.pub_esitmate()

    def got_sensor(self, msg):
        '''
        Got a sensor reading, do a correct step if imu_data has already been populated
        '''
        if self.imu_data is None:
            return
        
        x = msg.linear.x
        y = msg.linear.y
        q = self.imu_data.orientation
        theta = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        R = np.diag([1E-4, 1E-4, self.imu_data.orientation_covariance[8]])

        self.kf.correct([x, y, theta], R)
        self.imu_data = None

        self.pub_esitmate()
    
    def pub_esitmate(self):
        x = np.copy(self.kf.x).astype(np.float64)
        px = x[0]
        py = x[1]
        theta = x[2]

        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = rospy.Time.now()
        p.pose.position.x = px
        p.pose.position.y = py
        p.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, theta))

        print "x:", np.round(self.kf.x.T, 2)
        print "P:", np.round(self.kf.P.diagonal(), 4)
        print
        self.pub_est.publish(p)


if __name__ == "__main__":
    rospy.init_node("kf")

    num_sensors = 3
    num_dims = 6

    def F(x, u, dt):
        # x := [x0, x1, theta, x0dot, x1dot, thetadot]
        # u := [ax0, ax1]

        new_x = np.copy(x).astype(np.float64)

        c, s = np.cos(new_x[2]), np.sin(new_x[2]) 
        new_x[0] += 0.5 * u[0] * dt ** 2 + (c * x[3] - s * x[4]) * dt
        new_x[1] += 0.5 * u[1] * dt ** 2 + (s * x[3] + c * x[4]) * dt
        new_x[2] += x[5] * dt
        new_x[3] += (c * u[0] + s * u[1]) * dt
        new_x[4] += (-s * u[0] + c * u[1]) * dt
        new_x[5] = u[2]

        # Linearize around the previous esitmate (I think this is right)
        c, s = np.cos(x[2]), np.sin(x[2])
        dx0_dx2 = -s * x[3] * dt - c * x[4] * dt
        dx1_dx2 = -c * x[3] * dt - s * x[4] * dt
        dx3_dx2 = -s * u[0] * dt + c * u[1] * dt
        dx4_dx2 = -c * u[0] * dt - s * u[1] * dt 
        Jf = np.array([[1, 0, dx0_dx2, c * dt, -s * dt,  0],
                       [0, 1, dx1_dx2, s * dt,  c * dt,  0],
                       [0, 0,       1,      0,       0, dt],
                       [0, 0, dx3_dx2,      1,       0,  0],
                       [0, 0, dx4_dx2,      0,       1,  0],
                       [0, 0,       0,      0,       0,  0]])

        return new_x, Jf
    

    def H(x, z):
        # Return the error between the expected obs (H(x)) and the actual observation (z)
        # z := [x0, x1, theta]
        expected_z = np.zeros(3)
        expected_z[0] = x[3]
        expected_z[1] = x[4]
        expected_z[2] = x[2]
        
        # Jacobian of H function
        Jh = np.array([[0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 1, 0, 0, 0]])

        # Error
        y = z - expected_z.reshape(-1, 1)

        # Stupid orientation - compute that error here
        cx, sx = np.cos(x[2]), np.sin(x[2])
        cz, sz = np.cos(z[2]), np.sin(z[2])
        y[2] = np.arctan2(sz * cx - cz * sx, cz * cx + sz * sx)

        return y.reshape(-1, 1), Jh

    x0 = np.array([0, 0, 0, 0, 0, 0])
    P0 = np.diag([1, 1, 1, 1E-3, 1, 1])
     
    kf = KalmanFilter(F, H, x0, P0, num_dims, num_sensors)

    Aquire(kf)
    rospy.spin() 
