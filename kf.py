#!/usr/bin/env python
from __future__ import division
import numpy as np

import rospy
import tf

from geometry_msgs.msg import PoseStamped, Quaternion
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

        self._x, Jf = self.F(self.x, u, dt)
        self._P = Jf.dot(self.P).dot(Jf.T) + Q
        
    def correct(self, z, R):
        '''
        z := sensor vector
        R := sensor noise model
        '''
        z = np.array(z).reshape(self._sensors, 1)
        assert R.shape == (self._sensors, self._sensors)
        
        y, Jh = self.H(self._x, z)
        S = Jh.dot(self._P).dot(Jh.T) + R
        K = self._P.dot(Jh.T).dot(np.linalg.inv(S))

        self.x = self._x + K.dot(y)
        self.P = (np.eye(self._dims) - K.dot(Jh)).dot(self._P)


class Aquire(object):
    def __init__(self, kf):
        self.pub_est = rospy.Publisher("pose_est", PoseStamped, queue_size=10)
        self.kf = kf
        
        self.imu_data = None
        self.last_imu = None

        rospy.Subscriber("/imu", Imu, self.got_imu)
        rospy.Subscriber("/pos_sensor", PoseStamped, self.got_sensor)

    def got_imu(self, msg):
        '''
        Got an imu message, do a predict step and save correct data for sensor measurements
        '''
        if self.last_imu is None:
            self.last_imu = msg
            return

        u = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.angular_velocity]
        Q = np.zeros(6)
        Q[0] = Q[1] = msg.linear_acceleration_covariance[0]  # x0
        Q[3] = Q[4] = msg.linear_acceleration_covariance[4]  # x1
        Q[2] = Q[5] = msg.angular_velocity_covariance[8]

        dt = (msg.header.stamp - self.last_imu.header.stamp).to_sec()

        self.kf.predict(u, np.diag(Q), dt)
        self.imu_data = msg
        self.last_imu = msg

    def got_sensor(self, msg):
        '''
        Got a sensor reading, do a correct step if imu_data has already been populated
        '''
        if self.imu_data is None:
            return
        
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = self.imu_data.orientation
        theta = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        R = np.diag([1E-4, 1E-4, self.imu_data.orientation_covariance[8]])

        self.kf.correct([x, y, theta], R)
        self.pub_esitmate()
    
    def pub_esitmate(self):
        x = self.kf.x
        px = x[0]
        py = x[1]
        theta = x[2]

        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = rospy.Time.now()
        p.pose.position.x = px
        p.pose.position.y = py
        p.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, theta))

        np.set_printoptions(precision=2)
        print "x:", np.round(self.kf.x.T, 2)
        print "P:", np.round(self.kf.P.diagonal(), 2)
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

        new_x[2] += x[5] * dt
        c, s = np.cos(new_x[2]), np.sin(new_x[2]) 
        new_x[0] += 0.5 * u[0] * dt ** 2 + (c * x[3] - s * x[4]) * dt
        new_x[1] += 0.5 * u[1] * dt ** 2 + (s * x[3] + c * x[4]) * dt
        new_x[3] += (c * u[0] + s * u[1]) * dt
        new_x[4] += (-s * u[0] + c * u[1]) * dt
        new_x[5] = u[3]

        Jf = np.array([[1, 0, 0, c * dt, -s * dt,  0],
                       [0, 1, 0, s * dt,  c * dt,  0],
                       [0, 0, 1,      0,       0, dt],
                       [0, 0, 0,      1,       0,  0],
                       [0, 0, 0,      0,       1,  0],
                       [0, 0, 0,      0,       0,  1]])

        return new_x, Jf
    

    def H(x, z):
        # Return the error between the expected obs (H(x)) and the actual observation (z)
        # z := [x0, x1, theta, thetadot]
        expected_z = np.zeros(3)
        expected_z[0] = x[0]
        expected_z[1] = x[1]
        expected_z[2] = x[2]
        
        # Jacobian of H function
        Jh = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0]])

        # Error
        y = z - expected_z.reshape(-1, 1)

        # Stupid orientation - compute that error here
        c, s = np.cos(x[2]), np.sin(x[2])
        cz, sz = np.cos(z[2]), np.sin(z[2])
        y[2] = np.arctan2(sz * c - cz * s, cz * c + sz * s )

        return y.reshape(-1, 1), Jh

    x0 = np.array([0, 0, 0, 0, 0, 0])
    P0 = np.diag([1, 1, 1, 1, 1, 1])
     
    kf = KalmanFilter(F, H, x0, P0, num_dims, num_sensors)

    Aquire(kf)
    rospy.spin() 
