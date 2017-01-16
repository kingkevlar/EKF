#!/usr/bin/env python
from __future__ import division
import numpy as np

import rospy
import tf

from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from sensor_msgs.msg import Imu


class State(object):
    @classmethod
    def zeros(cls):
        return cls(*[0] * 14)

    def __init__(self, x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, t):
        self.x = x; self.y = y; self.z = z
        self.qx = qx; self.qy = qy; self.qz = qz; self.qw = qw
        self.vx = vx; self.vy = vy; self.vz = vz
        self.wx = wx; self.wy = wy; self.wz = wz;
        self.t = t

    def __repr__(self):
        return "<{s.x}, {s.y}, {s.z}>".format(s=self)

    def rotation_matrix(self):
        ''' Maps body -> world '''
        return tf.transformations.quaternion_matrix([self.qx, self.qy, self.qz, self.qw])[:3, :3]

    def world_v(self):
        ''' Returns the velocity in the world frame '''
        return self.rotation_matrix().dot([self.vx, self.vy, self.vz])

    def angular_velocity_quat(self, dt):
        ''' Returns '''
        r = np.zeros(4).astype(np.float64)
        w = np.array([self.wx, self.wy, self.wz]) * dt

        angle = np.linalg.norm(w)
        r[3] = np.cos(angle / 2)
        r[:3] = w * np.sin(angle / 2) / angle  # sinc(angle / 2) / 2
        return r

class InputVector(object):
    def __init__(self, ax, ay, az, wx, wy, wz):
        self.ax = ax; self.ay = ay; self.az = az
        self.wx = wx; self.wy = wy; self.wz = self.wz

    def __repr__(self):
        return "<{s.ax}, {s.ay}, {s.az}, {s.wx}, {s.wy}, {s.wz}>".format(s=self)

    def body_a(self, x):
        ''' Returns acceleration in the body frame of the x State '''
        rot = x.rotation_matrix()
        return rot.T.dot([self.ax, self.ay, self.az])


class ObservationVector(object):
    pass


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

        self.x = x0 
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
    num_dims = 14  # [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, t]

    def F(x, u, dt):
        # x := State
        # u := InputVector

        # ===============================================================
        # Compute new State =============================================
        # ===============================================================
        new_x = State.zeros() 

        # x = 1/2*a*t^2 + v*dt + x
        # But v is in body and a and x are in world frame.
        world_v = x.world_v()
        new_x.x = 0.5 * u.ax * dt ** 2 + world_v[0] * dt + x.y
        new_x.y = 0.5 * u.ay * dt ** 2 + world_v[1] * dt + x.x
        new_x.z = 0.5 * u.az * dt ** 2 + world_v[2] * dt + x.z
        
        # From oritools example
        dq = x.angular_velocity_quat(dt)
        q = trns.quaternion_multiply([x.qx, x.qy, x.qz, x.qw], dq) 
        new_x.qx = q[0]
        new_x.qy = q[1]
        new_x.qz = q[2]
        new_x.qw = q[3]

        # v = a*t + v
        # But again, v is in body and a is in world
        body_a = u.body_a(x)
        new_x.vx = body_a[0] * dt + x.vx 
        new_x.vy = body_a[1] * dt + x.vy 
        new_x.vz = body_a[2] * dt + x.vz 

        # Angular velocity comes straight from input 
        new_x.wx = u.wx
        new_y.wy = u.wy
        new_z.wz = u.wz

        # Time goes forward 1 dt
        new_x.t += dt

        # ===============================================================
        # Compute Jacobian for this state ===============================
        # ===============================================================

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
