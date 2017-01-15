#!/usr/env/bin python
import rospy
import numpy as np
from tf import transformations

from geometry_msgs.msg import PoseStamped, WrenchStamped, Quaternion, Vector3
from sensor_msgs.msg import Imu


class State(object):
    def __init__(self, x0, x1, theta, x0dot, x1dot, thetadot):
        self.x0 = x0  # World, m
        self.x1 = x1  # World, m
        self.theta = theta  # World, rads
        self.x0dot = x0dot  # Body, m/s
        self.x1dot = x1dot  # Body, m/s
        self.thetadot = thetadot  # World, rads/s

    def as_PoseStamped(self):
        p = PoseStamped()
        p.header.frame_id = "map"
        p.pose.position.x = self.x0
        p.pose.position.y = self.x1
        p.pose.orientation = Quaternion(*transformations.quaternion_from_euler(0, 0, self.theta))
        return p

    def apply_force(self, a, angular_a, dt):
        '''
        Apply a world frame force to this state.
        '''
        body_a = self._rot_mat(self.theta).T.dot(a[:2])  # body, m/s^2
        body_a *= dt  # body, m/s

        self.thetadot = angular_a * dt + self.thetadot  # body, rads/s

        self.x0dot = self.x0dot + body_a[0]
        self.x1dot = self.x1dot + body_a[1]

    def step_time(self, dt):
        '''
        Steps this state forward one timestep of `dt` seconds.
        '''
        world_v = self._rot_mat(self.theta).dot([self.x0dot, self.x1dot])  # world, m/s
        world_v *= dt  # world, m

        self.theta = self.theta + self.thetadot * dt
        
        self.x0 = self.x0 + world_v[0]
        self.x1 = self.x1 + world_v[1]

    def _rot_mat(self, theta):
        '''
        Converts body->world frame
        '''
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

class IMU(object):
    def __init__(self, s0):
        self.last_v = np.array([s0.x0dot, s0.x1dot]) 

    def orientation(self, state, cov=1E-1):
        '''
        Returns quaternion orentation with given covariance 
        '''
        theta = state.theta
        theta += np.random.normal(scale=cov)
        
        cov_mat = np.zeros(9, dtype=np.float64)
        cov_mat[8] = cov

        q = Quaternion(*transformations.quaternion_from_euler(0, 0, theta))
        return q, cov_mat

    def angular_vel(self, state, cov=1E-1):
        '''
        Returns angular velocity with given covariance 
        '''
        thetadot = state.thetadot
        thetadot += np.random.normal(scale=cov)
        
        cov_mat = np.zeros(9, dtype=np.float64)
        cov_mat[8] = cov

        return Vector3(x=0, y=0, z=thetadot), cov_mat

    def acceleration(self, state, dt, cov=1E-1):
        this_v = np.array([state.x0dot, state.x1dot])
        a = state._rot_mat(state.theta).dot(this_v - self.last_v) / dt + np.random.normal(scale=cov, size=2)

        cov_mat = np.zeros(9, dtype=np.float64)
        cov_mat[0] = cov
        cov_mat[4] = cov

        self.last_v = this_v
        return Vector3(x=a[0], y=a[1], z=0), cov_mat


if __name__ == "__main__":
    rospy.init_node("pose_controller")
    pub = rospy.Publisher("pose", PoseStamped, queue_size=10)
    pub_ori = rospy.Publisher("pose_ori", PoseStamped, queue_size=10)
    pub_imu = rospy.Publisher("imu", Imu, queue_size=10)
    pub_pos_sensor = rospy.Publisher("pos_sensor", PoseStamped, queue_size=10)

    s = State(0, 0, 0, 0, 0, 0)
    z = IMU(s)

    dt = .1
    apply_force = lambda w: s.apply_force([w.wrench.force.x, w.wrench.force.y], w.wrench.torque.z, dt)
    rospy.Subscriber("wrench", WrenchStamped, apply_force)
    
    while not rospy.is_shutdown():
        rospy.sleep(dt)
        s.step_time(dt)
        p = s.as_PoseStamped()

        cov = 1E-3
        # Make IMU measurements
        q, q_cov = z.orientation(s, cov)
        _, _, rz = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        pub_ori.publish(State(s.x0, s.x1, rz, s.x0dot, s.x1dot, s.thetadot).as_PoseStamped())
        rospy.sleep(dt)

        av, av_cov = z.angular_vel(s, cov)
        
        a, a_cov = z.acceleration(s, dt, cov)

        # Populate IMU message
        msg = Imu()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "/map"
        msg.orientation = q
        msg.orientation_covariance = q_cov
        msg.angular_velocity = av
        msg.angular_velocity_covariance = av_cov
        msg.linear_acceleration = a
        msg.linear_acceleration_covariance = a_cov
        pub_imu.publish(msg)

        pub.publish(p)

        # Read position sensor
        sensor_cov = 1E-4
        x = np.array([s.x0, s.x1]) + np.random.normal(scale=sensor_cov, size=2)
        p_sensor = p
        p_sensor.pose.position.x = x[0]
        p_sensor.pose.position.y = x[1]
        pub_pos_sensor.publish(p_sensor)
        
