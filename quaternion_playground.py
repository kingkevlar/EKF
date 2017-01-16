#!/usr/bin/env python
from __future__ import division
import rospy
import numpy as np
from tf import transformations as trns

from geometry_msgs.msg import Quaternion, PoseStamped


def normalize(q):
    return q / np.linalg.norm(q)

def get_dq(q, w):
    '''
    q := Quaternion with components [qx, qy, qz, qw] (standard tf convention)
    w := Angular velocity [wx, wy, wz]  # rad/s
    '''
    dq = np.zeros(4).astype(np.float64)
    dq[3] = 1/2 * (-(w[0] * q[0]) - (w[1] * q[1]) - (w[2] * q[2]))
    dq[0] = 1/2 * ( (w[0] * q[3]) - (w[1] * q[2]) + (w[2] * q[1]))
    dq[1] = 1/2 * ( (w[0] * q[2]) + (w[1] * q[3]) - (w[2] * q[0]))
    dq[2] = 1/2 * (-(w[0] * q[1]) + (w[1] * q[0]) + (w[2] * q[3]))
    return dq

if __name__ == "__main__":
    rospy.init_node("q_playground")
    pub = rospy.Publisher("/pose", PoseStamped, queue_size=1)

    q0 = trns.quaternion_from_euler(0, 0, 0)
    w = [0, 0, .01]  # x, y, z angular velocities (rad/s ?)
    # This seems to represent rads when the numbers are suffienctly small

    i = 0
    while not rospy.is_shutdown():
        # q' = q0 + dq * dt
        dq = get_dq(q0, w)
        q0 += dq

        q1 = q0#normalize(q0)
        print i, "======================="
        print q1
        print np.round(trns.euler_from_quaternion(q1), 5)
        i += 1

        # Ros visualization
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.orientation = Quaternion(*q1)
        pub.publish(pose)
        rospy.sleep(.05)

