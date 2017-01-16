#!/usr/bin/env python
from __future__ import division
import rospy
import numpy as np
from tf import transformations as trns

from geometry_msgs.msg import Quaternion, PoseStamped


def normalize(q):
    return q / np.linalg.norm(q)

def get_dq(v):
    r = np.zeros(4).astype(np.float64)
    angle = np.linalg.norm(v)
    r[3] = np.cos(angle / 2)
    r[:3] = np.array(v) * np.sin(angle / 2) / angle  # sinc(angle / 2) / 2
    return r

if __name__ == "__main__":
    rospy.init_node("q_playground")
    pub = rospy.Publisher("/pose", PoseStamped, queue_size=1)

    q0 = trns.quaternion_from_euler(0.9691, -0.76813, -1.83213)
    rot = trns.quaternion_matrix(q0)[:3, :3]
    w = [0, 0.1, 0]  # x, y, z angular velocities (rad/s ?)

    i = 0
    while not rospy.is_shutdown():
        # q' = q0 + dq * dt
        dq = get_dq(w)
        q0 = trns.quaternion_multiply(q0, dq)

        q1 = normalize(q0)
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
        rospy.sleep(.1)

