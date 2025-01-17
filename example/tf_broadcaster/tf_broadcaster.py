#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseStamped


class CameraTransformBroadcaster(object):
    def __init__(self):
        # Load transformations
        self.tf_sbr = StaticTransformBroadcaster()


    def broadcast_transforms(self, camera_pose_tf):
        try:
            self.tf_sbr.sendTransform([camera_pose_tf])
            rospy.sleep(0.1)
            rospy.loginfo('Broadcasted transforms')
            return 0
        except NotImplementedError as e:
            rospy.logwarn('Failed to broadcast transforms: {}'.format(e))

    def broadcast_transforms_pose(self, camera_pose):
        cam_tf = TransformStamped()
        cam_tf.header.frame_id = camera_pose.header.frame_id
        cam_tf.child_frame_id = "camera_color_optical_frame"
        cam_tf.transform.translation.x = camera_pose.pose.position.x
        cam_tf.transform.translation.y = camera_pose.pose.position.y
        cam_tf.transform.translation.z = camera_pose.pose.position.z
        cam_tf.transform.rotation.x = camera_pose.pose.orientation.x
        cam_tf.transform.rotation.y = camera_pose.pose.orientation.y
        cam_tf.transform.rotation.z = camera_pose.pose.orientation.z
        cam_tf.transform.rotation.w = camera_pose.pose.orientation.w
        try:
            self.tf_sbr.sendTransform([cam_tf])
            rospy.sleep(0.1)
            rospy.loginfo('Broadcasted transforms')
            return 0
        except NotImplementedError as e:
            rospy.logwarn('Failed to broadcast transforms: {}'.format(e))

if __name__ == '__main__':
    # init ros node
    rospy.init_node('camera_trasform_visualizer')

    # init visualizer
    visualizer = CameraTransformBroadcaster()

    # broadcast transforms
    visualizer.broadcast_transforms()
    rospy.spin()

        # self.cam_tf_broadcaster = CameraTransformBroadcaster()        
        # self.cam_tf_broadcaster.broadcast_transforms(self.camera_pose_tf)