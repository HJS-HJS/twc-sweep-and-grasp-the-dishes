#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import struct
import cv2

import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from nav_msgs.msg import Path
from moveit_msgs.msg import CartesianTrajectory
from geometry_msgs.msg import PoseStamped

# Push path module service client
from swipe_planner_interface.swipe_planner_interface import GetSwipeDishesPath
from manipulator_interface.motion_planner import MotionPlanner

# For vizualization
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from utils.utils import depth2pcd
from tf_broadcaster.tf_broadcaster import CameraTransformBroadcaster

# For loading example data
import pickle

# python 2/3 compatibility
try:
    input = raw_input
except NameError:
    print(
        'python 2/3 compatiblity: input() is changed to raw_input(). Delete this try-except block when you update to noetic.')
    pass


class SwipeDishExample(object):
    def __init__(self):

        # Initialize planner module
        self.init_swipe_planner()

        # Get pre-made example data
        # self.get_example_data()

        # Load camera transform and broadcast to /tf
        # self.cam_tf_broadcaster = CameraTransformBroadcaster()
        # self.cam_tf_broadcaster.broadcast_transforms_pose(self.camera_pose)

        self.cv_bridge = CvBridge()

        # publish for visualization
        self.push_moveit_pub = rospy.Publisher('/push_path', CartesianTrajectory, queue_size=2)
        self.push_path_pub = rospy.Publisher('/vis/push_path', Path, queue_size=2)
        self.point_cloud_pub = rospy.Publisher('/vis/point_cloud2', PointCloud2, queue_size=2)
        self.color_image_pub = rospy.Publisher('/vis/color_image', Image, queue_size=2)

        # self.visualize_example_scene_in_rviz()

    def init_swipe_planner(self):
        self.swipe_planner_client = GetSwipeDishesPath()
        rospy.loginfo('Push planner initialized.')

    @staticmethod
    def show_example_segmented_scene(scene_img):
        # Visualize segmentation result
        plt.figure()
        plt.imshow(scene_img)
        plt.show()

    def visualize_example_scene_in_rviz(self, depth_image, camera_info, color_image=None):

        '''Visualize point cloud & color segmask in rViz'''
        depth = self.depth_msg2image(depth_image)
        self.point_cloud_pub.publish(self.pcd_to_pointcloud2(depth2pcd(depth, np.array(camera_info.K).reshape(3, 3))))
        if color_image:
            self.color_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(color_image, encoding="passthrough"))

    def request_swipe_path(self, dish_segmentation, table_detection, depth_image, camera_info, camera_pose, target_id,
                           color_image=None, vis=True):

        # Visualize example scene (does not affect planning)
        # self.show_example_segmented_scene(self.color_image)
        if vis:
            self.visualize_example_scene_in_rviz(depth_image, camera_info, color_image)

        # Request push planning
        push_path, plan_successful, gripper_pose = self.swipe_planner_client.request(dish_segmentation,
                                                                                     table_detection,
                                                                                     depth_image,
                                                                                     camera_info,
                                                                                     camera_pose,
                                                                                     target_id)

        if not plan_successful:
            rospy.logerr('Push planning Failed.')
            return

        # Visualize planned push path in rViz
        if vis:
            self.push_moveit_pub.publish(push_path)
            self.push_path_pub.publish(self.moveit_cartesian_to_path(camera_pose, push_path))

        return push_path, plan_successful, gripper_pose

    def pcd_to_pointcloud2(self, pcd):
        _header = Header()
        _header.frame_id = "camera_color_optical_frame"
        _header.stamp = rospy.Time.now()

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]

        points = []
        for point in pcd:
            rgb = struct.unpack('I', struct.pack('BBBB', 255, 255, 255, 255))[0]
            points.append([point[0], point[1], point[2], rgb])

        return point_cloud2.create_cloud(_header, fields, points)

    def moveit_cartesian_to_path(self, camera_pose, moveit_path):

        eef_path_msg = Path()
        eef_path_msg.header.frame_id = camera_pose.header.frame_id
        eef_path_msg.header.stamp = rospy.Time.now()
        for each_point in moveit_path.points:
            _pose_stamped = PoseStamped()
            _pose_stamped.header.stamp = rospy.Time.now()
            _pose_stamped.header.frame_id = camera_pose.header.frame_id
            _pose_stamped.pose.position = each_point.point.pose.position
            _pose_stamped.pose.orientation = each_point.point.pose.orientation
            eef_path_msg.poses.append(_pose_stamped)

        return eef_path_msg

    def depth_msg2image(self, depth) -> np.ndarray:
        """Depth image from the subscribed depth image topic.

        Returns:
            `numpy.ndarray`: (H, W) with `float32` depth image.
        """
        if depth.encoding == '32FC1':
            img = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            img = self.cv_bridge.imgmsg_to_cv2(depth)
            img = (img / 1000.).astype(np.float32)
        else:
            img = self.cv_bridge.imgmsg_to_cv2(depth)

        # none to zero
        img = np.nan_to_num(img)

        # depth hole filling
        inpaint_mask = np.zeros(img.shape, dtype='uint8')
        inpaint_mask[img == 0] = 255
        restored_depth_image = cv2.inpaint(
            img,
            inpaint_mask,
            inpaintRadius=15,
            flags=cv2.INPAINT_NS
        )
        return restored_depth_image


if __name__ == '__main__':
    rospy.init_node('swipe_across_the_dishes_example')
    example = SwipeDishExample()

    # motion_planner = MotionPlanner(
    #     group_name='m1013_arm', pose_reference_frame='base_0')
    motion_planner = MotionPlanner(
        group_name='arm', pose_reference_frame='base_0', ns='dsr01m1013')
    motion_planner.move_group.set_max_velocity_scaling_factor(0.1)


    def open_pickle(filename):
        file_path = os.path.join(os.path.dirname(__file__), "service_req", filename)
        with open(file_path, 'rb') as f:
            return pickle.load(f)


    dish_segmentation = open_pickle('dish_segmentation.p')
    table_detection = open_pickle('table_detection.p')
    depth_image = open_pickle('depth_image.p')
    camera_info = open_pickle('camera_info.p')
    camera_pose = open_pickle('camera_pose.p')
    target_id = open_pickle('target_id.p')
    color_image = open_pickle('image.p')

    while True:
        user_input = input('Press enter to start task, q to quit...')
        if user_input == 'q':
            break
        elif user_input == '':
            pass
        else:
            continue

        push_path, plan_successful, gripper_pose = example.request_swipe_path(dish_segmentation, table_detection,
                                                                              depth_image, camera_info, camera_pose,
                                                                              target_id, color_image)
        motion_planner.run_swipe_path(push_path)