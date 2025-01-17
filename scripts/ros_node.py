#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import copy
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import List, Tuple

import rospy
import tf
import tf.transformations as tft
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point
from nav_msgs.msg import Path
from moveit_msgs.msg import CartesianTrajectory, CartesianTrajectoryPoint

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.abspath(current_directory + "/third_party/quasi_static_push/scripts/"))

from utils.dish_simulation import DishSimulation
from sweep_and_grasp_the_dishes.srv import GetSweepGraspDishesPath, GetSweepGraspDishesPathRequest, GetSweepGraspDishesPathResponse
from sweep_and_grasp_the_dishes.utils.edge_sampler import EdgeSampler
from sweep_and_grasp_the_dishes.utils.ellipse import Ellipse

class SweepGraspDishesServer(object):
    
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.tf = tf.TransformerROS()
        
        # Get parameters.
        self.planner_config = rospy.get_param("~planner")
        self.gripper_config = rospy.get_param("~gripper")[self.planner_config["gripper"]]

        # Print param to terminal.
        rospy.loginfo("planner config: {}".format(self.planner_config))
        rospy.loginfo("gripper config: {}".format(self.gripper_config))

        # Initialize ros service.
        rospy.Service(
            '/swipe_across_ths_dishes/get_swipe_dish_path',
            GetSweepGraspDishesPath,
            self.get_swipe_dish_path_handler
            )

        # Publisher for visualization
        if self.planner_config["publish_vis_topic"]:
            self.push_path_origin_pub = rospy.Publisher(
                '/swipe_across_ths_dishes/push_path_origin', Path, queue_size=2)
            self.push_path_origin_second_pub = rospy.Publisher(
                '/swipe_across_ths_dishes/push_path_origin_second', Path, queue_size=2)
            self.push_path_origin_eef_pub = rospy.Publisher(
                '/swipe_across_ths_dishes/push_path_origin_eef', Path, queue_size=2)
            self.push_path_moveit = rospy.Publisher(
                '/swipe_across_ths_dishes/push_path', CartesianTrajectory, queue_size=2)
            self.dish_edge_pub = rospy.Publisher(
                '/swipe_across_ths_dishes/dish_edge', MarkerArray, queue_size=2)
            
        self.sim = DishSimulation(
            visualize='human',
            state="linear",
            action_skip=8,
        )

        # Print info message to terminal when push server is ready.
        rospy.loginfo('SweepGraspDishesServer is ready to serve.')
    
    def get_swipe_dish_path_handler(self, request:GetSweepGraspDishesPathRequest) -> GetSweepGraspDishesPathResponse:
        """Response to ROS service. make push path and gripper pose by using trained model(push net).

        Args:
            request (GetSweepGraspDishesPathRequest): ROS service from stable task

        Returns:
            GetSweepGraspDishesPathResponse: generated push_path(moveit_msgs::CartesianTrajectory()), plan_successful(bool), gripper pose(float32[angle, width])
        """

        assert isinstance(request, GetSweepGraspDishesPathRequest)
        # Save service request data.
        dish_seg_msg          = request.dish_segmentation  # vision_msgs/Detection2DArray
        table_det_msg         = request.table_detection    # vision_msgs/BoundingBox3D
        depth_img_msg         = request.depth_image        # sensor_msgs/Image
        camera_info_msg       = request.camera_info        # sensor_msgs/CameraInfo
        camera_pose_msg       = request.camera_pose        # geometry_msgs/PoseStamped
        target_dish_id        = request.target_id          # std_msgs/Int32
        rospy.loginfo("Received request.")
        
        # Parse segmentation image data.
        # Convert segmentation image list from vision_msgs/Detection2DArray to segmask list and id list.
        target_segmask, segmask_list = self.parse_dish_segmentation_msg(dish_seg_msg, target_dish_id.data)

        # Parse table (map) data.
        # Convert table_detection from vision_msgs/BoundingBox3D to map corner and table normal vector matrix.
        map_corners, table_center, table_rotation, rot_matrix = self.parse_table_detection_msg(table_det_msg) # min_x, max_x, min_y, max_y

        # Parse camera data.
        # Convert camera extrinsic type from geometry_msgs/PoseStamped to extrinsic tf.
        cam_pos_tran = [camera_pose_msg.pose.position.x, camera_pose_msg.pose.position.y, camera_pose_msg.pose.position.z]
        cam_pos_quat = [camera_pose_msg.pose.orientation.x, camera_pose_msg.pose.orientation.y, camera_pose_msg.pose.orientation.z, camera_pose_msg.pose.orientation.w]
        cam_pos = self.tf.fromTranslationRotation(cam_pos_tran, cam_pos_quat)
        # Convert depth image type from sensor_msgs/Image to cv2.
        depth_img = self.depth_msg2image(depth_img_msg)
        # Convert camera intrinsic type from sensor_msgs/CameraInfo to matrix.
        cam_intr = np.array(camera_info_msg.K).reshape(3, 3)

        # Edge Sampler
        cps = EdgeSampler(cam_intr,cam_pos)

        # param for simulation
        table_size  = np.array([table_center[0] - map_corners[0], table_center[1] - map_corners[2]]) * 2
        pusher_pose = None
        slider_pose = []
        slider_num  = None

        # target dish
        masked_depth_image = np.multiply(depth_img, target_segmask)

        # Sample the edge points where the dishes can be pushed.
        target_edge = cps.sample(masked_depth_image)
        target_ellipse = Ellipse(target_edge.edge_xyz[:,0], target_edge.edge_xyz[:,1])
        slider_pose.append(target_ellipse.q)

        # Sample the obs edge points where the dishes can be pushed.
        obs_edge_list=[]
        for obs in segmask_list:
            obs_edge_list.append(cps.sample(np.multiply(depth_img, obs)))

        obs_ellipse_list=[]
        for _obs in obs_edge_list:
            _obs_ellipse = Ellipse(_obs.edge_xyz[:,0], _obs.edge_xyz[:,1])
            obs_ellipse_list.append(_obs_ellipse)
            slider_pose.append(_obs_ellipse.q)
        
        # Notice the target dish and obstacles 
        # Target dish
        rospy.loginfo("target dish [m]: \t x: {:.3f}, y: {:.3f}".format(target_ellipse.center[0], target_ellipse.center[1]))
        # Obstacle dish
        if len(obs_ellipse_list) == 0:
            return self.path_failed("obstacle dish not exist")
        else:
            rospy.loginfo("total obstacle dish num: {0}".format(len(obs_ellipse_list)))
        for _obs in obs_ellipse_list:
            rospy.loginfo("obstacle dish [m]: \t x: {:.3f}, y: {:.3f}".format(_obs.center[0], _obs.center[1]))

        # Publish edge of the dishes
        # if self.planner_config["publish_vis_topic"]:
            _edge_marker_list = MarkerArray()
            _id = 0
            _edge_marker = Marker()
            _edge_marker.header.frame_id = camera_pose_msg.header.frame_id
            _edge_marker.ns = "dish_edge_marker"
            _edge_marker.id = 0
            _edge_marker.type = Marker.LINE_STRIP
            _edge_marker.pose.position.x = 0
            _edge_marker.pose.position.y = 0
            _edge_marker.pose.position.z = 0
            _edge_marker.pose.orientation.x = 0
            _edge_marker.pose.orientation.y = 0
            _edge_marker.pose.orientation.z = 0
            _edge_marker.pose.orientation.w = 1
            _edge_marker.scale.x = 0.01
            _edge_marker.scale.y = 0.01
            _edge_marker.scale.z = 0.01
            _edge_marker.color.a = 1.0
            _edge_marker.color.r = 1.0
            _edge_marker.color.g = 1.0
            _edge_marker.color.b = 0.0
            _edge_marker.points = []
            # target dish
            _edge = copy.deepcopy(_edge_marker)
            for _point in target_ellipse.get_ellipse_pts(npts=20).T:
                _p = Point()
                _p.x, _p.y, _p.z = _point[0], _point[1], table_center[2] + 0.1
                _edge.points.append(_p)
            _edge.color.r = 0.0
            _edge.color.g = 0.0
            _edge.color.b = 1.0
            _edge.id = _id
            _id += 1
            _edge_marker_list.markers.append(_edge)
            # Obstacle dish
            for _dish in obs_ellipse_list:
                _edge = copy.deepcopy(_edge_marker)
                for _point in _dish.get_ellipse_pts(npts=20).T:
                    _p = Point()
                    _p.x, _p.y, _p.z = _point[0], _point[1], table_center[2] + 0.1
                    _edge.points.append(_p)
                _edge.id = _id
                _id += 1
                _edge_marker_list.markers.append(_edge)
            self.dish_edge_pub.publish(_edge_marker_list)
            rospy.loginfo("Publish the edge of the dishes as ROS topic.")
            
        # ready for simulation
        
        slider_pose = np.array(slider_pose)
        table_center = np.array(table_center)
        for ellipse in slider_pose:
            ellipse[0][:2] -= table_center[:2]

        simulation_settings = {"table_size": table_size, "pusher_pose": pusher_pose, "slider_pose": slider_pose, "slider_num": slider_num}

        for _ in range(4):
            self.sim.reset(simulation_settings)
            for i in range(400):
                action = np.random.random(4)
                state_next, reward, done = self.sim.env.step(action)
                if done: break


        res = GetSweepGraspDishesPathResponse()   
        res.path = CartesianTrajectory()
        res.path.points = []
        rospy.loginfo('Path generation failed\n')
        res.plan_successful = False
        res.gripper_pose = [self.gripper_config["width"]]
        return res

    def parse_dish_segmentation_msg(self, dish_segmentation_msg, target_id:int):
        ''' Parse dish segmentation msg to segmasks and ids.'''
        
        segmasks = []
        target_segmask = None

        for idx, detection in enumerate(dish_segmentation_msg.detections):
            # Get segmask
            segmask_msg = detection.source_img
            segmask = self.depth_msg2image(segmask_msg)
            if idx == target_id: target_segmask = segmask
            else: segmasks.append(segmask)
        
        return target_segmask, segmasks
    
    def parse_table_detection_msg(self, table_det_msg):
        ''' Parse table detection msg to table pose.'''
        
        self.position_msg = table_det_msg.center.position
        orientation_msg = table_det_msg.center.orientation
        self.size_msg = table_det_msg.size
        
        position = np.array([self.position_msg.x, self.position_msg.y, self.position_msg.z])
        orientation = np.array([orientation_msg.x, orientation_msg.y, orientation_msg.z, orientation_msg.w])
        
        rot_mat = tft.quaternion_matrix(orientation)[:3,:3]
        self.n_vector = rot_mat[:,2]
        
        # Get local positions of vertices 
        vertices_loc = []
        for x in [-self.size_msg.x/2, self.size_msg.x/2]:
            for y in [-self.size_msg.y/2, self.size_msg.y/2]:
                for z in [-self.size_msg.z/2, self.size_msg.z/2]:
                    vertices_loc.append([x,y,z])
        vertices_loc = np.array(vertices_loc)
        
        # Convert to world frame
        vertices_world = np.matmul(rot_mat, vertices_loc.T).T + position
        
        x_max, x_min = np.max(vertices_world[:,0]), np.min(vertices_world[:,0])
        y_max, y_min = np.max(vertices_world[:,1]), np.min(vertices_world[:,1])
        
        x_vector = rot_mat @ np.array([self.size_msg.x / 2,0,0])
        y_vector = rot_mat @ np.array([0,self.size_msg.y / 2,0])
        z_vector = rot_mat @ np.array([0,0,self.size_msg.z / 2])

        # return [x_min, x_max, y_min, y_max], tft.quaternion_matrix(orientation)
        return [x_min, x_max, y_min, y_max], [position[0], position[1], position[2]], [x_vector[0:3], y_vector[0:3], z_vector[0:3]], tft.quaternion_matrix(orientation)

    def is_bound_out_point(self, center, ellipse_list, push_angle, push_width, table_center_xy, table_vectors_xy):
        ''' Check if the dished is out of the table.'''
        _temp = []
        for ellipse in ellipse_list:
            c_vector = ellipse.center - center
            c_vector = c_vector / np.linalg.norm(c_vector) * push_width
            rot_matrix = np.array([
                [np.cos(push_angle), -np.sin(push_angle)],
                [np.sin(push_angle), np.cos(push_angle)],
            ])
            t_vector = rot_matrix @ c_vector + ellipse.center - table_center_xy
            _x = t_vector @ table_vectors_xy[0][0:2] / np.linalg.norm(table_vectors_xy[0][0:2])
            _y = t_vector @ table_vectors_xy[1][0:2] / np.linalg.norm(table_vectors_xy[1][0:2])
            _temp.append([_x + table_center_xy[0],
                          _y
                          ])
        return _temp

    def cal_path_height(self, x, y):
        ''' Parse table detection msg to table pose.'''
        
        _z = self.position_msg.z - self.n_vector[0] / self.n_vector[2] * (x - self.position_msg.x) - self.n_vector[1] / self.n_vector[2] * (y - self.position_msg.y) + self.size_msg.z/2

        return _z

    def path_failed(self, log:str):
        res = GetSweepGraspDishesPathResponse()   
        rospy.logwarn('Path generation failed: %s\n', log)
        res.plan_successful = False
        res.gripper_pose = [self.gripper_config["width"]]
        return res
    
    def depth_msg2image(self, depth) -> np.ndarray:
        """Depth image from the subscribed depth image topic.

        Returns:
            `numpy.ndarray`: (H, W) with `float32` depth image.
        """
        if depth.encoding == '32FC1':
            img = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            img = self.cv_bridge.imgmsg_to_cv2(depth)
            img = (img/1000.).astype(np.float32)
        else:
            img = self.cv_bridge.imgmsg_to_cv2(depth)

        return img

if __name__ == '__main__':
    rospy.init_node('stable_push_net_server')
    server = SweepGraspDishesServer()
    
    rospy.spin()