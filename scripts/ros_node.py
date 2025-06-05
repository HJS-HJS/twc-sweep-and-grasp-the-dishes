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
import torch
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
so_file_path = os.path.abspath("../cpp")
sys.path.append(so_file_path)

from sweep_and_grasp_the_dishes.srv import GetSweepGraspDishesPath, GetSweepGraspDishesPathRequest, GetSweepGraspDishesPathResponse
from utils.model import ActorNetwork
from utils.dish_simulation import DishSimulation
from utils.edge_sampler import EdgeSampler
from utils.ellipse import Ellipse
from utils.utils import load_model

device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

class SweepGraspDishesServer(object):
    
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.tf = tf.TransformerROS()
        
        # Get parameters.
        self.planner_config = rospy.get_param("~planner")
        self.gripper_config = rospy.get_param("~gripper")[self.planner_config["gripper"]]
        self.simulat_config = rospy.get_param("~simulator")

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        model_path = current_directory + "/../model"
        model_name = self.planner_config["model"]
        self.start_point = np.array(self.planner_config["start_point"])

        self.actor = load_model(ActorNetwork().to(device), model_path, "actor", model_name)

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
        table_corners, table_center, table_rotation, rot_matrix = self.parse_table_detection_msg(table_det_msg) # min_x, max_x, min_y, max_y

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
        work_space = np.array(self.planner_config["work_space"])
        table_data_x = np.clip(table_corners[:2], work_space[0,0], work_space[0,1])
        table_data_y = np.clip(table_corners[2:], work_space[1,0], work_space[1,1])
        table_size  = np.array([table_data_x[1] - table_data_x[0], table_data_y[1] - table_data_y[0]])
        table_center = np.array([table_data_x[0], table_data_y[0]]) + table_size / 2
        slider_pose = []

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
            if _obs_ellipse.q[0] < table_data_x[0]: continue
            elif _obs_ellipse.q[0] > table_data_x[1]: continue
            elif _obs_ellipse.q[1] < table_data_y[0]: continue
            elif _obs_ellipse.q[1] > table_data_y[1]: continue
            obs_ellipse_list.append(_obs_ellipse)
            slider_pose.append(_obs_ellipse.q)

        # Notice the target dish and obstacles 
        # Target dish
        rospy.loginfo("target dish [m]: \t x: {:.3f}, y: {:.3f}".format(target_ellipse.center[0], target_ellipse.center[1]))
        # Obstacle dish
        if len(obs_ellipse_list) == 0:
            rospy.loginfo("obstacle dish not exist")
        else:
            rospy.loginfo("total obstacle dish num: {0}".format(len(obs_ellipse_list)))
        for _obs in obs_ellipse_list:
            rospy.loginfo("obstacle dish [m]: \t x: {:.3f}, y: {:.3f}".format(_obs.center[0], _obs.center[1]))
        
        slider_pose = np.array(slider_pose)
        table_center = np.array(table_center)
        for ellipse in slider_pose:
            ellipse[:2] -= table_center[:2]
        
        pusher_pose = copy.deepcopy(self.start_point)
        pusher_pose[:,:2] -= table_center[:2]
        pusher_pose[:,2] *= (np.pi / 180)
        
        success = False
        rospy.loginfo("generate simulation env")
        sim = DishSimulation(
            visualize=self.simulat_config["visualize"],
            state="linear",
            action_skip=self.simulat_config["frame_skip"],
        )
        
        path = []
        move_idx = None
        for start_idx in range(4):
            rospy.loginfo("generating path {}".format(start_idx))
            path = []
            move_idx = None
            state_curr, _, _, mode = sim.reset(
                mode = True,
                setting = {
                    "table_size":table_size, 
                    "slider_state":slider_pose,
                    "slider_num":None
                    }
                )
            _, _, _, _ = sim.env.step([0.9, 0.9, 0, 0], 0)
            state_curr, _, _, _ = sim.env.step([0, 0, 0, 0], 1)

            state_curr1, state_curr2 = state_curr
            state_curr1 = torch.tensor(state_curr1, dtype=torch.float32, device=device).unsqueeze(0)
            state_curr2 = torch.tensor(state_curr2.T, dtype=torch.float32, device=device)

            image_start = sim.env.image_without_gripper()
            # image_mid = sim.env.get_image()
            # image_end = sim.env.get_image()
            
            rospy.loginfo("start simulation")
            # Running one episode
            if start_idx == 0:
                for step in range(1, 150):
                    # 1. Get action from policy network
                    with torch.no_grad():
                        action = self.actor(state_curr1, state_curr2.unsqueeze(0), torch.tensor([mode], device=device).unsqueeze(0))

                    if step > 10:
                        rand = (2 * np.random.random(action.size) - 1) * (step / 150)
                        rand[2:] *= 2
                        action = np.clip(action + rand, -0.9999, 0.9999)

                    # 2. Run simulation 1 step (Execute action and observe reward)
                    state_next, reward, done, mode = sim.env.step(action, mode)

                    if mode == 1:
                        state_next1, state_next2 = state_next
                        state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
                        state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)
                        state_curr1 = state_next1
                        state_curr2 = state_next2
                        break
            else:
                for step in range(1, 150):
                    # 1. Get action from policy network
                    action = np.array([-0.9, 0.9, 0, 0.5])
                    if start_idx % 2 == 0:
                        action[1] *= -1
                    elif start_idx == 2:
                        action = np.array([-0.9, 0.0, 0.0, 0.5])

                    if step > 1:
                        rand = (2 * np.random.random(action.size) - 1) * (step / 150)
                        rand[2:] *= 2
                        action = np.clip(action + rand, -0.9999, 0.9999)

                    # 2. Run simulation 1 step (Execute action and observe reward)
                    state_next, reward, done, mode = sim.env.step(action, mode)

                    if mode == 1:
                        state_next1, state_next2 = state_next
                        state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
                        state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)
                        state_curr1 = state_next1
                        state_curr2 = state_next2
                        break
                

            for step in range(1, 150):
                with torch.no_grad():
                    action = self.actor(state_curr1, state_curr2.unsqueeze(0), torch.tensor([mode], device=device).unsqueeze(0))
                state_next, reward, done, mode = sim.env.step(action, mode)
                
                path.append(sim.env.gripper_pose())
                    
                state_next1, state_next2 = state_next
                state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
                state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)
                state_curr1 = state_next1
                state_curr2 = state_next2
                
                if reward < 0 and move_idx is None:
                    move_idx = step
                    image_mid = sim.env.get_image()
                    
                if done: break

            image_end = sim.env.get_image()
            if reward > 3: 
                success = True
                break

        rospy.loginfo("simulation finished")
        del sim
        
        print("move_idx", move_idx)
        if move_idx is None:
            move_idx = len(path) - 1 if len(path) > 1 else len(path)
            image_mid = image_end
        else: move_idx = move_idx - 1 if move_idx > 1 else move_idx
        print("move_idx", move_idx)
        print("path lengh", len(path))
        path = np.array(path)[move_idx:]
        path[:,:2] += table_center[:2]
        _spent_time = rospy.Duration.from_sec(step * self.simulat_config["frame_skip"] / self.simulat_config["fps"])

        # vis
        if self.planner_config["visualize"]:
            fig = plt.figure(figsize=(10,10))
            ax1 = fig.add_subplot(231)
            ax2 = fig.add_subplot(232)
            fig1 = fig.add_subplot(234)
            fig2 = fig.add_subplot(235)
            fig3 = fig.add_subplot(236)
            ax1.grid(True)
            ax2.grid(True)
            
            # Draw table
            ax1.set_xlim([table_corners[0] - 0.1, table_corners[1] + 0.1])
            ax1.set_ylim([table_corners[2] - 0.1, table_corners[3] + 0.1])
            ax2.set_xlim([table_corners[0] - 0.1, table_corners[1] + 0.1])
            ax2.set_ylim([table_corners[2] - 0.1, table_corners[3] + 0.1])

            # Draw target
            x, y = target_edge.edge_xyz[:,0], target_edge.edge_xyz[:,1]
            ax1.plot(x, y, color='black')
            for obs in obs_edge_list:
                x, y = obs.edge_xyz[:,0], obs.edge_xyz[:,1]
                ax1.plot(x, y, color='black')
            
            origin_target_ellipse = Ellipse(target_edge.edge_xyz[:,0], target_edge.edge_xyz[:,1])
            x, y = origin_target_ellipse.get_ellipse_pts()
            ax2.plot(x, y, color='black')
            for obs in obs_ellipse_list:
                x, y = obs.get_ellipse_pts()
                ax2.plot(x, y, color='black')
            

            # Draw path
            ax2.plot(path[:,0], path[:,1], 'red', linewidth=4)

            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            fig1.imshow(image_start[:, :, ::-1])
            fig2.imshow(image_mid[:, :, ::-1])
            fig3.imshow(image_end[:, :, ::-1])
                
            plt.show()

        path_msg = CartesianTrajectory()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = camera_pose_msg.header.frame_id # base link of doosan m1013
        path_msg.tracked_frame = "end_effector" # end effector of gripper
        path_msg.points =[]
        
        eef_path = PoseArray()
        eef_path.header.stamp = rospy.Time.now()
        eef_path.header.frame_id = camera_pose_msg.header.frame_id # base link of doosan m1013

        for point in path:
            _pose = Pose()
            # finger position x, y
            _pose.position.x, _pose.position.y = point[0], point[1]
            # finger position z along table pose
            _pose.position.z = self.planner_config['height'] + self.cal_path_height(point[0], point[1])
            # finger orientation matrix
            path_rot_matrix = np.dot(rot_matrix, tft.euler_matrix(point[2] + np.deg2rad(self.gripper_config["z_angle"]), 0, 0, axes='rzxy'))
            # finger orientation x, y, z, w
            _pose.orientation.x, _pose.orientation.y, _pose.orientation.z, _pose.orientation.w = tft.quaternion_from_matrix(path_rot_matrix)
            eef_path.poses.append(_pose)
            
            _point = CartesianTrajectoryPoint()
            # whole spent time
            _point.time_from_start = _spent_time
            # point position
            _point.point.pose.position = _pose.position
            _point.point.pose.position.z += self.gripper_config['height']
            # apply gripper tilt angle (table angle, gripper push tilt angle)
            path_rot_matrix = np.dot(rot_matrix, tft.euler_matrix(point[2] + np.deg2rad(self.gripper_config["z_angle"] + self.gripper_config["finger_angle"]), -np.pi, 0, axes='rzxy'))
            # gripper orientation
            _point.point.pose.orientation.x, _point.point.pose.orientation.y, _point.point.pose.orientation.z, _point.point.pose.orientation.w = tft.quaternion_from_matrix(path_rot_matrix)
            path_msg.points.append(_point)
            
        rospy.loginfo("Swipe ROS path generation finished")

        res = GetSweepGraspDishesPathResponse()   
        res.path = path_msg
        res.plan_successful = success
        # res.gripper_pose = [self.gripper_config["width"]]
        res.gripper_pose = path[:,3].tolist()
        if success: rospy.loginfo('Path generation successed\n')
        else: rospy.loginfo('Path generation failed\n')
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