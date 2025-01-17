#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author : Sungwon Seo(ssw0536@g.skku.edu)
Date   : 2023-01-22
"""
import os
import sys
import rospkg
import numpy as np
import rospy
import tf
import tf.transformations
import moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState, DisplayTrajectory, Constraints, PositionIKRequest
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from nav_msgs.msg import Path

# python 2/3 compatibility
try:
    input = input
except NameError:
    print(
        'python 2/3 compatiblity: input() is changed to input(). Delete this try-except block when you update to noetic.')
    pass

# import doosan robot module
rospack = rospkg.RosPack()
rospack.list()
pkg_path = rospack.get_path('common')
sys.path.append(os.path.join(pkg_path, "imp"))
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m1013"
import DR_init

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
import DSR_ROBOT as drfl
import DR_common as dr_com


class MotionPlanner(object):
    def __init__(self, group_name, pose_reference_frame, ns=''):
        """Initialize MotionPlanner.

        Args:
            group_name (`str`): MoveIt! group name.
            pose_reference_frame_id (`str`): Pose reference frame id.
        """
        # get init variables
        self.group_name = group_name
        self.pose_reference_frame = pose_reference_frame

        # Initialize tf
        self.tf = tf.TransformListener()

        # Initialize move group
        self.move_group = moveit_commander.MoveGroupCommander(group_name, robot_description=ns + '/robot_description',
                                                              ns=ns, wait_for_servers=20.0)
        # self.move_group = moveit_commander.MoveGroupCommander(group_name)
        self.move_group.set_pose_reference_frame(self.pose_reference_frame)

        # publishers
        self.display_planned_path_pub = rospy.Publisher(
            '/move_group/display_planned_path',
            DisplayTrajectory,
            queue_size=2)

        # Initialize services
        self.fk_srv = rospy.ServiceProxy(ns + '/compute_fk', GetPositionFK)
        self.fk_srv.wait_for_service()
        self.ik_srv = rospy.ServiceProxy(ns + '/compute_ik', GetPositionIK)
        self.ik_srv.wait_for_service()

        # sleep
        rospy.sleep(1)

    def run_swipe_path(self, path):
        """Run swipe path.
        """

        # Convert moveit cartesian to nav path
        push_path = Path()
        push_path.header = path.header
        for _path in path.points:
            _pose = PoseStamped()
            _pose.header = path.header
            _pose.pose = _path.point.pose
            push_path.poses.append(_pose)

        # Set approach pose
        approach_pose = self.offset_pose_relative(
            push_path.poses[0],
            offset=[0.0, 0.0, -0.2, 0.0, 0.0, 0.0])
        approach_posx = self.convert_to_drfl_posx(approach_pose, 0.0)
        # Set end pose
        end_pose = self.offset_pose_relative(
            push_path.poses[-1],
            offset=[0.0, 0.0, -0.2, 0.0, 0.0, 0.0])
        end_posx = self.convert_to_drfl_posx(end_pose, 0.0)

        # Move m1013
        max_vel = 30  # mm/s
        max_acc = 30  # mm/s^2
        drfl.set_robot_mode(1)  # set robot mode to auto

        # 0) Move to home pose
        pose = drfl.posj(0, 0, 90, 0, 90, 0)
        drfl.movej(pose, vel=30, acc=30)

        # 1) Move to approach pose
        rospy.loginfo('Move to approach pose.')
        drfl.movel(approach_posx, vel=max_vel, acc=max_acc,
                   ref=drfl.DR_BASE, mod=drfl.DR_MV_MOD_ABS)

        # 2) Move along path
        rospy.loginfo('Move along path.')
        self.move_group.set_start_state_to_current_state()

        push_poses_cartesian = []
        for pose in push_path.poses:
            pose_cartesian_list = self.convert_to_drfl_posx(pose, 0.0)
            pose_cartesian = dr_com.posx(pose_cartesian_list[0], pose_cartesian_list[1], pose_cartesian_list[2],
                                         pose_cartesian_list[3], pose_cartesian_list[4], pose_cartesian_list[5])
            push_poses_cartesian.append(pose_cartesian)

        drfl.amovesx(push_poses_cartesian, vel=max_vel, acc=max_acc,
                     ref=drfl.DR_BASE, mod=drfl.DR_MV_MOD_ABS, vel_opt=drfl.DR_MVS_VEL_CONST)

        # Check move along path done
        while drfl.check_motion() == 2:
            rospy.sleep(0.01)

        # 3) Move up
        rospy.loginfo('Move to up.')
        drfl.movel(end_posx, vel=max_vel, acc=max_acc,
                   ref=drfl.DR_BASE, mod=drfl.DR_MV_MOD_ABS)

    def is_pose_reachable(self, pose, eef_link=None, verbose=False):
        """Check if the pose is reachable.

        Args:
            pose (`geometry_msgs/PoseStamped`): Pose to be checked.
            eef_link (`str`): End effector link. If None, use the default end effector link.
            verbose (`bool`): Verbose mode. Default is False.

        Returns:
            `bool`: True if the pose is reachable.
        """
        if eef_link is not None:
            assert isinstance(eef_link, str), 'eef_link should be a string'
            self.move_group.set_end_effector_link(eef_link)

        # check if the pose is reachable
        ik = self.get_inverse_kinematics(pose, timeout=0.05)
        if ik is None:
            rospy.logwarn('The pose is not reachable.')
            return False
        else:
            if verbose:
                rospy.loginfo('The pose is reachable.')
            return True

    def get_inverse_kinematics(
            self,
            pose,
            init_state=None,
            contraints=None,
            avoid_collisions=True,
            timeout=0.5,
            eef_link=None):
        """Get the inverse kinematics.

        Args:
            pose (`geometry_msgs/PoseStamped`): A pose target.
            init_state (`moveit_msgs/RobotState`, optional): A robot state. If None, use the current state. Defaults to None.
            contraints (`moveit_msgs/Constraints`, optional): A constraint. If None, use the default constraint. Defaults to None.
            avoid_collisions (`bool`, optional): If True, avoid collisions. Defaults to True.
            timeout (`float`, optional): Timeout in sec. Defaults to 0.5.
            eef_link (`str`, optional): End effector link. Defaults to None.

        Returns:
            `RobotState`: A robot state.
        """
        # check arguments
        if eef_link is not None:
            assert isinstance(eef_link, str), 'eef_link should be a string'
            self.move_group.set_end_effector_link(eef_link)
        assert isinstance(pose, PoseStamped), 'pose should be a PoseStamped'
        if init_state is None:
            init_state = self.move_group.get_current_state()
        else:
            assert isinstance(init_state, RobotState), 'init_state should be a RobotState'
        if contraints is None:
            contraints = self.move_group.get_path_constraints()
        else:
            assert isinstance(contraints, Constraints), 'contraints should be a Constraints'

        # get request arguments
        ik_request = PositionIKRequest()
        ik_request.group_name = self.group_name
        ik_request.robot_state = init_state
        ik_request.constraints = contraints
        ik_request.avoid_collisions = True
        ik_request.ik_link_name = self.move_group.get_end_effector_link()
        ik_request.pose_stamped = pose
        ik_request.timeout = rospy.Duration(timeout)

        # get inverse kinematics
        try:
            resp = self.ik_srv(ik_request)
            if resp.error_code.val == 1:
                return resp.solution
            else:
                rospy.logerr('Inverse kinematics failed with error code: {}'.format(resp.error_code.val))
                return None
        except rospy.ServiceException as e:
            rospy.logerr('Invert kinematic service call failed: {}'.format(e))
            return None

    def offset_pose_along_approach(self, pose, offset=-0.15):
        """Offset pose along approach direction.

        Args:
            pose (`PoseStamped`): A pose.
            offset (`float`, optional): Offset in [m]. Defaults to -0.15.

        Returns:
            `PoseStamped`: A pose with offset.
        """
        assert isinstance(pose, PoseStamped)
        assert isinstance(offset, float)

        # get target translation matrix
        pose_trans = [
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ]
        pose_trans_mat = tf.transformations.translation_matrix(pose_trans)

        # get pose rotation matrix
        pose_quat = [
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ]
        pose_rot_mat = tf.transformations.quaternion_matrix(pose_quat)

        # get z-axis (-) offset translation matrix
        approach_trans_mat = tf.transformations.translation_matrix([0., 0., offset])

        # get approach pose
        approach_trans_mat = np.linalg.multi_dot([pose_trans_mat, pose_rot_mat, approach_trans_mat])
        approach_pose = PoseStamped()
        approach_pose.header = pose.header
        approach_pose.pose.orientation = pose.pose.orientation
        approach_pose.pose.position.x = approach_trans_mat[0, 3]
        approach_pose.pose.position.y = approach_trans_mat[1, 3]
        approach_pose.pose.position.z = approach_trans_mat[2, 3]
        return approach_pose

    def offset_pose_relative(self, pose, offset):
        """Offset pose relative to the pose.

        Args:
            pose (`PoseStamped`): A pose.
            offset (`list`): Offset in [x, y, z, a, b, c] in [m] and [rad].

        Returns:
            `PoseStamped`: A pose with offset.
        """
        # pose: PoseStamped
        # offset: [x, y, z, a, b, c]
        # a, b, c: euler zyx angles
        assert isinstance(pose, PoseStamped)

        # get transformation matrix
        pose_tf = tf.transformations.translation_matrix(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        pose_rot = tf.transformations.quaternion_matrix(
            [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])
        # print(tf.transformations.euler_from_matrix(pose_rot, 'rzxy'))
        # print(np.rad2deg(np.pi + tf.transformations.euler_from_matrix(pose_rot, 'rzxy')[2]))
        _rad = np.pi + tf.transformations.euler_from_matrix(pose_rot, 'rzxy')[2]
        pose_tf = np.linalg.multi_dot([pose_tf, pose_rot])

        # get offset matrix
        offset_tf = tf.transformations.translation_matrix(offset[:3])
        # offset_tf = tf.transformations.translation_matrix([offset[0] * np.arccos(_rad), offset[1], offset[0] * np.arcsin(_rad)])
        offset_rot = tf.transformations.euler_matrix(offset[3], offset[4], offset[5], axes='sxzy')
        # print(offset_tf, offset_rot)
        offset_tf = np.linalg.multi_dot([offset_tf, offset_rot])
        # print(offset_tf)

        # get new pose
        new_pose_tf = np.linalg.multi_dot([pose_tf, offset_tf])
        # print(new_pose_tf)
        new_pose_quat = tf.transformations.quaternion_from_matrix(new_pose_tf)
        new_pose = PoseStamped()
        new_pose.header = pose.header
        new_pose.pose.position.x = new_pose_tf[0, 3]
        new_pose.pose.position.y = new_pose_tf[1, 3]
        new_pose.pose.position.z = new_pose_tf[2, 3]
        new_pose.pose.orientation.x = new_pose_quat[0]
        new_pose.pose.orientation.y = new_pose_quat[1]
        new_pose.pose.orientation.z = new_pose_quat[2]
        new_pose.pose.orientation.w = new_pose_quat[3]
        return new_pose

    def convert_to_drfl_posx(self, pose, z_offset: float = -0.266):
        """Convert pose to drfl_posx.

        Args:
            pose (PoseStamped): Pose.
            z_offset (float): Vertical distance between the gripper end and the robot end effector. robotiq_2f: -0.266, hanyang: -0.343421
        """
        assert isinstance(pose, PoseStamped)
        assert isinstance(z_offset, float)

        # robotiq_2f: -0.266, hanyang: -0.343421
        pose_link6 = self.offset_pose_along_approach(pose, offset=z_offset)

        # translation: m to mm
        trans = [pose_link6.pose.position.x * 1000,
                 pose_link6.pose.position.y * 1000,
                 pose_link6.pose.position.z * 1000]

        # rotation: quaternion to euler (rzyz) with degree
        quat = [pose_link6.pose.orientation.x,
                pose_link6.pose.orientation.y,
                pose_link6.pose.orientation.z,
                pose_link6.pose.orientation.w]
        euler = tf.transformations.euler_from_quaternion(
            quat, axes='rzyz')
        euler = np.rad2deg(euler)

        # convert to drfl posx
        posx = [
            trans[0],
            trans[1],
            trans[2],
            euler[0],
            euler[1],
            euler[2]
        ]
        return posx
