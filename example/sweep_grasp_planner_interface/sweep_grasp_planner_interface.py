#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author : Sungwon Seo(ssw0536@g.skku.edu)
Date   : 2023-01-22
"""
import rospy
import sweep_and_grasp_the_dishes.srv as swipe_dishes_srv


class GetSweepGraspDishesPath(object):
    def __init__(self):
        # register service
        service_name = '/swipe_across_ths_dishes/get_swipe_dish_path'
        rospy.wait_for_service(service_name)
        self.get_stable_push_path = rospy.ServiceProxy(
            service_name, swipe_dishes_srv.GetSweepGraspDishesPath())
        rospy.loginfo("Service `%s` is ready" % service_name)

    def request(self, dish_segmentation, table_detection, depth_image, camera_info, camera_pose, target_id):
        """Request stable push path from min_icr_push_planner.

        Parameters
        ----------
        depth_image : sensor_msgs/Image
            Depth image.
        segmask : sensor_msgs/Image
            Segmentation mask.
        camera_info : sensor_msgs/CameraInfo
            Camera info.
        camera_pose : geometry_msgs/PoseStamped
            Camera pose in robot base frame.
        map_info : MapInfo
            Map info msg.
        goal_pose : geometry_msgs/PoseStamped
            Goal pose in robot base frame.
        push_direction_range : [float, float]
            Push direction range in radian.

        Returns
        -------
        push_path : nav_msgs/Path
            Push path.
        push_contact : list of ContactPoint
            Push contact points.
        """
        rospy.loginfo("Requesting stable push path")
        start_time = rospy.Time.now()
        try:
            req = swipe_dishes_srv.GetSweepGraspDishesPathRequest()
            req.dish_segmentation = dish_segmentation
            req.table_detection = table_detection
            req.depth_image = depth_image
            req.camera_info = camera_info
            req.camera_pose = camera_pose
            req.target_id = target_id
            res = self.get_stable_push_path(req)
            rospy.loginfo('Service call succeeded. Elapsed time: {}'.format(
                (rospy.Time.now() - start_time).to_sec()))
            return res.path, res.plan_successful, res.gripper_pose
        
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            rospy.logwarn("Service call failed")