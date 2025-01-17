import math
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseArray, Pose

def distance(pose1x, pose1y, pose2x, pose2y):
    return math.sqrt((pose1x - pose2x)**2 + (pose1y - pose2y)**2)

def calculateAngle(pose_x, pose_y):
    angle = math.atan2(pose_y, pose_x)

    if angle > math.pi :
        # Ensure the angle is in range [0, 2pi] 
        angle -= 2 * math.pi        
        
    return angle

def calculateGripperCP(pose1x, pose1y, pose2x, pose2y, R, clock_wise):
    d = distance(pose1x, pose1y, pose2x, pose2y)

    # Check if a circle can be constructed
    if d > 2 * R:
        raise ValueError("given radius is not valid to compute")
    
    # determine two center point using vector
    Mx = (pose1x + pose2x) / 2
    My = (pose1y + pose2y) / 2

    h = math.sqrt(R**2 - (d/2)**2)

    dx = pose2x - pose1x
    dy = pose2y - pose1y

    norm = math.sqrt(dx**2 + dy**2)
    perp_dx = -dy / norm
    perp_dy =  dx / norm

    cp1x = Mx + perp_dx * h
    cp1y = My + perp_dy * h
    cp2x = Mx - perp_dx * h
    cp2y = My - perp_dy * h

    # Calculate cross product to determine which is left or right center point
    if dx*(cp1y-pose2y) - dy*(cp1x-pose2x) > 0 and not clock_wise:
        return [cp1x, cp1y]
    elif dx*(cp1y-pose2y) - dy*(cp1x-pose2x) > 0 and clock_wise:
        return [cp2x, cp2y]
    elif dx*(cp1y-pose2y) - dy*(cp1x-pose2x) < 0 and not clock_wise:
        return [cp1x, cp1y]
    else:
        return [cp2x, cp2y]
        

def euler2Quaternion(roll = 0, pitch = 0, yaw = 0):
    # Convert roll, pitch, yaw to quaternion
    return quaternion_from_euler(roll, pitch, yaw)


def cartesianTraj2EETraj(cartesian_traj, gripper_radius, margin_angle, alpha = 0.01, clock_wise = False):
    """
    Change given xyz cartesian trajectory to EE pose
    Inputs:
    - cartesian_traj: geometry_msgs.msg/PoseArray format trajectory that represent finger_front pose (orientation value will be not used)
    - gripper_radius: distance between center point of gripper to gripper finger
    - margin_angle: Make buffer distance to grasp object safely
    - alpha: Extract trajectory waypoint that finger_behind will pass through using distance error value alpha 
    - clock_wise: if trajectory moves along clockwise direction, reverse should be True value. If not, reverse should be False

    Outputs:
    - EETraj: geometry_msgs.msg/PoseArray format trajectory that represent EE pose / Orientation value will be inside [0, 2*pi], change orientation range if you want!!
    - BHTraj: geometry_msgs.msg/PoseArray format trajectory that represent Behind Finger pose / Orientation value is not inserted!!

    Caution:
    - This function assume that trajectory start from right to left... if EE start from left to right and margin_angle is given
    finger_behind will go through inside of dishes (Be careful!!)
    """

    traj_EE_x = []
    traj_EE_y = []
    traj_EE_z = []
    traj_EE_qx = []
    traj_EE_qy = []
    traj_EE_qz = []
    traj_EE_qw = []
    
    traj_BH_x = []
    traj_BH_y = []
    traj_BH_z = []

    cartesian_traj_x = []
    cartesian_traj_y = []
    cartesian_traj_z = []

    effective_raius = math.sqrt(3) * gripper_radius

    # Extract x and y coordinates from the PoseArray
    cartesian_traj_x_ = [pose.position.x for pose in cartesian_traj.poses]
    cartesian_traj_y_ = [pose.position.y for pose in cartesian_traj.poses]
    cartesian_traj_z_ = [pose.position.z for pose in cartesian_traj.poses]

    # Interpolate Trajectory (xn)
    n = 8
    for i in range(len(cartesian_traj_x_) - 1):
        for j in range(n):
            cartesian_traj_x.append((n-j)/n * cartesian_traj_x_[i] + (j)/n * cartesian_traj_x_[i+1])
            cartesian_traj_y.append((n-j)/n * cartesian_traj_y_[i] + (j)/n * cartesian_traj_y_[i+1])
            cartesian_traj_z.append((n-j)/n * cartesian_traj_z_[i] + (j)/n * cartesian_traj_z_[i+1])

    cartesian_traj_x.append(cartesian_traj_x_[-1])
    cartesian_traj_y.append(cartesian_traj_y_[-1])
    cartesian_traj_z.append(cartesian_traj_z_[-1])

    # Extract behind finger pose from trajectory
    reverse_vec = [cartesian_traj_x[0] - cartesian_traj_x[1], cartesian_traj_y[0] - cartesian_traj_y[1]]
    for index, (finger_front_x, finger_front_y) in enumerate(zip(cartesian_traj_x, cartesian_traj_y)):        
        waypoint_n = index
        k = 1

        # Find finger_behind pose
        while(True):
            waypoint_n = waypoint_n - 1
            # if previous waypoint that satisfy condition not exist, reverse vector from inital x, y will be used.
            if waypoint_n < 0:
                finger_behind_x = cartesian_traj_x[0] + k * reverse_vec[0]
                finger_behind_y = cartesian_traj_y[0] + k * reverse_vec[1]
                k = k + 1
                if k > 10000:
                    raise ValueError("alpha value must be larger")
            else:
                finger_behind_x = cartesian_traj_x[waypoint_n]
                finger_behind_y = cartesian_traj_y[waypoint_n]
            
            # assume that second finger lies on passed trajectory 
            if abs(distance(finger_front_x, finger_front_y, finger_behind_x, finger_behind_y) - effective_raius) < alpha:
                finger_behind = [finger_front_x + math.cos(margin_angle)*(finger_behind_x - finger_front_x) - math.sin(margin_angle)*(finger_behind_y - finger_front_y),
                                 finger_front_y + math.sin(margin_angle)*(finger_behind_x - finger_front_x) + math.cos(margin_angle)*(finger_behind_y - finger_front_y)]
                finger_behind_x = finger_behind[0]
                finger_behind_y = finger_behind[1]
                break
        
        yaw = calculateAngle(finger_front_x - finger_behind_x, finger_front_y - finger_behind_y) 
        center_x, center_y = calculateGripperCP(finger_front_x, finger_front_y, finger_behind[0], finger_behind[1], gripper_radius, clock_wise)

        if(index%n == 0):
            traj_EE_x.append(center_x)
            traj_EE_y.append(center_y)
            traj_EE_z.append(cartesian_traj_z[index])

            quaternion = euler2Quaternion(0, 0, yaw)
            traj_EE_qx.append(quaternion[0])
            traj_EE_qy.append(quaternion[1])
            traj_EE_qz.append(quaternion[2])
            traj_EE_qw.append(quaternion[3])

            traj_BH_x.append(finger_behind_x)
            traj_BH_y.append(finger_behind_y)
            traj_BH_z.append(cartesian_traj_z[index])
    
    # if reverse == True:
    #     traj_EE_x.reverse()
    #     traj_EE_y.reverse()
    #     traj_EE_z.reverse()

    EETraj = PoseArray()
    BHTraj = PoseArray()

    for i in range(len(traj_EE_x)):
        # Create Pose for EETraj & BHTraj
        ee_pose = Pose()
        ee_pose.position.x = traj_EE_x[i]
        ee_pose.position.y = traj_EE_y[i]
        ee_pose.position.z = traj_EE_z[i]
        ee_pose.orientation.x = traj_EE_qx[i]
        ee_pose.orientation.y = traj_EE_qy[i]
        ee_pose.orientation.z = traj_EE_qz[i]
        ee_pose.orientation.w = traj_EE_qw[i]

        bh_pose = Pose()
        bh_pose.position.x = traj_BH_x[i]
        bh_pose.position.y = traj_BH_y[i]
        bh_pose.position.z = traj_BH_z[i]

        EETraj.poses.append(ee_pose)
        BHTraj.poses.append(bh_pose)

    return EETraj, BHTraj

