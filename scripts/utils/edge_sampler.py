import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

class Edge(object):
    def __init__(self, edge_xyz, edge_uv):
        """Push contact point.
        Args:
            edge_xyz (numpy.ndarray): (N, 3) array of edge points in world frame.
            edge_uv (numpy.ndarray): (N, 2) array of edge points in image coordinates.
        """
        
        self.edge_xyz = edge_xyz
        self.edge_uv = edge_uv
    
    @property
    def center(self):
        return self.edge_xyz.mean(0)

    def visualize_on_image(self, image):
        # get random 1000 point on edge_uv
        rand_idx = np.random.randint(0, len(self.edge_uv), 1000)

        # visualize contact point
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap='gray')
        plt.plot(self.edge_uv[rand_idx, 0], self.edge_uv[rand_idx, 1], 'ko')
        plt.show()

    def visualize_on_cartesian(self):
        # get random 1000 point on edge_xy
        rand_idx = np.random.randint(0, len(self.edge_xyz), 1000)
        
        fig = plt.figure()
        # plot edge points
        plt.plot(self.edge_xyz[rand_idx, 0], self.edge_xyz[rand_idx, 1], 'ko')
        plt.show()   

    @property
    def sampled_edge_xy(self, sample:int=1000):
        # get random 1000 point on edge_xy
        rand_idx = np.random.randint(0, len(self.edge_xyz), sample)
        # plot edge points
        return self.edge_xyz[rand_idx, 0], self.edge_xyz[rand_idx, 1]
        
class EdgeSampler(object):
    '''
    Samples edge points from a masked depth image
    Masked depth image -> Edge points 
    
    '''
    def __init__(self, camera_intr, camera_extr):
        self.camera_intr = camera_intr
        self.camera_extr = camera_extr
        
    def sample(self, masked_depth_image):
        # Get point cloud of the object only
        pcd_object = self.depth_to_pcd(masked_depth_image, self.camera_intr)
        
        # Transform point cloud to world frame
        pcd_w = (np.matmul(self.camera_extr[:3,:3], pcd_object[:,:3].T) + self.camera_extr[:3,3].reshape(3,1)).T
        
        # Height Thresholding
        
        max_height = np.max(pcd_w[:,2]) - (np.max(pcd_w[:,2]) - np.min(pcd_w[:,2])) * 0.1
        pcd_w = pcd_w[np.where(pcd_w[:,2] < max_height)[0]]
        min_height = np.min(pcd_w[:,2]) + (np.max(pcd_w[:,2]) - np.min(pcd_w[:,2])) * 0.05
        pcd_w = pcd_w[np.where(pcd_w[:,2] > min_height)[0]]

        # Edge Detection - alpha shape algorithm
        pcd_w_2d = pcd_w[:,:2]
        hull = ConvexHull(pcd_w_2d)
        outermost_indices = hull.vertices
        
        # Interpolate
        num_interpolated_points = 1000
        outermost_indices = np.append(outermost_indices, outermost_indices[0])
        edge_list_xyz = self.interpolate_with_even_distance(pcd_w[outermost_indices], num_interpolated_points)

        # Get uv coordinates of the edge list
        edge_list_xyz_camera = (np.matmul(np.linalg.inv(self.camera_extr)[:3,:3], edge_list_xyz[:,:3].T) + np.linalg.inv(self.camera_extr)[:3,3].reshape(3,1)).T
        edge_list_uvd = edge_list_xyz_camera @ self.camera_intr.T
        edge_list_uv = edge_list_uvd[:,:2] / edge_list_uvd[:,2].reshape(-1,1)
        edge_list_uv = edge_list_uv.astype(int)
        
        return Edge(edge_list_xyz, edge_list_uv)

    @staticmethod
    def remove_outliers(array, threshold=3):
        # Calculate the mean and standard deviation of the array
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)

        # Calculate the Z-scores for each data point
        z_scores = np.abs((array - mean) / std)

        # Filter out the outliers based on the threshold
        filtered_array = array[(z_scores < threshold).all(axis=1)]

        return filtered_array

    @staticmethod
    def interpolate_with_even_distance(trajectory, num_sample):
        '''
        From a trajectory, interpolate the points with even Euclidean distances (xy-plane).
        
        Args:
            trajectory (N,3): Trajectory points
            num_sample (int): Number of points to be sampled
        Returns:
            interpolated_trajectory (num_sample,3): Interpolated trajectory points
        '''
        # Extract the x and y coordinates from the trajectory
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]

        # Compute the cumulative distance along the trajectory
        distances = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        distances = np.insert(distances, 0, 0)  # Prepend a zero for the initial position

        # Create an interpolation function for x and y coordinates
        interp_func_x = interp1d(distances, x, kind='linear')
        interp_func_y = interp1d(distances, y, kind='linear')
        interp_func_z = interp1d(distances, z, kind='linear')

        # Generate evenly spaced distances for the interpolated points
        target_distances = np.linspace(0, distances[-1], num_sample)

        # Interpolate the x and y coordinates at the target distances
        interpolated_x = interp_func_x(target_distances)
        interpolated_y = interp_func_y(target_distances)
        interpolated_z = interp_func_z(target_distances)

        # Return the interpolated x and y coordinates as a (m, 2) trajectory
        interpolated_trajectory = np.column_stack((interpolated_x, interpolated_y, interpolated_z))
        return interpolated_trajectory
    
    @staticmethod
    def depth_to_pcd(depth_image, camera_intr):
        height, width = depth_image.shape
        row_indices = np.arange(height)
        col_indices = np.arange(width)
        pixel_grid = np.meshgrid(col_indices, row_indices)
        pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
        pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
        depth_arr = np.tile(depth_image.flatten(), [3, 1])
        point_cloud = depth_arr * np.linalg.inv(camera_intr).dot(pixels_homog)
        return point_cloud.transpose()