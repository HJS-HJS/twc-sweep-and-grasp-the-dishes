import numpy as np

def depth2pcd(depth: np.array, intr: np.array) ->np.array:
    """Convert depth image to pointcloud data.

    Args:
        depth_image (np.array): (H, W) depth image to convert.
        intr (np.array): (3, 3) camera intrinsic matrix.
        extr (np.array): (3, 3) camera extrinsic matrix.

    Returns:
        np.array: (N, 3) pointcloud data array converted from depth image
    """
    height, width = depth.shape
    row_indices = np.arange(height)
    col_indices = np.arange(width)
    pixel_grid = np.meshgrid(col_indices, row_indices)
    pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
    pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
    depth_arr = np.tile(depth.flatten(), [3, 1])
    point_cloud = depth_arr * np.linalg.inv(intr).dot(pixels_homog)

    return point_cloud.transpose()
