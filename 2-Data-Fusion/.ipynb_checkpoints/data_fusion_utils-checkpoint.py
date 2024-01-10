import open3d as o3d 
import numpy as np


def visualize_cloud(obj_list):
    """
    Function modifies the Open3D visualizer for better display. 

    Parameters:
    obj_list (list): A list of objects to visualize this can be multiple point clouds, bounding box data, etc

    Returns:
    type: Void
    """
    
    # Create a visualizer object
    viz = o3d.visualization.Visualizer()
    viz.create_window()
    
    # Add each object to the vizualizer
    for obj in obj_list:
        viz.add_geometry(obj)
    
    #Set the background color to black
    opt = viz.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # RGB values for black
    
    #Run the visualizer
    viz.run()
    viz.destroy_window()



def RANSAC(point_cloud, distance_threshold=0.11, ransac_n=5, num_iterations=1000, downsample=False, voxel_size = 0.08):
    """
    Helper function for plane segmentation in open3d using ransac. 

    Parameters:
    point_cloud (obj): An open3D PointCloud obj
    distance_threshold (float): the maximum distance a point can have to an estimated plane to be considered an inlier
    ransac_n (int): the number of points that are randomly sampled to estimate a plane
    num_iterations (int):  how often a random plane is sampled and verified
    

    Returns:
    An point cloud object of the outlier cloud
    """
    #down smample cloud if voxel down sample
    if downsample:
        poin_cloud = point_cloud.voxel_down_sample(voxel_size=0.08)

    _, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
    return point_cloud.select_by_index(inliers, invert=True)
                                       


def project_points(P, R_0, R_t, X):
    """
    This function projects 3D lidar points to its 2D pixel coordinates  

    Parameters:
    P (numpy array): A 3x4 numpy array of the extrinsics and intrinsic parameters of the camera
    R_0 (numpy array): A 3x3 numpy array of the rectification matrix for stereo vision, if no stereo this should be the identity matrix
    R_t (numpy array): A 3x4 numpy array of the rotation and translation matrix to from lidar to camera coordinates
    X (numpy array): A Nx3 numpy array of the lidar points where N is the number of points
    
    

    Returns:
    A Nx2 numpy array "Y" of the lidar points in 3D projected to 2D  
    """
    #Convert to homogenous coord
    R_0_homogenous = np.concatenate((R_0, np.array([[0,0,0]])), axis=0)
    X_homogenous = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    #Transpose X
    X_homogenous_transpose = X_homogenous.T
    #Peform matrix multiplication
    Y_homogenous = (P@R_0_homogenous@R_t@X_homogenous_transpose).T

    return Y_homogenous[:, 0:2] / Y_homogenous[:, 2].reshape(-1,1)