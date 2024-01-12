import open3d as o3d 
import numpy as np
import re
import json
import matplotlib.pyplot as plt


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

def get_semantic_image(img, pred_masks, pred_labels, color_map):
    #Create sematic map
    semantic_image = np.zeros_like(img)
    
    for mask, label in zip(pred_masks, pred_labels):
        #Threshold mask as output was float32 then convert to np.uint8
        mask = mask  > 0.5
        mask = mask.astype(np.uint8)
        semantic_image[mask == 1] = color_map[label]
    
    #Set any left over black pixels from 0 to calss 0 which is just background or noise
    # This will set all pixels that are black (0, 0, 0) to white (255, 255, 255)
    is_black = np.all(semantic_image == 0, axis=-1)
    semantic_image[is_black] = color_map[0]

    return semantic_image
    


def get_coco_classes_and_color_map():
    filename = 'coco_classes.json'

    #Load the dictionary from the JSON file
    with open(filename, 'r') as file:
        coco_classes = json.load(file)
    #Convert keys back to integers as they are loaded as string in json
    coco_classes = {int(k): v for k, v in coco_classes.items()}
    #Create custom color map
    #Manually define colors for specific classes the most likely to be found in the images
    specific_colors = {
        'person': (255, 0, 0),     # Red
        'bicycle': (0, 255, 0),    # Green
        'car': (0, 0, 255),        # Blue
        'truck': (255, 255, 0),     # Yellow
        'motorcyle': (0, 255, 255),
        'background': (255, 0, 255)  #purple 
    }
    
    # Generate a colormap for other classes
    num_classes = len(coco_classes)
    cmap = plt.colormaps['hsv'](np.linspace(0, 1, num_classes))
    
    # Create a dictionary to store colors
    class_colors = {}
    
    # Assign a unique color from the colormap or specific color to each class
    for class_id, class_name in coco_classes.items():
        if class_name in specific_colors:
            # Use the specific color for important classes
            class_colors[class_id] = specific_colors[class_name]
        else:
            # Normalize class ID to fit in the colormap index range
            normalized_id = class_id % num_classes
            # Convert RGBA to RGB for other classes
            rgba_color = cmap[normalized_id]
            rgb_color = (int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255))
            class_colors[class_id] = rgb_color
    return coco_classes, class_colors




def load_calibrations():
    #Calibrations paths 
    camera_calibration_path = "../DATA/calibrations/calib_cam_to_cam.txt"
    #Set path for rotation and translation matrix from velodyne to camera
    velodyne_to_camera_calibration_path = "../DATA/calibrations/calib_velo_to_cam.txt"
    #Patterns
    P_pattern = "P_rect_02"
    R_0_pattern = "R_rect_02"
    R_pattern = "R:"
    t_pattern = "T:"
    #Matrix init 
    P = None
    R_0 = None
    R = None
    t = None
    #Load camera calibrations matrices P and R 
    with open(camera_calibration_path) as file:
        for line in file:
            try:
                #First see if it is P using re.search
                if re.search(P_pattern, line):
                    #lets split line at spaces and end of line \n
                    split_list = re.split(f'[ \n]', line)
                    #Remove first element and last to contain just numbers
                    split_list.pop(0)
                    split_list.pop(-1)
                    float_numbers = [float(number) for number in split_list]
                    #Convert to numpy array and reshape to (3, 4 matrix)
                    P = np.array(float_numbers).reshape(3,4)
                    print(f"Matrix P {P.shape}:")
                    print(P)
                #Second see if it is R using re.search
                elif re.search(R_0_pattern, line):
                    #lets split line at spaces and end of line \n
                    split_list = re.split(f'[ \n]', line)
                    #Remove first element and last to contain just numbers
                    split_list.pop(0)
                    split_list.pop(-1)
                    float_numbers = [float(number) for number in split_list]
                    #Convert to numpy array and reshape to (3, 3 matrix)
                    R_0 = np.array(float_numbers).reshape(3,3)
                    print(f"Matrix R0 {R_0.shape}:")
                    print(R_0)
            except ValueError as ve:
                print(f"ValueError occurred: {ve}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        #load lidar-camera clibration matrix R|t
        with open(velodyne_to_camera_calibration_path) as file:
            for line in file:
                try:
                    #First see if it is R using re.search
                    if re.search(R_pattern, line):
                        #lets split line at spaces and end of line \n
                        split_list = re.split(f'[ \n]', line)
                        #Remove first element and last to contain just numbers
                        split_list.pop(0)
                        split_list.pop(-1)
                        float_numbers = [float(number) for number in split_list]
                        #Convert to numpy array and reshape to (3, 4 matrix)
                        R = np.array(float_numbers).reshape(3,3)
                        print(f"Matrix R {R.shape}:")
                        print(R)
                    #Second see if it is T using re.search
                    elif re.search(t_pattern, line):
                        #lets split line at spaces and end of line \n
                        split_list = re.split(f'[ \n]', line)
                        #Remove first element and last to contain just numbers
                        split_list.pop(0)
                        split_list.pop(-1)
                        float_numbers = [float(number) for number in split_list]
                        #Convert to numpy array and reshape to vector
                        t = np.array(float_numbers).reshape(-1,1)
                        print(f"Vector t {t.shape}:")
                        print(t)
                except ValueError as ve:
                    print(f"ValueError occurred: {ve}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
            #Let us now combine the rotation translation vector to give us R_t 3x4 
            R_t = np.concatenate((R, t), axis=1)
            print(f"Matrix R|t {R_t.shape}:")
            print(R_t)
        return P, R_0, R_t