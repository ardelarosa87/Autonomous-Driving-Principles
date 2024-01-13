# Part I: Projection of LiDAR Data from 3D to 2D Pixel Space 

In this first section, we delve into the process of projecting LiDAR data from a three-dimensional space into a two-dimensional pixel space. We will explore a range of techniques rooted in linear algebra and computer graphics to effectively achieve this transformation. This tutorial will encompass the understanding of essential concepts and the application of various computational methods to bridge the gap between 3D LiDAR data and 2D imagery.

### Projection formula

To derive a 2D pixel coordinate from a 3D LiDAR point, we use the following projection formula:

$$
Y = P \cdot R_0 \cdot [R | t] \cdot X
$$

Where:
- \(Y\) is the 2D pixel coordinate (Nx2 matrix, N being the number of points).
- \(P\) is a 3x4 matrix containing the 3x3 intrinsic matrix and a 1x4 extrinsic vector.
- \(R_0\) is the 3x3 Stereo Rectification Matrix.
- \([R | t]\) is a 3x4 matrix combining rotation and translation from the LiDAR to the camera coordinate frame.
- \(X\) is an Nx3 matrix of 3D points from the LiDAR sensor.
Using this projection formula we can now go between pixels in 2D from an images to lidar points in 3D. Below shows the results of projecting lidar points unto the images after applying RANSAC. 

<img src="lidar_projection.gif" alt="Lidar-Camera-Projection" />


# Part II: Data Fusion Application Via PointPainting

In this second section, we apply a similar concept to that in [PointPainting: Sequential Fusion for 3D Object Detection](https://arxiv.org/abs/1911.10150). Like the paper, I pass an image through a segmentation algorithm. Here, `maskrcnn_resnet50_fpn_v2` from PyTorch trained on the COCO dataset is used to obtain our segmentation mask and labels. These outputs are then utilized to decorate the point cloud, assigning a class to each point within the camera's frustum. All technical aspects of this section are complete, with documentation for the Jupyter notebook and helper Python functions and classes underway. Meanwhile, check out the results below for a glimpse of what to expect. Note: We focus solely on the data fusion aspect and not on the 3D object detection head using PointPillars, which might be a topic for future exploration.



### Point Painting Approach
![Alt text](../Doc_Images/PointPainting_Overview.png)


### My Implementation Results

#### Semantic Map and Enhanced LiDAR Point Clouds in 2D Visualization
<img src="semantic_scene_map.gif" alt="Lidar-Camera-Projection" />

#### Comprehensive 3D Point Cloud Enhancement Using Semantic Mapping
![Alt text](../Doc_Images/DATAFUSION_DOC_IMAGES/part_II_overall_stack.png)

