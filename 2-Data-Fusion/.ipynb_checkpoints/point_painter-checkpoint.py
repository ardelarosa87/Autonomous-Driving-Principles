import cv2
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T



class PointPainter():

    def __init__(self, P, R_0, R_t, conf=0.85, filename = 'coco_classes.json'):
        #Define input parameters
        self.P = P
        self.R_0 = R_0
        self.R_t = R_t
        self.conf = conf
        self.filename = filename
        
        #Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Load model 
        self.model = maskrcnn_resnet50_fpn_v2(MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.transform = T.ToTensor()
        #Set color mapping 
        self.__set_coco_class_and_color_map()
    
    def paintCloud(self, image, X):
        #Lets first get dections
        pred_masks, pred_labels, pred_scores = self.get_maskrcnn_predictions(image)
        #Get segmantic image and per pixel score
        semantic_image, per_pixel_score = self.get_semantic_image_and_per_pixel_score(image, pred_masks, pred_labels)
        #Decorate point cloud
        lidar_labels, lidar_color_map, lidar_image = self.decorate_point_cloud(image, X, per_pixel_score = per_pixel_score)
        #Now crete dictionary with everything
        point_painting_detections ={
            "pred_mask": pred_masks,
            "pred_labels": pred_labels,
            "pred_scores": pred_scores,
            "semantic_image": semantic_image,
            "per_pixel_score": per_pixel_score,
            "lidar_labels": lidar_labels,
            "lidar_color_map": lidar_color_map,
            "lidar_image": lidar_image

        }
        return point_painting_detections


    
    def get_maskrcnn_predictions(self, image):
        input_image = self.transform(image)
        self.model.eval()
        
        #Get predictions
        # Get predictions
        with torch.no_grad():
            prediction = self.model([input_image])
        # The prediction is a list of dictionaries, and we can extract the masks, labels, and boxes
        pred_masks = prediction[0]['masks'].cpu().squeeze().numpy()
        pred_labels = prediction[0]['labels'].cpu().numpy()
        pred_scores = prediction[0]['scores'].cpu().numpy()
        #Get only those images specefied by confidence desired
        high_conf_indices = np.where(pred_scores >= self.conf)[0]
        
        pred_masks = pred_masks[high_conf_indices]
        pred_labels = pred_labels[high_conf_indices]
        pred_scores = pred_scores[high_conf_indices]
        return pred_masks, pred_labels, pred_scores
    
    def get_semantic_image_and_per_pixel_score(self, image, pred_masks = None, pred_labels = None):
        # Get predictions

        # Check if pred masks and pred labels is None or is empty
        if pred_masks is None or not len(pred_masks) or pred_labels is None or not len(pred_labels):
            pred_masks, pred_labels, _= self.get_maskrcnn_predictions(image)
        # Create empty sematic map
        semantic_image = np.zeros_like(image)
        #Create empty per pixel class map 
        per_pixel_score = np.zeros_like(pred_masks[0])
        
        for mask, label in zip(pred_masks, pred_labels):
            #Threshold mask as output was float32 then convert to np.uint8
            mask = mask  > 0.5
            mask = mask.astype(np.uint8)
            semantic_image[mask == 1] = self.color_map[label]
            per_pixel_score[mask == 1] = label
        
        #Set any left over black pixels from 0 to calss 0 which is just background or noise
        # This will set all pixels that are black (0, 0, 0) to white (255, 255, 255)
        zeros = np.all(semantic_image == 0, axis=-1)
        semantic_image[zeros] = self.color_map[0]

        return semantic_image, per_pixel_score
    
    def decorate_point_cloud(self, image, X, per_pixel_score = None):
        self.X = X
        #Project points
        self.__project_points()
        #Set label map to zeros
        labels = np.zeros(self.Y.shape[0]).astype(np.uint16)
        #Set Y to be int only 
        Y_int = self.Y.astype(np.uint16)
        #Get pixel score it empty then get segmeantic image by infereing
        if per_pixel_score is None or not len(per_pixel_score):
            _, per_pixel_score = self.get_semantic_image_and_per_pixel_score(image)
        #Decorate lidar points
        for y in range(per_pixel_score.shape[0]):
            for x in range(per_pixel_score.shape[1]):
                #Get indices where any pixels in Y_frustum_int are the current y,x pixel
                indices = np.where((Y_int[:, 0] == x) & (Y_int[:, 1] == y))[0]
                labels[indices]  = per_pixel_score[y, x]
        #Get color map 
        lidar_color_map = np.array([self.color_map[label] for label in labels])
        #Get lidar image
        lidar_image = np.zeros_like(image)
        for i, (x, y) in enumerate(Y_int):
            #Draw a circle with radius 1 at (x, y) position
            cv2.circle(lidar_image, (x, y), radius=1, color=self.color_map[labels[i]], thickness=-1) 

        
        return labels, lidar_color_map, lidar_image
    
        
    
    def __project_points(self):
        #Convert to homogenous coord
        R_0_homogenous = np.concatenate((self.R_0, np.array([[0,0,0]])), axis=0)
        X_homogenous = np.concatenate((self.X, np.ones((self.X.shape[0], 1))), axis=1)
        #Transpose X
        X_homogenous_transpose = X_homogenous.T
        #Peform matrix multiplication
        Y_homogenous = (self.P@R_0_homogenous@self.R_t@X_homogenous_transpose).T

        self.Y = Y_homogenous[:, 0:2] / Y_homogenous[:, 2].reshape(-1,1)



    def __set_coco_class_and_color_map(self):
        #Load the dictionary from the JSON file
        with open(self.filename, 'r') as file:
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
        
        self.coco_classes = coco_classes
        self.color_map = class_colors


